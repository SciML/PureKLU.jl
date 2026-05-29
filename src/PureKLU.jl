"""
    PureKLU

Pure-Julia translation of SuiteSparse's KLU sparse LU solver. Mirrors
the public API of `KLU.jl` (the ccall-based wrapper) so existing code
can swap modules without changes, but contains no foreign-language
dependencies.

The implementation is a direct port of `BTF`, the `klu_kernel` /
`klu_factor` / `klu_refactor` / `klu_solve` / `klu_tsolve` and the
symbolic-analysis routines from SuiteSparse. AMD/COLAMD per-block
ordering is currently approximated by natural ordering; blocks of size
≤3 use natural ordering identically to SuiteSparse.
"""
module PureKLU

using SparseArrays
using SparseArrays: SparseMatrixCSC
using LinearAlgebra
using MuladdMacro: @muladd  # kept for future opt-in FMA; see src/Kernel.jl
# Full `using` (not a selective `using PrecompileTools: ...`) so the bare
# `PrecompileTools` name is in scope: the `@setup_workload`/`@compile_workload`
# macro expansion in PrecompileTools 1.0.x references it directly, so a
# selective import breaks loading on the lower compat bound (see Downgrade CI).
using PrecompileTools
import SparseArrays: nnz, nonzeros
import Base: (\), size, getproperty, setproperty!, propertynames, show

export klu, klu!
export klu_factor!, klu_refactor!, klu_analyze!, solve!
export nonzeros

include("Common.jl")
include("BTF.jl")
include("AMD.jl")
include("Symbolic.jl")
include("Kernel.jl")
include("Solve.jl")

# Type aliases / shims for KLU.jl-style symbols expected by tests.
const klu_common = KLUCommon{Int32}
const klu_l_common = KLUCommon{Int64}
const klu_symbolic = KLUSymbolic{Int32}
const klu_l_symbolic = KLUSymbolic{Int64}
const klu_numeric = KLUNumeric{Float64, Int32, Float64}
const klu_l_numeric = KLUNumeric{Float64, Int64, Float64}

const AdjointFact = isdefined(LinearAlgebra, :AdjointFactorization) ? LinearAlgebra.AdjointFactorization : Adjoint
const TransposeFact = isdefined(LinearAlgebra, :TransposeFactorization) ? LinearAlgebra.TransposeFactorization : Transpose

# `KLUTypes` is the legacy KLU.jl alias; `klu(...)` actually dispatches
# on `KLUGenericTypes`, which is broad enough to cover `ForwardDiff.Dual`
# (`<:Real` but not `<:AbstractFloat`) and any other user number type.
const KLUTypes = Union{Float64, ComplexF64}
const KLUGenericTypes = Union{Real, Complex}

if sizeof(Int) == 4
    const KLUITypes = Int32
    const KLUIndexTypes = (:Int32,)
else
    const KLUITypes = Union{Int32, Int64}
    const KLUIndexTypes = (:Int32, :Int64)
end

# Convert from 1-based to 0-based indices
function decrement!(A::AbstractArray{T}) where {T <: Integer}
    for i in eachindex(A)
        A[i] -= oneunit(T)
    end
    return A
end
decrement(A::AbstractArray{<:Integer}) = decrement!(copy(A))

function increment!(A::AbstractArray{T}) where {T <: Integer}
    for i in eachindex(A)
        A[i] += oneunit(T)
    end
    return A
end
increment(A::AbstractArray{<:Integer}) = increment!(copy(A))

function kluerror(status::KLUStatus)
    if status == KLU_OK
        return
    elseif status == KLU_SINGULAR
        throw(LinearAlgebra.SingularException(0))
    elseif status == KLU_OUT_OF_MEMORY
        throw(OutOfMemoryError())
    elseif status == KLU_INVALID
        throw(ArgumentError("Invalid Status"))
    elseif status == KLU_TOO_LARGE
        throw(OverflowError("Integer overflow has occurred"))
    else
        throw(ErrorException("Unknown KLU error code: $status"))
    end
end

kluerror(common::KLUCommon) = kluerror(common.status)

abstract type AbstractKLUFactorization{Tv, Ti} <: LinearAlgebra.Factorization{Tv} end

"""
    KLUFactorization{Tv, Ti}

Matrix factorization type for KLU. Constructed by [`klu`](@ref).
Mirrors the field names of `KLU.jl`'s `KLUFactorization` so the same
property accessors (`F.L`, `F.U`, `F.F`, `F.p`, `F.q`, `F.Rs`, …) work.
"""
mutable struct KLUFactorization{Tv, Ti <: Integer, Tr <: Real} <: AbstractKLUFactorization{Tv, Ti}
    common::KLUCommon{Ti}
    # `symbolic.n == 0` is the "not yet analyzed" sentinel; `numeric.n ==
    # 0` is "not yet factored".  Concrete fields (no `Union{..., Nothing}`)
    # so hot-path accesses are type-stable.  `Tr` is a third type
    # parameter for the same reason on `numeric`.
    symbolic::KLUSymbolic{Ti}
    numeric::KLUNumeric{Tv, Ti, Tr}
    n::Int
    colptr::Vector{Ti}    # 0-based
    rowval::Vector{Ti}    # 0-based
    nzval::Vector{Tv}

    function KLUFactorization(
            n::Integer, colptr::Vector{Ti}, rowval::Vector{Ti},
            nzval::Vector{Tv}
        ) where {Ti <: Integer, Tv}
        Tr = _real_eltype(Tv)
        common = KLUCommon{Ti}()
        sym = KLUSymbolic{Ti}()
        num = KLUNumeric{Tv, Ti, Tr}()
        return new{Tv, Ti, Tr}(common, sym, num, Int(n), colptr, rowval, nzval)
    end
end

function KLUFactorization(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <: Integer}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("KLU only accepts square matrices."))
    return KLUFactorization(n, decrement(A.colptr), decrement(A.rowval), copy(A.nzval))
end

size(K::AbstractKLUFactorization) = (K.n, K.n)
function size(K::AbstractKLUFactorization, dim::Integer)
    if dim < 1
        throw(ArgumentError("size: dimension $dim out of range"))
    elseif dim == 1 || dim == 2
        return Int(K.n)
    else
        return 1
    end
end

nnz(K::AbstractKLUFactorization) = K.lnz + K.unz + K.nzoff
"""The nonzeros of the factorization `K`."""
nonzeros(K::AbstractKLUFactorization) = K.nzval

if !isdefined(LinearAlgebra, :AdjointFactorization)
    Base.adjoint(K::AbstractKLUFactorization) = Adjoint(K)
end
Base.transpose(K::AbstractKLUFactorization) = TransposeFact(K)

# --- analyze / factor / refactor entry points -----------------------------

_is_analyzed(K::KLUFactorization) = getfield(K, :symbolic).n != 0
_is_factored(K::KLUFactorization) = getfield(K, :numeric).n != 0

"""
    klu_analyze!(K) -> K
    klu_analyze!(K, P, Q) -> K

Run the symbolic analysis (BTF + per-block ordering) on `K`, storing the
result in `K.symbolic`. Subsequent factorisations reuse it. The two-arg
form accepts user-supplied row/column permutations.
"""
function klu_analyze!(K::KLUFactorization{Tv, Ti}; check::Bool = true) where {Tv, Ti}
    _is_analyzed(K) && return K
    Sym = klu_analyze(K.n, K.colptr, K.rowval, K.common)
    if Sym === nothing && check
        kluerror(K.common)
    elseif Sym !== nothing
        setfield!(K, :symbolic, Sym)
    end
    return K
end

function klu_analyze!(
        K::KLUFactorization{Tv, Ti}, P::Vector{Ti}, Q::Vector{Ti};
        check::Bool = true
    ) where {Tv, Ti}
    _is_analyzed(K) && return K
    Sym = klu_analyze(K.n, K.colptr, K.rowval, K.common; given_P = P, given_Q = Q)
    if Sym === nothing && check
        kluerror(K.common)
    elseif Sym !== nothing
        setfield!(K, :symbolic, Sym)
    end
    return K
end

"""
    klu_factor!(K; check=true, allowsingular=false) -> K

Numeric factorisation. Runs `klu_analyze!` first if it hasn't already
been done. Mirrors `KLU.jl::klu_factor!`.
"""
function klu_factor!(
        K::KLUFactorization{Tv, Ti}; check::Bool = true,
        allowsingular::Bool = false
    ) where {Tv, Ti}
    if !_is_analyzed(K) && K.common.status >= KLU_OK
        klu_analyze!(K; check)
    end
    if _is_analyzed(K) && K.common.status >= KLU_OK
        K.common.halt_if_singular = (!allowsingular && check) ? Cint(1) : Cint(0)
        # Reuse the existing numeric struct (and its preallocated block
        # capacities + kernel workspace) when possible so subsequent
        # `klu_factor!` calls on the same `KLUFactorization` allocate
        # nothing beyond the (possible) geometric grow path in the kernel.
        Sym = getfield(K, :symbolic)
        existing = getfield(K, :numeric)
        Num = klu_factor!(
            Sym, K.colptr, K.rowval, K.nzval, K.common,
            existing; allowsingular
        )
        K.common.halt_if_singular = Cint(1)
        if K.common.status < KLU_OK && check
            kluerror(K.common)
        end
        if K.common.status == KLU_SINGULAR && !allowsingular && check
            kluerror(K.common)
        end
        setfield!(K, :numeric, Num)
    else
        if check
            kluerror(K.common)
        end
    end
    return K
end

"""
    klu!(K::KLUFactorization, nzval::Vector; check=true, allowsingular=false) -> K
    klu!(K::KLUFactorization, S::SparseMatrixCSC; check=true, allowsingular=false) -> K

Refactor an already-factored `K` in place with new numeric values that share its
existing sparsity pattern, reusing the symbolic analysis and preallocated numeric
workspace. `K` must already have been factored (via [`klu`](@ref) or
[`klu_factor!`](@ref)). New values are given either as a `nzval` vector matching the
length of `K`'s stored values, or as a `SparseMatrixCSC` whose pattern matches `K`
(a mismatched pattern throws). Mirrors `KLU.jl`'s `klu!`.
"""
function klu!(
        K::KLUFactorization{Tv, Ti}, nzval::Vector{Tv};
        check::Bool = true, allowsingular::Bool = false
    ) where {Tv, Ti}
    length(nzval) != length(K.nzval) && throw(DimensionMismatch())
    _is_factored(K) || throw(ArgumentError("KLUFactorization has not been factored yet. Call `klu_factor!` first."))
    K.nzval = nzval
    K.common.halt_if_singular = (!allowsingular && check) ? Cint(1) : Cint(0)
    klu_refactor!(
        getfield(K, :symbolic), getfield(K, :numeric),
        K.colptr, K.rowval, K.nzval, K.common; allowsingular
    )
    K.common.halt_if_singular = Cint(1)
    if K.common.status < KLU_OK && check
        kluerror(K.common)
    end
    if K.common.status == KLU_SINGULAR && !allowsingular && check
        kluerror(K.common)
    end
    return K
end

# Eltype-mismatched fallback: convert rather than throw.
function klu!(
        K::KLUFactorization{Tv, Ti}, nzval::Vector{U};
        check::Bool = true, allowsingular::Bool = false
    ) where {Tv, Ti, U}
    return klu!(K, convert(Vector{Tv}, nzval); check, allowsingular)
end

function klu!(
        K::KLUFactorization{Tv, Ti}, S::SparseMatrixCSC;
        check::Bool = true, allowsingular::Bool = false
    ) where {Tv, Ti}
    size(K) == size(S) || throw(ArgumentError("Sizes of K and S must match."))
    increment!(K.colptr); increment!(K.rowval)
    pattern_ok = K.colptr == S.colptr && K.rowval == S.rowval
    decrement!(K.colptr); decrement!(K.rowval)
    pattern_ok || throw(ArgumentError("The pattern of the original matrix must match the pattern of the refactor."))
    return klu!(K, S.nzval; check, allowsingular)
end

"""
    klu_refactor!(K, vals; check=true, allowsingular=false) -> K

Alias for [`klu!`](@ref): refactor an existing factorization in place with new
numeric values that share the original sparsity pattern. Provided for naming
parity with the analyze/factor/refactor entry points.
"""
klu_refactor!(args...; kwargs...) = klu!(args...; kwargs...)

# The `use_fma` kwarg accepts either a `Bool` or a `Val{true}/Val{false}`.
# Both are normalised to a `Val` internally (via `_as_val`) and stored on
# `K.common.use_fma`, so dispatch through the kernel hot loops is type-
# stable end-to-end.
#
#   * `use_fma=true` (default) and `use_fma=Val(true)` are equivalent.
#   * `use_fma=false` and `use_fma=Val(false)` opt out of FMA fusion and
#     give bit-for-bit results identical to SuiteSparse `KLU.jl`.
"""
    klu(A; check=true, allowsingular=false, full_factor=true) -> K
    klu(n, colptr, rowval, nzval; ...) -> K

Compute the LU factorisation of a sparse matrix using KLU.
"""
function klu(
        n::Integer, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv};
        check::Bool = true, allowsingular::Bool = false,
        full_factor::Bool = true, use_fma = true,
        fully_preallocated::Union{Bool, Nothing} = nothing,
        detect_banded::Bool = false,
    ) where {Ti <: KLUITypes, Tv <: KLUGenericTypes}
    K = KLUFactorization(n, colptr, rowval, nzval)
    K.common.use_fma = _as_val(use_fma)
    K.common.detect_banded = detect_banded
    if fully_preallocated isa Bool
        K.common.fully_preallocated = fully_preallocated
        return full_factor ? klu_factor!(K; check, allowsingular) : klu_analyze!(K; check)
    end
    klu_analyze!(K; check)
    K.common.fully_preallocated = Int(getfield(K, :symbolic).maxblock) <= AUTO_PREALLOC_MAXBLOCK
    return full_factor ? klu_factor!(K; check, allowsingular) : K
end

function klu(
        A::SparseMatrixCSC{Tv, Ti}; check::Bool = true,
        allowsingular::Bool = false, full_factor::Bool = true,
        use_fma = true, fully_preallocated::Union{Bool, Nothing} = nothing,
        detect_banded::Bool = false,
    ) where {Tv <: KLUGenericTypes, Ti <: KLUITypes}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch())
    return klu(
        n, decrement(A.colptr), decrement(A.rowval), A.nzval;
        check, allowsingular, full_factor, use_fma, fully_preallocated, detect_banded
    )
end

# --- solve API -------------------------------------------------------------

"""
    solve!(K, B; check=true) -> B

In-place solve. `B` is overwritten with the solution.
"""
function solve!(
        K::AbstractKLUFactorization{Tv, Ti}, B::StridedVecOrMat{Tv};
        check::Bool = true
    ) where {Tv, Ti}
    stride(B, 1) == 1 || throw(ArgumentError("B must have unit strides"))
    _is_factored(K) || klu_factor!(K)
    size(B, 1) == size(K, 1) || throw(DimensionMismatch())
    klu_solve!(getfield(K, :symbolic), getfield(K, :numeric), B, K.common)
    if K.common.status != KLU_OK && check
        kluerror(K.common)
    end
    return B
end

function solve!(
        K::AdjointFact{Tv, KF}, B::StridedVecOrMat{Tv};
        check::Bool = true
    ) where {Tv, Ti, KF <: AbstractKLUFactorization{Tv, Ti}}
    parent_K = parent(K)
    stride(B, 1) == 1 || throw(ArgumentError("B must have unit strides"))
    _is_factored(parent_K) || klu_factor!(parent_K)
    size(B, 1) == size(parent_K, 1) || throw(DimensionMismatch())
    klu_tsolve!(
        getfield(parent_K, :symbolic), getfield(parent_K, :numeric),
        B, parent_K.common; conj_solve = (Tv <: Complex)
    )
    if parent_K.common.status != KLU_OK && check
        kluerror(parent_K.common)
    end
    return B
end

function solve!(
        K::TransposeFact{Tv, KF}, B::StridedVecOrMat{Tv};
        check::Bool = true
    ) where {Tv, Ti, KF <: AbstractKLUFactorization{Tv, Ti}}
    parent_K = parent(K)
    stride(B, 1) == 1 || throw(ArgumentError("B must have unit strides"))
    _is_factored(parent_K) || klu_factor!(parent_K)
    size(B, 1) == size(parent_K, 1) || throw(DimensionMismatch())
    klu_tsolve!(
        getfield(parent_K, :symbolic), getfield(parent_K, :numeric),
        B, parent_K.common; conj_solve = false
    )
    if parent_K.common.status != KLU_OK && check
        kluerror(parent_K.common)
    end
    return B
end

solve(K, B; check::Bool = true) = solve!(K, copy(B); check)

LinearAlgebra.ldiv!(K::AbstractKLUFactorization{Tv}, B::StridedVecOrMat{Tv}) where {Tv} =
    solve!(K, B)
LinearAlgebra.ldiv!(K::Union{AdjointFact{Tv, KF}, TransposeFact{Tv, KF}}, B::StridedVecOrMat{Tv}) where {Tv, Ti, KF <: AbstractKLUFactorization{Tv, Ti}} =
    solve!(K, B)
function LinearAlgebra.ldiv!(
        K::AbstractKLUFactorization{<:AbstractFloat},
        B::StridedVecOrMat{<:Complex}
    )
    imagX = solve(K, imag(B))
    realX = solve(K, real(B))
    return map!(complex, B, realX, imagX)
end
function LinearAlgebra.ldiv!(
        K::Union{AdjointFact{Tv, KF}, TransposeFact{Tv, KF}},
        B::StridedVecOrMat{<:Complex}
    ) where {Tv <: AbstractFloat, Ti, KF <: AbstractKLUFactorization{Tv, Ti}}
    imagX = solve(K, imag(B))
    realX = solve(K, real(B))
    return map!(complex, B, realX, imagX)
end

# Generic Base.\ falls through to ldiv! via LinearAlgebra.Factorization.

function LinearAlgebra.issuccess(K::AbstractKLUFactorization; allowsingular::Bool = false)
    return (allowsingular ? K.common.status >= KLU_OK : K.common.status == KLU_OK) &&
        _is_factored(K)
end

# --- property access (mirrors KLU.jl) --------------------------------------

function Base.propertynames(::AbstractKLUFactorization, private::Bool = false)
    publicnames = (:lnz, :unz, :nzoff, :L, :U, :F, :q, :p, :Rs, :symbolic, :numeric)
    privatenames = (:nblocks, :maxblock)
    if private
        return (publicnames..., privatenames...)
    else
        return publicnames
    end
end

function getproperty(K::AbstractKLUFactorization{Tv, Ti}, s::Symbol) where {Tv, Ti}
    if s ∈ (:lnz, :unz, :nzoff)
        _is_factored(K) || throw(ArgumentError("This KLUFactorization has not yet been factored. Try `klu_factor!`."))
        Num = getfield(K, :numeric)
        s === :lnz && return Int(Num.lnz)
        s === :unz && return Int(Num.unz)
        s === :nzoff && return Int(Num.nzoff)
    end
    if s ∈ (:nblocks, :maxblock)
        _is_analyzed(K) || throw(ArgumentError("This KLUFactorization has not yet been analyzed. Try `klu_analyze!`."))
        Sym = getfield(K, :symbolic)
        s === :nblocks && return Int(Sym.nblocks)
        s === :maxblock && return Int(Sym.maxblock)
    end
    if s === :symbolic
        _is_analyzed(K) || throw(ArgumentError("This KLUFactorization has not yet been analyzed. Try `klu_analyze!`."))
        return getfield(K, :symbolic)
    end
    if s === :numeric
        _is_factored(K) || throw(ArgumentError("This KLUFactorization has not yet been factored. Try `klu_factor!`."))
        return getfield(K, :numeric)
    end
    if s ∉ (:L, :U, :F, :p, :q, :R, :Rs)
        return getfield(K, s)
    end

    _is_analyzed(K) || throw(ArgumentError("Not yet analyzed."))
    _is_factored(K) || throw(ArgumentError("Not yet factored."))
    Sym = getfield(K, :symbolic); Num = getfield(K, :numeric)

    if s === :p
        out = copy(Num.Pnum); out .+= one(Ti); return out
    elseif s === :q
        out = copy(Sym.Q); out .+= one(Ti); return out
    elseif s === :R
        nb = Int(Sym.nblocks)
        out = Sym.R[1:(nb + 1)]; out .+= one(Ti); return out
    elseif s === :Rs
        Tr = _real_eltype(Tv)
        return isempty(Num.Rs) ? fill(one(Tr), K.n) : copy(Num.Rs)
    elseif s === :L
        return _extract_L(K)
    elseif s === :U
        return _extract_U(K)
    elseif s === :F
        return _extract_F(K)
    end
    return getfield(K, s)
end

function setproperty!(K::AbstractKLUFactorization, s::Symbol, x)
    return setfield!(K, s, x)
end

function _extract_L(K::AbstractKLUFactorization{Tv, Ti}) where {Tv, Ti}
    Sym = K.symbolic; Num = K.numeric
    # KLU.jl's _extract! calls klu_sort, which sorts the internal L and U
    # storage in place. Match that behaviour so subsequent refactors see
    # identical FP-accumulation order to KLU.jl's.
    klu_sort!(Sym, Num)
    n = Int(Sym.n); nblocks = Int(Sym.nblocks)
    nz = Int(Num.lnz)
    Lp = Vector{Ti}(undef, n + 1)
    Li = Vector{Ti}(undef, nz)
    Lx = Vector{Tv}(undef, nz)
    p = 0
    for block in 1:nblocks
        k1 = Int(Sym.R[block]); k2 = Int(Sym.R[block + 1])
        nk = k2 - k1
        if nk == 1
            Lp[k1 + 1] = Ti(p)
            Li[p + 1] = Ti(k1)
            Lx[p + 1] = one(Tv)
            p += 1
        else
            bk = Num.LUbx[block]
            for kk in 0:(nk - 1)
                Lp[k1 + kk + 1] = Ti(p)
                col_start = p
                Li[p + 1] = Ti(k1 + kk)  # unit diagonal first
                Lx[p + 1] = one(Tv)
                p += 1
                lip = Int(Num.Lip[k1 + kk + 1])
                llen = Int(Num.Llen[k1 + kk + 1])
                for q in 0:(llen - 1)
                    Li[p + 1] = Ti(k1 + Int(bk.Li[lip + q + 1]))
                    Lx[p + 1] = bk.Lx[lip + q + 1]
                    p += 1
                end
                # sort by row within the column
                _sort_col!(Li, Lx, col_start + 1, p)
            end
        end
    end
    Lp[n + 1] = Ti(p)
    return SparseMatrixCSC(n, n, increment!(Lp), increment!(Li), Lx)
end

function _extract_U(K::AbstractKLUFactorization{Tv, Ti}) where {Tv, Ti}
    Sym = K.symbolic; Num = K.numeric
    klu_sort!(Sym, Num)
    n = Int(Sym.n); nblocks = Int(Sym.nblocks)
    nz = Int(Num.unz)
    Up = Vector{Ti}(undef, n + 1)
    Ui = Vector{Ti}(undef, nz)
    Ux = Vector{Tv}(undef, nz)
    p = 0
    for block in 1:nblocks
        k1 = Int(Sym.R[block]); k2 = Int(Sym.R[block + 1])
        nk = k2 - k1
        if nk == 1
            Up[k1 + 1] = Ti(p)
            Ui[p + 1] = Ti(k1)
            Ux[p + 1] = Num.Udiag[k1 + 1]
            p += 1
        else
            bk = Num.LUbx[block]
            for kk in 0:(nk - 1)
                Up[k1 + kk + 1] = Ti(p)
                col_start = p
                uip = Int(Num.Uip[k1 + kk + 1])
                ulen = Int(Num.Ulen[k1 + kk + 1])
                for q in 0:(ulen - 1)
                    Ui[p + 1] = Ti(k1 + Int(bk.Ui[uip + q + 1]))
                    Ux[p + 1] = bk.Ux[uip + q + 1]
                    p += 1
                end
                Ui[p + 1] = Ti(k1 + kk)
                Ux[p + 1] = Num.Udiag[k1 + kk + 1]
                p += 1
                # sort by row within the column
                _sort_col!(Ui, Ux, col_start + 1, p)
            end
        end
    end
    Up[n + 1] = Ti(p)
    return SparseMatrixCSC(n, n, increment!(Up), increment!(Ui), Ux)
end

function _sort_col!(I::Vector, X::Vector, lo::Int, hi::Int)
    lo >= hi && return nothing
    perm = sortperm(view(I, lo:hi))
    Iv = I[lo:hi][perm]
    Xv = X[lo:hi][perm]
    @inbounds for i in 1:length(perm)
        I[lo + i - 1] = Iv[i]
        X[lo + i - 1] = Xv[i]
    end
    return nothing
end

function _extract_F(K::AbstractKLUFactorization{Tv, Ti}) where {Tv, Ti}
    Sym = K.symbolic; Num = K.numeric
    n = Int(Sym.n)
    nzoff = Int(Num.nzoff)
    if nzoff == 0
        Fp = zeros(Ti, n + 1)
        Fi = Ti[]
        Fx = Tv[]
        return SparseMatrixCSC(n, n, increment!(Fp), increment!(Fi), Fx)
    end
    Fp = copy(Num.Offp)
    Fi = copy(Num.Offi)
    Fx = copy(Num.Offx)
    # KLU's F is not sorted, sort columns
    for col in 1:n
        first = Int(Fp[col]) + 1
        last = Int(Fp[col + 1])
        first > last && continue
        Fiview = view(Fi, first:last)
        Fxview = view(Fx, first:last)
        perm = sortperm(Fiview)
        Fiview .= Fiview[perm]
        Fxview .= Fxview[perm]
    end
    return SparseMatrixCSC(n, n, increment!(Fp), increment!(Fi), Fx)
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, K::AbstractKLUFactorization)
    summary(io, K); println(io)
    return if _is_factored(K)
        println(io, "L factor:")
        show(io, mime, K.L)
        println(io, "\nU factor:")
        show(io, mime, K.U)
        F = K.F
        if F !== nothing
            println(io, "\nF factor:")
            show(io, mime, K.F)
        end
    else
        println(io, "Incomplete Factorization, please try `klu_factor!(K)`.")
    end
end

# Precompile the four BLAS eltypes (Float64, Float32, ComplexF64, ComplexF32)
# crossed with the standard index types (Int32, Int64). Generic Real / Complex
# (e.g. ForwardDiff.Dual, BigFloat) intentionally JIT on first use.
@setup_workload begin
    _itypes = sizeof(Int) == 4 ? (Int32,) : (Int32, Int64)
    @compile_workload begin
        for Tv in (Float64, Float32, ComplexF64, ComplexF32)
            for Ti in _itypes
                colptr = Ti[1, 2, 3, 4, 5, 6, 7, 8, 9]
                rowval = Ti[1, 2, 3, 4, 5, 6, 7, 8]
                nzval = ones(Tv, 8)
                A = SparseMatrixCSC{Tv, Ti}(8, 8, colptr, rowval, nzval)
                K = klu(A)
                b = ones(Tv, 8)
                solve!(K, copy(b))
                solve!(K', copy(b))
                solve!(transpose(K), copy(b))
                klu!(K, A.nzval)
                klu(A; fully_preallocated = true)
                klu(A; use_fma = false)
            end
        end
    end
end

end # module
