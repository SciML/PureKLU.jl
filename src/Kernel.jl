# ----------------------------------------------------------------------
# FMA toggle
# ----------------------------------------------------------------------
# Every kernel hot loop computes the multiply-subtract update
#     X[i] = X[i] - L_ij * y_j
# via `_mulsub(X[i], L_ij, y_j, fma_val)`, where `fma_val::Val{Bool}` is
# threaded through from `common.use_fma`.  This lets the compiler
# specialise each loop on a single instruction:
#
#   * `Val(true)`  ->  `muladd(-b, c, a)`, which LLVM lowers to a single
#                      `vfnmadd*sd` (FMA) on hardware with FMA support.
#                      Slightly faster and one less rounding error per
#                      iteration; results differ from `KLU.jl` by up to
#                      one ULP per scalar update.
#   * `Val(false)` ->  `a - b*c`, two separate roundings; bit-for-bit
#                      identical to SuiteSparse's `libklu.so` binary
#                      (which is built SSE2-only, see the long note in
#                      `_kernel_lsolve_numeric!`).
#
# Default is `Val(true)`.  Use `klu(A; use_fma=false)` or set
# `K.common.use_fma = false` before factoring/solving to opt out.
@inline _mulsub(a, b, c, ::Val{true})  = muladd(-b, c, a)
@inline _mulsub(a, b, c, ::Val{false}) = a - b*c

# Complex specialisations.  Julia's built-in `*(::Complex, ::Complex)`
# uses `muladd` internally, so plain `a - b*c` would silently emit FMA
# for complex arguments even when `Val(false)`.  These explicit forms
# mirror SuiteSparse's `MULT_SUB(c, a, b)` macro for complex `Entry`
# (see KLU/Include/klu_version.h): four bare `mul`s followed by two
# `sub`s, in the same operand order as the C code.
@inline function _mulsub(a::Complex, b::Complex, c::Complex, ::Val{false})
    ar = real(a); ai = imag(a)
    br = real(b); bi = imag(b)
    cr = real(c); ci = imag(c)
    return Complex(ar - (br*cr - bi*ci),
                   ai - (bi*cr + br*ci))
end
# For Val(true) we *want* the FMA fusion that Julia's `*` already provides,
# so the generic `a - b*c` is fine (it'll route through complex_mul +
# subtraction, with muladd inside).

# Complex abs.  KLU uses `SuiteSparse_hypot` which computes
# `x * sqrt(1 + (y/x)^2)` (with x = max(|a.Real|, |a.Imag|)).  Julia's
# `hypot(::Float64, ::Float64)` uses a more accurate algorithm that
# differs by 1 ULP for inputs like `hypot(3, 2)` (Julia returns
# `sqrt(13)` exactly; SuiteSparse returns one ULP higher).  Mirror
# SuiteSparse exactly when `Val(false)`; let Julia's better algorithm
# run when `Val(true)`.
@inline _ssabs(a, ::Val) = abs(a)
@inline _ssabs(a::Real, ::Val) = abs(a)
@inline function _ssabs(a::Complex, ::Val{false})
    x = abs(real(a)); y = abs(imag(a))
    if x >= y
        x + y == x && return x
        r = y / x
        return x * sqrt(1.0 + r*r)
    else
        y + x == y && return y
        r = x / y
        return y * sqrt(1.0 + r*r)
    end
end
@inline _ssabs(a::Complex, ::Val{true}) = abs(a)

# Complex division.  KLU's `DIV` macro for complex uses ACM Algorithm 116
# (R. L. Smith, 1962) verbatim; Julia's `/(::ComplexF64, ::ComplexF64)` uses
# the same algorithm but adds over/underflow scaling and replaces the two
# `numerator / den` divisions with a single `t = 1/den` followed by two
# multiplications.  For normal-magnitude inputs that's still a different
# rounding sequence, costing 1 ULP.  When `Val(false)`, we use exactly
# KLU's macro expansion; when `Val(true)`, we let Julia's faster default
# do its FMA-friendly thing.
@inline _cdiv(a, b, ::Val) = a / b
@inline _cdiv(a::Complex, b::Complex, ::Val{true}) = a / b

# Per-nonzero scale application.  `scale_val::Val{Bool}` is
# `Val(common.scale > 0)`; threading it as a `Val` lets the compiler
# specialise away the `if scale > 0` branch in the inner construction
# loop.  Always uses direct division to keep bit-for-bit parity with
# KLU.jl (`x / y` is one rounding; `x * inv(y)` is two).
@inline _scale_aik(ax, _rs, ::Val{false}) = ax
@inline _scale_aik(ax, rs::Real, ::Val{true}) = ax / rs

@inline _rs_at(_, _, ::Val{false}) = nothing
@inline _rs_at(v, i, ::Val{true}) = @inbounds v[i]
@inline function _cdiv(a::Complex, b::Complex, ::Val{false})
    ar = real(a); ai = imag(a)
    br = real(b); bi = imag(b)
    if abs(br) >= abs(bi)
        r = bi / br
        den = br + r * bi
        return Complex((ar + ai * r) / den, (ai - ar * r) / den)
    else
        r = br / bi
        den = r * br + bi
        return Complex((ar * r + ai) / den, (ai * r - ar) / den)
    end
end

"""
    KernelWorkspace{Tv, Ti}

Scratch storage used by a single block's LU kernel.  `firstrow_ref` is
persistent across blocks so `_kernel_lpivot!` can update / read it
without allocating a fresh `Ref` per column.
"""
mutable struct KernelWorkspace{Tv, Ti}
    Pinv::Vector{Ti}
    Stack::Vector{Ti}
    Flag::Vector{Ti}
    Lpend::Vector{Ti}
    Ap_pos::Vector{Ti}
    X::Vector{Tv}
    firstrow_ref::Base.RefValue{Int}
end

KernelWorkspace{Tv, Ti}(n::Integer) where {Tv, Ti} = KernelWorkspace{Tv, Ti}(
    Vector{Ti}(undef, n),
    Vector{Ti}(undef, n),
    Vector{Ti}(undef, n),
    Vector{Ti}(undef, n),
    Vector{Ti}(undef, n),
    zeros(Tv, n),
    Ref(0),
)

"""
    KLUNumericBlock{Tv, Ti}

Per-block LU storage. `Li`/`Lx` hold the strict-lower-triangular entries
of `L` for every column of the block, packed contiguously; column `k`
(0-based in the block) starts at offset `Lip[k+1]+1` in Julia indexing and
has length `Llen[k+1]`. Same convention for `Ui`/`Ux`. `L` is unit lower
triangular and its diagonal is not stored.

`Li`/`Lx` and `Ui`/`Ux` are sized to *capacity*: the actual number of
populated slots is `Li_used` / `Ui_used`.  `length(Li)` (== `length(Lx)`)
is the current capacity; `klu_factor!` overwrites the prefix and only
calls `resize!` on the geometric grow path when the AMD-estimated
capacity proves insufficient.
"""
mutable struct KLUNumericBlock{Tv, Ti<:Integer}
    Li::Vector{Ti}
    Lx::Vector{Tv}
    Li_used::Int
    Ui::Vector{Ti}
    Ux::Vector{Tv}
    Ui_used::Int
end

KLUNumericBlock{Tv, Ti}() where {Tv, Ti} = KLUNumericBlock{Tv, Ti}(
    Ti[], Tv[], 0, Ti[], Tv[], 0
)

"""
    KLUNumeric{Tv, Ti}

Pure-Julia analogue of SuiteSparse's `klu_numeric`. Mirrors the field
names exposed by `KLU.jl` (`Pnum`, `Pinv`, `Lip`, `Uip`, `Llen`, `Ulen`,
`LUbx`, `Udiag`, `Rs`, `Offp`, `Offi`, `Offx`) so the same accessor logic
applies.
"""
_real_eltype(::Type{Tv}) where {Tv} = typeof(real(zero(Tv)))

mutable struct KLUNumeric{Tv, Ti<:Integer, Tr<:Real}
    n::Ti
    nblocks::Ti
    lnz::Ti
    unz::Ti
    max_lnz_block::Ti
    max_unz_block::Ti
    Pnum::Vector{Ti}
    Pinv::Vector{Ti}
    Lip::Vector{Ti}
    Uip::Vector{Ti}
    Llen::Vector{Ti}
    Ulen::Vector{Ti}
    LUbx::Vector{KLUNumericBlock{Tv, Ti}}
    Udiag::Vector{Tv}
    Rs::Vector{Tr}
    Offp::Vector{Ti}
    Offi::Vector{Ti}
    Offx::Vector{Tv}
    nzoff::Ti
    Xwork::Vector{Tv}
    # Allocated only when `common.scale > 0`; empty otherwise -- the
    # `Rs[k] = Rs[Pnum[k]]` shuffle at the tail of factor/refactor is
    # skipped entirely in the unscaled case.
    Xtmp::Vector{Tr}
    # Cached kernel workspace and per-block pivot permutation buffer.
    # Sized to `max(1, maxblock)` at numeric-alloc time so re-factor
    # calls (`klu_factor!`) reuse them instead of allocating fresh
    # length-`maxblock` arrays each call.
    wk::KernelWorkspace{Tv, Ti}
    Pblock::Vector{Ti}
end

KLUNumeric{Tv, Ti}(args...) where {Tv, Ti} =
    KLUNumeric{Tv, Ti, _real_eltype(Tv)}(args...)

# Empty placeholder; `n == 0` is the "not yet factored" sentinel checked
# throughout the package.  Lets `KLUFactorization.numeric` be a concrete
# non-Union field, and gives `_prepare_numeric_for_reuse!` a uniform path
# (`n == 0` ⇒ allocate fresh; otherwise reuse buffers).
KLUNumeric{Tv, Ti, Tr}() where {Tv, Ti<:Integer, Tr<:Real} = KLUNumeric{Tv, Ti, Tr}(
    Ti(0), Ti(0), Ti(0), Ti(0), Ti(0), Ti(0),
    Ti[], Ti[], Ti[], Ti[], Ti[], Ti[],
    KLUNumericBlock{Tv, Ti}[],
    Tv[], Tr[],
    Ti[], Ti[], Tv[],
    Ti(0),
    Tv[], Tr[],
    KernelWorkspace{Tv, Ti}(0),
    Ti[],
)

# Multiple of the block's input nonzero count used as an upper ceiling on
# the AMD-estimated initial capacity.  For highly-structured matrices
# (banded, arrow, ...) AMD predicts the block fills in near-densely and
# `Lnz` is `nk*(nk+1)/2`, which over-allocates the `Li/Lx/Ui/Ux` buffers
# by 1-3 orders of magnitude versus the actual fill (e.g. arrow_n500:
# 125k estimated vs 499 used).  Real fill rarely exceeds ~20x the input
# nonzeros for these orderings, so we cap the *initial* allocation at
# `FILL_CEILING_MULT * innz` and let the geometric `_klu_grow!` absorb the
# rare underestimate.  Chosen so the well-estimated dense/random cases
# (fill ratio ~10-20x, where AMD is accurate) still get a cap >= their
# final size and incur no grows.  Does not change numerical results.
const FILL_CEILING_MULT = 28

# `nk*(nk+1)/2` is a provable upper bound on `block.Li_used + nk` (the
# kernel's reserve check): column `k` writes at most `nk-1-k` off-
# diagonals, so the cumulative sum is maximised at `k = nk-1`.  Symmetric
# for `Ui`.  Sizing to this bound makes `_klu_grow!` provably unreachable.
#
# `innz` is a (possibly loose) upper bound on the block's input nonzeros;
# `FILL_CEILING_MULT * innz + nk` ceilings the AMD estimate so structured
# blocks are not over-allocated.  The ceiling is never applied below `nk`
# (a single full column) to keep at least a column of slack.
@inline function _block_cap(nk::Int, lnz_est::Float64, initmem_amd::Float64,
                            fully_preallocated::Bool, innz::Int)
    if fully_preallocated
        return (nk * (nk + 1)) >>> 1
    end
    est = lnz_est
    if !(est >= 0)
        est = Float64(nk) * Float64(nk) / 4
    end
    cap = ceil(Int, initmem_amd * est) + nk
    ceil_cap = FILL_CEILING_MULT * innz + nk
    return min(cap, max(ceil_cap, nk))
end

function _alloc_numeric(::Type{Tv}, Sym::KLUSymbolic{Ti}, common::KLUCommon{Ti},
                        Ap::Union{Vector{Ti}, Nothing}=nothing) where {Tv, Ti}
    return _alloc_numeric(Tv, Sym, common, common.fully_preallocated, Ap)
end

function _alloc_numeric(::Type{Tv}, Sym::KLUSymbolic{Ti}, common::KLUCommon{Ti},
                        fully_preallocated::Bool,
                        Ap::Union{Vector{Ti}, Nothing}=nothing) where {Tv, Ti}
    n = Int(Sym.n)
    nblocks = Int(Sym.nblocks)
    nzoff = Int(Sym.nzoff)
    maxblock = max(1, Int(Sym.maxblock))
    scale = Int(common.scale)
    Tr = _real_eltype(Tv)
    initmem_amd = common.initmem_amd
    Q = Sym.Q
    LUbx = Vector{KLUNumericBlock{Tv, Ti}}(undef, nblocks)
    @inbounds for b in 1:nblocks
        k1 = Int(Sym.R[b]); k2 = Int(Sym.R[b+1])
        nk = k2 - k1
        bk = KLUNumericBlock{Tv, Ti}()
        if nk > 1
            # Loose upper bound on the block's input nonzeros: total entries
            # in the block's columns (in original-column space via `Q`),
            # counting all rows.  Used only to ceiling the AMD estimate.  When
            # `Ap` is unavailable, pass `nk*nk` so the ceiling is inert.
            innz = nk * nk
            if Ap !== nothing && !fully_preallocated
                cnt = 0
                for k in k1:(k2-1)
                    oldcol = Int(Q[k+1])
                    cnt += Int(Ap[oldcol+2]) - Int(Ap[oldcol+1])
                end
                innz = cnt
            end
            cap = _block_cap(nk, Sym.Lnz[b], initmem_amd, fully_preallocated, innz)
            resize!(bk.Li, cap); resize!(bk.Lx, cap)
            resize!(bk.Ui, cap); resize!(bk.Ux, cap)
        end
        LUbx[b] = bk
    end
    Num = KLUNumeric{Tv, Ti, Tr}(
        Sym.n, Sym.nblocks, Ti(0), Ti(0), Ti(1), Ti(1),
        Vector{Ti}(undef, n),
        Vector{Ti}(undef, n),
        Vector{Ti}(undef, n),
        Vector{Ti}(undef, n),
        Vector{Ti}(undef, n),
        Vector{Ti}(undef, n),
        LUbx,
        Vector{Tv}(undef, n),
        scale > 0 ? zeros(Tr, n) : Tr[],
        zeros(Ti, n + 1),
        Vector{Ti}(undef, max(nzoff, 0)),
        Vector{Tv}(undef, max(nzoff, 0)),
        Sym.nzoff,
        Vector{Tv}(undef, n),
        scale > 0 ? Vector{Tr}(undef, n) : Tr[],
        KernelWorkspace{Tv, Ti}(maxblock),
        Vector{Ti}(undef, maxblock),
    )
    return Num
end

# Reuse an existing `KLUNumeric` for a fresh factor on the same symbolic
# structure.  Zeros the `Offp` prefix and the `Rs` accumulator when
# scaling is enabled (matching `_alloc_numeric`'s initial state); other
# fields are overwritten by the factor proper.  Adjusts the `Rs`/`Xtmp`
# buffers if `common.scale` changed since the previous factor.
function _prepare_numeric_for_reuse!(Num::KLUNumeric{Tv, Ti, Tr},
                                     Sym::KLUSymbolic{Ti},
                                     common::KLUCommon{Ti}) where {Tv, Ti, Tr}
    n = Int(Sym.n)
    scale = Int(common.scale)
    if scale > 0
        if length(Num.Rs) != n
            resize!(Num.Rs, n)
        end
        @inbounds for k in 1:n
            Num.Rs[k] = zero(Tr)
        end
        if length(Num.Xtmp) != n
            resize!(Num.Xtmp, n)
        end
    else
        if !isempty(Num.Rs)
            empty!(Num.Rs); empty!(Num.Xtmp)
        end
    end
    @inbounds for i in eachindex(Num.Offp)
        Num.Offp[i] = Ti(0)
    end
    return Num
end

# Geometric grow for `Li`/`Lx` (`which = :L`) or `Ui`/`Ux` (`which = :U`).
# `target` is the minimum new capacity required.  We grow by
# `max(target, ceil(memgrow * current_cap))` so amortised cost stays
# linear even when AMD's estimate is consistently low.
@noinline function _klu_grow!(block::KLUNumericBlock{Tv, Ti}, which::Symbol,
                              target::Int, memgrow::Float64) where {Tv, Ti}
    if which === :L
        cur = length(block.Li)
        new = max(target, ceil(Int, memgrow * cur))
        resize!(block.Li, new)
        resize!(block.Lx, new)
    else
        cur = length(block.Ui)
        new = max(target, ceil(Int, memgrow * cur))
        resize!(block.Ui, new)
        resize!(block.Ux, new)
    end
    return nothing
end

# ---------- scaling --------------------------------------------------------

"""
    _validate_pattern!(n, Ap, Ai, W, common) -> Bool

Validate the CSC pattern of an input matrix: `Ap[1]==0`, monotone column
pointers, row indices in `[0, n)`, and (when `W !== nothing`) no
duplicate entries within a single column. Sets `common.status` to
`KLU_INVALID` and returns `false` on the first violation. Extracted from
the original `klu_scale.c` so the factor path can validate the pattern
once without paying the row-norm work, and so the refactor path can
skip the whole walk when `scale == 0` (pattern is unchanged after
factor).
"""
function _validate_pattern!(n::Integer,
                            Ap::Vector{Ti}, Ai::Vector{Ti},
                            W::Union{Vector{Ti},Nothing},
                            common::KLUCommon{Ti}) where {Ti}
    if n <= 0
        common.status = KLU_INVALID
        return false
    end
    if Ap[1] != 0 || Ap[n+1] < 0
        common.status = KLU_INVALID
        return false
    end
    for col in 1:n
        if Ap[col] > Ap[col+1]
            common.status = KLU_INVALID
            return false
        end
    end
    check_duplicates = W !== nothing
    if check_duplicates
        @inbounds for row in 1:n
            W[row] = Ti(EMPTY)
        end
    end
    for col in 1:n
        pend = Int(Ap[col+1])
        for p in (Int(Ap[col])+1):pend
            row = Int(Ai[p]) + 1
            if row < 1 || row > n
                common.status = KLU_INVALID
                return false
            end
            if check_duplicates
                if W[row] == Ti(col - 1)
                    common.status = KLU_INVALID
                    return false
                end
                W[row] = Ti(col - 1)
            end
        end
    end
    return true
end

"""
    klu_scale!(scale, n, Ap, Ai, Ax, Rs, W, common) -> Bool

Row-scale a matrix in CSC form. `scale = 0` validates the matrix but
does not scale; `scale = 1` divides each row by its 1-norm; `scale ≥ 2`
divides by its ∞-norm. Mirrors `klu_scale.c`.
"""
function klu_scale!(scale::Integer, n::Integer,
                    Ap::Vector{Ti}, Ai::Vector{Ti}, Ax::Vector{Tv},
                    Rs::Union{Vector{<:Real},Nothing},
                    W::Union{Vector{Ti},Nothing},
                    common::KLUCommon{Ti}) where {Tv, Ti}
    fma_val = common.use_fma
    common.status = KLU_OK
    scale < 0 && return true
    if n <= 0 || (scale > 0 && Rs === nothing)
        common.status = KLU_INVALID
        return false
    end
    if Ap[1] != 0 || Ap[n+1] < 0
        common.status = KLU_INVALID
        return false
    end
    for col in 1:n
        if Ap[col] > Ap[col+1]
            common.status = KLU_INVALID
            return false
        end
    end
    if scale > 0
        Tr = eltype(Rs)
        z = zero(Tr)
        @inbounds for row in 1:n
            Rs[row] = z
        end
    end
    check_duplicates = W !== nothing
    if check_duplicates
        @inbounds for row in 1:n
            W[row] = Ti(EMPTY)
        end
    end
    for col in 1:n
        pend = Int(Ap[col+1])
        for p in (Int(Ap[col])+1):pend
            row = Int(Ai[p]) + 1
            if row < 1 || row > n
                common.status = KLU_INVALID
                return false
            end
            if check_duplicates
                if W[row] == Ti(col - 1)
                    common.status = KLU_INVALID
                    return false
                end
                W[row] = Ti(col - 1)
            end
            if scale > 0
                a = _ssabs(Ax[p], fma_val)
                if scale == 1
                    Rs[row] += a
                else
                    Rs[row] = max(Rs[row], a)
                end
            end
        end
    end
    if scale > 0
        Tr = eltype(Rs)
        zr = zero(Tr); one_r = one(Tr)
        @inbounds for row in 1:n
            if Rs[row] == zr
                Rs[row] = one_r
            end
        end
    end
    return true
end

# ---------- LU kernel ------------------------------------------------------

@inline _kflip(k::Integer) = -k - 2
@inline _kunflip(k::Integer) = k < -1 ? _kflip(k) : k

# DFS for the symbolic Lsolve; ports klu_kernel.c::dfs.  Writes new L row
# indices for column k into `block_Li` at offsets `Lip[k+1]+1 .. Lip[k+1]+l_length`.
# Returns `(top, l_length)` rather than mutating a Ref so the compiler can
# keep `l_length` in a register across the inlined call boundary.
@inline function _kernel_dfs!(j_in::Int, k::Int,
                      Pinv::Vector{Ti}, Llen::AbstractVector{Ti}, Lip::AbstractVector{Ti},
                      Stack::Vector{Ti}, Flag::Vector{Ti}, Lpend::Vector{Ti},
                      top_in::Int,
                      block_Li::Vector{Ti}, Lip_k::Int,
                      l_length::Int,
                      Ap_pos::Vector{Ti}) where {Ti}
    head = 0
    @inbounds Stack[1] = Ti(j_in)
    top = top_in
    kT = Ti(k)
    emptyT = Ti(EMPTY)
    # `j`, `jnew`, `lip_jnew` describe the node currently at the top of the
    # DFS stack.  They only change when `head` changes (push/pop), so we hoist
    # them out of the per-node visit and refresh them via `@goto restart`
    # instead of reloading from `Stack`/`Pinv`/`Lip` on every outer iteration.
    # `pos` (the scan cursor for the current node) is kept in a register and
    # spilled to `Ap_pos[head+1]` only when we push, then reloaded on pop.
    j = j_in
    @label restart
    @inbounds begin
        jnew = Int(Pinv[j+1])
        lip_jnew = Int(Lip[jnew+1])
        if Flag[j+1] != kT
            Flag[j+1] = kT
            lp = Lpend[jnew+1]
            pos = Int((lp == emptyT) ? Llen[jnew+1] : lp) - 1
        else
            pos = Int(Ap_pos[head+1]) - 1
        end
        while true
            broke = false
            while pos >= 0
                i = Int(block_Li[lip_jnew + pos + 1])
                if Flag[i+1] != kT
                    if Pinv[i+1] >= 0
                        # Push child `i`: save our resume cursor, then restart
                        # with the child as the new top-of-stack node.
                        Ap_pos[head+1] = Ti(pos)
                        head += 1
                        Stack[head+1] = Ti(i)
                        j = i
                        broke = true
                        break
                    else
                        Flag[i+1] = kT
                        block_Li[Lip_k + l_length + 1] = Ti(i)
                        l_length += 1
                    end
                end
                pos -= 1
            end
            broke && @goto restart
            # Node fully explored: emit it and pop.
            head -= 1
            top -= 1
            Stack[top+1] = Ti(j)
            head < 0 && break
            j = Int(Stack[head+1])
            jnew = Int(Pinv[j+1])
            lip_jnew = Int(Lip[jnew+1])
            pos = Int(Ap_pos[head+1]) - 1
        end
    end
    return top, l_length
end

# Port of klu_kernel.c::lsolve_symbolic
function _kernel_lsolve_symbolic!(n::Int, k::Int,
                                  Ap::Vector{Ti}, Ai::Vector{Ti}, Q::Vector{Ti},
                                  Pinv::Vector{Ti}, Stack::Vector{Ti},
                                  Flag::Vector{Ti}, Lpend::Vector{Ti},
                                  Ap_pos::Vector{Ti},
                                  block_Li::Vector{Ti},
                                  Llen::AbstractVector{Ti}, Lip::AbstractVector{Ti},
                                  k1::Int, PSinv::Vector{Ti}) where {Ti}
    top = n
    l_length = 0
    kglobal = k + k1
    @inbounds Lip_k = Int(Lip[k+1])
    @inbounds oldcol = Int(Q[kglobal+1])
    @inbounds pend = Int(Ap[oldcol+2])
    @inbounds pstart = Int(Ap[oldcol+1])
    kT = Ti(k)
    @inbounds for p in pstart:(pend-1)
        i = Int(PSinv[Int(Ai[p+1])+1]) - k1
        if i < 0
            continue
        end
        if Flag[i+1] != kT
            if Pinv[i+1] >= 0
                top, l_length = _kernel_dfs!(i, k, Pinv, Llen, Lip, Stack, Flag, Lpend,
                                             top, block_Li, Lip_k, l_length, Ap_pos)
            else
                Flag[i+1] = kT
                block_Li[Lip_k + l_length + 1] = Ti(i)
                l_length += 1
            end
        end
    end
    @inbounds Llen[k+1] = Ti(l_length)
    return top
end

# Port of klu_kernel.c::construct_column.  Branches statically on
# `(scale_val, fma_val)` so each instantiation collapses to the
# appropriate inner loop with no per-iteration test (matches the
# monomorphic C compile).
@inline function _kernel_construct_column!(k::Int, Ap::Vector{Ti}, Ai::Vector{Ti},
                                           Ax::Vector{Tv}, Q::Vector{Ti}, X::Vector{Tv},
                                           k1::Int, PSinv::Vector{Ti},
                                           Rs::Vector{Tr},
                                           scale_val::Val,
                                           Offp::Vector{Ti}, Offi::Vector{Ti},
                                           Offx::Vector{Tv}) where {Tv, Ti, Tr<:Real}
    kglobal = k + k1
    @inbounds poff = Int(Offp[kglobal+1])
    @inbounds oldcol = Int(Q[kglobal+1])
    @inbounds pend = Int(Ap[oldcol+2])
    @inbounds pstart = Int(Ap[oldcol+1])
    if scale_val === Val(false)
        @inbounds for p in pstart:(pend-1)
            oldrow = Int(Ai[p+1])
            i = Int(PSinv[oldrow+1]) - k1
            aik = Ax[p+1]
            if i < 0
                Offi[poff+1] = Ti(oldrow); Offx[poff+1] = aik; poff += 1
            else
                X[i+1] = aik
            end
        end
    else
        @inbounds for p in pstart:(pend-1)
            oldrow = Int(Ai[p+1])
            i = Int(PSinv[oldrow+1]) - k1
            aik = Ax[p+1] / Rs[oldrow+1]
            if i < 0
                Offi[poff+1] = Ti(oldrow); Offx[poff+1] = aik; poff += 1
            else
                X[i+1] = aik
            end
        end
    end
    @inbounds Offp[kglobal+2] = Ti(poff)
    return nothing
end

# Port of klu_kernel.c::lsolve_numeric.
#
# Note on FMA: this loop is the canonical multiply-subtract hot spot.
# SuiteSparse_jll's `libklu.so` ships an SSE2-only baseline build
# (BinaryBuilder targets baseline x86_64-linux-gnu), so the compiler
# emits a pair of separate `mulsd`/`subsd` instructions for
# `X[Li[p]] -= Lx[p] * xj`, never a fused `vfnmadd*sd`.  Concretely,
# the disassembly of `KLU_kernel` in the shipped library reads:
#
#     mulsd  %xmm2, %xmm1       ; xmm1 = Lx[p] * xj
#     movsd  (%rdx), %xmm0      ; xmm0 = X[Li[p]]
#     subsd  %xmm1, %xmm0       ; xmm0 -= xmm1
#     movsd  %xmm0, (%rdx)      ; X[Li[p]] = xmm0
#
# When `fma_val == Val(false)` we match that exactly (two roundings).
# When `fma_val == Val(true)` (default) we use `muladd`, which LLVM
# lowers to `vfnmadd*sd` on hardware with FMA -- one rounding, slightly
# more accurate, but L/U entries can differ from KLU.jl by 1 ULP.
function _kernel_lsolve_numeric!(Pinv::Vector{Ti},
                                 block_Li::Vector{Ti}, block_Lx::Vector{Tv},
                                 Stack::Vector{Ti}, Lip::AbstractVector{Ti},
                                 top::Int, n::Int, Llen::AbstractVector{Ti},
                                 X::Vector{Tv}, fma_val::Val) where {Tv, Ti}
    @inbounds for s in top:(n-1)
        j = Int(Stack[s+1])
        jnew = Int(Pinv[j+1])
        xj = X[j+1]
        lip = Int(Lip[jnew+1])
        len = Int(Llen[jnew+1])
        # `block_Li[lip..lip+len-1]` are the row indices for one column of L,
        # which are pairwise distinct by construction (sparse LU never inserts
        # a row twice into a column).  The compiler can't see that, so we
        # promise it via `@simd ivdep` -- the scatter `X[i] -= ...` has no
        # loop-carried memory dependency.
        @simd ivdep for p in 0:(len-1)
            i = Int(block_Li[lip+p+1])
            X[i+1] = _mulsub(X[i+1], block_Lx[lip+p+1], xj, fma_val)
        end
    end
end

# Port of klu_kernel.c::lpivot
function _kernel_lpivot!(diagrow::Int, k::Int, n::Int,
                         tol::Float64, X::Vector{Tv},
                         block_Li::Vector{Ti}, block_Lx::Vector{Tv},
                         Lip::AbstractVector{Ti}, Llen::AbstractVector{Ti},
                         Pinv::Vector{Ti}, firstrow_ref::Base.RefValue{Int},
                         common::KLUCommon{Ti}, fma_val::Val) where {Tv, Ti}
    Tr = _real_eltype(Tv)
    if Llen[k+1] == 0
        if common.halt_if_singular != 0
            return false, -1, zero(Tv), zero(Tr)
        end
        for firstrow in firstrow_ref[]:(n-1)
            if Pinv[firstrow+1] < 0
                firstrow_ref[] = firstrow
                # Clear X at any non-pivotal row (matches C's behavior to leave X zero)
                return false, firstrow, zero(Tv), zero(Tr)
            end
        end
        return false, -1, zero(Tv), zero(Tr)
    end

    pdiag = -1
    ppivrow = -1
    abs_pivot = -one(Tr)
    @inbounds lip = Int(Lip[k+1])
    @inbounds len_full = Int(Llen[k+1])
    @inbounds last_row_index = Int(block_Li[lip + len_full])

    @inbounds Llen[k+1] = Ti(len_full - 1)
    len = len_full - 1
    diagrowT = Ti(diagrow)

    @inbounds for p in 0:(len-1)
        i = block_Li[lip+p+1]
        x = X[Int(i)+1]
        X[Int(i)+1] = zero(Tv)
        block_Lx[lip+p+1] = x
        xabs = abs(x)
        pdiag = ifelse(i == diagrowT, p, pdiag)
        if xabs > abs_pivot
            abs_pivot = xabs
            ppivrow = p
        end
    end

    xabs_last = abs(X[last_row_index+1])
    if xabs_last > abs_pivot
        abs_pivot = xabs_last
        ppivrow = -1
    end

    if last_row_index == diagrow
        if xabs_last >= tol * abs_pivot
            abs_pivot = xabs_last
            ppivrow = -1
        end
    elseif pdiag != -1
        xabs_d = abs(block_Lx[lip + pdiag + 1])
        if xabs_d >= tol * abs_pivot
            abs_pivot = xabs_d
            ppivrow = pdiag
        end
    end

    pivrow = 0
    pivot = zero(Tv)
    if ppivrow != -1
        pivrow = Int(block_Li[lip + ppivrow + 1])
        pivot = block_Lx[lip + ppivrow + 1]
        block_Li[lip + ppivrow + 1] = Ti(last_row_index)
        block_Lx[lip + ppivrow + 1] = X[last_row_index+1]
    else
        pivrow = last_row_index
        pivot = X[last_row_index+1]
    end
    X[last_row_index+1] = zero(Tv)

    if iszero(pivot) && common.halt_if_singular != 0
        return false, pivrow, pivot, abs_pivot
    end

    # Scaling pass: each entry is divided by the same pivot, no cross-iteration
    # dependency.  `@simd ivdep` is safe because the writes are to consecutive
    # storage locations.  `len` equals the post-decrement `Llen[k+1]`.
    @inbounds @simd ivdep for p in 0:(len-1)
        block_Lx[lip+p+1] = _cdiv(block_Lx[lip+p+1], pivot, fma_val)
    end

    return true, pivrow, pivot, abs_pivot
end

# Port of klu_kernel.c::prune
function _kernel_prune!(Lpend::Vector{Ti}, Pinv::Vector{Ti}, k::Int, pivrow::Int,
                        block_Li::Vector{Ti}, block_Lx::Vector{Tv},
                        block_Ui::Vector{Ti},
                        Uip::AbstractVector{Ti}, Lip::AbstractVector{Ti},
                        Ulen::AbstractVector{Ti}, Llen::AbstractVector{Ti}) where {Tv, Ti}
    @inbounds uip_k = Int(Uip[k+1])
    @inbounds ulen_k = Int(Ulen[k+1])
    emptyT = Ti(EMPTY)
    pivrowT = Ti(pivrow)
    @inbounds for p in 0:(ulen_k-1)
        j = Int(block_Ui[uip_k+p+1])
        if Lpend[j+1] == emptyT
            lip_j = Int(Lip[j+1])
            llen_j = Int(Llen[j+1])
            found = -1
            for p2 in 0:(llen_j-1)
                if block_Li[lip_j+p2+1] == pivrowT
                    found = p2
                    break
                end
            end
            if found >= 0
                phead = 0
                ptail = llen_j
                while phead < ptail
                    i = block_Li[lip_j+phead+1]
                    if Pinv[Int(i)+1] >= 0
                        phead += 1
                    else
                        ptail -= 1
                        block_Li[lip_j+phead+1] = block_Li[lip_j+ptail+1]
                        block_Li[lip_j+ptail+1] = i
                        x = block_Lx[lip_j+phead+1]
                        block_Lx[lip_j+phead+1] = block_Lx[lip_j+ptail+1]
                        block_Lx[lip_j+ptail+1] = x
                    end
                end
                Lpend[j+1] = Ti(ptail)
            end
        end
    end
end

"""
    klu_kernel!(...) -> (lnz_block, unz_block)

Port of `KLU_kernel`. Factors a single block (already pre-permuted by
the BTF symbolic pass) into `L` (unit lower, no diagonal stored) and `U`
(upper, diagonal in `Udiag`). Off-diagonal entries that point to rows
outside the current block are appended to `Offp`/`Offi`/`Offx`.

`Lip`/`Llen`/`Uip`/`Ulen`/`Udiag` are views into the global per-column
tables (sliced to the current block) and are modified in place.
"""
function klu_kernel!(nk::Int, Ap::Vector{Ti}, Ai::Vector{Ti}, Ax::Vector{Tv},
                    Q::Vector{Ti},
                    block::KLUNumericBlock{Tv, Ti},
                    Udiag::AbstractVector{Tv},
                    Llen::AbstractVector{Ti}, Ulen::AbstractVector{Ti},
                    Lip::AbstractVector{Ti}, Uip::AbstractVector{Ti},
                    Pblock::Vector{Ti},
                    wk::KernelWorkspace{Tv, Ti},
                    k1::Int, PSinv::Vector{Ti},
                    Rs::Vector{Tr},
                    scale_val::Val,
                    Offp::Vector{Ti}, Offi::Vector{Ti}, Offx::Vector{Tv},
                    common::KLUCommon{Ti}, fma_val::Val) where {Tv, Ti, Tr<:Real}
    n = nk
    tol = common.tol
    @inbounds for k in 1:n
        wk.X[k] = zero(Tv)
        wk.Flag[k] = Ti(EMPTY)
        wk.Lpend[k] = Ti(EMPTY)
    end
    @inbounds for k in 1:n
        Pblock[k] = Ti(k-1)
        wk.Pinv[k] = Ti(_kflip(k-1))
    end

    # Reset per-block used counts.  The underlying Vector capacities are
    # already sized to AMD's lnz estimate by `_alloc_numeric` and persist
    # across factor/refactor; we overwrite the live prefix in place.
    block.Li_used = 0
    block.Ui_used = 0
    memgrow = common.memgrow

    lnz = 0
    unz = 0
    firstrow_ref = wk.firstrow_ref
    firstrow_ref[] = 0

    for k in 0:(n-1)
        # Reserve `n` slots for column k's L pattern.  If capacity is
        # insufficient, grow geometrically; otherwise just record the
        # starting offset.
        old_len = block.Li_used
        @inbounds Lip[k+1] = Ti(old_len)
        if old_len + n > length(block.Li)
            _klu_grow!(block, :L, old_len + n, memgrow)
        end

        top = _kernel_lsolve_symbolic!(n, k, Ap, Ai, Q, wk.Pinv, wk.Stack,
                                       wk.Flag, wk.Lpend, wk.Ap_pos,
                                       block.Li, Llen, Lip, k1, PSinv)

        _kernel_construct_column!(k, Ap, Ai, Ax, Q, wk.X, k1, PSinv,
                                  Rs, scale_val,
                                  Offp, Offi, Offx)
        _kernel_lsolve_numeric!(wk.Pinv, block.Li, block.Lx, wk.Stack, Lip,
                                top, n, Llen, wk.X, fma_val)

        @inbounds diagrow = Int(Pblock[k+1])
        ok, pivrow, pivot, _ = _kernel_lpivot!(diagrow, k, n, tol, wk.X,
                                               block.Li, block.Lx, Lip, Llen,
                                               wk.Pinv, firstrow_ref, common,
                                               fma_val)
        if !ok
            common.status = KLU_SINGULAR
            if common.numerical_rank == Ti(EMPTY)
                common.numerical_rank = Ti(k + k1)
                @inbounds common.singular_col = Q[k + k1 + 1]
            end
            if common.halt_if_singular != 0
                return lnz, unz
            end
        end

        # Record actual L slots used (no shrink -- the slack is just
        # unused capacity for the next column).
        @inbounds lip_k = Int(Lip[k+1])
        @inbounds llen_k = Int(Llen[k+1])
        block.Li_used = lip_k + llen_k

        # Build U for this column from Stack[top..n-1] and X.  Index-write
        # path: reserve `ulen` slots after the current used prefix.
        u_off = block.Ui_used
        @inbounds Uip[k+1] = Ti(u_off)
        ulen = n - top
        @inbounds Ulen[k+1] = Ti(ulen)
        if u_off + ulen > length(block.Ui)
            _klu_grow!(block, :U, u_off + ulen, memgrow)
        end
        @inbounds for s in top:(n-1)
            j = Int(wk.Stack[s+1])
            idx = u_off + (s - top) + 1
            block.Ui[idx] = wk.Pinv[j+1]
            block.Ux[idx] = wk.X[j+1]
            wk.X[j+1] = zero(Tv)
        end
        block.Ui_used = u_off + ulen

        Udiag[k+1] = pivot

        if pivrow != diagrow
            common.noffdiag += Ti(1)
            if wk.Pinv[diagrow+1] < 0
                kbar = _kflip(Int(wk.Pinv[pivrow+1]))
                Pblock[kbar+1] = Ti(diagrow)
                wk.Pinv[diagrow+1] = Ti(_kflip(kbar))
            end
        end
        Pblock[k+1] = Ti(pivrow)
        wk.Pinv[pivrow+1] = Ti(k)

        _kernel_prune!(wk.Lpend, wk.Pinv, k, pivrow, block.Li, block.Lx,
                       block.Ui, Uip, Lip, Ulen, Llen)

        lnz += llen_k + 1
        unz += ulen + 1
    end

    # Remap L row indices using the final Pinv.  Each iteration updates a
    # distinct slot in block.Li (consecutive `p` values), so the writes
    # don't alias the reads in a later iteration.  Safe to vectorise.
    @inbounds for k in 0:(n-1)
        lip_k = Int(Lip[k+1])
        llen_k = Int(Llen[k+1])
        @simd ivdep for p in 0:(llen_k-1)
            block.Li[lip_k+p+1] = Ti(wk.Pinv[Int(block.Li[lip_k+p+1])+1])
        end
    end

    return lnz, unz
end

"""
    klu_factor!(Sym, Ap, Ai, Ax, common; allowsingular=false) -> KLUNumeric

Top-level numeric factorisation. Builds a `KLUNumeric`, scales the input
if requested and dispatches each BTF block either to a singleton path or
to `klu_kernel!`.
"""
function klu_factor!(Sym::KLUSymbolic{Ti}, Ap::Vector{Ti}, Ai::Vector{Ti},
                     Ax::Vector{Tv}, common::KLUCommon{Ti};
                     allowsingular::Bool=false,
                     ) where {Tv, Ti}
    Tr = _real_eltype(Tv)
    scale_val = Int(common.scale) > 0 ? Val(true) : Val(false)
    return _klu_factor_impl!(Sym, Ap, Ai, Ax, common, common.use_fma,
                             KLUNumeric{Tv, Ti, Tr}(), allowsingular,
                             scale_val)
end

# Variant that reuses an existing `KLUNumeric` (its block capacities and
# kernel workspace).  Used by the `KLUFactorization` top-level entry
# point so subsequent `klu_factor!` calls allocate nothing beyond the
# (possible) geometric grow path in the kernel.  Pass an empty
# `KLUNumeric{Tv, Ti, Tr}()` (the default field value on a fresh
# `KLUFactorization`) on the cold call.
function klu_factor!(Sym::KLUSymbolic{Ti}, Ap::Vector{Ti}, Ai::Vector{Ti},
                     Ax::Vector{Tv}, common::KLUCommon{Ti},
                     reuse::KLUNumeric{Tv, Ti};
                     allowsingular::Bool=false,
                     ) where {Tv, Ti}
    scale_val = Int(common.scale) > 0 ? Val(true) : Val(false)
    return _klu_factor_impl!(Sym, Ap, Ai, Ax, common, common.use_fma,
                             reuse, allowsingular, scale_val)
end

function _klu_factor_impl!(Sym::KLUSymbolic{Ti}, Ap::Vector{Ti}, Ai::Vector{Ti},
                           Ax::Vector{Tv}, common::KLUCommon{Ti},
                           fma_val::Val,
                           reuse::KLUNumeric{Tv, Ti, Tr},
                           allowsingular::Bool,
                           scale_val::Val) where {Tv, Ti, Tr}
    common.status = KLU_OK
    common.numerical_rank = Ti(EMPTY)
    common.singular_col = Ti(EMPTY)
    common.noffdiag = Ti(0)

    Num = reuse.n == Sym.n ?
        _prepare_numeric_for_reuse!(reuse, Sym, common) :
        _alloc_numeric(Tv, Sym, common, Ap)
    n = Int(Sym.n)
    nblocks = Int(Sym.nblocks)
    nzoff = Int(Sym.nzoff)

    P = Sym.P
    Q = Sym.Q
    R = Sym.R

    Pinv = Num.Pinv
    @inbounds for k in 1:n
        Pinv[Int(P[k])+1] = Ti(k-1)
    end

    scale = Int(common.scale)
    if scale > 0
        ok = klu_scale!(scale, n, Ap, Ai, Ax,
                        Num.Rs,
                        Num.Pnum,
                        common)
        if !ok
            return Num
        end
    elseif scale == 0
        if !_validate_pattern!(n, Ap, Ai, Num.Pnum, common)
            return Num
        end
    end
    Num.Offp[1] = Ti(0)
    lnz_total = 0
    unz_total = 0
    max_lnz_block = 1
    max_unz_block = 1

    wk = Num.wk
    Pblock = Num.Pblock

    for block in 1:nblocks
        k1 = Int(R[block])
        k2 = Int(R[block+1])
        nk = k2 - k1

        if nk == 1
            poff = Int(Num.Offp[k1+1])
            oldcol = Int(Q[k1+1])
            pend = Int(Ap[oldcol+2])
            s = zero(Tv)
            for p in Int(Ap[oldcol+1]):(pend-1)
                oldrow = Int(Ai[p+1])
                newrow = Int(Pinv[oldrow+1])
                aik = _scale_aik(Ax[p+1],
                                 _rs_at(Num.Rs, oldrow+1, scale_val),
                                 scale_val)
                if newrow < k1
                    Num.Offi[poff+1] = Ti(oldrow)
                    Num.Offx[poff+1] = aik
                    poff += 1
                else
                    s = aik
                end
            end
            Num.Udiag[k1+1] = s
            if iszero(s)
                common.status = KLU_SINGULAR
                if common.numerical_rank == Ti(EMPTY)
                    common.numerical_rank = Ti(k1)
                    common.singular_col = Ti(oldcol)
                end
                if common.halt_if_singular != 0
                    return Num
                end
            end
            Num.Offp[k1+2] = Ti(poff)
            Num.Pnum[k1+1] = P[k1+1]
            Num.Lip[k1+1] = Ti(0)
            Num.Uip[k1+1] = Ti(0)
            Num.Llen[k1+1] = Ti(0)
            Num.Ulen[k1+1] = Ti(0)
            lnz_total += 1
            unz_total += 1
        else
            block_lnz, block_unz = klu_kernel!(
                nk, Ap, Ai, Ax, Q,
                Num.LUbx[block],
                view(Num.Udiag, k1+1:k2),
                view(Num.Llen, k1+1:k2),
                view(Num.Ulen, k1+1:k2),
                view(Num.Lip, k1+1:k2),
                view(Num.Uip, k1+1:k2),
                Pblock,
                wk, k1, Pinv, Num.Rs, scale_val,
                Num.Offp, Num.Offi, Num.Offx,
                common, fma_val,
            )

            if common.status < KLU_OK ||
               (common.status == KLU_SINGULAR && common.halt_if_singular != 0)
                return Num
            end

            lnz_total += block_lnz
            unz_total += block_unz
            max_lnz_block = max(max_lnz_block, block_lnz)
            max_unz_block = max(max_unz_block, block_unz)

            @inbounds for k in 0:(nk-1)
                Num.Pnum[k + k1 + 1] = P[Int(Pblock[k+1]) + k1 + 1]
            end
        end
    end

    Num.lnz = Ti(lnz_total)
    Num.unz = Ti(unz_total)
    Num.max_lnz_block = Ti(max_lnz_block)
    Num.max_unz_block = Ti(max_unz_block)

    @inbounds for k in 1:n
        Pinv[Int(Num.Pnum[k])+1] = Ti(k-1)
    end

    if scale > 0
        # `Num.Xtmp` is preallocated to length `n` at factor time when
        # scale > 0; reuse it for the Rs permutation shuffle instead of
        # allocating fresh.
        Xtmp = Num.Xtmp
        @inbounds for k in 1:n
            Xtmp[k] = Num.Rs[Int(Num.Pnum[k])+1]
        end
        @inbounds for k in 1:n
            Num.Rs[k] = Xtmp[k]
        end
    end

    @inbounds for p in 1:nzoff
        Num.Offi[p] = Ti(Pinv[Int(Num.Offi[p])+1])
    end

    return Num
end
