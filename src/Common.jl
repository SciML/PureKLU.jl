const EMPTY = -1
const EMPTY_FLOAT = -1.0

"""
    KLUStatus

Status codes mirroring SuiteSparse's KLU return values. Stored in
[`KLUCommon`](@ref)'s `status` field. Non-negative values (`KLU_OK`,
`KLU_SINGULAR`) indicate success or a benign singularity; negative
values indicate hard errors, so `status >= KLU_OK` tests for "no error".
"""
@enum KLUStatus::Cint begin
    KLU_OK = 0
    KLU_SINGULAR = 1
    KLU_OUT_OF_MEMORY = -2
    KLU_INVALID = -3
    KLU_TOO_LARGE = -4
end

"""
    KLUCommon{Ti}

Pure-Julia analogue of SuiteSparse's `klu_common`/`klu_l_common` struct.
Carries the same tunable parameters and the per-call statistics that KLU
mutates. `Ti` is `Int32` or `Int64`, matching the integer width of the
index arrays.
"""
mutable struct KLUCommon{Ti <: Integer}
    tol::Float64
    memgrow::Float64
    initmem_amd::Float64
    initmem::Float64
    maxwork::Float64
    btf::Cint
    ordering::Cint
    scale::Cint
    halt_if_singular::Cint
    status::KLUStatus
    nrealloc::Cint
    structural_rank::Ti
    numerical_rank::Ti
    singular_col::Ti
    noffdiag::Ti
    flops::Float64
    rcond::Float64
    condest::Float64
    rgrowth::Float64
    work::Float64
    # PureKLU-only: when `Val(true)` (default) the kernel hot loops compute
    # `a - b*c` via `muladd(-b, c, a)` so that LLVM emits a fused
    # multiply-add (FMA) on supported hardware. Faster and slightly more
    # accurate, but the results differ from SuiteSparse's `libklu.so`
    # binary by up to 1 ULP (the binary is shipped SSE2-only, with no
    # FMA instructions). Set to `Val(false)` to opt out and get bit-for-bit
    # identical results to `KLU.jl`.
    #
    # Stored as `Union{Val{true},Val{false}}` so the kernel can dispatch
    # on it directly via a function barrier, giving each loop a fully
    # specialised compilation.  Users may pass either `Bool` or `Val`
    # to the `use_fma` kwarg of [`klu`](@ref); both forms are normalised
    # here.
    use_fma::Union{Val{true}, Val{false}}
    # Auto-mode (`fully_preallocated=nothing` at the `klu(...)` kwarg) is
    # resolved to a concrete `Bool` in `klu(...)` after `klu_analyze!` runs;
    # the direct `klu_factor!(K)` path uses whatever is set here (default
    # `false`).
    fully_preallocated::Bool
    # Opt-in: when `true`, the analyze phase detects per-BTF-block narrow,
    # dense bands and orders them naturally (skipping AMD) -- a large analyze
    # speedup for genuinely banded systems.  Default `false` keeps the AMD
    # ordering, so the factorization stays bit-for-bit identical to SuiteSparse
    # KLU (the natural ordering is equally valid and equal-fill, but its `Q`/L/U
    # differ from AMD's, which would break the strict KLU-parity guarantee).
    detect_banded::Bool
end

function KLUCommon{Ti}() where {Ti <: Integer}
    C = KLUCommon{Ti}(
        0.001, 1.2, 1.2, 10.0, 0.0,
        Cint(1), Cint(0), Cint(2),
        Cint(1), KLU_OK, Cint(0),
        Ti(EMPTY), Ti(EMPTY), Ti(EMPTY), Ti(0),
        EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_FLOAT, 0.0,
        Val(true),
        false,
        false,
    )
    return C
end

"""
    _as_val(x) -> Val{true} or Val{false}

Normalise a user-supplied FMA flag.  Accepts either `Bool` or
`Val{Bool}`.
"""
@inline _as_val(x::Val{true}) = Val(true)
@inline _as_val(x::Val{false}) = Val(false)
@inline _as_val(x::Bool) = x ? Val(true) : Val(false)

function klu_defaults!(C::KLUCommon{Ti}) where {Ti}
    C.tol = 0.001
    C.memgrow = 1.2
    C.initmem_amd = 1.2
    C.initmem = 10.0
    C.maxwork = 0.0
    C.btf = Cint(1)
    C.ordering = Cint(0)
    C.scale = Cint(2)
    C.halt_if_singular = Cint(1)
    C.status = KLU_OK
    C.nrealloc = Cint(0)
    C.structural_rank = Ti(EMPTY)
    C.numerical_rank = Ti(EMPTY)
    C.singular_col = Ti(EMPTY)
    C.noffdiag = Ti(0)
    C.flops = EMPTY_FLOAT
    C.rcond = EMPTY_FLOAT
    C.condest = EMPTY_FLOAT
    C.rgrowth = EMPTY_FLOAT
    C.work = 0.0
    C.use_fma = Val(true)
    C.fully_preallocated = false
    C.detect_banded = false
    return C
end

# At nk=64 the hard upper bound `nk*(nk+1)/2 = 2080` entries (~32 KB for
# Float64+Int64) caps per-block over-allocation; above this, the bound
# grows O(nk^2) relative to AMD's `Lnz` estimate.
const AUTO_PREALLOC_MAXBLOCK = 64
