# Opt-in structure-specialized solver for single-spike arrowhead / bordered
# systems.  This is NOT a faithful KLU port: it solves the arrowhead by
# block elimination (Schur complement of the diagonal core), an O(n) factor
# + O(n) solve that sidesteps AMD, the symbolic phase, and per-column
# pivoting.  It produces a solution that matches KLU within rounding (it is
# *not* bit-identical to libklu, since the elimination order differs), so it
# is exposed only behind an explicit opt-in and never replaces the default
# byte-identical KLU path.
#
# Structure (spike at index `s`):
#   A[i,i]  = d[i]      core diagonal,  i != s
#   A[s,s]  = alpha     corner
#   A[i,s]  = c[i]      dense spike column, i != s
#   A[s,i]  = r[i]      dense spike row,    i != s
#   A[i,j]  = 0         otherwise (i != j, i != s, j != s)
#
# Schur scalar:  sigma = alpha - sum_i r[i] c[i] / d[i].
# Solve A x = b:
#   xs       = (b[s] - sum_i r[i] b[i]/d[i]) / sigma
#   x[i]     = (b[i] - c[i] xs) / d[i]          (i != s)

"""
    ArrowheadFactorization{Tv, Ti}

Structure-specialized factorization of a single-spike arrowhead matrix
produced by `klu(A; structured = true)` when the detector fires. Solves via
Schur-complement block elimination of the diagonal core. Subtypes
`AbstractKLUFactorization` so `\\`, `ldiv!`, and `solve!` dispatch to the
bordered solve. Not bit-identical to SuiteSparse KLU (different elimination
order), but matches it within floating-point rounding.
"""
struct ArrowheadFactorization{Tv, Ti <: Integer} <: AbstractKLUFactorization{Tv, Ti}
    n::Int
    s::Int              # spike index (1-based)
    d::Vector{Tv}       # core diagonal (d[s] unused)
    c::Vector{Tv}       # spike column (c[s] unused)
    r::Vector{Tv}       # spike row    (r[s] unused)
    alpha::Tv
    sigma::Tv           # Schur complement scalar
    common::KLUCommon{Ti}
end

size(K::ArrowheadFactorization) = (K.n, K.n)
_is_factored(::ArrowheadFactorization) = true

# Detect a single-spike arrowhead.  Returns the spike index, or 0 if the
# pattern is not a single-spike arrowhead (purely diagonal/banded matrices,
# matrices with two or more dense columns, etc. all return 0).
function _detect_arrowhead(n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti}
    n >= 3 || return 0
    spike = 0
    @inbounds for j in 1:n
        len = Int(colptr[j + 1] - colptr[j])
        if len > 2
            spike == 0 || return 0          # second dense column -> not single-spike
            spike = j
        end
    end
    return spike                            # 0 if no dense column at all
end

# Build the arrowhead factorization from 0-based CSC arrays.  Returns nothing
# if the numeric structure does not actually hold (off-structure entry, zero
# core diagonal, or a core diagonal too small to be a stable no-pivot pivot --
# in which case KLU would pivot and the bordered answer would diverge, so we
# fall back).  `colptr`/`rowval` are 0-based (as stored in KLUFactorization).
function _build_arrowhead(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv},
        spike::Int, common::KLUCommon{Ti}
    ) where {Tv, Ti}
    piv_tol = common.tol                    # KLU partial-pivot tol (default 1e-3)
    d = zeros(Tv, n); c = zeros(Tv, n); r = zeros(Tv, n)
    alpha = zero(Tv)
    @inbounds for j in 1:n
        for p in (Int(colptr[j]) + 1):Int(colptr[j + 1])
            i = Int(rowval[p]) + 1          # 0-based -> 1-based
            v = nzval[p]
            if i == j
                if i == spike
                    alpha = v
                else
                    d[i] = v
                end
            elseif j == spike
                c[i] = v
            elseif i == spike
                r[j] = v
            else
                return nothing              # off-structure -> not an arrowhead
            end
        end
    end
    @inbounds for i in 1:n
        i == spike && continue
        di = d[i]
        iszero(di) && return nothing
        # No-pivot safety: in the AMD-reordered arrowhead, eliminating core
        # column i has candidate pivots {d[i] (diag), c[i] (spike)}; KLU keeps
        # the diagonal iff |d[i]| >= tol*|c[i]|.  Require that (plus the row
        # entry, conservatively) so our diagonal-pivot order matches KLU's.
        (abs(di) >= piv_tol * abs(c[i]) && abs(di) >= piv_tol * abs(r[i])) ||
            return nothing
    end
    sigma = alpha
    @inbounds for i in 1:n
        i == spike && continue
        sigma -= r[i] * c[i] / d[i]
    end
    iszero(sigma) && return nothing
    return ArrowheadFactorization{Tv, Ti}(n, spike, d, c, r, alpha, sigma, common)
end

# Try to build an arrowhead factorization; returns nothing on any non-match.
function _try_arrowhead(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv},
        common::KLUCommon{Ti}
    ) where {Tv, Ti}
    spike = _detect_arrowhead(n, colptr, rowval)
    spike == 0 && return nothing
    return _build_arrowhead(n, colptr, rowval, nzval, spike, common)
end

# Forward solve A x = b (in place on a copy of the RHS column(s)).
function _arrowhead_solve!(K::ArrowheadFactorization{Tv}, B::StridedVecOrMat{Tv}) where {Tv}
    n = K.n; s = K.s; d = K.d; c = K.c; r = K.r; sigma = K.sigma
    @inbounds for col in 1:size(B, 2)
        acc = B[s, col]
        for i in 1:n
            i == s && continue
            yi = B[i, col] / d[i]
            B[i, col] = yi
            acc -= r[i] * yi
        end
        xs = acc / sigma
        for i in 1:n
            i == s && continue
            B[i, col] = B[i, col] - c[i] * xs / d[i]
        end
        B[s, col] = xs
    end
    return B
end

# Transpose solve A' x = b.  A' is the arrowhead with row/col swapped:
# (A')[i,s] = r[i], (A')[s,i] = c[i], same diagonal/corner; Schur scalar is
# identical (sigma uses r[i]*c[i] symmetrically), so swap the roles of r, c.
function _arrowhead_tsolve!(K::ArrowheadFactorization{Tv}, B::StridedVecOrMat{Tv}; conj_solve::Bool = false) where {Tv}
    n = K.n; s = K.s; d = K.d
    c = conj_solve ? conj.(K.c) : K.c
    r = conj_solve ? conj.(K.r) : K.r
    dd = conj_solve ? conj.(K.d) : K.d
    sigma = conj_solve ? conj(K.sigma) : K.sigma
    @inbounds for col in 1:size(B, 2)
        acc = B[s, col]
        for i in 1:n
            i == s && continue
            yi = B[i, col] / dd[i]
            B[i, col] = yi
            acc -= c[i] * yi          # transpose: row of A' is c
        end
        xs = acc / sigma
        for i in 1:n
            i == s && continue
            B[i, col] = B[i, col] - r[i] * xs / dd[i]
        end
        B[s, col] = xs
    end
    return B
end

function solve!(K::ArrowheadFactorization{Tv}, B::StridedVecOrMat{Tv}; check::Bool = true) where {Tv}
    stride(B, 1) == 1 || throw(ArgumentError("B must have unit strides"))
    size(B, 1) == K.n || throw(DimensionMismatch())
    return _arrowhead_solve!(K, B)
end

function solve!(
        K::AdjointFact{Tv, KF}, B::StridedVecOrMat{Tv}; check::Bool = true
    ) where {Tv, Ti, KF <: ArrowheadFactorization{Tv, Ti}}
    P = parent(K)
    stride(B, 1) == 1 || throw(ArgumentError("B must have unit strides"))
    size(B, 1) == P.n || throw(DimensionMismatch())
    return _arrowhead_tsolve!(P, B; conj_solve = (Tv <: Complex))
end

function solve!(
        K::TransposeFact{Tv, KF}, B::StridedVecOrMat{Tv}; check::Bool = true
    ) where {Tv, Ti, KF <: ArrowheadFactorization{Tv, Ti}}
    P = parent(K)
    stride(B, 1) == 1 || throw(ArgumentError("B must have unit strides"))
    size(B, 1) == P.n || throw(DimensionMismatch())
    return _arrowhead_tsolve!(P, B; conj_solve = false)
end
