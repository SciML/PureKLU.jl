module PureKLUForwardDiffExt

using PureKLU: PureKLU, KLUFactorization, solve!
using LinearAlgebra: LinearAlgebra
using ForwardDiff: ForwardDiff, Dual, Partials

# Read row `i`'s partial columns (2:N+1) of `B` into a tuple. A function barrier so the
# `ntuple` closure captures arguments rather than the boxed loop index `i` of the caller
# (which would otherwise allocate per element).
@inline function _row_partials(B, i, ::Val{N}, ::Type{V}) where {N, V}
    return ntuple(j -> V(@inbounds B[i, j + 1]), Val(N))
end

# Backsolve a real (primal) KLU factorization against a Dual right-hand side.
#
# A factorization is a linear operator, so `K \ Dual(v, p₁…p_N) = Dual(K\v, K\p₁ … K\p_N)`.
# A `KLUFactorization{<:AbstractFloat}` stores its L/U in real arithmetic, so the value and
# every partial column can be solved in one real multi-RHS `solve!` without ever promoting
# the matrix to a Dual — the factorization stays real and the duals ride only through the
# back-substitution. This mirrors the existing real-factor / Complex-RHS `ldiv!` (a Complex
# is a Dual with one imaginary "partial").
#
# Non-allocating: the `n × (N+1)` real RHS is laid out in `K.solve_scratch`, a buffer that
# lives on the factorization and is grown on demand, so repeated solves (e.g. one per Newton
# step under an implicit ODE) allocate nothing after warmup.
#
# The factor eltype is constrained to `AbstractFloat` (a `Dual` is `<: Real` but not
# `<: AbstractFloat`) so this never collides with PureKLU's own
# `ldiv!(::KLUFactorization{Tv}, ::VecOrMat{Tv})` on the dual-factor path.
# `KLUFactorization{Tv, Ti, Tv}` pins the numeric eltype == real eltype, i.e. a genuinely
# real factor; this also makes `K.solve_scratch::Matrix{Tv}` concretely typed here.
function LinearAlgebra.ldiv!(
        K::KLUFactorization{Tv, Ti, Tv},
        b::AbstractVector{Dual{T, V, N}}
    ) where {Tv <: AbstractFloat, Ti, T, V, N}
    n = length(b)
    n == size(K, 1) || throw(DimensionMismatch("RHS length $n ≠ factor size $(size(K, 1))"))

    B = getfield(K, :solve_scratch)
    if size(B) != (n, N + 1)
        B = similar(B, n, N + 1) # only when the column count changes
        setfield!(K, :solve_scratch, B)
    end

    @inbounds for i in 1:n
        bi = b[i]
        B[i, 1] = ForwardDiff.value(bi)
        p = ForwardDiff.partials(bi)
        for j in 1:N
            B[i, j + 1] = p[j]
        end
    end

    solve!(K, B) # one real multi-RHS solve; reuses the primal factorization

    @inbounds for i in 1:n
        b[i] = Dual{T, V, N}(B[i, 1], Partials{N, V}(_row_partials(B, i, Val(N), V)))
    end
    return b
end

# 3-arg form: `LinearSolve` calls `ldiv!(y, F, b)`. Solve in place on the scratch-backed
# 2-arg path, then write the Dual solution into `y` (`y` and `b` may alias).
function LinearAlgebra.ldiv!(
        y::AbstractVector{Dual{T, V, N}},
        K::KLUFactorization{Tv, Ti, Tv},
        b::AbstractVector{Dual{T, V, N}}
    ) where {Tv <: AbstractFloat, Ti, T, V, N}
    y === b || copyto!(y, b)
    return LinearAlgebra.ldiv!(K, y)
end

end # module
