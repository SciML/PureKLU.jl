using PureKLU, BenchmarkTools
using StableRNGs, SparseArrays, LinearAlgebra

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Sparse matrices via sprand
A500 = sprand(rng, 500, 500, 0.02) + 5I
A2000 = sprand(rng, 2000, 2000, 0.005) + 10I
b500 = rand(rng, 500)
b2000 = rand(rng, 2000)

# =============================================================================
# Factorization
# =============================================================================

SUITE["factorize"] = BenchmarkGroup()

SUITE["factorize"]["klu_500"] = @benchmarkable klu($A500)
SUITE["factorize"]["klu_2000"] = @benchmarkable klu($A2000)
SUITE["factorize"]["klu_analyze_2000"] = @benchmarkable klu_analyze!(k) setup = (
    k = klu($A2000)
)

K500 = klu(A500)
K2000 = klu(A2000)

# =============================================================================
# Solve
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["ldiv_500"] = @benchmarkable $K500 \ $b500
SUITE["solve"]["ldiv_2000"] = @benchmarkable $K2000 \ $b2000

# Refactorization of a matrix with the same sparsity pattern
A500b = copy(A500)
A500b.nzval .*= 1.01
SUITE["solve"]["klu_refactor_500"] = @benchmarkable klu_refactor!($K500, $A500b)
