using PureKLU
using PureKLU: klu, klu!, klu_factor!, klu_analyze!, solve!
using SparseArrays
using LinearAlgebra
using Random
using Test

function _small_test_matrix(seed::Int=1)
    rng = MersenneTwister(seed)
    n = 200
    A = sprand(rng, n, n, 0.05) + n*I
    return SparseMatrixCSC(A)
end

@testset "Zero-allocation hot paths (post-warmup)" begin
    A = _small_test_matrix()
    n = size(A, 1)

    # Warm everything up so JIT/specialisation is done.
    K = klu(A)
    b = randn(n)
    bw = copy(b);            solve!(K, bw)
    bw = copy(b);            solve!(K', bw)
    bw = copy(b);            solve!(transpose(K), bw)
    klu!(K, A.nzval)

    # solve!
    b1 = copy(b)
    @test (@allocated solve!(K, b1)) == 0

    # adjoint solve
    b2 = copy(b)
    @test (@allocated solve!(K', b2)) == 0

    # transpose solve
    b3 = copy(b)
    @test (@allocated solve!(transpose(K), b3)) == 0

    # refactor (same pattern, new values)
    @test (@allocated klu!(K, A.nzval)) == 0

    # Sanity: each path still produced the right answer.
    @test A * b1 ≈ b
    @test A' * b2 ≈ b
    @test transpose(A) * b3 ≈ b
end
