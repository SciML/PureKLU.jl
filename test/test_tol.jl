# Trivial test that the `tol` kwarg (diagonal-vs-partial pivoting threshold)
# is actually honored, not just accepted and ignored.
using Test
using SparseArrays
using LinearAlgebra
using PureKLU

@testset "tol kwarg: stored on common, solves correctly" begin
    A = sparse([2.0 1.0; 1.0 3.0])
    b = [1.0, 2.0]
    for tol in (0.0, 0.001, 0.5, 1.0)
        K = PureKLU.klu(A; tol)
        @test K.common.tol == tol
        @test A * (K \ b) ≈ b
    end
end

@testset "tol kwarg: extreme values change pivot choice" begin
    # Column 1 has a tiny diagonal entry (0.001) and a much larger
    # off-diagonal entry (1.0) in row 2. tol = 0.0 always pivots on the
    # diagonal; tol = 1.0 is full partial pivoting and must pick the larger
    # entry instead, so the two runs choose different row pivots.
    A = sparse([0.001 1.0; 1.0 1.0])
    b = [1.0, 1.0]

    K_diag = PureKLU.klu(A; tol = 0.0, detect_banded = false)
    K_partial = PureKLU.klu(A; tol = 1.0, detect_banded = false)

    @test K_diag.p == [1, 2]      # diagonal pivot kept despite tiny magnitude
    @test K_partial.p == [2, 1]   # partial pivoting swaps to the larger entry
    @test A * (K_diag \ b) ≈ b
    @test A * (K_partial \ b) ≈ b
end
