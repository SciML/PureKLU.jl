# The `scale` kwarg selects KLU's row scaling and must be honored, not just
# accepted and ignored.
using Test
using SparseArrays
using LinearAlgebra
using PureKLU

@testset "scale kwarg: stored on common, solves correctly" begin
    A = sparse([2.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 4.0])
    b = [1.0, 2.0, 3.0]
    for scale in (-1, 0, 1, 2)
        K = PureKLU.klu(A; scale)
        @test K.common.scale == scale
        @test A * (K \ b) ≈ b
        @test (K.Rs == ones(3)) == (scale <= 0)
        # `klu!` keeps the scaling chosen at factorization.
        PureKLU.klu!(K, 2 .* A)
        @test K.common.scale == scale
        @test 2 .* A * (K \ b) ≈ b
    end
    @test PureKLU.klu(A).common.scale == 2
    @test PureKLU.klu(
        size(A, 1), A.colptr .- 1, A.rowval .- 1, A.nzval;
        scale = 0
    ).common.scale == 0
end

@testset "scale kwarg: row scaling changes the pivot choice" begin
    # Unscaled, row 2 holds the largest entry of column 1 and is chosen as
    # pivot under partial pivoting. Dividing each row by its largest absolute
    # entry turns column 1 into (0.5, 0.001), so row 1 is chosen instead.
    A = sparse([1.0 2.0; 1.0e3 1.0e6])
    b = [1.0, 1.0]

    K0 = PureKLU.klu(A; scale = 0, tol = 1.0, detect_banded = false)
    K2 = PureKLU.klu(A; scale = 2, tol = 1.0, detect_banded = false)

    @test K0.p == [2, 1]
    @test K2.p == [1, 2]
    @test K2.Rs == [2.0, 1.0e6]
    @test A * (K0 \ b) ≈ b
    @test A * (K2 \ b) ≈ b
end
