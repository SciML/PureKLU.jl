using PureKLU, SparseArrays, LinearAlgebra, Test

@testset "New pivots with a retained symbolic analysis" begin
    for T in (Float64, ComplexF64), Ti in Base.uniontypes(PureKLU.KLUITypes),
            matrix_input in (false, true)
        A = SparseMatrixCSC{T, Ti}(sparse(T[1 1; 1 2]))
        factor = klu(A)
        expected = T[1, 2]
        ordering = copy(factor.q)
        @test factor \ (A * expected) ≈ expected
        for pivot in (1.0e-10, 1.0e-16, 0.0, 1.0)
            B = copy(A)
            B[1, 1] = pivot
            values = matrix_input ? B : nonzeros(B)
            @test klu!(factor, values; reuse_pivots = false) === factor
            @test factor.q == ordering
            @test issuccess(factor)
            result = factor \ (B * expected)
            @test result ≈ expected rtol = 1.0e-12 atol = 1.0e-12
            @test B * result ≈ B * expected rtol = 1.0e-12 atol = 1.0e-12
        end
    end
    factor = klu(sparse([1.0 1.0; 1.0 2.0]))
    @test klu!(factor, Float32[0, 1, 1, 2]; reuse_pivots = false) === factor
    @test factor \ [2.0, 5.0] ≈ [1.0, 2.0]
end
