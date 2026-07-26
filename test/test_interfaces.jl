using LinearAlgebra
using PureKLU
using SparseArrays
using Test

@testset "LinearAlgebra factorization interface" begin
    A = sparse([2.0 1.0; 1.0 3.0])
    F = klu(A)

    @test F isa Factorization
    @test size(F) == size(A)
    @test size(F, 1) == size(A, 1)
    @test issuccess(F)
    @test length(nonzeros(F)) == nnz(A)

    b = [1.0, 2.0]
    @test F \ b ≈ A \ b

    inplace_b = copy(b)
    @test ldiv!(F, inplace_b) === inplace_b
    @test inplace_b ≈ A \ b

    transpose_b = [3.0, 1.0]
    @test adjoint(F) \ transpose_b ≈ adjoint(A) \ transpose_b
    @test transpose(F) \ transpose_b ≈ transpose(A) \ transpose_b
end
