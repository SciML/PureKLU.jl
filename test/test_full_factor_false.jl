using PureKLU
using PureKLU: klu_factor!
using Test
using SparseArrays: sparse, nonzeros

@testset "full_factor = false" begin
    for T in (Float64, ComplexF64, Float32, ComplexF32)
        A = sparse(Matrix{T}([1 0; 1 1]))
        nonzeros(A) .= zero(T)
        F = PureKLU.klu(A; full_factor = false)
        nonzeros(A) .= one(T)
        nonzeros(F) .= one(T)
        klu_factor!(F)
        b = ones(T, 2)
        @test F \ b ≈ A \ b
    end
end
