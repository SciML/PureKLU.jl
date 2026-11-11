using PureKLU, SparseArrays, LinearAlgebra, Random, Test
import KLU

# `libklu` ships SSE2-only on x86_64 and FMA-enabled on aarch64, so matching
# its arithmetic is what makes the exact comparison below meaningful.
const USE_FMA = Sys.ARCH === :aarch64

function cases()
    Random.seed!(7)
    return [
        "2x2 dense" => sparse([2.0 1.0; 1.0 3.0]),
        "identity" => sparse(1.0I, 5, 5),
        "tridiagonal" => spdiagm(-1 => ones(9), 0 => fill(4.0, 10), 1 => ones(9)),
        "diagonally dominant" => sprand(60, 60, 0.12) + 20I,
        "near-singular" => sprand(80, 80, 0.06) + 0.01I,
        "block triangular" => vcat(
            hcat(sprand(6, 6, 0.5) + 5I, sprand(6, 6, 0.3)),
            hcat(spzeros(6, 6), sprand(6, 6, 0.5) + 5I)
        ),
        "complex" => sprand(ComplexF64, 40, 40, 0.1) + 8I,
    ]
end

@testset "rgrowth matches SuiteSparse" begin
    for (label, A) in cases(), scale in (0, 1, 2)
        pure = klu(sparse(A); check = false, use_fma = USE_FMA)
        suite = KLU.klu(sparse(A); check = false)
        pure.common.scale = suite.common.scale = Cint(scale)
        klu_factor!(pure; check = false)
        KLU.klu_factor!(suite; check = false)
        @test PureKLU.rgrowth(pure) ≈ KLU.rgrowth(suite) rtol = 1.0e-12
        @test pure.common.rgrowth == PureKLU.rgrowth(pure)
    end
end

@testset "rgrowth after a reused-pivot refactorization" begin
    A = spdiagm(-1 => ones(19), 0 => fill(3.0, 20), 1 => ones(19))
    values = copy(nonzeros(A))
    values[1:3:end] .*= 1.0e-10
    pure = klu(A; check = false, use_fma = USE_FMA)
    suite = KLU.klu(A; check = false)
    klu!(pure, values; check = false)
    KLU.klu!(suite, values; check = false)
    @test PureKLU.rgrowth(pure) ≈ KLU.rgrowth(suite) rtol = 1.0e-10
end

# The growth test is only worth anything if it separates the refactorization
# that has to be redone from the one that does not.
@testset "rgrowth flags pivots that reuse has made unstable" begin
    A = sparse([1.0 1.0; 1.0 2.0])
    factor = klu(A)
    @test PureKLU.rgrowth(factor) > 0.1

    B = copy(A)
    B[1, 1] = 1.0e-14
    klu!(factor, nonzeros(B); check = false)
    @test PureKLU.rgrowth(factor) < sqrt(eps(Float64))
    klu_factor!(factor; check = false)
    @test PureKLU.rgrowth(factor) > 0.1
    @test factor \ (B * [1.0, 2.0]) ≈ [1.0, 2.0] rtol = 1.0e-12
end

@testset "rgrowth needs a numeric factorization" begin
    K = PureKLU.KLUFactorization(sparse([2.0 1.0; 1.0 3.0]))
    @test_throws ArgumentError PureKLU.rgrowth(K)
    klu_analyze!(K)
    @test_throws ArgumentError PureKLU.rgrowth(K)
    klu_factor!(K)
    @test PureKLU.rgrowth(K) > 0.1
end
