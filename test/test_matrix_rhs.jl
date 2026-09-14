using Test
using PureKLU
using SparseArrays
using LinearAlgebra
using Random

# Compare against repeated vector solves using the very same numeric factors.
function column_solve!(K, B)
    for j in axes(B, 2)
        solve!(K, view(B, :, j))
    end
    return B
end

@testset "Blocked matrix RHS" begin
    rng = MersenneTwister(714)
    # Two irreducible blocks plus a singleton and off-diagonal BTF couplings.
    # Small first diagonal forces pivoting; row magnitudes exercise scaling.
    A = sparse([0.01 2 3 0 1; 4 5 0 2 0; 0 0 6 1 2; 0 0 2 7 0; 0 0 0 0 8.0])
    p = [3, 1, 5, 2, 4]
    q = [2, 5, 1, 4, 3]
    A = A[p, q]
    configurations = (
        (Float64, Int, true, 2), (Float64, Int, false, 0),
        (Float32, Int32, true, 1), (Float32, Int32, false, 2),
        (Float64, Int32, false, 1), (Float32, Int, true, 0),
    )
    for (Tv, Ti, fma, scale) in configurations
        boundaries = (Tv, Ti, fma, scale) == first(configurations)
        At = SparseMatrixCSC{Tv, Ti}(A)
        K = klu(At; use_fma = fma, full_factor = false, detect_banded = false)
        K.common.scale = scale
        klu_factor!(K)
        @test K.nblocks == 3
        @test K.nzoff > 0
        @test isempty(K.numeric.Xrhs)
        counts = boundaries ? (0, 1, 7, 8, 16, 63, 64, 65, 129, 9, 256, 8) : (65,)
        for nrhs in counts
            B = randn(rng, Tv, 5, nrhs)
            expected = column_solve!(K, copy(B))
            X = copy(B)
            solve!(K, X)
            @test X == expected
            @test At * X ≈ B
        end
        X = ones(Tv, 5, 65)
        @test solve!(K, X) === X
        @test (@allocated solve!(K, X)) == 0
        boundaries || continue
        scratch = K.numeric.Xrhs
        K2 = deepcopy(K)
        @test K2.numeric.Xrhs !== scratch
        @test klu(At).numeric.Xrhs !== scratch
        for refactor! in (klu!, (F, A) -> klu_factor!(F))
            refactor!(K, At)
            @test K.numeric.Xrhs === scratch
            B = randn(rng, Tv, 5, 65)
            @test solve!(K, copy(B)) == column_solve!(K2, copy(B))
        end
        changed = 2 * At
        klu!(K, changed)
        B = randn(rng, Tv, 5, 65)
        @test changed * solve!(K, copy(B)) ≈ B
        @test At * solve!(K2, copy(B)) ≈ B

        storage = randn(rng, Tv, 7, 130)
        X = view(storage, 2:6, 1:2:130)
        expected = column_solve!(K, copy(X))
        @test solve!(K, X) === X
        @test X == expected
        @test_throws ArgumentError solve!(K, view(storage, 1:2:7, :))
        @test_throws DimensionMismatch solve!(K, zeros(Tv, 4, 65))
    end
end

@testset "Matrix RHS fallbacks and singular status" begin
    for Tv in (BigFloat, ComplexF64)
        A = sparse(Tv[4 1; 2 3])
        K = klu(A)
        B = ones(Tv, 2, 65)
        @test solve!(K, copy(B)) == column_solve!(K, copy(B))
        @test isempty(K.numeric.Xrhs)
        @test A' * solve!(K', copy(B)) ≈ B
        @test transpose(A) * solve!(transpose(K), copy(B)) ≈ B
    end
    A = sparse([1.0 0; 0 0])
    for check in (true, false)
        K = klu(A; allowsingular = true)
        rank, col = K.common.numerical_rank, K.common.singular_col
        B = ones(2, 65)
        expected = column_solve!(K, copy(B))
        @test isequal(solve!(K, B; check), expected)
        @test K.common.status == PureKLU.KLU_SINGULAR
        @test (K.common.numerical_rank, K.common.singular_col) == (rank, col)
    end
end
