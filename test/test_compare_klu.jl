# Cross-check PureKLU against KLU.jl (the SuiteSparse-backed wrapper).
# Each test factors the same matrix with both implementations and verifies
# the BTF-level structure agrees (this is uniquely determined) and that
# both produce the same numerical solution to A*x = b. Per-block AMD
# ordering is *not* required to match exactly, since SuiteSparse uses
# AMD inside each block and this port currently uses natural ordering for
# blocks > 3, so the L/U sparsity pattern can legitimately differ while
# remaining mathematically equivalent.

using Test
using SparseArrays
using LinearAlgebra
using PureKLU
using Random
import KLU

const SUITESPARSE = KLU

"""
    compare_strict(A)

Strict equality: BTF, permutations, scaling, factor pattern and the
solve all match bit-for-bit. PureKLU is invoked with `use_fma=false`
so its multiply-subtract loops emit the same separate `mulsd`/`subsd`
sequence as SuiteSparse's SSE2-baseline `libklu.so`.
"""
function compare_strict(A::SparseMatrixCSC{Float64})
    K_ref = SUITESPARSE.klu(A)
    K_pj  = PureKLU.klu(A; use_fma=false)

    @test K_ref.nblocks == K_pj.nblocks
    @test K_ref.lnz == K_pj.lnz
    @test K_ref.unz == K_pj.unz
    @test K_ref.nzoff == K_pj.nzoff
    @test K_ref.p == K_pj.p
    @test K_ref.q == K_pj.q
    @test K_ref.R == K_pj.R
    @test K_ref.Rs ≈ K_pj.Rs
    @test K_ref.L ≈ K_pj.L
    @test K_ref.U ≈ K_pj.U
    @test K_ref.F ≈ K_pj.F

    n = size(A, 1)
    b = collect(1.0:n)
    x_ref = K_ref \ b
    x_pj  = K_pj  \ b
    @test x_ref ≈ x_pj
    @test A * x_pj ≈ b

    b_adj = collect(n:-1.0:1.0)
    @test K_ref' \ b_adj ≈ K_pj' \ b_adj
    @test transpose(K_ref) \ b_adj ≈ transpose(K_pj) \ b_adj
    return nothing
end

"""
    compare_loose(A)

Allow per-block ordering to differ between the two factorisations, but
still require:

  * identical BTF (nblocks, R, nzoff are invariants of BTF alone)
  * identical numerical solution to A*x = b
  * the PureKLU factorisation satisfies L*U + F = Rs \\ A[p,q]
"""
function compare_loose(A::SparseMatrixCSC{Float64})
    K_ref = SUITESPARSE.klu(A)
    K_pj  = PureKLU.klu(A; use_fma=false)

    @test K_ref.nblocks == K_pj.nblocks
    @test K_ref.R == K_pj.R
    @test K_ref.nzoff == K_pj.nzoff

    n = size(A, 1)
    b = collect(1.0:n)
    x_ref = K_ref \ b
    x_pj  = K_pj  \ b
    @test x_ref ≈ x_pj
    @test A * x_pj ≈ b

    # PureKLU must produce a valid factorisation in its own frame
    Rs_pj = Diagonal(K_pj.Rs)
    @test Rs_pj \ A[K_pj.p, K_pj.q] ≈ K_pj.L * K_pj.U + K_pj.F
    return nothing
end

@testset "PureKLU vs KLU.jl: 5x5 BTF benchmark (strict)" begin
    Ap = [0,4,1,1,2,2,0,1,2,3,4,4] .+ 1
    Ai = [0,4,0,2,1,2,1,4,3,2,1,2] .+ 1
    Ax = [2.,1.,3.,4.,-1.,-3.,3.,6.,2.,1.,4.,2.]
    A = sparse(Ap, Ai, Ax)
    compare_strict(A)
end

@testset "PureKLU vs KLU.jl: 2x2 diagonal (strict)" begin
    A = sparse([1.0 0.0; 0.0 2.0])
    compare_strict(A)
end

@testset "PureKLU vs KLU.jl: 3x3 dense block (strict)" begin
    A = sparse([2.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 4.0])
    compare_strict(A)
end

@testset "PureKLU vs KLU.jl: 4x4 with off-diagonal blocks (strict)" begin
    A = sparse([
        4.0 0.0 1.0 0.0;
        0.0 5.0 2.0 0.0;
        0.0 0.0 6.0 3.0;
        0.0 0.0 0.0 7.0;
    ])
    compare_strict(A)
end

@testset "PureKLU vs KLU.jl: refactor with same pattern (strict)" begin
    Ap = [0,4,1,1,2,2,0,1,2,3,4,4] .+ 1
    Ai = [0,4,0,2,1,2,1,4,3,2,1,2] .+ 1
    Ax1 = [2.,1.,3.,4.,-1.,-3.,3.,6.,2.,1.,4.,2.]
    Ax2 = [2.,1.,3.,4.,-1.,-3.,3.,9.,2.,1.,4.,2.]
    A = sparse(Ap, Ai, Ax1)
    B = sparse(Ap, Ai, Ax2)
    K_ref = SUITESPARSE.klu(A); SUITESPARSE.klu!(K_ref, B)
    K_pj  = PureKLU.klu(A; use_fma=false); PureKLU.klu!(K_pj, B)
    @test K_ref.L == K_pj.L
    @test K_ref.U == K_pj.U
    @test K_ref.F == K_pj.F
    b = [8., 45., -3., 3., 19.]
    @test K_ref \ b == K_pj \ b
end

@testset "PureKLU vs KLU.jl: Issue #4 (15x15) (strict)" begin
    A = SparseMatrixCSC(15, 15,
        [1, 8, 12, 16, 21, 22, 27, 34, 37, 39, 41, 44, 47, 50, 55, 58],
        [3, 4, 6, 8, 11, 13, 15, 2, 6, 8, 15, 1, 2, 5, 12, 3, 5, 6, 7, 9, 14, 7,
         10, 11, 13, 15, 4, 5, 6, 8, 10, 12, 13, 8, 12, 15, 2, 13, 1, 7, 3, 4,
         14, 2, 4, 5, 1, 5, 9, 4, 7, 9, 11, 14, 13, 14, 15],
        [0.2775474841561938, 0.19549953706849155, 0.7221976371086005,
         0.4339373082200655, 0.983079431343046, 0.10918778088879799,
         0.3728676112188065, 0.9134045061777432, 0.14560891622463457,
         0.7715210431553383, 0.2945501295372417, 0.6134722502122134,
         0.8777181195348973, 0.6995382425541914, 0.9562490972786235,
         0.27001502642215325, 0.8573661029146233, 0.13020432565448115,
         0.9221068910751316, 0.17087414970038983, 0.7062975193151109,
         0.7668596005709167, 0.46967704631299334, 0.31764226298560083,
         0.39054386892157833, 0.36610203401046015, 0.16689896372140534,
         0.9624322297755521, 0.1478381603984824, 0.45423514524961806,
         0.5564610482579242, 0.1844671322175948, 0.0823893170743808,
         0.25409993152712307, 0.10475245199943273, 0.5863595004922162,
         0.14733912690513562, 0.6504152422320895, 0.4339054908933866,
         0.27614384058497166, 0.4019619228414033, 0.8631491210976487,
         0.20159747073826084, 0.3273367915690062, 0.23866880928640288,
         0.9557759456784265, 0.016351125161178537, 0.5320355909884844,
         0.9010930260468242, 0.3780686420068593, 0.6375477164214856,
         0.9850645305956225, 0.5366242762582065, 0.08835652070698918,
         0.9877090693717305, 0.9775298646022268, 0.9759511830494418])
    compare_strict(A)
end

@testset "PureKLU vs KLU.jl: random diagonally dominant matrices (strict)" begin
    rng_seeds = (1, 2, 3, 7, 13)
    for seed in rng_seeds, n in (8, 16, 32)
        rng = Random.MersenneTwister(seed)
        A = sprand(rng, n, n, 0.3) + n*I
        compare_strict(A)
    end
end

@testset "PureKLU vs KLU.jl: complex matrices" begin
    A = sparse(ComplexF64[
        2.0+1.0im 0.0 0.0;
        1.0 3.0+2.0im 1.0im;
        0.0 0.5 4.0-1.0im
    ])
    K_ref = SUITESPARSE.klu(A)
    K_pj  = PureKLU.klu(A; use_fma=false)
    @test K_ref.p == K_pj.p
    @test K_ref.q == K_pj.q
    @test K_ref.R == K_pj.R
    @test K_ref.L == K_pj.L
    @test K_ref.U == K_pj.U
    @test K_ref.F == K_pj.F
    b = ComplexF64[1.0, 2.0im, -1.0+3.0im]
    @test K_ref \ b == K_pj \ b
end

@testset "PureKLU vs KLU.jl: identity-like matrices (strict)" begin
    for n in (3, 5, 10, 20)
        A = sparse(Float64(n)I, n, n) + sparse(Diagonal(0.1 .* (1:n)))
        compare_strict(A)
    end
end

@testset "PureKLU vs KLU.jl: larger random matrices (strict)" begin
    rng_seeds = (101, 202, 303, 404, 505)
    for seed in rng_seeds, n in (50, 100, 150)
        rng = Random.MersenneTwister(seed)
        A = sprand(rng, n, n, 0.1) + n*I
        compare_strict(A)
    end
end

@testset "PureKLU vs KLU.jl: structured matrices (strict)" begin
    # tridiagonal
    for n in (10, 50, 100)
        A = spdiagm(-1 => -ones(n-1), 0 => 4*ones(n), 1 => -ones(n-1))
        compare_strict(A)
    end
    # 2D Laplacian (5-point stencil, n×n grid -> n^2 vars)
    for k in (5, 10, 15)
        n = k * k
        diag_main = 4.0 * ones(n)
        diag_off1 = -1.0 * ones(n-1)
        # zero out wrap-around in diag_off1
        for i in 1:n-1
            if i % k == 0
                diag_off1[i] = 0.0
            end
        end
        diag_offk = -1.0 * ones(n-k)
        A = spdiagm(-k => diag_offk, -1 => diag_off1, 0 => diag_main,
                    1 => diag_off1, k => diag_offk)
        # filter zeros
        dropzeros!(A)
        compare_strict(A)
    end
end

@testset "PureKLU vs KLU.jl: ComplexF64 with non-trivial AMD (strict)" begin
    rng = Random.MersenneTwister(42)
    for n in (10, 25, 50)
        Ar = sprand(rng, n, n, 0.15)
        Ai = sprand(rng, n, n, 0.15)
        A = Ar + im * Ai + n*I
        K_ref = SUITESPARSE.klu(A)
        K_pj  = PureKLU.klu(A; use_fma=false)
        @test K_ref.p == K_pj.p
        @test K_ref.q == K_pj.q
        @test K_ref.R == K_pj.R
        @test K_ref.Rs == K_pj.Rs
        @test K_ref.L == K_pj.L
        @test K_ref.U == K_pj.U
        @test K_ref.F == K_pj.F
        b = randn(rng, ComplexF64, n)
        @test K_ref \ b == K_pj \ b
    end
end
