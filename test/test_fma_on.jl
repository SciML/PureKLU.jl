# Tests with FMA fusion enabled (the default `use_fma=true`).
#
# Multiply-subtract kernels emit `muladd(-b, c, a)` which LLVM lowers to a
# fused multiply-add (`vfnmadd*sd`) on x86/AVX-FMA. Results therefore
# differ from SuiteSparse's SSE2-only `libklu.so` by up to 1 ULP per
# scalar operation. We use `≈` for value comparisons, but structural
# fields (`p`, `q`, `R`, `nzoff`, `nblocks`, etc.) are still strict --
# FMA doesn't affect pivot choices or BTF, only the rounding of the
# accumulated values.

using Test
using SparseArrays
using LinearAlgebra
using PureKLU
using Random
import KLU

const SUITESPARSE = KLU

function compare_fma_on(A::SparseMatrixCSC)
    K_ref = SUITESPARSE.klu(A)
    K_pj  = PureKLU.klu(A)  # use_fma=true is default
    # Structural fields are unchanged by FMA -- still strict
    @test K_ref.p == K_pj.p
    @test K_ref.q == K_pj.q
    @test K_ref.R == K_pj.R
    @test K_ref.nblocks == K_pj.nblocks
    @test K_ref.nzoff == K_pj.nzoff
    @test K_ref.lnz == K_pj.lnz
    @test K_ref.unz == K_pj.unz
    @test K_ref.Rs == K_pj.Rs                 # Rs is computed pre-LU, no FMA path
    # Numerical values can differ by ~ULP
    @test K_ref.L ≈ K_pj.L
    @test K_ref.U ≈ K_pj.U
    @test K_ref.F ≈ K_pj.F
    return K_ref, K_pj
end

@testset "FMA-on default: 5x5 BTF reference" begin
    Ap = [0,4,1,1,2,2,0,1,2,3,4,4] .+ 1
    Ai = [0,4,0,2,1,2,1,4,3,2,1,2] .+ 1
    Ax = [2.,1.,3.,4.,-1.,-3.,3.,6.,2.,1.,4.,2.]
    A = sparse(Ap, Ai, Ax)
    K_ref, K_pj = compare_fma_on(A)
    b = [8., 45., -3., 3., 19.]
    @test K_ref \ b ≈ K_pj \ b
    @test A * (K_pj \ b) ≈ b
end

@testset "FMA-on default: small dense blocks (3x3, 4x4)" begin
    A1 = sparse([2.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 4.0])
    compare_fma_on(A1)

    A2 = sparse([
        4.0 0.0 1.0 0.0;
        0.0 5.0 2.0 0.0;
        0.0 0.0 6.0 3.0;
        0.0 0.0 0.0 7.0;
    ])
    compare_fma_on(A2)
end

@testset "FMA-on default: random diagonally dominant matrices" begin
    Random.seed!(7)
    for n in (10, 25, 50, 100), density in (0.05, 0.1, 0.3)
        A = sprand(n, n, density) + n*I
        dropzeros!(A)
        K_ref, K_pj = compare_fma_on(A)
        b = randn(n)
        @test K_ref \ b ≈ K_pj \ b
        @test A * (K_pj \ b) ≈ b
    end
end

@testset "FMA-on default: 2D Laplacian grids" begin
    for k in (5, 10, 15, 20)
        n = k * k
        diag_main = 4.0 * ones(n)
        diag_off1 = -1.0 * ones(n-1)
        for i in 1:n-1
            if i % k == 0
                diag_off1[i] = 0.0
            end
        end
        diag_offk = -1.0 * ones(n-k)
        A = spdiagm(-k => diag_offk, -1 => diag_off1, 0 => diag_main,
                    1 => diag_off1, k => diag_offk)
        dropzeros!(A)
        K_ref, K_pj = compare_fma_on(A)
        b = ones(n)
        @test K_ref \ b ≈ K_pj \ b
    end
end

@testset "FMA-on default: ComplexF64" begin
    Random.seed!(42)
    for n in (10, 25, 50)
        Ar = sprand(n, n, 0.15)
        Ai = sprand(n, n, 0.15)
        A = Ar + im * Ai + n*I
        K_ref = KLU.klu(A); K_pj = PureKLU.klu(A)
        @test K_ref.p == K_pj.p
        @test K_ref.q == K_pj.q
        @test K_ref.L ≈ K_pj.L
        @test K_ref.U ≈ K_pj.U
        b = randn(ComplexF64, n)
        @test K_ref \ b ≈ K_pj \ b
    end
end

@testset "FMA-on default: solve and tsolve" begin
    Random.seed!(13)
    A = sprand(40, 40, 0.15) + 40*I
    K_ref = KLU.klu(A); K_pj = PureKLU.klu(A)
    b = randn(40)
    B = randn(40, 5)
    @test K_ref \ b ≈ K_pj \ b
    @test K_ref \ B ≈ K_pj \ B
    @test K_ref' \ b ≈ K_pj' \ b
    @test transpose(K_ref) \ b ≈ transpose(K_pj) \ b
    @test K_ref' \ B ≈ K_pj' \ B
end

@testset "FMA-on default: refactor (single and multi)" begin
    Random.seed!(99)
    n = 30
    proto = sprand(n, n, 0.2) + n*I
    I_idx, J_idx, _ = findnz(proto)
    K_ref = KLU.klu(proto); K_pj = PureKLU.klu(proto)
    for iter in 1:5
        V = randn(length(I_idx)) .+ 0.7
        B = sparse(I_idx, J_idx, V, n, n)
        KLU.klu!(K_ref, B); PureKLU.klu!(K_pj, B)
        @test K_ref.L ≈ K_pj.L
        @test K_ref.U ≈ K_pj.U
        @test K_ref.F ≈ K_pj.F
        @test K_ref.Rs == K_pj.Rs
        b = randn(n)
        @test K_ref \ b ≈ K_pj \ b
    end
end

@testset "FMA-on default: 200x200 random matrices" begin
    Random.seed!(2024)
    for seed in (1, 2, 3), density in (0.02, 0.05)
        rng = MersenneTwister(seed)
        A = sprand(rng, 200, 200, density) + 200*I
        dropzeros!(A)
        compare_fma_on(A)
    end
end

@testset "FMA-on: self-consistency invariants" begin
    # FMA mode should still produce a valid factorisation
    Random.seed!(303)
    for n in (10, 50, 100)
        A = sprand(n, n, 0.15) + n*I
        dropzeros!(A)
        K_pj = PureKLU.klu(A)  # use_fma=true
        Rs = Diagonal(K_pj.Rs)
        @test Rs \ A[K_pj.p, K_pj.q] ≈ K_pj.L * K_pj.U + K_pj.F
        b = randn(n)
        x = K_pj \ b
        @test A * x ≈ b
    end
end

@testset "FMA-on vs FMA-off: structural identity" begin
    # On vs off must agree on EVERYTHING except L/U/F values
    Random.seed!(404)
    for n in (10, 30, 100)
        A = sprand(n, n, 0.15) + n*I
        dropzeros!(A)
        K_on  = PureKLU.klu(A; use_fma=true)
        K_off = PureKLU.klu(A; use_fma=false)
        @test K_on.p == K_off.p
        @test K_on.q == K_off.q
        @test K_on.R == K_off.R
        @test K_on.nblocks == K_off.nblocks
        @test K_on.nzoff == K_off.nzoff
        @test K_on.lnz == K_off.lnz
        @test K_on.unz == K_off.unz
        @test K_on.Rs == K_off.Rs
        @test K_on.L ≈ K_off.L
        @test K_on.U ≈ K_off.U
        @test K_on.F == K_off.F  # F entries don't go through FMA
    end
end

@testset "use_fma kwarg accepts Val(true)/Val(false) and Bool" begin
    A = sparse([2.0 1.0; 0.0 3.0])
    K_bool_true  = PureKLU.klu(A; use_fma=true)
    K_val_true   = PureKLU.klu(A; use_fma=Val(true))
    K_bool_false = PureKLU.klu(A; use_fma=false)
    K_val_false  = PureKLU.klu(A; use_fma=Val(false))
    @test K_bool_true.common.use_fma === Val(true)
    @test K_val_true.common.use_fma === Val(true)
    @test K_bool_false.common.use_fma === Val(false)
    @test K_val_false.common.use_fma === Val(false)
    @test K_bool_true.L == K_val_true.L
    @test K_bool_false.L == K_val_false.L
end

@testset "FMA-on vs FMA-off: explicit divergence on identical input" begin
    # Verify the two modes actually take different paths -- at least one
    # nontrivial matrix should produce non-bit-identical L while both
    # remain valid. (If they always matched, the toggle would be a no-op.)
    Random.seed!(505)
    saw_difference = false
    for _ in 1:5
        A = sprand(40, 40, 0.2) + 40*I
        dropzeros!(A)
        K_on  = PureKLU.klu(A; use_fma=true)
        K_off = PureKLU.klu(A; use_fma=false)
        if K_on.L != K_off.L
            saw_difference = true
            break
        end
    end
    @test saw_difference
end
