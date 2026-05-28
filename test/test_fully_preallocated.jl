using PureKLU
using PureKLU: klu, klu!, klu_factor!, klu_analyze!, solve!, KLUFactorization
using SparseArrays
using LinearAlgebra
using Random
using Test

# Scales AMD's `Lnz` estimate down to force the default-mode grow path;
# isolates the grow-path branch from matrix-specific quirks.
const _TIGHT_INITMEM = 0.1

@testset "fully_preallocated: bit-identical results" begin
    rng = MersenneTwister(1234)
    matrices = Any[
        ("sprand_50_0.1",  sprand(rng, 50, 50, 0.1) + 50*I),
        ("sprand_100_0.1", sprand(rng, 100, 100, 0.1) + 100*I),
        ("sprand_200_0.05", sprand(rng, 200, 200, 0.05) + 200*I),
    ]
    for (name, A) in matrices
        n = size(A, 1)
        b = randn(rng, n)
        K_def = klu(A)
        K_pre = klu(A; fully_preallocated=true)
        x_def = K_def \ copy(b)
        x_pre = K_pre \ copy(b)
        @test x_def == x_pre
        @test K_def.lnz == K_pre.lnz
        @test K_def.unz == K_pre.unz
        @test K_def.nzoff == K_pre.nzoff
    end
end

@testset "fully_preallocated: zero allocations even when default path grows" begin
    rng = MersenneTwister(7)
    A = sprand(rng, 100, 100, 0.2) + 100*I

    K_def = KLUFactorization(A)
    K_def.common.initmem_amd = _TIGHT_INITMEM
    klu_analyze!(K_def); klu_factor!(K_def)

    # Confirm the default path actually grew a block -- otherwise the
    # zero-alloc check below would pass for the wrong reason.
    Sym = K_def.symbolic; Num = K_def.numeric
    did_grow = false
    for b in 1:Int(Sym.nblocks)
        k1 = Int(Sym.R[b]); k2 = Int(Sym.R[b+1]); nk = k2 - k1
        nk <= 1 && continue
        est = Sym.Lnz[b]
        init_cap = est >= 0 ?
            ceil(Int, _TIGHT_INITMEM*est) + nk :
            ceil(Int, Float64(nk)^2/4 * _TIGHT_INITMEM) + nk
        bk = Num.LUbx[b]
        if length(bk.Li) > init_cap || length(bk.Ui) > init_cap
            did_grow = true
            break
        end
    end
    @test did_grow

    K_pre = KLUFactorization(A)
    K_pre.common.initmem_amd = _TIGHT_INITMEM
    K_pre.common.fully_preallocated = true
    klu_analyze!(K_pre); klu_factor!(K_pre)
    klu_factor!(K_pre)

    @test (@allocated klu_factor!(K_pre)) == 0
    nz = A.nzval
    @test (@allocated klu!(K_pre, nz)) == 0

    b = randn(MersenneTwister(99), size(A, 1))
    b1 = copy(b)
    solve!(K_pre, b1)
    b1 = copy(b)
    @test (@allocated solve!(K_pre, b1)) == 0
    @test A * b1 ≈ b
end

@testset "fully_preallocated: capacity bound is sufficient on tough matrices" begin
    rng = MersenneTwister(31415)
    for (n, d) in [(30, 0.5), (50, 0.3), (100, 0.2), (200, 0.1)]
        A = sprand(rng, n, n, d) + n*I
        K = klu(A; fully_preallocated=true)
        Sym = K.symbolic; Num = K.numeric
        for b in 1:Int(Sym.nblocks)
            k1 = Int(Sym.R[b]); k2 = Int(Sym.R[b+1]); nk = k2 - k1
            nk <= 1 && continue
            bk = Num.LUbx[b]
            bound = (nk * (nk + 1)) >> 1
            @test bk.Li_used <= bound
            @test bk.Ui_used <= bound
            @test length(bk.Li) == bound
            @test length(bk.Ui) == bound
        end
    end
end

@testset "fully_preallocated: kwarg accepts true, false, and nothing (auto)" begin
    A = sprand(MersenneTwister(2), 30, 30, 0.1) + 30*I
    K1 = klu(A; fully_preallocated=true)
    K2 = klu(A; fully_preallocated=false)
    K3 = klu(A; fully_preallocated=nothing)
    K4 = klu(A)
    b = randn(MersenneTwister(3), 30)
    @test K1 \ copy(b) == K2 \ copy(b)
    @test K1 \ copy(b) == K3 \ copy(b)
    @test K3 \ copy(b) == K4 \ copy(b)
end

@testset "fully_preallocated: auto heuristic flips on small-maxblock" begin
    A_small = sprand(MersenneTwister(11), 50, 50, 0.05) + 50*I
    K_small = klu(A_small)
    @test Int(K_small.symbolic.maxblock) <= PureKLU.AUTO_PREALLOC_MAXBLOCK

    A_dense = sparse(Matrix{Float64}(I, 80, 80) .+ randn(MersenneTwister(7), 80, 80))
    K_dense = klu(A_dense)
    @test Int(K_dense.symbolic.maxblock) > PureKLU.AUTO_PREALLOC_MAXBLOCK

    b1 = randn(MersenneTwister(8), 50);  @test A_small * (K_small \ copy(b1)) ≈ b1
    b2 = randn(MersenneTwister(9), 80);  @test A_dense * (K_dense \ copy(b2)) ≈ b2
end
