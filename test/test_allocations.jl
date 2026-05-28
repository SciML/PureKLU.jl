# Locks in the zero-heap-allocation property of the user-visible solve
# and factor paths.  `klu_factor!`'s per-block L/U storage is pre-sized
# to AMD's `Lnz` estimate at analyze time, so the second and subsequent
# factor calls only walk pre-allocated buffers (no `resize!`/`push!`).

using PureKLU
using PureKLU: klu, klu!, klu_factor!, klu_analyze!, solve!
using SparseArrays
using LinearAlgebra
using Random
using Test

function _small_test_matrix(seed::Int=1)
    rng = MersenneTwister(seed)
    n = 200
    A = sprand(rng, n, n, 0.05) + n*I
    return SparseMatrixCSC(A)
end

@testset "Zero-allocation hot paths (post-warmup)" begin
    A = _small_test_matrix()
    n = size(A, 1)

    # Warm everything up so JIT/specialisation is done.
    K = klu(A)
    b = randn(n)
    bw = copy(b);            solve!(K, bw)
    bw = copy(b);            solve!(K', bw)
    bw = copy(b);            solve!(transpose(K), bw)
    klu!(K, A.nzval)
    klu_factor!(K)

    # solve!
    b1 = copy(b)
    @test (@allocated solve!(K, b1)) == 0

    # adjoint solve
    b2 = copy(b)
    @test (@allocated solve!(K', b2)) == 0

    # transpose solve
    b3 = copy(b)
    @test (@allocated solve!(transpose(K), b3)) == 0

    # refactor (same pattern, new values)
    @test (@allocated klu!(K, A.nzval)) == 0

    # factor (re-factor on the same factorisation object); after the
    # initial warm-up call has populated all block buffers, subsequent
    # `klu_factor!`s should be zero-allocation.
    @test (@allocated klu_factor!(K)) == 0

    # Sanity: each path still produced the right answer.
    @test A * b1 ≈ b
    @test A' * b2 ≈ b
    @test transpose(A) * b3 ≈ b
end

@testset "Zero-allocation klu_factor! on representative matrices" begin
    function _laplacian_2d(k::Int)
        nn = k * k
        diag_main = 4.0 * ones(nn)
        diag_off1 = -1.0 * ones(nn-1)
        for i in 1:nn-1
            if i % k == 0; diag_off1[i] = 0.0; end
        end
        diag_offk = -1.0 * ones(nn-k)
        A = spdiagm(-k => diag_offk, -1 => diag_off1, 0 => diag_main,
                    1 => diag_off1, k => diag_offk)
        dropzeros!(A); return A
    end
    function _band(n::Int, bw::Int)
        diags = Dict{Int, Vector{Float64}}()
        diags[0] = Float64(n) .* ones(n)
        for d in 1:bw
            diags[ d] = -ones(n-d); diags[-d] = -ones(n-d)
        end
        A = spdiagm((k => v for (k, v) in diags)...)
        dropzeros!(A); return A
    end
    function _arrow(n::Int)
        Ir = vcat(fill(1, n-1), 2:n, 1:n)
        Jr = vcat(2:n, fill(1, n-1), 1:n)
        Vr = vcat(ones(n-1), ones(n-1), Float64(n) .* ones(n))
        return sparse(Ir, Jr, Vr, n, n)
    end

    matrices = Any[
        ("laplacian_20x20", _laplacian_2d(20)),
        ("band_n500_bw5",   _band(500, 5)),
        ("band_n2000_bw5",  _band(2000, 5)),
        ("arrow_n500",      _arrow(500)),
    ]
    for (name, A) in matrices
        K = klu(A; full_factor=false)
        # Warm both paths: factor populates the per-block buffers,
        # refactor specialises the refactor call.
        klu_factor!(K)
        nz = A.nzval  # hoist out the SparseMatrixCSC.nzval getproperty
        klu!(K, nz)
        @test (@allocated klu_factor!(K)) == 0
        @test (@allocated klu!(K, nz)) == 0
    end
end
