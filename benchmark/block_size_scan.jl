# Standalone investigation script (not part of `SUITE`).
#
# Measures PureKLU vs KLU.jl refactor cost as a function of BTF block
# size, using block-diagonal banded matrices that decompose into many
# identical sub-blocks.
#
# This was used to test the hypothesis "PureKLU's gap to KLU.jl on small-
# structured matrices is dominated by per-column overhead in klu_kernel!".
# That hypothesis predicted the PureFMA/KLU ratio would be worst at the
# smallest block sizes and shrink as nk grows.  Measurement shows the
# ratio is essentially flat at ~1.20 across nk in 5..500, so the gap is
# *not* small-block-specific and a dedicated small-nk fast path would not
# help.  See PR fast-path-small-nk for the full investigation log.
#
# Run with:
#     julia --project=benchmark -O3 benchmark/block_size_scan.jl

using SparseArrays
using LinearAlgebra
using Random
using BenchmarkTools
using PureKLU
import KLU

# Build a block-diagonal matrix where each diagonal block is itself
# a banded sparse block of size nk x nk with bandwidth bw.  Because the
# off-diagonal between blocks is empty, BTF returns exactly `num_blocks`
# blocks of size `nk` (verified empirically).
function block_diag_band(num_blocks::Int, nk::Int, bw::Int=2; seed::Int=1)
    rng = MersenneTwister(seed)
    n = num_blocks * nk
    rows = Int[]; cols = Int[]; vals = Float64[]
    for b in 0:(num_blocks-1)
        off = b * nk
        for j in 1:nk
            for i in max(1, j-bw):min(nk, j+bw)
                push!(rows, off + i)
                push!(cols, off + j)
                if i == j
                    push!(vals, Float64(nk) + 2*bw + rand(rng))
                else
                    push!(vals, -rand(rng))
                end
            end
        end
    end
    return sparse(rows, cols, vals, n, n)
end

const CONFIGS = [
    (100, 5),    # n=500, 100 tiny blocks
    (50,  10),
    (25,  20),
    (20,  25),
    (10,  50),
    (5,   100),
    (2,   250),
    (1,   500),  # n=500, single big block
]

function bench_one(num_blocks::Int, nk::Int; bw::Int=2)
    A = block_diag_band(num_blocks, nk, bw)
    n = size(A, 1)
    K_pure       = PureKLU.klu(A)
    K_pure_nofma = PureKLU.klu(A; use_fma=Val(false))
    K_klu        = KLU.klu(A)

    # Sanity: same answer.
    b = randn(n)
    rel = norm((K_pure \ b) - (K_klu \ b)) / norm(K_klu \ b)
    rel < 1e-8 || @warn "Mismatch for ($num_blocks, $nk): relerr=$rel"

    nzval = copy(A.nzval)
    b_pure       = @benchmark PureKLU.klu!($K_pure, $nzval)       samples=200 evals=3 seconds=2
    b_pure_nofma = @benchmark PureKLU.klu!($K_pure_nofma, $nzval) samples=200 evals=3 seconds=2
    b_klu        = @benchmark KLU.klu!($K_klu, $nzval)            samples=200 evals=3 seconds=2

    return (n=n, num_blocks=num_blocks, nk=nk,
            klu_us       = minimum(b_klu).time / 1e3,
            pure_fma_us  = minimum(b_pure).time / 1e3,
            pure_nofma_us= minimum(b_pure_nofma).time / 1e3)
end

println("Refactor timings vs BTF block size  --  $(Sys.CPU_NAME), Julia $VERSION")
println(rpad("config", 12), rpad("n", 6), rpad("KLU(us)", 12),
        rpad("Pure_fma(us)", 14), rpad("Pure_nofma(us)", 16),
        rpad("PureFMA/KLU", 14), rpad("KLU us/col", 14), rpad("Pure us/col", 14))
println("-"^110)
for (nb, nk) in CONFIGS
    r = bench_one(nb, nk)
    println(rpad("$(nb)x$(nk)", 12),
            rpad(string(r.n), 6),
            rpad(string(round(r.klu_us, digits=2)), 12),
            rpad(string(round(r.pure_fma_us, digits=2)), 14),
            rpad(string(round(r.pure_nofma_us, digits=2)), 16),
            rpad(string(round(r.pure_fma_us / r.klu_us, digits=3)), 14),
            rpad(string(round(r.klu_us / r.n, digits=4)), 14),
            rpad(string(round(r.pure_fma_us / r.n, digits=4)), 14),
        )
end
