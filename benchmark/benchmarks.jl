# AirspeedVelocity / BenchmarkTools suite for PureKLU.
#
# Conventions follow https://github.com/MilesCranmer/AirspeedVelocity.jl:
# this file defines a top-level `SUITE::BenchmarkGroup` so `benchpkg`
# can load and run it against any tagged version of the package.
#
# Run locally:
#     julia --project=benchmark benchmark/benchmarks.jl
#
# Or via AirspeedVelocity to compare branches/tags:
#     benchpkg PureKLU --rev=main,HEAD
#
# Groups:
#   ["analyze_factor"][matrix][backend]   -- full klu(A) timing
#   ["refactor"][matrix][backend]         -- klu!(K, A) (reuse pattern)
#   ["solve"][matrix][backend]            -- K \ b
#   ["tsolve"][matrix][backend]           -- K' \ b  (adjoint solve)
#   ["analyze"][matrix][backend]          -- analyze-only
#   ["factor_only"][matrix][backend]      -- factor after analyze
#
# `backend` is one of "klu_jl" (SuiteSparse, reference), "pureklu_fma"
# (PureKLU with FMA fusion, default), "pureklu_nofma" (PureKLU matching
# SuiteSparse bit-for-bit).

using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Random
using PureKLU
import KLU

const SUITE = BenchmarkGroup()

# ---------- matrix zoo ------------------------------------------------------

function _laplacian_2d(k::Int)
    n = k * k
    diag_main = 4.0 * ones(n)
    diag_off1 = -1.0 * ones(n - 1)
    for i in 1:(n - 1)
        if i % k == 0
            diag_off1[i] = 0.0
        end
    end
    diag_offk = -1.0 * ones(n - k)
    A = spdiagm(
        -k => diag_offk, -1 => diag_off1, 0 => diag_main,
        1 => diag_off1, k => diag_offk
    )
    dropzeros!(A)
    return A
end

function _random(n::Int, density::Float64, seed::Int)
    rng = MersenneTwister(seed)
    A = sprand(rng, n, n, density) + n * I
    dropzeros!(A)
    return A
end

function _band(n::Int, bw::Int)
    diags = Dict{Int, Vector{Float64}}()
    diags[0] = Float64(n) .* ones(n)
    for d in 1:bw
        diags[d] = -ones(n - d)
        diags[-d] = -ones(n - d)
    end
    A = spdiagm((k => v for (k, v) in diags)...)
    dropzeros!(A)
    return A
end

function _arrow(n::Int)
    Ir = vcat(fill(1, n - 1), 2:n, 1:n)
    Jr = vcat(2:n, fill(1, n - 1), 1:n)
    Vr = vcat(ones(n - 1), ones(n - 1), Float64(n) .* ones(n))
    return sparse(Ir, Jr, Vr, n, n)
end

const MATRICES = (
    "laplacian_20x20" => _laplacian_2d(20),    # n=400
    "laplacian_40x40" => _laplacian_2d(40),    # n=1600
    "laplacian_60x60" => _laplacian_2d(60),    # n=3600
    "random_n100_d10" => _random(100, 0.1, 1),
    "random_n500_d02" => _random(500, 0.02, 1),
    "random_n1000_d01" => _random(1000, 0.01, 1),
    "random_n2000_d005" => _random(2000, 0.005, 1),
    "band_n500_bw5" => _band(500, 5),
    "band_n2000_bw5" => _band(2000, 5),
    "arrow_n500" => _arrow(500),
)

# ---------- analyze + factor (full klu(A) call) -----------------------------

let g = (SUITE["analyze_factor"] = BenchmarkGroup())
    for (name, A) in MATRICES
        g[name] = BenchmarkGroup()
        g[name]["klu_jl"] = @benchmarkable KLU.klu($A)
        g[name]["pureklu_fma"] = @benchmarkable PureKLU.klu($A; use_fma = Val(true))
        g[name]["pureklu_nofma"] = @benchmarkable PureKLU.klu($A; use_fma = Val(false))
    end
end

# ---------- analyze only (klu_analyze!) -------------------------------------

let g = (SUITE["analyze"] = BenchmarkGroup())
    for (name, A) in MATRICES
        g[name] = BenchmarkGroup()
        g[name]["klu_jl"] = @benchmarkable begin
            K = KLU.KLUFactorization($A)
            KLU.klu_analyze!(K)
        end
        g[name]["pureklu"] = @benchmarkable begin
            K = PureKLU.KLUFactorization($A)
            PureKLU.klu_analyze!(K)
        end
    end
end

# ---------- factor only (after analyze, full LU with pivoting) -------------

let g = (SUITE["factor_only"] = BenchmarkGroup())
    for (name, A) in MATRICES
        g[name] = BenchmarkGroup()
        g[name]["klu_jl"] = @benchmarkable KLU.klu_factor!(K) setup = (K = (KLU.klu_analyze!(KLU.KLUFactorization($A))))
        g[name]["pureklu_fma"] = @benchmarkable PureKLU.klu_factor!(K) setup = (
            K = (
                begin
                    K = PureKLU.KLUFactorization($A); K.common.use_fma = Val(true)
                    PureKLU.klu_analyze!(K); K
                end
            )
        )
        g[name]["pureklu_nofma"] = @benchmarkable PureKLU.klu_factor!(K) setup = (
            K = (
                begin
                    K = PureKLU.KLUFactorization($A); K.common.use_fma = Val(false)
                    PureKLU.klu_analyze!(K); K
                end
            )
        )
    end
end

# ---------- refactor (klu!(K, A)) -------------------------------------------

let g = (SUITE["refactor"] = BenchmarkGroup())
    for (name, A) in MATRICES
        g[name] = BenchmarkGroup()
        g[name]["klu_jl"] = @benchmarkable KLU.klu!(K, $A) setup = (K = KLU.klu($A))
        g[name]["pureklu_fma"] = @benchmarkable PureKLU.klu!(K, $A) setup = (K = PureKLU.klu($A; use_fma = Val(true)))
        g[name]["pureklu_nofma"] = @benchmarkable PureKLU.klu!(K, $A) setup = (K = PureKLU.klu($A; use_fma = Val(false)))
    end
end

# ---------- solve (K \ b, vector RHS) ---------------------------------------

let g = (SUITE["solve"] = BenchmarkGroup())
    for (name, A) in MATRICES
        n = size(A, 1)
        Random.seed!(0x00C0FFEE); b = randn(n)
        g[name] = BenchmarkGroup()
        g[name]["klu_jl"] = @benchmarkable $(KLU.klu(A)) \ $b
        g[name]["pureklu_fma"] = @benchmarkable $(PureKLU.klu(A; use_fma = Val(true))) \ $b
        g[name]["pureklu_nofma"] = @benchmarkable $(PureKLU.klu(A; use_fma = Val(false))) \ $b
    end
end

# ---------- adjoint solve (K' \ b) ------------------------------------------

let g = (SUITE["tsolve"] = BenchmarkGroup())
    for (name, A) in MATRICES
        n = size(A, 1)
        Random.seed!(0x00C0FFEE); b = randn(n)
        g[name] = BenchmarkGroup()
        g[name]["klu_jl"] = @benchmarkable $(KLU.klu(A))' \ $b
        g[name]["pureklu_fma"] = @benchmarkable $(PureKLU.klu(A; use_fma = Val(true)))' \ $b
        g[name]["pureklu_nofma"] = @benchmarkable $(PureKLU.klu(A; use_fma = Val(false)))' \ $b
    end
end

# ---------- entry point: when run directly, execute and print summary ------

function _fmt(x; w::Int = 12, d::Int = 1)
    isnan(x) && return lpad("--", w)
    rounded = round(x; digits = d)
    return lpad(string(rounded), w)
end
_fmts(s; w::Int = 12) = lpad(s, w)

if abspath(PROGRAM_FILE) == @__FILE__
    println("Tuning suite (warmup)...")
    tune!(SUITE; verbose = false)
    println("Running suite...")
    results = run(SUITE; verbose = false, seconds = 0.5, samples = 5)

    println()
    println("===========================================================================================")
    println("PureKLU vs SuiteSparse KLU.jl  --  $(Sys.CPU_NAME), Julia $VERSION")
    println("===========================================================================================")

    for phase in ("analyze_factor", "analyze", "factor_only", "refactor", "solve", "tsolve")
        haskey(results, phase) || continue
        println("\n>>> $phase")
        println(
            rpad("matrix", 22), " | ",
            _fmts("klu_jl (μs)"), " ",
            _fmts("pureklu_on"), " ",
            _fmts("pureklu_off"), " | ",
            _fmts("on/ref"; w = 8), " ",
            _fmts("off/ref"; w = 8)
        )
        for (name, _) in MATRICES
            r = results[phase][name]
            ref = "klu_jl" in keys(r) ? r["klu_jl"] : nothing
            on = "pureklu_fma" in keys(r) ? r["pureklu_fma"] : (("pureklu" in keys(r)) ? r["pureklu"] : nothing)
            off = "pureklu_nofma" in keys(r) ? r["pureklu_nofma"] : nothing
            ref === nothing && continue
            tref = minimum(ref).time / 1.0e3
            ton = on === nothing ? NaN : minimum(on).time / 1.0e3
            toff = off === nothing ? NaN : minimum(off).time / 1.0e3
            ron = isnan(ton) ? NaN : ton / tref
            roff = isnan(toff) ? NaN : toff / tref
            println(
                rpad(name, 22), " | ",
                _fmt(tref), " ",
                _fmt(ton), " ",
                _fmt(toff), " | ",
                _fmt(ron; w = 8, d = 2), " ",
                _fmt(roff; w = 8, d = 2)
            )
        end
    end
end
