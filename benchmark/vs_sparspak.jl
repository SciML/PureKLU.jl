# Three-way pure-Julia-vs-C sparse-LU benchmark:
#   PureKLU (pure Julia) | KLU.jl (C/SuiteSparse) | Sparspak.jl (pure Julia)
#
# Run:
#     julia --project=benchmark benchmark/vs_sparspak.jl
#
# Measures min-of-N wall time (μs) for three phases:
#   factor   -- full analyze + numeric factorization from scratch
#   refactor -- numeric refactor reusing symbolic factorization (same pattern)
#   solve    -- K \ b for a fixed RHS
#
# Notes on Sparspak: `sparspaklu!(lu, A2)` with an unchanged sparsity pattern
# reuses the ordering and symbolic factorization (only re-runs inmatrix!/factor!),
# so it is a genuine numeric-only refactor directly comparable to KLU's klu!.

using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Random
using PureKLU
import KLU
using Sparspak

# ---------- matrix zoo (copied from benchmarks.jl) --------------------------

function _laplacian_2d(k::Int)
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
    return A
end

function _random(n::Int, density::Float64, seed::Int)
    rng = MersenneTwister(seed)
    A = sprand(rng, n, n, density) + n*I
    dropzeros!(A)
    return A
end

function _band(n::Int, bw::Int)
    diags = Dict{Int, Vector{Float64}}()
    diags[0] = Float64(n) .* ones(n)
    for d in 1:bw
        diags[ d] = -ones(n-d)
        diags[-d] = -ones(n-d)
    end
    A = spdiagm((k => v for (k, v) in diags)...)
    dropzeros!(A)
    return A
end

function _arrow(n::Int)
    Ir = vcat(fill(1, n-1), 2:n, 1:n)
    Jr = vcat(2:n, fill(1, n-1), 1:n)
    Vr = vcat(ones(n-1), ones(n-1), Float64(n) .* ones(n))
    return sparse(Ir, Jr, Vr, n, n)
end

const MATRICES = (
    "laplacian_20x20"   => _laplacian_2d(20),    # n=400
    "laplacian_40x40"   => _laplacian_2d(40),    # n=1600
    "laplacian_60x60"   => _laplacian_2d(60),    # n=3600
    "random_n100_d10"   => _random(100,  0.10, 1),
    "random_n500_d02"   => _random(500,  0.02, 1),
    "random_n1000_d01"  => _random(1000, 0.01, 1),
    "random_n2000_d005" => _random(2000, 0.005, 1),
    "band_n500_bw5"     => _band(500,  5),
    "band_n2000_bw5"    => _band(2000, 5),
    "arrow_n500"        => _arrow(500),
)

# ---------- correctness check (timing only correct solves) ------------------

const RTOL = 1e-8

function _check(name, A, b)
    # PureKLU
    Kp = PureKLU.klu(A)
    xp = Kp \ b
    rp = norm(A*xp - b) / norm(b)
    @assert rp < RTOL "PureKLU residual $rp for $name"
    # KLU.jl
    Kk = KLU.klu(A)
    xk = Kk \ b
    rk = norm(A*xk - b) / norm(b)
    @assert rk < RTOL "KLU.jl residual $rk for $name"
    # Sparspak
    Ks = sparspaklu(A)
    xs = Ks \ b
    rs = norm(A*xs - b) / norm(b)
    @assert rs < RTOL "Sparspak residual $rs for $name"
    return (rp, rk, rs)
end

# ---------- per-matrix timing ------------------------------------------------

# `seconds`/`samples` budget: generous for small, capped runtime for large.
_belapsed(ex) = ex   # placeholder; we use @belapsed macro directly below

function bench_matrix(name, A)
    n = size(A, 1)
    Random.seed!(0xC0FFEE)
    b = randn(n)

    # correctness gate
    rp, rk, rs = _check(name, A, b)

    # perturbed values for refactor (same pattern, new values)
    nz2 = A.nzval .* 1.0000001 .+ 1e-9
    A2 = copy(A); A2.nzval .= nz2

    # ----- warm up every solver -----
    let
        Kp = PureKLU.klu(A); PureKLU.klu!(Kp, copy(nz2)); Kp \ b
        Kk = KLU.klu(A);     KLU.klu!(Kk, copy(nz2));     Kk \ b
        Ks = sparspaklu(A);  sparspaklu!(Ks, A2);         Ks \ b
    end

    # ----- factor (analyze + factor from scratch) -----
    f_pk  = @belapsed PureKLU.klu($A)
    f_klu = @belapsed KLU.klu($A)
    f_spk = @belapsed sparspaklu($A)

    # ----- refactor (numeric only, same pattern) -----
    r_pk  = @belapsed PureKLU.klu!(K, nz) setup=(K = PureKLU.klu($A); nz = copy($nz2)) evals=1
    r_klu = @belapsed KLU.klu!(K, nz)     setup=(K = KLU.klu($A);     nz = copy($nz2)) evals=1
    r_spk = @belapsed sparspaklu!(K, $A2) setup=(K = sparspaklu($A))                   evals=1

    # ----- solve (K \ b) -----
    Kp = PureKLU.klu(A); Kk = KLU.klu(A); Ks = sparspaklu(A)
    s_pk  = @belapsed $Kp \ $b
    s_klu = @belapsed $Kk \ $b
    s_spk = @belapsed $Ks \ $b

    return (; n,
            resid = (rp, rk, rs),
            factor   = (f_pk, f_klu, f_spk),
            refactor = (r_pk, r_klu, r_spk),
            solve    = (s_pk, s_klu, s_spk))
end

# ---------- driver / printing -----------------------------------------------

_us(x) = x * 1e6  # seconds -> microseconds
_fmt(x; w=11, d=2) = lpad(string(round(_us(x); digits=d)), w)

function main()
    println("Tuning / warming up...")
    results = Pair{String,Any}[]
    for (name, A) in MATRICES
        print("  $name ... "); flush(stdout)
        push!(results, name => bench_matrix(name, A))
        println("done")
    end

    println()
    println("="^95)
    println("PureKLU (pure Julia) vs KLU.jl (C/SuiteSparse) vs Sparspak.jl (pure Julia)")
    println("$(Sys.CPU_NAME), Julia $VERSION   --  times in μs (min-of-N wall time)")
    println("="^95)

    for phase in (:factor, :refactor, :solve)
        println("\n>>> $phase")
        println(rpad("matrix", 20), " ", lpad("n", 6), " | ",
                lpad("PureKLU", 11), " ", lpad("KLU.jl", 11), " ", lpad("Sparspak", 11),
                " | ", lpad("PK/KLU", 8), " ", lpad("PK/Spk", 8))
        for (name, r) in results
            t = getfield(r, phase)
            pk, klu, spk = t
            println(rpad(name, 20), " ", lpad(r.n, 6), " | ",
                    _fmt(pk), " ", _fmt(klu), " ", _fmt(spk), " | ",
                    lpad(string(round(pk/klu; digits=2)), 8), " ",
                    lpad(string(round(pk/spk; digits=2)), 8))
        end
    end

    println("\nResiduals (PureKLU, KLU.jl, Sparspak), all < $RTOL:")
    for (name, r) in results
        println("  ", rpad(name, 20), " ", r.resid)
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
