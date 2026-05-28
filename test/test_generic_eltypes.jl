# Generic-eltype support: Float32, BigFloat, Complex{Float32},
# Complex{BigFloat}, and ForwardDiff.Dual.  None of these go through KLU.jl
# as an oracle (the C library is Float64/ComplexF64-only); instead we
# verify either against a Float64 PureKLU oracle or via the structural
# invariant `A * x ≈ b` / `A' * x ≈ b`.

using Test
using SparseArrays
using LinearAlgebra
using PureKLU
using Random
using ForwardDiff

# Small 5x5 BTF test matrix (same nonzero pattern as the runtests.jl
# example) and a 4x4 with a dense block.
const _AX_5x5 = Float64[2.0, 1.0, 3.0, 4.0, -1.0, -3.0, 3.0, 6.0, 2.0, 1.0, 4.0, 2.0]
const _AP_5x5 = [0, 4, 1, 1, 2, 2, 0, 1, 2, 3, 4, 4] .+ 1
const _AI_5x5 = [0, 4, 0, 2, 1, 2, 1, 4, 3, 2, 1, 2] .+ 1

_a64_5x5() = sparse(_AP_5x5, _AI_5x5, _AX_5x5)

_a64_4x4_dense() = sparse(
    Float64[
        4.0 1.0 1.0 0.5;
        0.5 5.0 2.0 0.0;
        0.1 0.0 6.0 3.0;
        0.0 0.2 0.4 7.0;
    ]
)

function _matrices(::Type{Tv}) where {Tv <: Real}
    A1 = SparseMatrixCSC{Tv, Int}(_a64_5x5())
    A2 = SparseMatrixCSC{Tv, Int}(_a64_4x4_dense())
    A3 = let
        rng = Random.MersenneTwister(7)
        n = 12
        Tv.(sprand(rng, n, n, 0.3) + n * I)
    end
    return (A1, A2, A3)
end

function _matrices(::Type{Complex{R}}) where {R <: Real}
    A1f = _a64_5x5()
    A2f = _a64_4x4_dense()
    A1 = SparseMatrixCSC{Complex{R}, Int}(complex.(A1f, A1f * R(0.1)))
    A2 = SparseMatrixCSC{Complex{R}, Int}(complex.(A2f, A2f * R(0.2)))
    A3 = let
        rng = Random.MersenneTwister(11)
        n = 10
        Ar = sprand(rng, n, n, 0.3)
        Ai = sprand(rng, n, n, 0.3)
        Af = Ar + im * Ai + n * I
        SparseMatrixCSC{Complex{R}, Int}(Af)
    end
    return (A1, A2, A3)
end

# Real-coefficient real eltypes
@testset "Generic eltypes: Float32" begin
    for A in _matrices(Float32)
        n = size(A, 1)
        K = PureKLU.klu(A)
        @test getfield(K, :numeric) isa PureKLU.KLUNumeric{Float32, <:Integer, Float32}
        @test eltype(K.Rs) === Float32
        b = Float32.(1:n)
        x = K \ b
        @test eltype(x) === Float32
        @test maximum(abs.(A * x .- b)) <= 1.0e-4
        xa = K' \ b
        @test eltype(xa) === Float32
        @test maximum(abs.(A' * xa .- b)) <= 1.0e-4
        # klu!
        B = copy(A); B.nzval .*= 2.0f0
        PureKLU.klu!(K, B)
        x = K \ b
        @test maximum(abs.(B * x .- b)) <= 1.0e-4
        # Float64 oracle: solution close to Float64-precision answer
        A64 = SparseMatrixCSC{Float64, Int}(A)
        K64 = PureKLU.klu(A64)
        x64 = K64 \ Float64.(b)
        # restored from initial values:
        Krestored = PureKLU.klu(A)
        xrestored = Krestored \ b
        @test maximum(abs.(Float64.(xrestored) .- x64)) <= 1.0e-4
    end
end

@testset "Generic eltypes: BigFloat" begin
    for A in _matrices(BigFloat)
        n = size(A, 1)
        K = PureKLU.klu(A)
        @test getfield(K, :numeric) isa PureKLU.KLUNumeric{BigFloat, <:Integer, BigFloat}
        @test eltype(K.Rs) === BigFloat
        b = BigFloat.(1:n)
        x = K \ b
        @test eltype(x) === BigFloat
        # BigFloat has high precision; check tight residual
        @test maximum(abs.(A * x .- b)) <= BigFloat(1.0e-50)
        xa = K' \ b
        @test eltype(xa) === BigFloat
        @test maximum(abs.(A' * xa .- b)) <= BigFloat(1.0e-50)
        # klu!
        B = copy(A); B.nzval .*= BigFloat(2)
        PureKLU.klu!(K, B)
        x = K \ b
        @test maximum(abs.(B * x .- b)) <= BigFloat(1.0e-50)
        # Float64 oracle agrees to Float64 precision
        A64 = SparseMatrixCSC{Float64, Int}(Float64.(A))
        K64 = PureKLU.klu(A64)
        b64 = Float64.(1:n)
        x64 = K64 \ b64
        Krestored = PureKLU.klu(A)
        xrestored = Krestored \ b
        @test maximum(abs.(Float64.(xrestored) .- x64)) <= 1.0e-12
    end
end

@testset "Generic eltypes: ComplexF32" begin
    for A in _matrices(ComplexF32)
        n = size(A, 1)
        K = PureKLU.klu(A)
        @test getfield(K, :numeric) isa PureKLU.KLUNumeric{ComplexF32, <:Integer, Float32}
        @test eltype(K.Rs) === Float32
        b = ComplexF32.(1:n) .+ 0.5f0im
        x = K \ b
        @test eltype(x) === ComplexF32
        @test maximum(abs.(A * x .- b)) <= 1.0f-4
        xa = K' \ b
        @test eltype(xa) === ComplexF32
        @test maximum(abs.(A' * xa .- b)) <= 1.0f-4
        # klu!
        B = copy(A); B.nzval .*= ComplexF32(2)
        PureKLU.klu!(K, B)
        x = K \ b
        @test maximum(abs.(B * x .- b)) <= 1.0f-4
    end
end

@testset "Generic eltypes: Complex{BigFloat}" begin
    for A in _matrices(Complex{BigFloat})
        n = size(A, 1)
        K = PureKLU.klu(A)
        @test getfield(K, :numeric) isa PureKLU.KLUNumeric{Complex{BigFloat}, <:Integer, BigFloat}
        @test eltype(K.Rs) === BigFloat
        b = Complex{BigFloat}.(1:n) .+ BigFloat(0.5)im
        x = K \ b
        @test eltype(x) === Complex{BigFloat}
        @test maximum(abs.(A * x .- b)) <= BigFloat(1.0e-50)
        xa = K' \ b
        @test eltype(xa) === Complex{BigFloat}
        @test maximum(abs.(A' * xa .- b)) <= BigFloat(1.0e-50)
        # klu!
        B = copy(A); B.nzval .*= Complex{BigFloat}(2)
        PureKLU.klu!(K, B)
        x = K \ b
        @test maximum(abs.(B * x .- b)) <= BigFloat(1.0e-50)
    end
end

# ForwardDiff.Dual: verify partials match finite-difference oracle.
@testset "Generic eltypes: ForwardDiff.Dual" begin
    rng = Random.MersenneTwister(101)
    n = 8

    # Build a problem `A(p) * x = b` where A depends linearly on p,
    # so that x(p) is a differentiable function of p.  Solve with
    # Dual numbers and check the resulting partials against a finite
    # difference of the same Float64 solve.
    Aproto = sprand(rng, n, n, 0.3) + n * I
    I_idx, J_idx, _ = findnz(Aproto)
    nnz_ = length(I_idx)

    function A_of_p(p::AbstractVector{T}, basevals::Vector{Float64}) where {T}
        # entries vary as basevals[k] + p[k mod npar + 1] * 0.1
        npar = length(p)
        V = Vector{T}(undef, nnz_)
        for k in 1:nnz_
            j = mod(k - 1, npar) + 1
            V[k] = T(basevals[k]) + T(0.1) * p[j]
        end
        return sparse(I_idx, J_idx, V, n, n)
    end

    basevals = randn(rng, nnz_) .+ 1.0
    b_real = randn(rng, n)

    for N in (1, 3)
        tag = typeof(ForwardDiff.Tag(typeof(identity), Float64))
        DT = ForwardDiff.Dual{tag, Float64, N}
        p_dual = DT[ForwardDiff.Dual{tag}(0.0, ntuple(i -> i == k ? 1.0 : 0.0, N)...) for k in 1:N]

        # Solve with Dual matrix and Dual RHS
        Ad = A_of_p(p_dual, basevals)
        b_dual = DT.(b_real)
        K = PureKLU.klu(Ad)
        @test getfield(K, :numeric) isa PureKLU.KLUNumeric{DT, <:Integer, DT}
        @test eltype(K.Rs) === DT
        x = K \ b_dual
        @test eltype(x) === DT
        @test maximum(abs.(ForwardDiff.value.(Ad * x .- b_dual))) <= 1.0e-10

        # Compare partials to finite-difference oracle
        p0 = zeros(N)
        eps = 1.0e-5
        A0 = A_of_p(p0, basevals)
        K0 = PureKLU.klu(A0)
        x0 = K0 \ b_real
        fd_partials = Matrix{Float64}(undef, n, N)
        for j in 1:N
            pp = copy(p0); pp[j] += eps
            pm = copy(p0); pm[j] -= eps
            Kp = PureKLU.klu(A_of_p(pp, basevals))
            Km = PureKLU.klu(A_of_p(pm, basevals))
            fd_partials[:, j] = (Kp \ b_real .- Km \ b_real) ./ (2 * eps)
        end
        # value matches Float64 solve
        @test maximum(abs.(ForwardDiff.value.(x) .- x0)) <= 1.0e-10
        # each partial matches finite-difference
        for j in 1:N
            ad_partial = [ForwardDiff.partials(x[i])[j] for i in 1:n]
            @test maximum(abs.(ad_partial .- fd_partials[:, j])) <= 1.0e-6
        end

        # adjoint solve also works with Dual
        xa = K' \ b_dual
        @test eltype(xa) === DT
        @test maximum(abs.(ForwardDiff.value.(Ad' * xa .- b_dual))) <= 1.0e-10

        # klu! refactor with new Dual values
        p_dual2 = DT[ForwardDiff.Dual{tag}(0.5, ntuple(i -> i == k ? 1.0 : 0.0, N)...) for k in 1:N]
        Ad2 = A_of_p(p_dual2, basevals)
        PureKLU.klu!(K, Ad2)
        x2 = K \ b_dual
        @test maximum(abs.(ForwardDiff.value.(Ad2 * x2 .- b_dual))) <= 1.0e-10
    end
end
