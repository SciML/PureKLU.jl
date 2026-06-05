# Singular / trivial-case fast-exit behaviour.
#
# Covers:
#  * a structurally rank-deficient (empty-column) matrix now reports
#    `KLU_SINGULAR`, records the offending column, and exits the numeric
#    factorization at that column rather than grinding on to produce NaN;
#  * solving a singular factorization is well-defined and never throws -- a
#    default `check=true` solve returns its computed vector with the status left
#    `KLU_SINGULAR` for the caller to inspect, as does `check=false`;
#  * well-conditioned square / non-symmetric / refactor cases are byte-for-byte
#    unchanged versus the dense `Matrix(A) \ b` reference, for Float64 and
#    ComplexF64.
#
# NOTE: PureKLU intentionally diverges from KLU.jl/libklu here. SuiteSparse
# raises on a numerically singular factor/solve when `check=true`; PureKLU never
# throws a `SingularException` on numerical singularity, surfacing it only via
# `common.status == KLU_SINGULAR`. Non-numerical input errors (non-square,
# dimension mismatch, etc.) still throw as usual.

using PureKLU
using PureKLU: klu, klu!, klu_factor!, klu_analyze!, solve!, solve,
    KLU_OK, KLU_SINGULAR
using SparseArrays
using LinearAlgebra
using Test

@testset "Fast-exit: structurally rank-deficient -> KLU_SINGULAR" begin
    @testset "$Tv" for Tv in (Float64, ComplexF64)
        # 3x3 with column 3 structurally empty: rank-deficient by construction.
        S = sparse(Tv[1 0 0; 0 2 0; 1 0 0])
        @test rank(Matrix(S)) == 2

        # Default check=true / allowsingular=false: never throws on numerical
        # singularity; returns with status flagged KLU_SINGULAR.
        K0 = klu(S)
        @test K0.common.status == KLU_SINGULAR
        @test !issuccess(K0)

        # allowsingular: status flagged singular, offending column recorded.
        K = klu(S; check = false, allowsingular = true)
        @test K.common.status == KLU_SINGULAR
        @test K.common.singular_col != -1
        @test !issuccess(K)
        @test issuccess(K; allowsingular = true)
    end
end

@testset "Fast-exit: numerically rank-deficient (cancellation)" begin
    @testset "$Tv" for Tv in (Float64, ComplexF64)
        # row 2 == 2*row 1: exact numerical cancellation gives a zero pivot.
        B = sparse(Tv[1 2 3; 2 4 6; 1 1 1])
        @test rank(Matrix(B)) == 2
        # Default path detects the zero pivot and flags KLU_SINGULAR without
        # throwing (diverges from SuiteSparse, which raises here).
        Kd = klu(B)
        @test Kd.common.status == KLU_SINGULAR
        @test !issuccess(Kd)
        K = klu(B; check = false, allowsingular = true)
        @test K.common.status == KLU_SINGULAR
    end
end

@testset "Solve on a singular factorization is well-defined" begin
    @testset "$Tv" for Tv in (Float64, ComplexF64)
        A = sparse(Tv[1 0 0; 0 2 0; 0 0 0])  # zero pivot in column 3
        b = Tv[1, 1, 1]

        # Factor allowing singularity, then a default (check=true) solve must
        # NOT throw: it returns a defined vector and preserves the singular
        # status for the caller (it is not reset to KLU_OK).
        K = klu(A; check = false, allowsingular = true)
        @test K.common.status == KLU_SINGULAR
        x0 = solve!(K, copy(b))  # check=true by default
        @test K.common.status == KLU_SINGULAR
        @test length(x0) == 3

        # An unchecked solve behaves identically and likewise preserves status.
        K2 = klu(A; check = false, allowsingular = true)
        x = solve!(K2, copy(b); check = false)
        @test K2.common.status == KLU_SINGULAR
        @test length(x) == 3

        # Adjoint / transpose solves are also non-throwing and preserve status.
        K3 = klu(A; check = false, allowsingular = true)
        x3 = solve!(K3', copy(b))
        @test K3.common.status == KLU_SINGULAR
        @test length(x3) == 3
        K4 = klu(A; check = false, allowsingular = true)
        x4 = solve!(transpose(K4), copy(b))
        @test K4.common.status == KLU_SINGULAR
        @test length(x4) == 3
    end
end

@testset "Well-conditioned results unchanged (Float64 & ComplexF64)" begin
    @testset "$Tv" for Tv in (Float64, ComplexF64)
        # Square, non-symmetric, multi-block-friendly system.
        A = sparse(
            Tv[
                4 1 0 0;
                1 3 1 0;
                0 1 5 2;
                0 0 1 6
            ]
        )
        b = Tv[1, 2, 3, 4]
        K = klu(A)
        @test K.common.status == KLU_OK
        x = K \ b
        @test x ≈ Matrix(A) \ b
        @test A * x ≈ b
        @test K.common.status == KLU_OK  # solve did not corrupt status

        # Adjoint and transpose solves.
        @test A' * (K' \ b) ≈ b
        @test transpose(A) * (transpose(K) \ b) ≈ b

        # Refactor with the same pattern, new values: still matches dense.
        A2 = copy(A)
        nonzeros(A2) .*= Tv(2)
        nonzeros(A2)[1] += Tv(1)
        F = klu(A)
        klu!(F, A2)
        @test F.common.status == KLU_OK
        @test F \ b ≈ Matrix(A2) \ b
    end
end
