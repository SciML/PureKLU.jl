using PureKLU
using PureKLU: increment!, decrement, klu!, KLUFactorization, klu_analyze!, klu_factor!,
    KLUITypes
using Test
using SparseArrays: SparseMatrixCSC, sparse, nnz
using LinearAlgebra

@testset "KLU Wrappers" begin
    Ap = increment!([0, 4, 1, 1, 2, 2, 0, 1, 2, 3, 4, 4])
    Ai = increment!([0, 4, 0, 2, 1, 2, 1, 4, 3, 2, 1, 2])
    Ax = [2.0, 1.0, 3.0, 4.0, -1.0, -3.0, 3.0, 6.0, 2.0, 1.0, 4.0, 2.0]
    A0 = sparse(Ap, Ai, Ax)
    A1 = sparse(
        increment!([0, 4, 1, 1, 2, 2, 0, 1, 2, 3, 4, 4]),
        increment!([0, 4, 0, 2, 1, 2, 1, 4, 3, 2, 1, 2]),
        [2.0, 1.0, 3.0, 4.0, -1.0, -3.0, 3.0, 9.0, 2.0, 1.0, 4.0, 2.0], 5, 5
    )
    @testset "Core functionality for $Tv elements" for Tv in (Float64, ComplexF64)
        @testset "Core functionality for $Ti indices" for Ti in Base.uniontypes(KLUITypes)
            A = convert(SparseMatrixCSC{Tv, Ti}, A0)
            klua = PureKLU.klu(size(A, 1), decrement(A.colptr), decrement(A.rowval), A.nzval)
            @test klua.F == PureKLU.klu(A).F
            @test nnz(klua) == 18
            R = Diagonal(Tv == ComplexF64 ? complex.(klua.Rs) : klua.Rs)
            @test R \ A[klua.p, klua.q] ≈ (klua.L * klua.U + klua.F)

            b = [8.0, 45.0, -3.0, 3.0, 19.0]
            x = klua \ b
            @test x ≈ float([1:5;])
            @test A * x ≈ b

            z = complex.(b)
            x = ldiv!(klua, z)
            @test x ≈ float([1:5;])
            @test z === x
            @test A * x ≈ b

            b = [8.0, 20.0, 13.0, 6.0, 17.0]
            x = klua' \ b
            @test x ≈ float([1:5;])

            @test A' * x ≈ b
            z = complex.(b)
            x = ldiv!(adjoint(klua), z)
            @test x ≈ float([1:5;])
            @test x === z

            @test A' * x ≈ b
            x = transpose(klua) \ b
            @test x ≈ float([1:5;])
            @test transpose(A) * x ≈ b

            x = ldiv!(transpose(klua), complex.(b))
            @test x ≈ float([1:5;])
            @test transpose(A) * x ≈ b

            @inferred klua \ fill(1, size(A, 2))

            @testset "Permutation vectors" begin
                @test Int(klu_analyze!(klua, klua.p, klua.q).common.status) == 0
                klu_factor!(klua)
                x = klua \ b
                @test A * x ≈ b
            end
            @testset "Utility functions" begin
                K = KLUFactorization(A)
                @test size(K) == (5, 5)
                @test size(K, 3) == 1
                @test_throws ArgumentError K.symbolic
                @test_throws ArgumentError K.numeric
                klu_analyze!(K)
                @test K.symbolic.nz == 12
                @test K.symbolic.nzoff == 4
                klu_factor!(K)
                @test K.numeric.lnz == 7 == K.numeric.unz
                @test K.nblocks == 3
                @test propertynames(K) == (:lnz, :unz, :nzoff, :L, :U, :F, :q, :p, :Rs, :symbolic, :numeric)
                @test_throws ArgumentError size(K, -1)
                @test size(K, 3) == 1
            end
            @testset "Refactorization" begin
                B = convert(SparseMatrixCSC{Tv, Ti}, A1)
                b = Tv[8.0, 45.0, -3.0, 3.0, 19.0]
                F = PureKLU.klu(A)
                klu!(F, B)
                @test F \ b ≈ B \ b ≈ Matrix(B) \ b
                @test klu!(F, B.nzval) \ b ≈ B \ b ≈ Matrix(B) \ b
            end

            @testset "Singular matrix" begin
                S = sparse([1 2; 0 0])
                S = convert(SparseMatrixCSC{Tv, Ti}, S)
                # PureKLU never throws on numerical singularity (diverges from
                # KLU.jl/libklu, which raise here). The singular factor is
                # surfaced via `common.status == KLU_SINGULAR` instead.
                K = PureKLU.klu(S)
                @test K.common.status == PureKLU.KLU_SINGULAR
                @test !issuccess(K)
            end
        end
    end
    @testset "REPL printing of KLU" begin
        A = sparse([1, 2], [1, 2], Float64[1.0, 1.0])
        F = PureKLU.klu(A)
        facstring = sprint((t, s) -> show(t, "text/plain", s), F)
        lstring = sprint((t, s) -> show(t, "text/plain", s), F.L)
        ustring = sprint((t, s) -> show(t, "text/plain", s), F.U)
        fstring = sprint((t, s) -> show(t, "text/plain", s), F.F)
        @test facstring == "$(summary(F))\nL factor:\n$lstring\nU factor:\n$ustring\nF factor:\n$fstring"
    end
end
