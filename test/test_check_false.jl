using PureKLU
using Test
using SparseArrays: sparse
using LinearAlgebra

@testset "check=false" begin
    for A in sparse.((Float64[1 2; 0 0], ComplexF64[1 2; 0 0]))
        # Even the default check=true path does not throw on numerical
        # singularity; it reports KLU_SINGULAR (diverges from KLU.jl/libklu).
        @test PureKLU.klu(A).common.status == PureKLU.KLU_SINGULAR
        @test !issuccess(PureKLU.klu(A))
        @test !issuccess(PureKLU.klu(A; check = false))
        @test issuccess(PureKLU.klu(A; allowsingular = true); allowsingular = true)
    end
end
