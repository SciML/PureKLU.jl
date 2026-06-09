using PureKLU
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(PureKLU)
end

@testset "JET" begin
    JET.test_package(PureKLU; target_defined_modules = true)
end
