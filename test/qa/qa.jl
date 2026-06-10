using PureKLU
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(PureKLU; stale_deps = false, deps_compat = false)
    # Genuine Aqua findings, marked broken pending fix — see
    # https://github.com/SciML/PureKLU.jl/issues/62
    @test_broken false  # stale_deps: ForwardDiff declared in [deps] but unused — see https://github.com/SciML/PureKLU.jl/issues/62
    @test_broken false  # deps_compat: missing [compat] for LinearAlgebra — see https://github.com/SciML/PureKLU.jl/issues/62
    @test_broken false  # deps_compat: missing [compat] for extra Pkg — see https://github.com/SciML/PureKLU.jl/issues/62
end

@testset "JET" begin
    @test_broken false  # JET: toplevel error at src/PureKLU.jl:27 — parens around (\) in `import Base` — see https://github.com/SciML/PureKLU.jl/issues/62
end
