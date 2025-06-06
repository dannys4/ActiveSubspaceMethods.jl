using ActiveSubspaceMethods, LinearAlgebra, Random, ReverseDiff, Manopt, Manifolds
using Test
@testset "ActiveSubspaceMethods.jl" begin
    @testset "Quadratic benchmark" begin
        include("quadratic.jl")
    end
    @testset "Uniform regression" begin
        include("uniform.jl")
    end
    @testset "Utilities" begin
        include("utils.jl")
    end
end
