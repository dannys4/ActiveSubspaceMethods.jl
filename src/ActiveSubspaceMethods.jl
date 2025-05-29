module ActiveSubspaceMethods
using ArgCheck, Random, LinearAlgebra, Statistics, StatsFuns


abstract type AbstractActiveSubspaces end
abstract type AbstractActiveSubspacesXX<:AbstractActiveSubspaces end
abstract type AbstractActiveSubspacesOutput end

include("data.jl")
include("methods.jl")
include("bounds.jl")
include("uniform_to_gaussian.jl")

end # ActiveSubspaceMethods