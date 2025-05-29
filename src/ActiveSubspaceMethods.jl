module ActiveSubspaceMethods
using ArgCheck, Random, LinearAlgebra, Statistics, StatsFuns, FastGaussQuadrature, DifferentiationInterface


abstract type AbstractActiveSubspaces end
abstract type AbstractActiveSubspacesXX<:AbstractActiveSubspaces end
abstract type AbstractActiveSubspacesInput end
abstract type AbstractActiveSubspacesOutput end

include("data.jl")
include("methods.jl")
include("bounds.jl")
include("uniform_to_gaussian.jl")

end # ActiveSubspaceMethods