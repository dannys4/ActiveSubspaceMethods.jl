export ActiveSubspacesInput, ActiveSubspacesOutput

struct ActiveSubspacesInput
    mean_f::Float64
    eval_f::Vector{Float64}
    grad_f::Matrix{Float64}
    samples::Matrix{Float64}
    d::Int
    N::Int
    function ActiveSubspacesInput(mean_f, eval_f, grad_f, samples)
        d, N = size(samples)
        @argcheck length(eval_f) == N
        @argcheck size(grad_f) == (d, N)
        return new(mean_f, eval_f, grad_f, samples, d, N)
    end
end

struct ActiveSubspacesOutput <: AbstractActiveSubspacesOutput
    vals::Vector{Float64}
    vecs::Matrix{Float64}
end

function ActiveSubspacesOutput(eig::Union{Eigen,GeneralizedEigen})
    vals, vecs = eig
    large_to_small = sortperm(vals; rev=true)
    vals = vals[large_to_small]
    vecs = vecs[:, large_to_small]
    return ActiveSubspacesOutput(vals, vecs)
end

(as::ActiveSubspacesOutput)(r::Int) = (as.vals[r], as.vecs[:, 1:r])
