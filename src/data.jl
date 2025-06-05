export ActiveSubspacesInput,
    ActiveSubspacesOutput, MCActiveSubspacesInput, QuadratureActiveSubspacesInput
using Base: getindex, iterate
struct InputUnweighted <: AbstractActiveSubspacesInput
    mean_f::Float64
    eval_f::Vector{Float64}
    grad_f::Matrix{Float64}
    points::Matrix{Float64}
    d::Int
    N::Int
    corrected::Bool
end

struct InputWeighted <: AbstractActiveSubspacesInput
    mean_f::Float64
    eval_f::Vector{Float64}
    grad_f::Matrix{Float64}
    points::Matrix{Float64}
    weights::Vector{Float64}
    d::Int
    N::Int
end

function ActiveSubspacesInput(mean_f, eval_f, grad_f, points, ::Nothing; corrected=true)
    d, N = size(points)
    @argcheck length(eval_f) == N DimensionMismatch
    @argcheck size(grad_f) == (d, N) DimensionMismatch
    return InputUnweighted(mean_f, eval_f, grad_f, points, d, N, corrected)
end

function ActiveSubspacesInput(mean_f, eval_f, grad_f, points, weights::AbstractVector; _...)
    d, N = size(points)
    @argcheck length(eval_f) == N DimensionMismatch
    @argcheck size(grad_f) == (d, N) DimensionMismatch
    @argcheck length(weights) == N DimensionMismatch
    @argcheck isapprox(sum(weights), 1)
    return InputWeighted(mean_f, eval_f, grad_f, points, weights, d, N)
end

function Base.getindex(inp::InputUnweighted, point_idx::Int)
    @argcheck point_idx > 0 && point_idx <= inp.N
    eval_f = inp.eval_f[point_idx]
    grad_f = @view inp.grad_f[:, point_idx]
    point_f = @view inp.points[:, point_idx]
    weight = inp.corrected ? 1 / (inp.N - 1) : 1 / inp.N
    return eval_f, grad_f, point_f, weight
end

function Base.getindex(inp::InputWeighted, point_idx::Int)
    @argcheck point_idx > 0 && point_idx <= inp.N
    eval_f = inp.eval_f[point_idx]
    grad_f = @view inp.grad_f[:, point_idx]
    point_f = @view inp.points[:, point_idx]
    weight = inp.weights[point_idx]
    return eval_f, grad_f, point_f, weight
end

function Base.iterate(inp::AbstractActiveSubspacesInput, state::Int=1)
    return state <= inp.N ? (inp[state], state + 1) : nothing
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

function ActiveSubspacesInput(
    eval_grad_fcn!,
    points::AbstractMatrix{Float64},
    weights::Union{<:AbstractVector{Float64},Nothing}=nothing;
    kwargs...,
)
    d, N = size(points)
    @argcheck isnothing(weights) || N == length(weights) DimensionMismatch
    grad_out_tmp = similar(points, d)
    evals = Vector{Float64}(undef, N)
    grads = similar(points)
    for pt_idx in axes(points, 2)
        pt = points[:, pt_idx]
        evals[pt_idx] = eval_grad_fcn!(grad_out_tmp, pt)
        grads[:, pt_idx] .= grad_out_tmp
    end
    mean_fcn = mean(evals)
    return ActiveSubspacesInput(
        mean_fcn, evals .- mean_fcn, grads, points, weights; kwargs...
    )
end

function MCActiveSubspacesInput(
    eval_grad_fcn!, d::Int, N::Int; rand_fcn=randn, rng=Random.GLOBAL_RNG, corrected=true
)
    return ActiveSubspacesInput(eval_grad_fcn!, rand_fcn(rng, d, N); corrected)
end

gaussprobhermite(N::Int) = gausshermite(N; normalize=true)

function tensor_prod_quad(
    pts_wts_zipped::NTuple{d,<:Tuple{<:AbstractVector{Float64},<:AbstractVector{Float64}}}
) where {d}
    points_1d = [p for (p, _) in pts_wts_zipped]
    log_wts_vals = [[log(abs(w)) for w in wts] for (_, wts) in pts_wts_zipped]
    wts_signs = [[sign(w) for w in wts] for (_, wts) in pts_wts_zipped]
    # Create all indices for the tensor product rule
    lengths_1d = ntuple(k -> length(points_1d[k]), d)
    idxs = CartesianIndices(lengths_1d)
    points = Matrix{Float64}(undef, d, length(idxs))
    weights = zeros(Float64, length(idxs))
    @inbounds for (pt_idx, cart_idx) in enumerate(idxs)
        log_wt_sum = 0.0
        wt_sign = 1.0
        for dim_idx in 1:d
            points[dim_idx, pt_idx] = points_1d[dim_idx][cart_idx[dim_idx]]
            log_wt_sum += log_wts_vals[dim_idx][cart_idx[dim_idx]]
            wt_sign *= wts_signs[dim_idx][cart_idx[dim_idx]]
        end
        weights[pt_idx] = exp(log_wt_sum) * wt_sign
    end
    return points, weights
end

function QuadratureActiveSubspacesInput(
    eval_grad_fcn!, d::Int, tensor_order::Int; quad_fcn1d=gaussprobhermite, verbose=false
)
    pts1d, wts1d = quad_fcn1d(tensor_order)
    pts_wts_zip_1d = ntuple(_ -> (pts1d, wts1d), d)
    pts, wts = tensor_prod_quad(pts_wts_zip_1d)
    verbose && @info "Using $(length(wts)) quadrature points"
    return ActiveSubspacesInput(eval_grad_fcn!, pts, wts)
end

function QuadratureActiveSubspacesInput(
    eval_grad_fcn!, tensor_orders::NTuple{d,Int}; quad_fcns1d=gaussprobhermite, verbose=true
) where {d}
    pts_wts_zip_1d = nothing
    if isa(quad_fcn1d, Union{<:AbstractVector,Tuple})
        pts_wts_zip_1d = ntuple(dim_idx -> quad_fcns1d[dim_idx](tensor_orders[dim_idx]), d)
    else
        pts_wts_zip_1d = ntuple(dim_idx -> quad_fcns1d(tensor_orders[dim_idx]), d)
    end
    pts, wts = tensor_prod_quad(pts_wts_zip_1d)
    verbose && @info "Using $(length(wts)) quadrature points"
    return ActiveSubspacesInput(eval_grad_fcn!, pts, wts)
end