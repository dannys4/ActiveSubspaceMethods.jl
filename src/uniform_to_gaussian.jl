export GaussianizedUniformInputFunction

"""
    GaussianizedUniformInputFunction([f0], f1!, bounds; [test_eval])
Take a function that we wish to minimize L2 error w.r.t. uniform measure and transform it to be w.r.t. GaussianizedUniformInputFunction

# Arguments
- `f0(z)::Float64` Returns function eval at `z`. Will default to using `f1!` if not provided
- `f1!(grad, z)::Float64` Returns *same* function eval at `z` and puts function gradient at `z` into `grad`
- `bounds::AbstractVector{<:NTuple{2}}` A sequence of (lower, upper) bounds that is as long as the number of inputs.
- `test_eval::Bool` Whether to test function eval when constructing this functor, default `true`.
"""
struct GaussianizedUniformInputFunction{
    F0<:Function,F1<:Function,V<:AbstractVector{<:NTuple{2}}
} <: Function
    f0::F0
    f1!::F1
    bounds::V
end

function GaussianizedUniformInputFunction(
    f1!::_F1, bounds::_V; test_eval=true
) where {_F1,_V}
    @argcheck all(b[2] > b[1] for b in bounds)
    tmp_space = Vector{Float64}(undef, length(bounds))
    test_eval && f1!(tmp_space, map(b -> (b[2] + b[1]) / 2, bounds))
    f0 = x -> f1!(tmp_space, x)
    return GaussianizedUniformInputFunction{typeof(f0),_F1,_V}(f0, f1!, bounds)
end

function GaussianizedUniformInputFunction(
    f0::_F0, f1!::_F1, bounds::_V; test_eval=true
) where {_F0,_F1,_V}
    @argcheck all(b[2] > b[1] for b in bounds)
    if test_eval
        tmp_space = Vector{Float64}(undef, length(bounds))
        tmp_pt = map(b -> (b[2] + b[1]) / 2, bounds)
        out_f0 = f0(tmp_pt)
        out_f1 = f1!(tmp_space, tmp_pt)
        @assert out_f0 == out_f1
    end
    return GaussianizedUniformInputFunction{_F0,_F1,_V}(f0, f1!, bounds)
end

function (fcn::GaussianizedUniformInputFunction)(x)
    unif_pt = map(eachindex(fcn.bounds)) do i
        return normcdf(x[i]) * (fcn.bounds[i][2] - fcn.bounds[i][1]) + fcn.bounds[i][1]
    end
    return fcn.f0(unif_pt)
end

function (fcn::GaussianizedUniformInputFunction)(grad_eval, x)
    unif_pt = map(eachindex(fcn.bounds)) do dim_idx
        return normcdf(x[dim_idx]) * (fcn.bounds[dim_idx][2] - fcn.bounds[dim_idx][1]) +
               fcn.bounds[dim_idx][1]
    end
    eval_pt = fcn.f1!(grad_eval, unif_pt)
    for dim_idx in eachindex(fcn.bounds)
        grad_eval[dim_idx] *=
            normpdf(x[dim_idx]) * (fcn.bounds[dim_idx][2] - fcn.bounds[dim_idx][1]) +
            fcn.bounds[dim_idx][1]
    end
    return eval_pt
end