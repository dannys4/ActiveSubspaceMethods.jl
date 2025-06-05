export GaussianizedUniformInputFunction

"""
    GaussianizedUniformInputFunction(fcn!, bounds; [test_eval])
Take a function that we wish to minimize L2 error w.r.t. uniform measure and transform it to be w.r.t. GaussianizedUniformInputFunction

# Arguments
- `fcn!(grad, z)::Float64` Returns function eval at `z` and puts function gradient at `z` into `grad`
- `bounds::AbstractVector{<:NTuple{2}}` A sequence of (lower, upper) bounds that is as long as the number of inputs.
- `test_eval::Bool` Whether to test function eval when constructing this functor, default `true`.
"""
struct GaussianizedUniformInputFunction{F<:Function,V<:AbstractVector{<:NTuple{2}}}
    f::F
    bounds::V
    function GaussianizedUniformInputFunction(
        f::_F, bounds::_V; test_eval=true
    ) where {_F,_V}
        @argcheck all(b[2] > b[1] for b in bounds)
        test_eval && f(map(b -> (b[2] + b[1]) / 2, bounds))
        return new{_F,_V}(f, bounds)
    end
end

function (fcn!::GaussianizedUniformInputFunction)(x)
    unif_pt = map(eachindex(fcn.bounds)) do i
        return normcdf(x[i]) * (fcn.bounds[i][2] - fcn.bounds[i][1]) + fcn.bounds[i][1]
    end
    grad_eval = similar(unif_pt)
    return fcn!.f(grad_eval, unif_pt), grad_eval
end

function (fcn!::GaussianizedUniformInputFunction)(grad_eval, x)
    unif_pt = map(eachindex(fcn!.bounds)) do i
        return normcdf(x[i]) * (fcn!.bounds[i][2] - fcn!.bounds[i][1]) + fcn!.bounds[i][1]
    end
    eval_pt = fcn!.f(grad_eval, unif_pt)
    for dim_idx in eachindex(fcn!.bounds)
        grad_eval[dim_idx] *=
            normpdf(x[dim_idx]) * (fcn!.bounds[i][2] - fcn!.bounds[i][1]) +
            fcn!.bounds[i][1]
    end
    return eval_pt
end