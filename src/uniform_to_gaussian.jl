export GaussianizedUniformInputFunction

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

function (fcn::GaussianizedUniformInputFunction)(x)
    unif_pt = map(eachindex(fcn.bounds)) do i
        return normcdf(x[i]) * (fcn.bounds[i][2] - fcn.bounds[i][1]) + fcn.bounds[i][1]
    end
    return fcn.f(unif_pt)
end
