struct ADFunctionWrapper{F<:Function,P,B<:DifferentiationInterface.AbstractADType}<:Function
    fcn::F
    prep::P
    backend::B
end

"""
    ActiveSubspaceMethods.ADFunctionWrapper(fcn, d, backend; [init_space])
Wraps the evaluation of a **pure Julia** function `fcn` to use automatic differentiation.

# Arguments
- `fcn(z::AbstractVector)::Float64` function we want to reduce dimension of
- `d::Int` dimension of input
- `backend::ADTypes.AbstractADType` Backend for autodiff (e.g., ReverseDiff, Enzyme, etc.). Must be explicitly installed!
- `init_space::Vector` values to prepare the memory spaces for autodiff, default `zeros(d)`.
"""
function ADFunctionWrapper(
    fcn::F, d::Int, backend::B; init_space=zeros(d)
) where {F<:Function,B<:DifferentiationInterface.AbstractADType}
    prep = DifferentiationInterface.prepare_gradient(fcn, backend, init_space)
    return ADFunctionWrapper{F,typeof(prep),B}(fcn, prep, backend)
end

function (ad_fcn!::ADFunctionWrapper)(grad, pt)
    return DifferentiationInterface.value_and_gradient!(
        ad_fcn!.fcn, grad, ad_fcn!.prep, ad_fcn!.backend, pt
    )[1]
end

function (ad_fcn::ADFunctionWrapper)(pt)
    return DifferentiationInterface.value_and_gradient(
        ad_fcn.fcn, ad_fcn.prep, ad_fcn.backend, pt
    )
end