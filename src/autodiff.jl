struct ADFunctionWrapper{F<:Function,P,B<:DifferentiationInterface.AbstractADType}
    fcn::F
    prep::P
    backend::B
end

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