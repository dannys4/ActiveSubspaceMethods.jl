struct AutoDiffFcn{F<:Function,P,B<:DifferentiationInterface.AbstractADType}
    fcn::F
    prep::P
    backend::B
end

function AutoDiffFcn(fcn::F, d::Int, backend::B; init_space=zero(d)) where {F,B}
    prep = DifferentiationInterface.prepare_gradient(fcn, backend, init_space)
    AutoDiffFcn{F,typeof(prep),B}(fcn, prep, backend)
end

function (ad_fcn!::AutoDiffFcn)(grad, pt)
    DifferentiationInterface.value_and_gradient!(ad_fcn!.fcn, grad, ad_fcn!.prep, ad_fcn!.backend, pt)[1]
end