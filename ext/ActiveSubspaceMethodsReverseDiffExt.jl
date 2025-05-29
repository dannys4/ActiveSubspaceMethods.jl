module ActiveSubspaceMethodsReverseDiffExt
using ActiveSubspaces, ReverseDiff, Random
export GaussianMonteCarlo_ActiveSubspacesInputAD

function GaussianMonteCarlo_ActiveSubspacesInputAD(fcn, d::Int, N::Int; rng=Random.GLOBAL_RNG)
    grad_f = Matrix{Float64}(undef, d, N)
    eval_f = Vector{Float64}(undef, N)

    samples = randn(rng, d, N)
    println(samples[1])
    sample_1 = samples[:, 1]
    fcn_tape = ReverseDiff.GradientTape(fcn, sample_1)
    compiled_fcn_tape = ReverseDiff.compile(fcn_tape)
    results = similar(sample_1)
    eval_f[1] = fcn(sample_1)
    for k in 1:N
        sample_k = samples[:, k]
        eval_f[k] = fcn(sample_k)
        ReverseDiff.gradient!(results, compiled_fcn_tape, sample_k)
        grad_f[:, k] .= results
    end
    mean_f = mean(eval_f)
    ActiveSubspacesInput(mean_f, eval_f .- mean_f, grad_f, samples)
end

end # ActiveSubspaceMethodsReverseDiffExt