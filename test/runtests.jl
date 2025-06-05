using ActiveSubspaceMethods, LinearAlgebra, Random, ReverseDiff, Manopt, Manifolds
using Test

rng = Xoshiro(284028)
d, order, r = 6, 5, 3
N = order^d
A_sqrt = randn(rng, d, d)
A = A_sqrt'*A_sqrt
tr_A = tr(A)

function test_eval_grad!(grad, x)
    grad .= A*x
    return 0.5*(grad'x - tr_A)
end

function test_eval(x)
    return 0.5*(x'A*x-tr_A)
end

@testset "ActiveSubspaceMethods.jl" begin
    AD_eval_grad! = ActiveSubspaceMethods.ADFunctionWrapper(test_eval, d, AutoReverseDiff())

    inp_mc = MCActiveSubspacesInput(test_eval_grad!, d, N; rng)
    inp_quad = QuadratureActiveSubspacesInput(test_eval_grad!, d, order)
    inp_mc_AD = MCActiveSubspacesInput(AD_eval_grad!, d, N; rng)
    inp_quad_AD = QuadratureActiveSubspacesInput(AD_eval_grad!, d, order)
    for inp in [inp_mc, inp_quad, inp_mc_AD, inp_quad_AD]
        as = ActiveSubspaces(inp)
        out_as = as()
        U_perp_as = out_as(r)

        mod_as = ModifiedActiveSubspaces(inp)
        out_mod_as = mod_as()
        U_perp_mod_as = out_mod_as(r)

        asxx = ActiveSubspacesXXManopt(inp)
        out_asxx = asxx()
        U_perp_asxx = out_asxx(r)

    end
end
