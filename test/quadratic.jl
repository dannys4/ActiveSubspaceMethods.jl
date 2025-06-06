seed = 2840284
d, order, r = 6, 5, 3
N = order^d
mc_tol, quad_tol = 10/sqrt(N), 1e-14

rng = Xoshiro(seed)
A_sqrt = randn(rng, d, d)
A = A_sqrt'*A_sqrt
A_sq = A*A
tr_A = tr(A)

function test_eval_grad!(grad, x)
    grad .= A*x
    return 0.5*(grad'x - tr_A)
end

function test_eval(x)
    return 0.5*(x'A*x-tr_A)
end


AD_eval_grad! = ActiveSubspaceMethods.ADFunctionWrapper(test_eval, d, AutoReverseDiff())
rng = Xoshiro(seed)
inp_mc = MCActiveSubspacesInput(test_eval_grad!, d, N; rng)
rng = Xoshiro(seed)
inp_mc_AD = MCActiveSubspacesInput(AD_eval_grad!, d, N; rng)

inp_quad = QuadratureActiveSubspacesInput(test_eval_grad!, d, order)
inp_quad_AD = QuadratureActiveSubspacesInput(AD_eval_grad!, d, order)

for (i,inp) in enumerate([inp_mc, inp_mc_AD, inp_quad, inp_quad_AD])
    tol = i < 3 ? mc_tol : quad_tol
    as = ActiveSubspaces(inp)
    @test norm(as.C_AS - A_sq)/norm(A_sq) < tol
    out_as = as()
    U_perp_as = out_as(r)

    mod_as = ModifiedActiveSubspaces(inp)
    out_mod_as = mod_as()
    @test norm(mod_as.C_MAS - 0.5A_sq)/norm(0.5A_sq) < tol
    U_perp_mod_as = out_mod_as(r)

    asxx = ActiveSubspacesXXManopt(inp)
    out_asxx = asxx()
    @test norm(asxx.C_AS - A_sq)/norm(A_sq) < tol
    @test norm(asxx.C_ZZ - A)/norm(A) < tol
    U_perp_asxx = out_asxx(r)

end