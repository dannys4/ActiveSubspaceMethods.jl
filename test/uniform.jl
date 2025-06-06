function borehole_fcn(x)
    r_w, r, T_u, H_u, T_l, H_l, L, K_w = x
    logrrw = log(r/r_w)
    denom = logrrw*(1.5+ (2L*T_u)/(logrrw*r_w*r_w*K_w) + T_u/T_l)
    num = 2pi*T_u*(H_u - H_l)
    num/denom
end
const borehole_bounds = NTuple{2,Float64}[
    (0.05, 0.15), (100, 50_000), (63_070, 115_600), (990, 1_110),
    (63.1, 116), (700, 820), (1_120, 1_680), (9_855, 12_045)
]
d, r = length(borehole_bounds), 3
borehole_AD = ActiveSubspaceMethods.ADFunctionWrapper(borehole_fcn, d, AutoReverseDiff())
borehole_gauss = GaussianizedUniformInputFunction(borehole_AD, borehole_bounds)
inp_quad = QuadratureActiveSubspacesInput(borehole_gauss, d, 4)

##
as = ActiveSubspaces(inp_quad)
as_out = as()
U_perp_as = as_out(r)
@test norm(I - U_perp_as'U_perp_as)/norm(I(r)) < 1e-15

##
asxx = ActiveSubspacesXXManopt(inp_quad)
asxx_out = asxx()
U_perp_asxx = asxx_out(r)
@test norm(I - U_perp_asxx'U_perp_asxx)/norm(I(r)) < 1e-15

##
asxx_bound = ASXX_bound(asxx, U_perp_asxx)
as_bound = ASXX_bound(asxx, U_perp_as)
@test asxx_bound <= as_bound