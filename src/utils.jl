export create_Ur, MC_variance, nested_MC_err, GaussianizedUniformInputFunctionAD

"""
    orthogonalize(U)
Return matrix with orthonormal columns but same range as `U`
"""
function orthogonalize(U)
	qr(U).Q * Matrix(I, size(U)...)
end

"""
    create_Ur(U_perp, [initial_guess, rng])
Create a projector \$U_r\$ that projects onto orthogonal space to given \$U_\\perp\$.
"""
function create_Ur(U_perp, initial_guess=nothing, rng::AbstractRNG = Random.GLOBAL_RNG)
    d,perp = size(U_perp)
	@argcheck norm(I - U_perp'U_perp)/norm(I(perp)) < 1e-14
    r = d - perp
    A = isnothing(initial_guess) ? randn(rng, d, r) : initial_guess
    return make_orthogonal((I - U_perp*U_perp')*A)
end

"""
	MC_variance(fcn_eval,d,N; [rand_fcn,rng])
Approximate the `N` sample variance of function `fcn_eval` with `d` inputs
"""
function MC_variance(fcn_eval,d,N;rand_fcn::F=randn,rng=Random.GLOBAL_RNG) where {F}
	var(fcn_eval(rand_fcn(rng,d)) for _ in 1:N)
end

"""
    nested_MC_err(fcn_eval, U_r, U_perp, N_inner, N_outer; [rand_fcn, rng])
Calculates the conditional variance of subspaces using a pick-and-freeze nested MC method.

# Arguments
- `fcn_eval(z::Vector)::Float64` Evaluation of function taking in length-d vector
- `U_r` a size (d,r) orthogonal matrix
- `U_perp` a size (d,d-r) orthogonal matrix
- `N_inner::Int` number of inner Monte Carlo samples
- `N_outer::Int` number of outer Monte Carlo samples
- `rand_fcn(rng,k)::Vector{Float64}` sampler of random numbers, default `randn`
- `rng::AbstractRNG` random number generator
"""
function nested_MC_err(fcn_eval::Function, U_r::AbstractMatrix, U_perp::AbstractMatrix, N_inner::Int, N_outer::Int; rand_fcn::F=randn, rng::AbstractRNG = Random.GLOBAL_RNG) where {F}
	d,r = size(U_r)
    @argcheck size(U_perp) == (d,d-r) DimensionMismatch
	return mean(1:N_outer) do _
		x_outer = U_r*rand_fcn(rng, r)
		sum_f = 0.
		sum_sq_f = 0.
		x = similar(x_outer)
		for _ in 1:N_inner
			x_inner = U_perp*rand_fcn(rng, d-r)
			x .= x_outer + x_inner
			f_eval = fcn_eval(x)
			sum_f += f_eval
			sum_sq_f += f_eval*f_eval
		end
		return sum_sq_f/N_inner - (sum_f/N_inner)^2
	end
end

"""
    GaussianizedUniformInputFunctionAD(f0, bounds, backend; [test_eval])
Equivalent to `GaussianizedUniformInputFunction(f0, ADFunctionWrapper(f0, d, backend), bounds; test_eval)`
"""
function GaussianizedUniformInputFunctionAD(
    f0::_F0, bounds::_V, backend::DifferentiationInterface.AbstractADType; test_eval=true
) where {_F0,_V}
    d = length(bounds)
    f1! = ADFunctionWrapper(f0, d, backend)
    return GaussianizedUniformInputFunction(f0, f1!, bounds; test_eval)
end
