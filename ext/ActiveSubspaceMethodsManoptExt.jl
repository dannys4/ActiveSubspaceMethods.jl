module ActiveSubspaceMethodsManoptExt
using ActiveSubspaceMethods
using ManOpt, Manifolds, ArgCheck

struct ActiveSubspacesXXManopt{M<:AbstractMatrix} <: AbstractActiveSubspacesXX
    C_AS::M
    C_ZZ::M
    d::Int
    function ActiveSubspacesXXManopt(C_AS::Matrix, C_ZZ::Matrix)
        d = size(C_AS, 1)
        @argcheck size(C_AS) == (d, d)
        @argcheck size(C_ZZ) == (d, d)
        @argcheck issymmetric(C_AS)
        @argcheck norm(C_ZZ - C_ZZ') / norm(C_ZZ) < 1e-15
        return new{typeof(C_AS)}(C_AS, C_ZZ, d)
    end
end

function ActiveSubspacesXXManopt(inp::ActiveSubspacesInput)
    C_AS = sum(inp) do (_, pt_grad, _, wt)
        wt * pt_grad * pt_grad'
    end
    C_ZZ = sum(inp) do (pt_eval, _, pt, wt)
        wt * pt_eval * pt * pt'
    end
    return ActiveSubspacesXXManopt(C_AS, C_ZZ)
end

struct ActiveSubspacesXXManoptOutput{M<:Hermitian}
    as::ActiveSubspacesXXManopt
    start_mat::Matrix{Float64}
    U_perps::Dict{Int,Matrix{Float64}}
    d::Int
    function ActiveSubspacesXXManoptOutput(as::ActiveSubspacesXXManopt)
        (; C_AS, C_ZZ) = as
        d = size(C_AS, 1)
        @argcheck size(C_AS, 2) == d DimensionMismatch
        @argcheck size(C_ZZ) == (d, d) DimensionMismatch
        _, vecs = eigen(C_AS)
        U_perps = Dict{Int,Matrix{Float64}}()
        return new(as, vecs, U_perps)
    end
end

function (as::ActiveSubspacesXXManopt)()
    return ActiveSubspacesXXManoptOutput(as)
end

function ASXX_manopt_loss(U_perp, as::ActiveSubspacesXXManopt)
    C_AS_perp = U_perp' * as.C_AS * U_perp
    C_ZZ_perp = U_perp' * as.C_ZZ * U_perp
    as_bd = tr(C_AS_perp)
    r < as.d && (as_bd -= 0.5 * sum(abs2, C_ZZ_perp))
    return as_bd
end

function ASXX_manopt_riem_grad(U_perp, as::ActiveSubspacesXXManopt)
    C_ZZ_perp_half = as.C_ZZ * U_perp
    C_ZZ_perp = C_ZZ_perp_half * C_ZZ_perp_half'
    return 2 * (I - U_perp * U_perp') * (as.C_AS - C_ZZ_perp) * U_perp
end

function (out::ActiveSubspacesXXManoptOutput)(
    r::Int; start_mat=nothing, manopt_opt=quasi_Newton
)
    @argcheck r > 0 && r <= out.d BoundsError
    U_perp = get(out.U_perps, r, nothing)
    isnothing(U_perp) || return U_perp
    U_start = isnothing(start_mat) ? out.start_mat[:, 1:r] : start_mat
    @argcheck size(start_mat) == (out.d, r) DimensionMismatch
    loss = (_, U_perp) -> ASXX_manopt_loss(U_perp, out.as)
    grad_loss = (_, U_perp) -> ASXX_manopt_riem_grad(U_perp, out.as)
    manifold = Grassmann(out.d, r)
    U_perp_r = manopt_opt(manifold, loss, grad_loss, U_start)
    out.U_perps[r] = U_perp_r
    return U_perp_r
end

end # ActiveSubspaceMethodsManoptExt