module ActiveSubspaceMethodsManoptExt
using ActiveSubspaceMethods
using ActiveSubspaceMethods: AbstractActiveSubspacesXX, AbstractActiveSubspacesInput
using Manopt, Manifolds, ArgCheck, LinearAlgebra

struct __ActiveSubspacesXXManopt{M<:Hermitian} <: AbstractActiveSubspacesXX
    C_AS::M
    C_ZZ::M
    d::Int
    function __ActiveSubspacesXXManopt(C_AS::AbstractMatrix, C_ZZ::AbstractMatrix)
        d = size(C_AS, 1)
        @argcheck size(C_AS) == (d, d)
        @argcheck size(C_ZZ) == (d, d)
        @argcheck norm(C_AS - C_AS') / norm(C_AS) < 1e-15
        @argcheck norm(C_ZZ - C_ZZ') / norm(C_ZZ) < 1e-15
        return new{Hermitian{eltype(C_AS)}}(Hermitian(C_AS), Hermitian(C_ZZ), d)
    end
end

function ActiveSubspaceMethods.ActiveSubspacesXXManopt(inp::AbstractActiveSubspacesInput)
    C_AS = sum(inp) do (_, pt_grad, _, wt)
        wt * pt_grad * pt_grad'
    end
    C_ZZ = sum(inp) do (pt_eval, _, pt, wt)
        wt * pt_eval * pt * pt'
    end
    return __ActiveSubspacesXXManopt(C_AS, C_ZZ)
end

struct ActiveSubspacesXXManoptOutput{M<:Hermitian}
    as::__ActiveSubspacesXXManopt{M}
    start_mat::Matrix{Float64}
    U_perps::Dict{Int,Matrix{Float64}}
    d::Int
    function ActiveSubspacesXXManoptOutput(as::__ActiveSubspacesXXManopt{_M}) where {_M}
        (; C_AS, C_ZZ) = as
        d = size(C_AS, 1)
        @argcheck size(C_AS, 2) == d DimensionMismatch
        @argcheck size(C_ZZ) == (d, d) DimensionMismatch
        _, vecs = eigen(C_AS)
        U_perps = Dict{Int,Matrix{Float64}}()
        return new{_M}(as, vecs, U_perps, d)
    end
end

function (as::__ActiveSubspacesXXManopt)()
    return ActiveSubspacesXXManoptOutput(as)
end

function ASXX_manopt_loss(U_perp, as::__ActiveSubspacesXXManopt)
    C_AS_perp = U_perp' * as.C_AS * U_perp
    C_ZZ_perp = U_perp' * as.C_ZZ * U_perp
    as_bd = tr(C_AS_perp)
    if size(U_perp,2) < as.d
        as_bd -= 0.5 * sum(abs2, C_ZZ_perp)
    end
    return as_bd
end

function ASXX_manopt_riem_grad(U_perp, as::__ActiveSubspacesXXManopt)
    C_ZZ_perp_half = as.C_ZZ * U_perp
    C_ZZ_perp = C_ZZ_perp_half * C_ZZ_perp_half'
    return 2 * (I - U_perp * U_perp') * (as.C_AS - C_ZZ_perp) * U_perp
end

function (out::ActiveSubspacesXXManoptOutput)(
    r::Int; start_mat=nothing, manopt_opt=quasi_Newton
)
    @argcheck r > 0 && r <= out.d BoundsError
    perp_size = out.d-r
    U_perp = get(out.U_perps, r, nothing)
    isnothing(U_perp) || return U_perp
    U_start = isnothing(start_mat) ? out.start_mat[:, r+1:end] : start_mat
    @argcheck size(U_start) == (out.d, perp_size) DimensionMismatch
    loss = (_, U_perp) -> ASXX_manopt_loss(U_perp, out.as)
    grad_loss = (_, U_perp) -> ASXX_manopt_riem_grad(U_perp, out.as)
    manifold = Grassmann(out.d, perp_size)
    U_perp_r = manopt_opt(manifold, loss, grad_loss, U_start)
    out.U_perps[r] = U_perp_r
    return U_perp_r
end

end # ActiveSubspaceMethodsManoptExt