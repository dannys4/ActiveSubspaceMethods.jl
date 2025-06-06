export ActiveSubspaces, ModifiedActiveSubspaces, ActiveSubspacesXXManopt

struct ActiveSubspaces{M<:Hermitian} <: AbstractActiveSubspaces
    C_AS::M
    d::Int
    function ActiveSubspaces(C_AS::Matrix)
        d = size(C_AS, 1)
        @argcheck size(C_AS, 2) == d
        @argcheck norm(C_AS - C_AS') / norm(C_AS) <= 1e-15
        H_C_AS = Hermitian(C_AS)
        return new{typeof(H_C_AS)}(H_C_AS, d)
    end
end

struct ModifiedActiveSubspaces{M<:Hermitian}
    C_MAS::M
    d::Int
    function ModifiedActiveSubspaces(C_MAS::Matrix)
        d = size(C_MAS, 1)
        @argcheck size(C_MAS, 2) == d
        @argcheck norm(C_MAS - C_MAS') / norm(C_MAS) <= 1e-15
        H_C_MAS = Hermitian(C_MAS)
        return new{typeof(H_C_MAS)}(H_C_MAS, d)
    end
end

struct ActiveSubspacesXXGenEig{M<:AbstractMatrix} <: AbstractActiveSubspacesXX
    C_AS::M
    C_ZZ::M
    d::Int
    function ActiveSubspacesXXGenEig(C_AS::Matrix, C_ZZ::Matrix)
        d = size(C_AS, 1)
        @argcheck size(C_AS) == (d, d)
        @argcheck size(C_ZZ) == (d, d)
        @argcheck norm(C_AS - C_AS') / norm(C_AS) < 1e-15
        @argcheck norm(C_ZZ - C_ZZ') / norm(C_ZZ) < 1e-15
        return new{typeof(C_AS)}(C_AS, C_ZZ, d)
    end
end

"""
    ActiveSubspaces(input::AbstractActiveSubspacesInput)
Type to represent traditional Active Subspaces method
[[Russi, 2010](https://www.proquest.com/dissertations-theses/uncertainty-quantification-with-experimental-data/docview/749356529/se-2?accountid=12492)].
"""
function ActiveSubspaces(inp::AbstractActiveSubspacesInput)
    C_AS = sum(inp) do (_, pt_grad, _, wt)
        wt * pt_grad * pt_grad'
    end
    return ActiveSubspaces(C_AS)
end

function ActiveSubspacesXXGenEig(inp::AbstractActiveSubspacesInput)
    C_AS = sum(inp) do (_, pt_grad, _, wt)
        wt * pt_grad * pt_grad'
    end
    C_ZZ = sum(inp) do (pt_eval, _, pt, wt)
        wt * pt_eval * pt * pt'
    end
    return ActiveSubspacesXXGenEig(C_AS, C_ZZ)
end

"""
    ModifiedActiveSubspaces(input::AbstractActiveSubspacesInput)
Type to represent Modified Active Subspaces method
[[Lee, 2019](https://doi.org/10.1137/17M1140662)].
"""
function ModifiedActiveSubspaces(inp::AbstractActiveSubspacesInput)
    C_MAS = sum(inp) do (_, pt_grad, _, wt)
        wt * pt_grad * pt_grad'
    end
    mean_grad = sum(inp) do (_, pt_grad, _, wt)
        wt * pt_grad
    end
    C_MAS .= 0.5*C_MAS + 0.5*mean_grad*mean_grad'
    return ModifiedActiveSubspaces(C_MAS)
end

function (as::ActiveSubspaces)()
    return ActiveSubspacesOutput(eigen(as.C_AS))
end

function (as::ModifiedActiveSubspaces)()
    return ActiveSubspacesOutput(eigen(as.C_MAS))
end

function (as::ActiveSubspacesXXGenEig)()
    vals, vecs = eigen(as.C_AS, as.C_ZZ)
    sqrt_ZZ = sqrt(as.C_ZZ)
    @assert all(isreal, sqrt_ZZ) "We require all positive eigenvalues of C_ZZ"
    vecs_ortho = sqrt_ZZ * vecs
    return ActiveSubspacesOutput(vals, vecs_ortho)
end

"""
    ActiveSubspacesXXManopt(input::AbstractActiveSubspacesInput)
Type to represent Active Subspaces ++, proposed in [Li et al., 2025]. Only usable when Manopt and Manifolds are installed.
"""
function ActiveSubspacesXXManopt end;