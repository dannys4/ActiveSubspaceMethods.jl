export ActiveSubspaces, ModifiedActiveSubspaces, ActiveSubspacesXXGenEig

struct ActiveSubspaces{M<:Hermitian}<:AbstractActiveSubspaces
    C_AS::M
    d::Int
    function ActiveSubspaces(C_AS::Matrix)
        d = size(C_AS, 1)
        @argcheck size(C_AS, 2) == d
        @argcheck issymmetric(C_AS)
        H_C_AS = Hermitian(C_AS)
        new{typeof(H_C_AS)}(H_C_AS, d)
    end
end

struct ModifiedActiveSubspaces{M<:Hermitian}
    C_MAS::M
    d::Int
    function ModifiedActiveSubspaces(C_MAS::Matrix)
        d = size(C_MAS, 1)
        @argcheck size(C_MAS, 2) == d
        @argcheck issymmetric(C_MAS)
        H_C_MAS = Hermitian(C_MAS)
        new{typeof(H_C_MAS)}(H_C_MAS, d)
    end
end

struct ActiveSubspacesXXGenEig{M<:AbstractMatrix}<:AbstractActiveSubspacesXX
    C_AS::M
    C_ZZ::M
    d::Int
    function ActiveSubspacesXXGenEig(C_AS::Matrix, C_ZZ::Matrix)
        d = size(C_AS, 1)
        @argcheck size(C_AS) == (d, d)
        @argcheck size(C_ZZ) == (d, d)
        @argcheck issymmetric(C_AS)
        @argcheck norm(C_ZZ - C_ZZ')/norm(C_ZZ) < 1e-15
        new{typeof(C_AS)}(C_AS, C_ZZ, d)
    end
end

function ActiveSubspaces(inp::ActiveSubspacesInput)
    C_AS = inp.grad_f*(inp.grad_f') / N
    ActiveSubspaces(C_AS)
end

function ActiveSubspacesXXGenEig(inp::ActiveSubspacesInput)
    C_AS = inp.grad_f*(inp.grad_f') / N
    C_ZZ = mean(1:N) do k
        (inp.eval_f[k]*inp.samples[:, k])*(inp.samples[:, k]')
    end
    ActiveSubspacesXXGenEig(C_AS, C_ZZ)
end

function ModifiedActiveSubspaces(inp::ActiveSubspacesInput)
    C_MAS = inp.grad_f*(inp.grad_f') / N
    mean_grad = mean(inp.grad_f; dims=2)
    mul!(C_MAS, mean_grad, mean_grad', 0.5, 0.5)
    ModifiedActiveSubspaces(C_MAS)
end


function (as::ActiveSubspaces)()
    ActiveSubspacesOutput(eigen(as.C_AS))
end

function (as::ModifiedActiveSubspaces)()
    ActiveSubspacesOutput(eigen(as.C_MAS))
end

function (as::ActiveSubspacesXXGenEig)()
    vals, vecs = eigen(as.C_AS, as.C_ZZ)
    sqrt_ZZ = sqrt(as.C_ZZ)
    @assert all(isreal, sqrt_ZZ) "We require all positive eigenvalues of C_ZZ"
    vecs_ortho = sqrt_ZZ*vecs
    ActiveSubspacesOutput(vals, vecs_ortho)
end
