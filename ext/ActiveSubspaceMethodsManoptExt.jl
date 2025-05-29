module ActiveSubspaceMethodsManoptExt
using ActiveSubspaceMethods

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
    C_AS = inp.grad_f * (inp.grad_f') / N
    C_ZZ = mean(1:N) do k
        (inp.eval_f[k] * inp.samples[:, k]) * (inp.samples[:, k]')
    end
    return ActiveSubspacesXXManopt(C_AS, C_ZZ)
end

end # ActiveSubspaceMethodsManoptExt