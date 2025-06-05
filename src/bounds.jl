export AS_bound, ASXX_bound

"""
    ASXX_bound(as::AbstractActiveSubspacesXX, U_perp::AbstractMatrix)
Evaluate the active subspaces++ bound using the **perp** subspace.
"""
function ASXX_bound(as::AbstractActiveSubspacesXX, U_perp::AbstractMatrix)
    d,r = size(U_perp)
    @argcheck size(as.C_AS,1) == d DimensionMismatch
    C_AS_perp = U_perp' * as.C_AS * U_perp
    C_ZZ_perp = U_perp' * as.C_ZZ * U_perp
    as_bd = tr(C_AS_perp)
    if r < as.d
        as_bd -= 0.5 * sum(abs2, C_ZZ_perp)
    end
    return as_bd
end


function AS_bound(as::AbstractActiveSubspaces, U_perp::AbstractMatrix)
    d = size(as.C_AS,1)
    @argcheck size(U_perp,1) == d DimensionMismatch
    C_AS_perp = U_perp' * as.C_AS * U_perp
    return tr(C_AS_perp)
end
