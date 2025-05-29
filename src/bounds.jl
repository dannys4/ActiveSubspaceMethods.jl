export AS_bound, ASXX_bound

function ASXX_bound(as::AbstractActiveSubspacesXX, out::ActiveSubspacesOutput, r::Int)
    U_perp = out.vecs[:, (r + 1):end]
    C_AS_perp = U_perp' * as.C_AS * U_perp
    C_ZZ_perp = U_perp' * as.C_ZZ * U_perp
    as_bd = tr(C_AS_perp)
    if r < as.d
        as_bd -= 0.5 * sum(idx -> abs2(C_ZZ_perp[idx]), eachindex(C_ZZ_perp))
    end
    return as_bd
end

function AS_bound(as::AbstractActiveSubspaces, out::ActiveSubspacesOutput, r)
    U_perp = out.vecs[:, (r + 1):end]
    C_AS_perp = U_perp' * as.C_AS * U_perp
    return tr(C_AS_perp)
end
