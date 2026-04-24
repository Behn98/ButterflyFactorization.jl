#=There are exactly 2 ways to consider when adding Butterflys. In either case, the
corresponding Matrix blocks are of equal dimensions. Now in the first case, our Butterflies
are mapping from the very same Observer cluster to the very same source cluster and vice
versa. This requires the action just as described in the paper. Since behind the compression
scheme we have exactly one tree, the physical DoF will match and there by Q and P can be
matched in a meaningful way not disturbing the physics. In the 2nd case we add 2 Butterflies
representing two disjoint source and observer cluserts. A concatenation can not happen here
and thus we only need to join the Butterflies into a new struct. However, be aware, that
this new struct is of pure algebraic interest and has lost its physical meaningfulness just
as much as it would ve if we were to add the two matrices behind them. Also be aware that im
only tackling the symetric case of a BF.=#

function add_eqbf(BF1::BF, BF2::BF, tree)
    @assert BF1.NS == BF2.NS && BF1.NO == BF2.NO "rootids must match for addition."
    # --- Case 1: Same source and observer clusters ---
    R_new = Dict{Int,Dict{Int,AbstractMatrix{ComplexF64}}}()

    for nodeS in keys(BF1.R)
        for nodeO in keys(BF1.R[nodeS])
            if !haskey(R_new, nodeS)
                R_new[nodeS] = Dict{Int,AbstractMatrix{ComplexF64}}()
            end
            R_new[nodeS][nodeO] = sparse_blockdiag(BF1.R[nodeS][nodeO], BF2.R[nodeS][nodeO])
        end
    end

    Q_new = Dict{Int,AbstractMatrix{ComplexF64}}()
    for k in keys(BF1.Q)
        Q_new[k] = vcat(BF1.Q[k], BF2.Q[k])
    end

    P_new = Dict{Int,AbstractMatrix{ComplexF64}}()
    for k in keys(BF1.P)
        P_new[k] = hcat(BF1.P[k], BF2.P[k])
    end
    return recompress_BF(
        BF(Q_new, R_new, P_new, BF1.NS, BF1.NO, max(BF1.k, BF2.k), max(BF1.τ, BF2.τ), true),
        tree,
    )
end

function add_neqbf(BF1::BF, BF2::BF)
    return (BF1, BF2)
end
