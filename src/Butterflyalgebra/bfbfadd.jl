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

function add_eqbfs(BF1::BF, BF_2::BF, τ)
    @assert BF1.NS == BF_2.NS && BF1.NO == BF_2.NO "rootids must match for addition."
    # --- Case 1: Same source and observer clusters ---
    R_new = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}(
        undef, length(BF1.R)
    )
    for l in eachindex(BF1.R)
        R_new[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
        for nodeS in keys(BF1.R[l])
            for nodeO in keys(BF1.R[l][nodeS])
                if !haskey(R_new[l], nodeS)
                    R_new[l][nodeS] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
                end
                R_new[l][nodeS][nodeO] = sparse_blockdiag(
                    BF1.R[l][nodeS][nodeO], BF_2.R[l][nodeS][nodeO]
                )
            end
        end
    end
    Q_new = Dict{Int,AbstractMatrix{ComplexF64}}()
    for k in keys(BF1.Q)
        Q_new[k] = vcat(BF1.Q[k], BF_2.Q[k])
    end

    P_new = Dict{Int,AbstractMatrix{ComplexF64}}()
    for k in keys(BF1.P)
        P_new[k] = hcat(BF1.P[k], BF_2.P[k])
    end

    return recompress_BF(
        BF(
            Q_new,
            R_new,
            P_new,
            BF1.PermQ,
            BF1.PermP,
            BF1.dim,
            BF1.NS,
            BF1.NO,
            BF1.k,
            max(BF1.τ, BF_2.τ),
        ),
        τ,
    )
end

function add_neqbfs(BF1::BF, BF_2::BF)
    return (BF1, BF_2)   #insert struct here if needed
end
