function mulBFs(BF_1::BF2, BF_2::BF2)
    children = H2Tree.children
    trialT1 = H2Trees.trialtree(BF_1.tree)
    # Sequential join of the two BFs.
    R_new = Vector{Dict{Int,Dict{Int,Matrix{ComplexF64}}}}(
        undef, length(BF_1) + length(BF_2) - 3
    )
    RL1 = length(BF_1) - 2
    RL2 = length(BF_2) - 2
    @assert RL1 == RL2 "Currently only supports multiplication of BFs with the same number of levels, but got $(RL1) and $(RL2)"
    for l in 1:(RL1 - 1)
        R_new[l + RL2] = BF_1.R[l + 1]
    end
    for l in 1:(RL2 - 1)
        R_new[l] = BF_2.R[l]
    end
    Q_new = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(BF_1.Q)
        Q_new[k] = BF_2.Q[k]
    end

    P_new = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(BF_1.P)
        P_new[k] = BF_1.P[k]
    end

    #Compute M = Q_1 * P_1, Factorization now: Z = P_1 * R_1^RL1 * .... R^1 * M * R_2^RL2 * ... R_2^1 * Q_2
    #or in new indices: Z = P_1 * R_1^L_new * .... R_1^RL2+2 * M * R_2^RL2 * ... R_2^1 * Q_2
    M = Dict{Int,Matrix{ComplexF64}}()
    for nodeS in keys(BF_1.Q)
        M[nodeS] = BF_1.Q[nodeS] * BF_1.P[nodeS]
    end
    #Compute M = R_1^1 * M * R_2^RL2, Factorization now: Z = P_1 * R_1^L_new * .... R_1^RL2+1 * M * R_2^RL2-1 * ... R_2^1 * Q_2
    M2 = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
    for Snode in keys(BF_1.R[2])
        for Schild in children(trialT1, Snode)
            for Onode in keys(BF_1.R[1][Schild])
                for Snode2 in keys(BF_2.R[RL2])
                    if !haskey(M2, Snode2)
                        M2[Snode2] = Dict{Int,Matrix{ComplexF64}}()
                    end
                    update_mat =
                        BF_1.R[1][Schild][Onode] * M[Schild] * BF_2.R[RL2][Snode2][Schild]

                    dim1, dim2 = size(update_mat)

                    # Initialize with zeros if the key 'Onode' does not exist yet
                    get!(M2[Snode2], Onode) do
                        zeros(ComplexF64, dim1, dim2)
                    end

                    M2[Snode2][Onode] += update_mat
                end
            end
        end
    end
    R_new[RL2] = M2
    k = RL2
    for m in 1:(k - 1)
        for t in 1:M
            @views Barrowswap!(R_new, k + 1 - t)
            @views multiply_level_R_right!(R_new, k - m)
            recompress_FatBF!(P_new, R_new, Q_new)
        end
    end

    return BF2(
        Q_new,
        R_new,
        P_new,
        BF_1.tree,
        (size(BF_1, 1), size(BF_2, 2)), # Corrected Dim
        RL1 + RL2 + 1,                 # Corrected Level count
        BF_1.NS,
        BF_2.NO,
        k,
        τ,         # Usually keep original NS/NO
    )
end
