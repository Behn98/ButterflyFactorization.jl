# A generic factor is just Dict{RowKey, Dict{ColKey, Matrix}}
function swap_and_recompress(LeftFactor, RightFactor, τ)
    NewLeftFactor = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
    NewRightFactor = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
    LeftFactor = reverse_dict_keys(LeftFactor)
    RightFactor = reverse_dict_keys(RightFactor)
    # 1. Loop over the outer row space
    for row_node in keys(LeftFactor)

        # 2. Loop over the outer column space
        # (You have to find all possible col_nodes by scanning the RightFactor)
        target_col_nodes = unique([
            col for inner in keys(LeftFactor[row_node]) if haskey(RightFactor, inner) for
            col in keys(RightFactor[inner])
        ])

        for col_node in target_col_nodes
            local_block = nothing

            # 3. Sum over the shared middle index
            for inner_node in keys(LeftFactor[row_node])
                if haskey(RightFactor, inner_node) &&
                    haskey(RightFactor[inner_node], col_node)
                    update =
                        LeftFactor[row_node][inner_node] * RightFactor[inner_node][col_node]

                    if local_block === nothing
                        local_block = copy(update)
                    else
                        local_block += update
                    end
                end
            end

            # 4. RRQR and Split
            if local_block !== nothing
                @show local_block !== nothing
                QRA = pqr(local_block; rtol=τ)
                Q = QRA[1]
                R_qr = QRA[2][:, invperm(QRA[3])]
                rank_id = Int(hash((row_node, col_node)) & typemax(Int))
                # 5. Populate the new dictionaries based purely on the outer keys
                if !haskey(NewLeftFactor, row_node)
                    NewLeftFactor[row_node] = Dict{Int,Matrix{ComplexF64}}()
                end
                NewLeftFactor[row_node][rank_id] = Q

                # R_qr maps from col_node to col_node (it's a square-ish mixing matrix)
                if !haskey(NewRightFactor, rank_id)
                    NewRightFactor[rank_id] = Dict{Int,Matrix{ComplexF64}}()
                end
                NewRightFactor[rank_id][col_node] = R_qr
            end
        end
    end

    return NewLeftFactor, NewRightFactor
end

function mulBFs(BF_1::BF3, BF_2::BF3, τ::Float64)
    # 1. Initialization and Leaf Fusion
    # M maps Observer Leaves (Tree 2) to Source Leaves (Tree 1)
    @assert BF_1.level == BF_2.level "Both BFs must have the same number of levels"
    @assert BF_1.NS == BF_2.NO "Source and Observer dimensions must match for multiplication"
    M_messenger = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    for leaf in keys(BF_1.Q)
        M_messenger[BF_1.NO, leaf] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        # Initialize as a nested dict to work with swap_and_recompress
        M_messenger[BF_2.NO, leaf][leaf, BF_2.NS] = BF_1.Q[leaf] * BF_2.P[leaf]
    end

    L = BF_1.level # Number of R-levels
    M_messenger = mul_factors(BF_1.R[1], M_messenger)
    M_messenger = mul_factors(M_messenger, BF_2.R[L])

    # 4. Construct the BF2 Result
    # Q remains Q_2, P is P_new, R is the 3-key dictionary R_final
    return BF2(
        BF_2.Q,         # Q_final = Q_2
        R_final,       # R_final[level][Snode][Onode]
        P_new,          # Updated P
        BF_1.tree,      # Typically uses the structure of the first tree
        (size(BF_1, 1), size(BF_2, 2)),
        L,              # Resulting level count
        BF_2.NS,
        BF_1.NO,
        BF_1.k,         # Or recalculated k
        τ,
    )
end

function mul_factors(
    leftfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}},
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}},
)
    product = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    for row in keys(leftfactor)
        if !haskey(product, row)
            product[row] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        end
        for inner in keys(leftfactor[row])
            for col in keys(rightfactor[inner])
                if !haskey(product[row], col)
                    product[row][col] = Matrix{ComplexF64}(
                        undef,
                        size(leftfactor[row][inner])[1],
                        size(rightfactor[inner][col])[2],
                    )
                    @views mul!(
                        product[row][col], leftfactor[row][inner], rightfactor[inner][col]
                    )
                else
                    temp = Matrix{ComplexF64}(
                        undef,
                        size(leftfactor[row][inner])[1],
                        size(rightfactor[inner][col])[2],
                    )
                    @views mul!(temp, leftfactor[row][inner], rightfactor[inner][col])
                    product[row][col] += temp
                end
            end
        end
    end
    return product
end
