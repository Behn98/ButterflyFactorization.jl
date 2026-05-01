# A generic factor is just Dict{RowKey, Dict{ColKey, Matrix}}
function swap_and_recompress(LeftFactor, RightFactor, τ)
    NewLeftFactor = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
    NewRightFactor = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()

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
                Q, R_qr = pqr(local_block, τ)

                # 5. Populate the new dictionaries based purely on the outer keys
                if !haskey(NewLeftFactor, row_node)
                    NewLeftFactor[row_node] = Dict{Int,Matrix{ComplexF64}}()
                end
                NewLeftFactor[row_node][col_node] = Q

                # R_qr maps from col_node to col_node (it's a square-ish mixing matrix)
                if !haskey(NewRightFactor, col_node)
                    NewRightFactor[col_node] = Dict{Int,Matrix{ComplexF64}}()
                end
                NewRightFactor[col_node][col_node] = R_qr
            end
        end
    end

    return NewLeftFactor, NewRightFactor
end

function mulBFs(BF_1::BF2, BF_2::BF2, τ::Float64)
    # 1. Initialization and Leaf Fusion
    # M maps Observer Leaves (Tree 2) to Source Leaves (Tree 1)
    M_messenger = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
    for leaf in keys(BF_1.Q)
        # Initialize as a nested dict to work with swap_and_recompress
        M_messenger[leaf] = Dict(leaf => BF_1.Q[leaf] * BF_2.P[leaf])
    end

    L = length(BF_1.R) # Number of R-levels
    R_final = Dict{Int,Dict{Int,Dict{Int,Matrix{ComplexF64}}}}()

    # 2. Folding Loop: From Leaves to Root (i = 1 to L)
    # We "eat" R_1[i] and R_2[L-i+1] to produce R_final[i]
    for i in 1:L
        # Step A: Multiply Left Factor with current Messenger
        # R_1[i] maps: Snode -> O_child (where Snode is the leaf or parent level)
        # M_messenger maps: O_child -> O_child_from_T2
        temp_left, M_mid = swap_and_recompress(BF_1.R[i], M_messenger, τ)

        # Step B: Multiply Result with Right Factor
        # BF_2.R[L-i+1] maps: S_parent -> S_child (Tree 2)
        # M_mid maps: S_child -> O_child (Tree 1)
        temp_right, M_next = swap_and_recompress(M_mid, BF_2.R[L - i + 1], τ)

        # Step C: The "Fold" - Contract the two basis factors into one
        # These two share the same inner dimension after recompression
        R_level_i = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
        for row in keys(temp_left)
            for inner in keys(temp_left[row])
                if haskey(temp_right, inner)
                    for col in keys(temp_right[inner])
                        if !haskey(R_level_i, row)
                            R_level_i[row] = Dict{Int,Matrix{ComplexF64}}()
                        end

                        # Product of basis matrices to form the new level factor
                        prod = temp_left[row][inner] * temp_right[inner][col]

                        if !haskey(R_level_i[row], col)
                            R_level_i[row][col] = prod
                        else
                            R_level_i[row][col] += prod
                        end
                    end
                end
            end
        end

        # Store the level in the 3-key dictionary structure
        R_final[i] = R_level_i
        # Move the remaining coupling information to the next level
        M_messenger = M_next
    end

    # 3. Final Boundary Resolution
    # Absorb the final messenger into P_1
    P_new = Dict{Int,Matrix{ComplexF64}}()
    for row in keys(BF_1.P)
        # M_messenger is now at the root level, mapping root_children -> root_children
        for inner in keys(BF_1.P[row]) # This assumes BF_1.P is dict of dicts
            if haskey(M_messenger, inner)
                # In the final P, there is only one "col" (the root)
                for col in keys(M_messenger[inner])
                    update = BF_1.P[row][inner] * M_messenger[inner][col]
                    P_new[row] = haskey(P_new, row) ? P_new[row] + update : update
                end
            end
        end
    end

    # 4. Construct the BF2 Result
    # Q remains Q_2, P is P_new, R is the 3-key dictionary R_final
    return BF2(
        BF_2.Q,         # Q_final = Q_2
        R_final,       # R_final[level][Snode][Onode]
        P_new,          # Updated P
        H2Trees.BlockTree(H2Trees.testtree(BF_1.tree), H2Trees.trialtree(BF_2.tree)),      # Typically uses the structure of the first tree
        (size(BF_1, 1), size(BF_2, 2)),
        L,              # Resulting level count
        BF_1.NO,
        BF_2.NS,
        BF_1.k,         # Or recalculated k
        τ,
    )
end
