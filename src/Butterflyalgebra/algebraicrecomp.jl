function recompress_BF_right(Butterfly::BF, H2Blocktree)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NS = Butterfly.NS
    NO = Butterfly.NO
    k = Butterfly.k
    τ = Butterfly.τ

    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    center = H2Trees.center
    halfsize = H2Trees.halfsize
    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO

    source_is_frozen = false
    obs_is_frozen = false

    for l in 1:(L - 1)
        R_u = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)
        if source_is_frozen && obs_is_frozen
            break
        elseif !source_is_frozen && !obs_is_frozen
            for nodeS in treeS[l]
                for nodeO in treeO[LO - l]
                    first = true
                    for Schild in children(trialT, nodeS)
                        R_k = Vector{Matrix{ComplexF64}}()
                        row_spc = Vector{Int}()
                        i = 1
                        for Ochild in children(testT, nodeO)
                            push!(R_k, R[Schild][Ochild])
                            push!(row_spc, size(R_k[i], 1))

                            i += 1
                        end
                        A_k = vcat(R_k...)
                        QRA = pqr(A_k; rtol=τ)
                        if !haskey(R_u, Schild)
                            R_u[Schild] = Dict{Int,Matrix{ComplexF64}}()
                        end
                        R_u[Schild][nodeO] = QRA[2][:, invperm(QRA[3])]
                        last = 0
                        j = 1
                        for Ochild in children(testT, nodeO)
                            R[Schild][Ochild] = QRA[1][(last + 1):(last + row_spc[j]), :]
                            last += row_spc[j]
                            j += 1
                        end
                    end
                end
            end
            update_next_level_R_right!(R, Q, R_u, l, H2Blocktree, NS, NO)
        elseif !source_is_frozen && obs_is_frozen
        end
    end

    return BF(Q, R, P, NS, NO, k, τ, Butterfly.sym)
end

function update_next_level_R_right!(R, Q, R_u, l, H2Blocktree, NS, NO)
    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    center = H2Trees.center
    halfsize = H2Trees.halfsize
    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO
    if l < LS - 1
        for nodeS in treeS[l + 1]
            for nodeO in treeO[LO - l]
                for Schild in children(trialT, nodeS)
                    for Ochild in treeO[LO - l +2]
                        R[Schild][Ochild] = R_u[nodeS][nodeO] * R[Schild][Ochild]
                        #@views mul!(R[nodeS][Ochild], R_u[nodeS][nodeO], R[nodeS][Ochild])
                    end
                end
            end
        end
    else
        counter = 1
        for nodeS in treeS[LS]
            #Q[nodeS] = R_u[nodeS][NO] * Q[nodeS]
            @views mul!(Q[nodeS], R_u[nodeS][NO], Q[nodeS])
            counter += 1
        end
    end
end

function recompress_symBF_left(Butterfly::BF, H2Blocktree)
    return recompress_BF_right(
        Butterfly',
        H2Trees.BlockTree(H2Trees.trialtree(H2Blocktree), H2Trees.testtree(H2Blocktree)),
    )'
end

function recompress_symBF(Butterfly::BF, H2Blocktree)
    return recompress_symBF_left(recompress_BF_right(Butterfly, H2Blocktree), H2Blocktree)
end

function recompress_BF_left(Butterfly::BF, H2Blocktree)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NS = Butterfly.NS
    NO = Butterfly.NO
    k = Butterfly.k
    τ = Butterfly.τ

    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)
    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO

    # Reverse loop: Move DOWN the observer tree
    for l in (L - 1):-1:1
        R_u = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()

        source_is_frozen = l >= LS
        obs_is_frozen = l >= LO

        if source_is_frozen && obs_is_frozen
            continue # Safely skip out-of-bounds levels when traversing backwards

        elseif !source_is_frozen && !obs_is_frozen
            for nodeS in treeS[l]
                for nodeO in treeO[LO - l]

                    # --- build A via hcat ---
                    R_k = Vector{Matrix{ComplexF64}}()
                    col_spc = Vector{Int}()

                    for Ochild in children(testT, nodeO)
                        push!(R_k, adjoint(R[nodeS][Ochild]))
                        push!(col_spc, size(R_k[end], 2))
                    end

                    A = hcat(R_k...)
                    Q_new = Matrix{ComplexF64}(undef, 0, size(A, 2))
                    last = 0

                    for Schild in children(trialT, nodeS)
                        if l < LS - 1
                            row_spc = size(R[Schild][nodeO], 1)
                        else
                            row_spc = size(Q[Schild], 1)
                        end
                        QRA = pqr(A[(last + 1):(last + row_spc), :]; rtol=τ)
                        Q_new = vcat(Q_new, QRA[1])
                        if !haskey(R_u, Schild)
                            R_u[Schild] = Dict{Int,Matrix{ComplexF64}}()
                        end
                        R_u[Schild][nodeO] = QRA[2][:, invperm(QRA[3])]
                        last += row_spc
                    end

                    # --- write back ---
                    last = 0
                    j = 1
                    for Ochild in children(testT, nodeO)
                        R[nodeS][Ochild] = adjoint(Q_new[:, (last + 1):(last + col_spc[j])])
                        last += col_spc[j]
                        j += 1
                    end
                end
            end

            update_next_level_R_left!(R, P, R_u, l, H2Blocktree, NS, NO)
        end
    end

    return BF(Q, R, P, NS, NO, k, τ, BF.sym)
end

function update_next_level_R_left!(R, P, R_u, l, H2Blocktree, NS, NO)
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LO = length(treeO)

    if l > 1
        # Push transfer factor down to Ochild for the next loop iteration
        for nodeS in treeS[l - 1]
            for nodeO in treeO[LO - l] # This maps to Ochild from the main loop
                R_temp = Vector{Matrix{ComplexF64}}()

                R[nodeS][nodeO] = R[nodeS][nodeO] * R_u[nodeS][nodeO]
            end
        end
    else
        # Final stage at l = 1 -> Ochild is at the leaves (level LO), update P
        for nodeS in treeS[1] # Source root
            for nodeO in treeO[LO]
                if haskey(R_u, nodeS) && haskey(R_u[nodeS], nodeO)
                    P[nodeO] = P[nodeO] * R_u[nodeS][nodeO]
                end
            end
        end
    end
end

function recompress_BF(Butterfly::BF, tree)
    newBF = recompress_BF_right(Butterfly, tree)
    newBF2 = recompress_BF_left(newBF, tree)
    return newBF2
end
