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
                    R_k = Vector{Matrix{ComplexF64}}()
                    row_spc = Vector{Int}()
                    i = 1
                    for Ochild in children(testT, nodeO)
                        push!(R_k, R[nodeS][Ochild])
                        push!(row_spc, size(R_k[i], 1))
                        i += 1
                    end

                    A = vcat(R_k...)
                    Q_new = Matrix{ComplexF64}(undef, size(A, 1), 0)

                    last = 0

                    for Schild in children(trialT, nodeS)
                        if l < LS - 1
                            col_spc = size(R[Schild][nodeO], 1)
                        else
                            col_spc = size(Q[Schild], 1)
                        end
                        QRA = pqr(A[:, (last + 1):(last + col_spc)]; rtol=τ)
                        Q_new = hcat(Q_new, QRA[1])
                        if !haskey(R_u, Schild)
                            R_u[Schild] = Dict{Int,Matrix{ComplexF64}}()
                        end
                        R_u[Schild][nodeO] = QRA[2][:, invperm(QRA[3])]
                        last += col_spc
                    end

                    last = 0
                    j = 1
                    for Ochild in children(testT, nodeO)
                        R[nodeS][Ochild] = Q_new[(last + 1):(last + row_spc[j]), :]
                        last += row_spc[j]
                        j += 1
                    end
                end
            end
            update_next_level_R_right!(R, Q, R_u, l, H2Blocktree, NS, NO)
        elseif !source_is_frozen && obs_is_frozen
        end
    end

    return BF(Q, R, P, NS, NO, k, τ)
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

                    # --- build A via hcat ---
                    R_k = Vector{Matrix{ComplexF64}}()
                    col_spc = Vector{Int}()

                    i = 1
                    for Schild in children(trialT, nodeS)
                        if l < LS - 1
                            push!(R_k, R[Schild][nodeO])
                            push!(col_spc, size(R_k[i], 2))
                        else
                            push!(R_k, Q[Schild])
                            push!(col_spc, size(R_k[i], 2))
                        end
                        i += 1
                    end

                    A = hcat(R_k...)

                    Q_new = Matrix{ComplexF64}(undef, 0, size(A, 2))

                    last = 0

                    # --- split along ROWS ---
                    for Ochild in children(testT, nodeO)
                        if l < LO - 1
                            row_spc = size(R[nodeS][Ochild], 2)
                        else
                            row_spc = size(Q[nodeO], 2)
                        end

                        rows = (last + 1):(last + row_spc)

                        Ai = A[rows, :]

                        # --- RRQR on transpose ---
                        QRA = pqr(Ai'; rtol=τ)

                        Qi = QRA[1]
                        Ri = QRA[2][:, invperm(QRA[3])]

                        QiT = Qi'

                        Q_new = vcat(Q_new, QiT)

                        if !haskey(R_u, nodeS)
                            R_u[nodeS] = Dict{Int,Matrix{ComplexF64}}()
                        end
                        R_u[nodeS][Ochild] = Ri'

                        last += row_spc
                    end

                    # --- write back ---
                    last = 0
                    j = 1
                    for Schild in children(trialT, nodeS)
                        cols = (last + 1):(last + col_spc[j])
                        if l < LS - 1
                            R[Schild][nodeO] = Q_new[:, cols]
                        else
                            Q[Schild] = Q_new[:, cols]
                        end
                        last += col_spc[j]
                        j += 1
                    end
                end
            end

            update_next_level_R_left!(R, Q, R_u, l, H2Blocktree, NS, NO)
        end
    end

    return BF(Q, R, P, NS, NO, k, τ)
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
                R[nodeS][nodeO] = R_u[nodeS][nodeO] * R[nodeS][nodeO]
            end
        end
    else
        counter = 1
        for nodeS in treeS[LS]          #future: leaves(trialT)
            Q[nodeS] = R_u[nodeS][NO] * Q[nodeS]
            counter += 1
        end
    end
end

function update_next_level_R_left!(R, Q, R_u, l, H2Blocktree, NS, NO)
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)

    if l < LS - 1
        # propagate via RIGHT multiplication
        for nodeS in treeS[l + 1]
            for nodeO in treeO[LO - l - 1]
                if haskey(R_u, nodeS) && haskey(R_u[nodeS], nodeO)
                    R[nodeS][nodeO] = R[nodeS][nodeO] * R_u[nodeS][nodeO]
                end
            end
        end
    else
        # final stage → update Q
        for nodeO in treeO[1]
            if haskey(R_u, nodeO) && haskey(R_u[nodeO], nodeO)
                Q[nodeO] = Q[nodeO] * R_u[nodeO][nodeO]
            end
        end
    end
end

function recompress_BF(Butterfly::BF, tree)
    newBF = recompress_BF_right(Butterfly, tree)
    newBF2 = recompress_BF_left(newBF, tree)
    return newBF2
end
