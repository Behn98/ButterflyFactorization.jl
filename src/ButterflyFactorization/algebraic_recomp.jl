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
        R_u = Vector{Matrix{ComplexF64}}()
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
                    col_spc = Vector{Int}()
                    for Schild in children(trialT, nodeS)
                        push!(col_spc, size(R[Schild][nodeO], 1))
                    end
                    A = vcat(R_k...)
                    Q_new = Matrix{ComplexF64}(undef, size(A, 1), 0)

                    last = 0
                    # --- recompress A ---
                    for j in eachindex(col_spc)
                        QRA = pqr(A[:, (last + 1):(last + col_spc[j])]; rtol=τ)
                        Q_new = hcat(Q_new, QRA[1])
                        push!(R_u, QRA[2][:, invperm(QRA[3])])
                        last += col_spc[j]
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

function recompress_BF_left(Butterfly::BF, tree)
    return recompress_BF_right(Butterfly', tree)'
end

function recompress_BF(Butterfly::BF, tree)
    return recompress_BF_left(recompress_BF_right(Butterfly, tree), tree)
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
        counter = 1
        for nodeS in treeS[l + 1]
            for nodeO in treeO[LO - l - 1]
                R[nodeS][nodeO] = R_u[counter] * R[nodeS][nodeO]
                counter += 1
            end
        end
    else
        counter = 1
        for nodeS in treeS[LS]          #future: leaves(trialT)
            Q[nodeS] = R_u[counter] * Q[nodeS]
            counter += 1
        end
    end
end
