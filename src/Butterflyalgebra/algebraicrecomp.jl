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

    if l < LS - 1
        for nodeS in treeS[l + 1]
            for nodeO in treeO[LO - l]
                for Schild in children(trialT, nodeS)
                    R[Schild][nodeO] = R_u[nodeS][nodeO] * R[Schild][nodeO]
                    #@views mul!(R[nodeS][Ochild], R_u[nodeS][nodeO], R[nodeS][Ochild])
                end
            end
        end
    else
        for nodeS in treeS[LS]
            Q[nodeS] = R_u[nodeS][NO] * Q[nodeS]
        end
    end
end

function recompress_BF_left(Butterfly::BF, H2Blocktree)
    return recompress_BF_right(
        Butterfly',
        H2Trees.BlockTree(H2Trees.trialtree(H2Blocktree), H2Trees.testtree(H2Blocktree)),
    )'
end

function recompress_BF(Butterfly::BF, H2Blocktree)
    return recompress_BF_left(recompress_BF_right(Butterfly, H2Blocktree), H2Blocktree)
end
