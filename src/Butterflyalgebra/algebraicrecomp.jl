function recompress_BF_left(Butterfly::BF2, τ)
    return recompress_BF_right(Butterfly', τ)'
end

function recompress_BF(Butterfly::BF2, τ)
    return recompress_BF_left(recompress_BF_right(Butterfly, τ), τ)
end

function recompress_BF_left(Butterfly::BF3, τ)
    return recompress_BF_right(Butterfly', τ)'
end

function recompress_BF(Butterfly::BF3, τ)
    return recompress_BF_left(recompress_BF_right(Butterfly, τ), τ)
end

function find_rows_for_column(R::Dict{T,Dict{T,Matrix}}, col_idx::Int) where {T}
    rows = Vector{T}()
    for (row, inner_dict) in R
        if haskey(inner_dict, col_idx)
            push!(rows, row)
        end
    end
    return rows
end

function recompress_BF_right(Butterfly::BF3, τ)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NS = Butterfly.NS
    NO = Butterfly.NO
    k = Butterfly.k
    τ2 = Butterfly.τ

    # --- trees & helpers ---
    H2Blocktree = Butterfly.tree

    for l in eachindex(R)
        lold = Butterfly.level - l + 1
        R_u = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
        col = Vector{Tuple{Int,Int}}(undef, 0)
        for row in keys(R[lold])
            unique(append!(col, keys(R[lold][row])))
        end
        for col_idx in col
            rows_with_col = [
                row for (row, inner_dict) in R[lold] if haskey(inner_dict, col_idx)
            ]
            R_k = Vector{Matrix{ComplexF64}}()
            row_spc = Vector{Int}()
            i = 1
            for row in rows_with_col
                push!(R_k, R[lold][row][col_idx])
                push!(row_spc, size(R_k[i], 1))

                i += 1
            end
            A_k = vcat(R_k...)
            QRA = pqr(A_k; rtol=τ)
            if !haskey(R_u, col_idx[1])
                R_u[col_idx[1]] = Dict{Int,Matrix{ComplexF64}}()
            end
            R_u[col_idx[1]][col_idx[2]] = QRA[2][:, invperm(QRA[3])]
            last = 0
            j = 1
            for row in rows_with_col
                R[lold][row][col_idx] = QRA[1][(last + 1):(last + row_spc[j]), :]
                last += row_spc[j]
                j += 1
            end
        end
        if l < Butterfly.level - 1
            update_next_level_R_right!(R[lodl - 1], R_u)
        else
            update_next_level_R_right!(Q, R_u)
        end
    end

    return BF3(Q, R, P, H2Blocktree, Butterfly.dim, Butterfly.level, NS, NO, k, τ2)
end

@views function update_next_level_R_right!(
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}},
    R_u::Dict{Int,AbstractMatrix{ComplexF64}},
)
    for row in keys(rightfactor)
        for col in keys(rightfactor[row])
            rightfactor[row][col] = R_u[row[1]][row[2]] * rightfactor[row][col]
        end
    end
end

@views function update_next_level_R_right!(
    rightfactor::Dict{Int,AbstractMatrix{ComplexF64}},
    R_u::Dict{Int,AbstractMatrix{ComplexF64}},
)
    for nodeS in treeS[LS]
        Q[nodeS] = R_u[nodeS][NO] * Q[nodeS]
    end
end

function recompress_BF_right(Butterfly::BF2, τ)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NS = Butterfly.NS
    NO = Butterfly.NO
    k = Butterfly.k
    τ2 = Butterfly.τ

    # --- trees & helpers ---
    H2Blocktree = Butterfly.tree
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    children = H2Trees.children
    level = H2Trees.level

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)
    LevelS = level(trialT, NS)
    LevelO = level(testT, NO)
    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)

    source_is_frozen = false
    obs_is_frozen = false

    for l in 1:(Butterfly.level - 1)
        lold = Butterfly.level - l
        R_u = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)
        #test = collect(keys(R[lold]))[1]
        #lS = LevelS + 1 - level(trialT, test)
        #lO = LevelO + 1 - level(testT, collect(keys(R[lold][test]))[1])
        if source_is_frozen && obs_is_frozen
            break
        elseif !source_is_frozen && !obs_is_frozen
            for nodeS in treeS[l]
                for nodeO in treeO[LO - l]
                    for Schild in children(trialT, nodeS)
                        R_k = Vector{Matrix{ComplexF64}}()
                        row_spc = Vector{Int}()
                        i = 1
                        for Ochild in children(testT, nodeO)
                            push!(R_k, R[lold][Schild][Ochild])
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
                            R[lold][Schild][Ochild] = QRA[1][
                                (last + 1):(last + row_spc[j]), :,
                            ]
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

    return BF2(Q, R, P, H2Blocktree, Butterfly.dim, Butterfly.level, NS, NO, k, τ2)
end

@views function update_next_level_R_right!(R, Q, R_u, l, H2Blocktree, NS, NO)
    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)
    children = H2Trees.children
    level = H2Trees.level
    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)

    if l < LS - 1
        lold = max(LO, LS) - l
        test = collect(keys(R[lold]))[1]
        lS = level(trialT, test)
        lO = level(testT, collect(keys(R[lold][test]))[1])
        for nodeS in treeS[lS]
            for nodeO in treeO[lO - 1]
                for Schild in children(trialT, nodeS)
                    R[lold - 1][Schild][nodeO] =
                        R_u[nodeS][nodeO] * R[lold - 1][Schild][nodeO]
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
