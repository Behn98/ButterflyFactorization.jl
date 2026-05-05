struct algebraBF
    Q::Dict{Int,Matrix{ComplexF64}}
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}
    P::Dict{Int,Matrix{ComplexF64}}
end

function Base.adjoint(B::algebraBF)
    lr = length(B.R)
    R_adj = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}(
        undef, lr
    )
    for l in eachindex(B.R)
        newl = lr - l + 1
        R_adj[newl] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
        for nodeS in keys(B.R[l])
            for nodeO in keys(B.R[l][nodeS])
                if !haskey(R_adj[newl], reverse(nodeO))
                    R_adj[newl][reverse(nodeO)] = Dict{
                        Tuple{Int,Int},AbstractMatrix{ComplexF64}
                    }()
                end
                R_adj[newl][reverse(nodeO)][reverse(nodeS)] = adjoint(B.R[l][nodeS][nodeO])
            end
        end
    end

    Q_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.Q)
        Q_adj[k] = adjoint(B.Q[k])
    end

    P_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.P)
        P_adj[k] = adjoint(B.P[k])
    end
    return algebraBF(P_adj, R_adj, Q_adj)
end

function recompress_BF_left(Butterfly::algebraBF, τ)
    return recompress_BF_right(Butterfly', τ)'
end

function recompress_BF(Butterfly::algebraBF, τ)
    return recompress_BF_left(recompress_BF_right(Butterfly, τ), τ)
end

function recompress_BF(Butterfly::BF3, τ)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    BFalg = algebraBF(Q, R, P)
    BFalg = recompress_BF(BFalg, τ)
    return BF3(
        BFalg.Q,
        BFalg.R,
        BFalg.P,
        Butterfly.tree,
        Butterfly.dim,
        Butterfly.level,
        Butterfly.NS,
        Butterfly.NO,
        Butterfly.k,
        Butterfly.τ,
    )
end

function recompress_BF_right(Butterfly::algebraBF, τ)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    lr = length(R)
    for l in eachindex(R)
        lold = lr - l + 1
        R_u = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
        col = Vector{Tuple{Int,Int}}(undef, 0)
        for row in keys(R[lold])
            col = unique(append!(col, keys(R[lold][row])))
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
            #@show size(A_k)
            QRA = pqr(A_k; rtol=τ)
            if !haskey(R_u, col_idx[1])
                R_u[col_idx[1]] = Dict{Int,Matrix{ComplexF64}}()
            end
            if haskey(R_u[col_idx[1]], col_idx[2])
                @show col_idx
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
        if l < lr
            R[lold - 1] = update_next_level_R_right(R_u, R[lold - 1])
        else
            Q = update_next_level_R_right(R_u, Q)
        end
    end

    return algebraBF(Q, R, P)
end

@views function update_next_level_R_right(
    R_u::Dict{Int,Dict{Int,Matrix{ComplexF64}}},
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}},
)
    for row in keys(rightfactor)
        for col in keys(rightfactor[row])
            rightfactor[row][col] = R_u[row[1]][row[2]] * rightfactor[row][col]
        end
    end
    return rightfactor
end

@views function update_next_level_R_right(
    R_u::Dict{Int,Dict{Int,Matrix{ComplexF64}}}, rightfactor::Dict{Int,Matrix{ComplexF64}}
)
    NO = collect(keys(R_u))[1]
    for nodeS in keys(rightfactor)
        rightfactor[nodeS] = R_u[NO][nodeS] * rightfactor[nodeS]
    end
    return rightfactor
end
