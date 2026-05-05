function mulBFs(BF_1::BF3, BF_2::BF3, τ::Float64)
    # 1. Initialization and Leaf Fusion
    # M maps Observer Leaves (Tree 2) to Source Leaves (Tree 1)
    @assert BF_1.level == BF_2.level "Both BFs must have the same number of levels"
    @assert BF_1.NS == BF_2.NO "Source and Observer dimensions must match for multiplication"
    M_messenger = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    for leaf in keys(BF_1.Q)
        M_messenger[BF_1.NO, leaf] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
        # Initialize as a nested dict to work with swap_and_recompress
        M_messenger[BF_1.NO, leaf][leaf, BF_2.NS] = BF_1.Q[leaf] * BF_2.P[leaf]
    end

    L = BF_1.level # Number of R-levels
    M_messenger = mul_factors(BF_1.R[1], M_messenger)
    M_messenger = mul_factors(M_messenger, BF_2.R[L])

    result = AlgBF(
        (size(BF_1, 1), size(BF_2, 2)),
        BF_2.Q,
        vcat(BF_2.R[1:(L - 1)], [M_messenger], BF_1.R[2:L]),
        BF_1.P,
    )
    for m in 1:(L - 1)
        for t in 1:m
            result = swap_and_recompress(result, L + 2 - t, τ)
        end
        result = recompress_BF(mul_factors(result, L + 1 - m), τ)
    end
    # 4. Construct the BF2 Result
    # Q remains Q_2, P is P_new, R is the 3-key dictionary R_final
    return BF3(
        result.Q,         # Q_final = Q_2
        result.R,       # R_final[level][Snode][Onode]
        result.P,          # Updated P
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
    leftfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}},
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}},
)
    product = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    for row in keys(leftfactor)
        if !haskey(product, row)
            product[row] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
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

function mul_factors(BF::AlgBF, idx::Int)
    L = length(BF.R)
    if idx > 1 && idx < (L + 1)
        leftfactor = BF.R[L + 1 - (idx - 1)]
        rightfactor = BF.R[L + 1 - idx]
    elseif idx == 1
        @show "Multiplying P and R[1]"
        leftfactor = BF.P
        rightfactor = BF.R[L + 1 - idx]
    else
        @show "Multiplying R[end] and Q"
        leftfactor = BF.R[L + 1 - idx]
        rightfactor = BF.Q
    end
    product = mul_factors(leftfactor, rightfactor)

    return AlgBF(
        (size(BF, 1), size(BF, 2)),
        BF.Q,
        vcat(BF.R[1:(L - idx)], [product], BF.R[(L - idx + 3):length(BF.R)]),
        BF.P,
    )
end

function swap_and_recompress(BF::AlgBF, idx::Int, τ)
    L = length(BF.R)
    if idx > 1 && idx < (L + 1)
        leftfactor = BF.R[L + 1 - (idx - 1)]
        rightfactor = BF.R[L + 1 - idx]
    elseif idx == 1
        @show "Multiplying P and R[L]"
        leftfactor = BF.P
        rightfactor = BF.R[L + 1 - idx]
    else
        @show "Multiplying R[1] and Q"
        leftfactor = BF.R[L + 1 - idx]
        rightfactor = BF.Q
    end
    nlfactor, nrfactor = swap_and_recompress(leftfactor, rightfactor, τ)

    return AlgBF(
        (size(BF, 1), size(BF, 2)),
        BF.Q,
        vcat(BF.R[1:(L - idx)], [nrfactor, nlfactor], BF.R[(L - idx + 3):length(BF.R)]),
        BF.P,
    )
end

# A generic factor is just Dict{RowKey, Dict{ColKey, Matrix}}
function swap_and_recompress(LeftFactor, RightFactor, τ)
    NewLeftFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    NewRightFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    #LeftFactor = reverse_dict_keys(LeftFactor)
    #RightFactor = reverse_dict_keys(RightFactor)
    NewLeftFactor = mul_factors(LeftFactor, RightFactor)
    col = Vector{Tuple{Int,Int}}(undef, 0)
    for row in keys(NewLeftFactor)
        col = unique(append!(col, keys(NewLeftFactor[row])))
    end
    for col_idx in col
        rows_with_col = [
            row for (row, inner_dict) in NewLeftFactor if haskey(inner_dict, col_idx)
        ]
        R_k = Vector{Matrix{ComplexF64}}()
        row_spc = Vector{Int}()
        i = 1
        for row in rows_with_col
            push!(R_k, NewLeftFactor[row][col_idx])
            push!(row_spc, size(R_k[i], 1))

            i += 1
        end
        A_k = vcat(R_k...)
        #@show size(A_k)
        QRA = pqr(A_k; rtol=τ)
        if !haskey(NewRightFactor, col_idx)
            NewRightFactor[col_idx] = Dict{Int,AbstractMatrix{ComplexF64}}()
        end
        #=
        if haskey(R_u[col_idx], col_idx)
            @show "Warning: Collision in R_u at column index $col_idx"
        end
        =#
        NewRightFactor[col_idx][col_idx] = QRA[2][:, invperm(QRA[3])]
        last = 0
        j = 1
        for row in rows_with_col
            NewLeftFactor[row][col_idx] = QRA[1][(last + 1):(last + row_spc[j]), :]
            last += row_spc[j]
            j += 1
        end
    end

    return NewLeftFactor, NewRightFactor
end
