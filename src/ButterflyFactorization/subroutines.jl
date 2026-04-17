struct BF
    Q::Dict{Int,Matrix{ComplexF64}}
    R::Dict{Int,Dict{Int,Matrix{ComplexF64}}}
    P::Dict{Int,Matrix{ComplexF64}}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    BF(Q, R, P, NS, NO, k, τ) = new(Q, R, P, NS, NO, k, τ)
end

struct BF_Mats
    Q::AbstractMatrix{ComplexF64}       #AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices
    R::Vector{AbstractMatrix{ComplexF64}}       #AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices
    P::AbstractMatrix{ComplexF64}       #AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    PermP::Vector{Int}          #permutation of source indices for P blocks, needed for correct assembly of R blocks
    PermQ::Vector{Int}          #permutation of source indices for Q blocks, needed for correct assembly of R blocks
    BF_Mats(Q, R, P, NS, NO, k, τ, PermP, PermQ) = new(Q, R, P, NS, NO, k, τ, PermP, PermQ)
end

@inline function getsubdict!(D::Dict{Int,Dict{Int,T}}, k::Int) where {T}
    get!(D, k) do
        Dict{Int,T}()
    end
end

function subroutine_BF_approx_treeh2(
    farassembler,
    H2Blocktree,
    NO::Int,
    NS::Int,
    k::Float64,
    τ::Float64;
    Compressor=ButterflyFactorization.PartialQR(),
)

    # --- containers ---
    Q = Dict{Int,Matrix{ComplexF64}}()

    R = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
    K = Dict{Int,Dict{Int,Vector{Int}}}()
    U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions

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

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    for Sleaf in treeS[LS]
        srcindex = values(trialT, Sleaf)
        obsindex = values(testT, NO)
        c_s = center(trialT, Sleaf)
        c_o = center(testT, NO)
        a_s = halfsize(trialT, Sleaf)
        a_o = halfsize(testT, NO)

        n_otilde = estimate_rank_3d(k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3)
        q_ks, k_l, r_l = Compressor(farassembler, srcindex, obsindex, n_otilde, τ)

        Q[Sleaf] = q_ks
        getsubdict!(K, Sleaf)[NO] = k_l
    end

    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------------
    for l in 1:(L - 1)
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)

        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        if !source_is_frozen
            for Svert in treeS[LS - l]
                U_S = getsubdict!(U, Svert)

                for Overt in treeO[min(l, LO)]
                    temp = Int[]

                    for Schild in children(trialT, Svert)
                        Ks = getsubdict!(K, Schild)
                        ks = get(Ks, Overt, nothing)
                        append!(temp, ks)
                    end

                    U_S[Overt] = temp
                end
            end
        end

        # --------------------------------------------------------------
        # Compute R blocks
        # --------------------------------------------------------------
        if !source_is_frozen && !obs_is_frozen
            rowsizeR = 0
            for Overt in treeO[l]
                for Ochild in children(testT, Overt)
                    obsindex = values(testT, Ochild)
                    isempty(obsindex) && continue
                    c_o = center(testT, Ochild)
                    a_o = halfsize(testT, Ochild)
                    for Svert in treeS[LS - l]
                        srcindex = U[Svert][Overt]
                        c_s = center(trialT, Svert)
                        a_s = halfsize(trialT, Svert)
                        n_otilde = estimate_rank_3d(
                            k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3
                        )
                        q_ks, k_l, r_l = Compressor(
                            farassembler, srcindex, obsindex, n_otilde, τ
                        )
                        rowsizeR += size(q_ks, 1)
                        getsubdict!(R, Svert)[Ochild] = q_ks
                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                end
            end

        elseif source_is_frozen && !obs_is_frozen
            @show source_is_frozen
            for Overt in treeO[l]
                for Ochild in children(testT, Overt)
                    obsindex = values(testT, Ochild)
                    c_o = center(testT, Ochild)
                    a_o = halfsize(testT, Ochild)
                    for Svert in treeS[1]
                        srcindex = K[Svert][Overt]
                        c_s = center(trialT, Svert)
                        a_s = halfsize(trialT, Svert)

                        n_otilde = estimate_rank_3d(
                            k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3
                        )
                        q_ks, k_l, r_l = Compressor(
                            farassembler, srcindex, obsindex, n_otilde, τ
                        )
                        getsubdict!(R, Svert)[Ochild] = q_ks

                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Overt in treeO[LO]
                obsindex = values(testT, Overt)
                c_o = center(testT, Overt)
                a_o = halfsize(testT, Overt)
                for Svert in treeS[LS - l]
                    srcindex = U[Svert][Overt]

                    c_s = center(trialT, Svert)

                    a_s = halfsize(trialT, Svert)

                    n_otilde = estimate_rank_3d(
                        k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3
                    )
                    q_ks, k_l, r_l = Compressor(
                        farassembler, srcindex, obsindex, n_otilde, τ
                    )

                    getsubdict!(R, Svert)[Overt] = q_ks
                    getsubdict!(K, Svert)[Overt] = k_l
                end
            end

        else
            break
        end
    end

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------
    P = Dict{Int,Matrix{ComplexF64}}()
    for Oleaf in treeO[LO]
        col = K[NS][Oleaf]
        row = values(testT, Oleaf)

        Z = zeros(ComplexF64, length(row), length(col))
        farassembler(Z, row, col)

        P[Oleaf] = Z
    end
    result = BF(Q, R, P, NS, NO, τ, k)
    return result
end

function subroutine_BF_approx_treeh2_mats(
    farassembler,
    H2Blocktree,
    NO::Int,
    NS::Int,
    k::Float64,
    τ::Float64;
    Compressor=ButterflyFactorization.PartialQR(),
)

    # --- containers ---
    Q = Matrix{ComplexF64}(undef, 0, 0)
    R = Vector{AbstractMatrix{ComplexF64}}()            #AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices
    P = Matrix{ComplexF64}(undef, 0, 0)
    K = Dict{Int,Dict{Int,Vector{Int}}}()
    U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions

    PermQ = Vector{Int}()          #permutation of source indices for Q blocks, needed for correct assembly of R blocks
    PermP = Vector{Int}()          #permutation of source indices for P blocks, needed for correct assembly of R blocks

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

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    for Sleaf in treeS[LS]
        srcindex = values(trialT, Sleaf)
        push!(PermQ, srcindex...)
        obsindex = values(testT, NO)
        c_s = center(trialT, Sleaf)
        c_o = center(testT, NO)
        a_s = halfsize(trialT, Sleaf)
        a_o = halfsize(testT, NO)
        n_otilde = estimate_rank_3d(k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3)
        q_ks, k_l, r_l = Compressor(farassembler, srcindex, obsindex, n_otilde, τ)
        Q = sparse_blockdiag(Q, q_ks)               #SPARSITY: sparse_ or blocksparse_
        getsubdict!(K, Sleaf)[NO] = k_l
    end
    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------------
    for l in 1:(L - 1)
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)

        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        if !source_is_frozen
            for Svert in treeS[LS - l]
                U_S = getsubdict!(U, Svert)

                for Overt in treeO[min(l, LO)]
                    temp = Int[]

                    for Schild in children(trialT, Svert)
                        Ks = getsubdict!(K, Schild)
                        ks = get(Ks, Overt, nothing)
                        append!(temp, ks)
                    end

                    U_S[Overt] = temp
                end
            end
        end

        # --------------------------------------------------------------
        # Compute R blocks
        # --------------------------------------------------------------
        if !source_is_frozen && !obs_is_frozen
            R_temp1 = Matrix{ComplexF64}(undef, 0, 0)
            for Overt in treeO[l]
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()      #AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices
                for Ochild in children(testT, Overt)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                    obsindex = values(testT, Ochild)
                    c_o = center(testT, Ochild)
                    a_o = halfsize(testT, Ochild)
                    for Svert in treeS[LS - l]
                        srcindex = U[Svert][Overt]
                        c_s = center(trialT, Svert)
                        a_s = halfsize(trialT, Svert)

                        n_otilde = estimate_rank_3d(
                            k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3
                        )
                        q_ks, k_l, r_l = Compressor(
                            farassembler, srcindex, obsindex, n_otilde, τ
                        )
                        R_temp3 = sparse_blockdiag(R_temp3, q_ks)   #SPARSITY: sparse_ or blocksparse_
                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                    push!(R_temp2, R_temp3)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                end
                R_temp1 = sparse_blockdiag(R_temp1, sparse_vcat(R_temp2...))    #SPARSITY: sparse_ or blocksparse_
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()          #AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices
            end

            push!(R, R_temp1)

        elseif source_is_frozen && !obs_is_frozen
            @show source_is_frozen
            R_temp1 = Matrix{ComplexF64}(undef, 0, 0)
            for Overt in treeO[l]
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()          #AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices
                for Ochild in children(testT, Overt)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                    obsindex = values(testT, Ochild)
                    c_o = center(testT, Ochild)
                    a_o = halfsize(testT, Ochild)
                    for Svert in treeS[1]
                        srcindex = K[Svert][Overt]
                        c_s = center(trialT, Svert)
                        a_s = halfsize(trialT, Svert)

                        n_otilde = estimate_rank_3d(
                            k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3
                        )
                        q_ks, k_l, r_l = Compressor(
                            farassembler, srcindex, obsindex, n_otilde, τ
                        )
                        R_temp3 = sparse_blockdiag(R_temp3, q_ks)   #SPARSITY: sparse_ or blocksparse_

                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                    push!(R_temp2, R_temp3)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                end
                R_temp1 = sparse_blockdiag(R_temp1, sparse_vcat(R_temp2...))    #SPARSITY: sparse_ or blocksparse_
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()          #AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices
            end
            push!(R, R_temp1)

        elseif !source_is_frozen && obs_is_frozen
            R_temp1 = Matrix{ComplexF64}(undef, 0, 0)
            for Overt in treeO[LO]
                obsindex = values(testT, Overt)
                R_temp2 = Matrix{ComplexF64}(undef, 0, 0)
                c_o = center(testT, Overt)
                a_o = halfsize(testT, Overt)
                for Svert in treeS[LS - l]
                    srcindex = U[Svert][Overt]

                    c_s = center(trialT, Svert)
                    a_s = halfsize(trialT, Svert)

                    n_otilde = estimate_rank_3d(
                        k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3
                    )
                    q_ks, k_l, r_l = Compressor(
                        farassembler, srcindex, obsindex, n_otilde, τ
                    )
                    R_temp2 = sparse_blockdiag(R_temp2, q_ks)   #SPARSITY: sparse_ or blocksparse_

                    getsubdict!(K, Svert)[Overt] = k_l
                end
                R_temp1 = sparse_blockdiag(R_temp1, R_temp2)    #SPARSITY: sparse_ or blocksparse_
            end
            push!(R, R_temp1)
        else
            break
        end
    end

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------

    for Oleaf in treeO[LO]
        col = K[NS][Oleaf]
        row = values(testT, Oleaf)
        push!(PermP, row...)
        Z = zeros(ComplexF64, length(row), length(col))
        farassembler(Z, row, col)
        P = sparse_blockdiag(P, Z)              #SPARSITY: sparse_ or blocksparse_
    end
    return BF_Mats(Q, R, P, NS, NO, τ, k, PermP, PermQ)
end
