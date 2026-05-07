"""
The subroutines are the core for producing the butterfly factorization. They produce the Q,
R and P blocks as well as the permutations of the source and observer indices for the Q and
P blocks, which are needed for the correct MV products without saving the tree explicitly.
NO and NS are the IDs of the root nodes related by the Butterfly. The subroutine_BF produces
the Butterfly in a dictionary format, which is more intuitive and easier to debug. The
subroutine_BF_mats produces the Butterfly in a matrix format, which is more efficient for MV
products. The Compressor argument allows for different compression schemes to be used, with
the default being a partial QR decomposition. Note that the kernelmatrix is a function that
computes the matrix entries for given row and column indices, and is used in the compression
step to compute the low-rank approximations of the blocks. The subroutines traverse the H2
tree structure to compute the necessary blocks for the Butterfly factorization, starting
from the leaf level and moving up to the root of the source/trial tree and to the leafs of
the observer/test tree, while keeping track of the necessary permutations and unions of
skeletons. Be aware that also the wavenumber k plays a crucial role in the estimation of the
ranks and thus in the overall performance of the factorization. In terms of memory
efficiency, the matrices turn out to be less efficient than the dicitionaries, due to the
necessary overhead of saving the nonzero entry indices in the sparse/block-sparse format,
which is not needed in the dictionary format. However, the matrix format allows for much
faster MV products as well as providing a visualization of the structure of the Butterfly,
which can be helpful for debugging and understanding the factorization. The choice between
the two formats depends on the specific use case and requirements of the application.
Algebraic operations on the Butterfly factorization, such as addition and multiplication,
can be implemented using the dictionary format, as it allows for more flexible manipulation
of the blocks and their indices. The matrix format can be used for efficient application of
the Butterfly to vectors, but may not be as convenient for algebraic operations that require
access to individual blocks. Overall, the subroutines provide a way to construct the
Butterfly factorization from the H2 tree structure and the kernel matrix, with flexibility
in the choice of compression scheme and output format. Additionally unbalanced trees are
supported just as well as trees of different height. this allows for more efficiency when
compressing farinteractions.
"""

function subroutine_BF(
    kernelmatrix,
    H2Blocktree,
    NO::Int,
    NS::Int,
    k::Float64,
    τ::Float64;
    Compressor=ButterflyFactorization.PartialQR(),
)

    # --- containers ---
    Q = Dict{Int,Matrix{ComplexF64}}()
    K = Dict{Int,Dict{Int,Vector{Int}}}()
    U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions
    PermQ = Dict{Int,Vector{Int}}()
    PermP = Dict{Int,Vector{Int}}()
    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    center = H2Trees.center
    halfsize = H2Trees.halfsize
    children = H2Trees.children

    treeS = traverseandpad(trialT, NS)
    treeO = traverseandpad(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)
    R = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}(
        undef, L - 1
    )

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    for Sleaf in treeS[LS]  #--> watchout this does not take account of leaves being on
        #higher levels, but we assume the tree is balanced enough that this is not a problem
        srcindex = values(trialT, Sleaf)
        obsindex = values(testT, NO)
        c_s = center(trialT, Sleaf)
        c_o = center(testT, NO)
        a_s = halfsize(trialT, Sleaf)
        a_o = halfsize(testT, NO)
        PermQ[Sleaf] = srcindex
        n_otilde = estimate_rank_3d(k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3)
        q_ks, k_l, r_l = Compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

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
        if source_is_frozen && obs_is_frozen
            break
        else
            R[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
        end
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
                            kernelmatrix, srcindex, obsindex, n_otilde, τ
                        )
                        last = 0
                        for Schild in children(trialT, Svert)
                            ks = length(getsubdict!(K, Schild)[Overt])
                            getsubdict!(R[l], (Ochild, Svert))[(Overt, Schild)] = q_ks[
                                :, (last + 1):(last + ks)
                            ]
                            last += ks
                        end
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
                            kernelmatrix, srcindex, obsindex, n_otilde, τ
                        )
                        last = 0
                        for Schild in children(trialT, Svert)
                            ks = length(getsubdict!(K, Schild)[Overt])
                            getsubdict!(R[l], (Ochild, Svert))[(Overt, Schild)] = q_ks[
                                :, (last + 1):(last + ks)
                            ]
                            last += ks
                        end
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
                        kernelmatrix, srcindex, obsindex, n_otilde, τ
                    )

                    last = 0
                    for Schild in children(trialT, Svert)
                        ks = length(getsubdict!(K, Schild)[Overt])
                        getsubdict!(R[l], (Overt, Svert))[(Overt, Schild)] = q_ks[
                            :, (last + 1):(last + ks)
                        ]
                        last += ks
                    end
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
        kernelmatrix(Z, row, col)
        PermP[Oleaf] = row
        P[Oleaf] = Z
    end
    return BF(
        Q,
        R,
        P,
        PermQ,
        PermP,
        (length(values(testT, NO)), length(values(trialT, NS))),
        NS,
        NO,
        k,
        τ,
    )
end

function subroutine_BF_mats(
    kernelmatrix,
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

    treeS = traverseandpad(trialT, NS)
    treeO = traverseandpad(testT, NO)

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
        q_ks, k_l, r_l = Compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)
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
                            kernelmatrix, srcindex, obsindex, n_otilde, τ
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
                            kernelmatrix, srcindex, obsindex, n_otilde, τ
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
                        kernelmatrix, srcindex, obsindex, n_otilde, τ
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
        kernelmatrix(Z, row, col)
        P = sparse_blockdiag(P, Z)              #SPARSITY: sparse_ or blocksparse_
    end
    return BF_Mats(Q, R, P, NS, NO, τ, k, PermP, PermQ)
end
