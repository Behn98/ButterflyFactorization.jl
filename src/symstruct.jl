struct BF2
    Q::Dict{Int,AbstractMatrix{ComplexF64}}
    R::Dict{Int,Dict{Int,Dict{Int,AbstractMatrix{ComplexF64}}}}
    P::Dict{Int,AbstractMatrix{ComplexF64}}
    tree::H2Trees.BlockTree
    dim::Tuple{Int,Int}
    level::Int
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    BF2(Q, R, P, tree, dim, level, NS, NO, k, τ) =
        new(Q, R, P, tree, dim, level, NS, NO, k, τ)
end

function subroutine_BF(
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

    R = Dict{Int,Dict{Int,Dict{Int,Matrix{ComplexF64}}}}()
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
    L = max(LS, LO)

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
        if source_is_frozen && obs_is_frozen
            break
        else
            R[l] = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
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
                            farassembler, srcindex, obsindex, n_otilde, τ
                        )
                        last = 0
                        for Schild in children(trialT, Svert)
                            ks = length(getsubdict!(K, Schild)[Overt])
                            getsubdict!(R[l], Schild)[Ochild] = q_ks[
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
                            farassembler, srcindex, obsindex, n_otilde, τ
                        )
                        last = 0
                        for Schild in children(trialT, Svert)
                            ks = length(getsubdict!(K, Schild)[Overt])
                            getsubdict!(R[l], Schild)[Ochild] = q_ks[
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
                        farassembler, srcindex, obsindex, n_otilde, τ
                    )

                    last = 0
                    for Schild in children(trialT, Svert)
                        ks = length(getsubdict!(K, Schild)[Overt])
                        getsubdict!(R[l], Schild)[Overt] = q_ks[:, (last + 1):(last + ks)]
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
        farassembler(Z, row, col)

        P[Oleaf] = Z
    end
    return BF2(
        Q,
        R,
        P,
        H2Blocktree,
        (length(values(testT, NO)), length(values(trialT, NS))),
        L,
        NS,
        NO,
        k,
        τ,
    )
end
