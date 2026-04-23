function apply_butterflyh2(H2Blocktree, Butterfly::BF, v::Vector{ComplexF64})
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NO = Butterfly.NO
    NS = Butterfly.NS

    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO

    coefficients = Dict{Int,Dict{Int,Vector{ComplexF64}}}()

    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------
    # Leaf initialization
    # ------------------------------------------------------------
    for Sleaf in treeS[LS]
        srcvals = values(trialT, Sleaf)
        getsubdict!(coefficients, Sleaf)[NO] = Vector{ComplexF64}(undef, size(Q[Sleaf])[1])
        @views mul!(getsubdict!(coefficients, Sleaf)[NO], Q[Sleaf], v[srcvals])
    end

    # ------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------
    for l in 1:(L - 1)
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)

        if !source_is_frozen && !obs_is_frozen
            for Svert in treeS[LS - l]
                coeff_S = getsubdict!(coefficients, Svert)

                # ---- upward aggregation (source side) ----
                for Overt in treeO[l]
                    temp = ComplexF64[]

                    for Schild in children(trialT, Svert)
                        coeffs_child = coefficients[Schild][Overt]
                        append!(temp, coeffs_child)
                    end

                    coeff_S[Overt] = temp
                end

                # ---- downward projection (observer side) ----
                for Overt in treeO[l]
                    coeff_SO = coeff_S[Overt]

                    for Ochild in children(testT, Overt)
                        R_check = R[Svert][Ochild]
                        coeff_S[Ochild] = Vector{ComplexF64}(undef, size(R_check)[1])
                        @views mul!(coeff_S[Ochild], R_check, coeff_SO)
                    end
                end
            end

        elseif source_is_frozen && !obs_is_frozen
            for Svert in treeS[1]
                coeff_S = getsubdict!(coefficients, Svert)

                for Overt in treeO[LS]
                    coeff_SO = coeff_S[Overt]

                    for Ochild in h2treelevels(testT, Overt)[l - LS + 2]
                        R_check = R[Svert][Ochild]
                        coeff_S[Ochild] = Vector{ComplexF64}(undef, size(R_check)[1])
                        @views mul!(coeff_S[Ochild], R_check, coeff_SO)
                    end
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Svert in treeS[LS - l]
                coeff_S = getsubdict!(coefficients, Svert)

                # ---- upward aggregation ----
                for Overt in treeO[LO]
                    temp = ComplexF64[]

                    for Schild in children(trialT, Svert)
                        coeffs_child = coefficients[Schild][Overt]
                        append!(temp, coeffs_child)
                    end

                    coeff_S[Overt] = temp
                end

                # ---- frozen observer projection ----
                for Overt in treeO[LO]
                    R_check = R[Svert][Overt]
                    temp = coeff_S[Overt]
                    coeff_S[Overt] = Vector{ComplexF64}(undef, size(R_check)[1])
                    @views mul!(coeff_S[Overt], R_check, temp)
                end
            end

        else
            break
        end
    end

    # ------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------
    rootvals = values(testT, H2Trees.root(testT))
    result = zeros(ComplexF64, length(rootvals))
    if LS >= LO
        for Oleaf in treeO[LO]
            inds = values(testT, Oleaf)
            dest = @view result[inds]
            mul!(dest, P[Oleaf], coefficients[NS][Oleaf])
        end
    else
        for Oleaf in treeO[LO]
            inds = values(testT, Oleaf)
            result[inds] = coefficients[NS][Oleaf]
        end
    end

    return result
end

"""
    apply_butterflyh2_transpose(H2Blocktree, Butterfly::BF, v)

Compute `transpose(B)*v` for a dictionary-based butterfly `Butterfly`.
Assumes the same concatenation order as in `apply_butterflyh2`.
"""
function apply_butterflyh2_transpose(H2Blocktree, Butterfly::BF, v::Vector{ComplexF64})
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NO = Butterfly.NO
    NS = Butterfly.NS

    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO

    coefficients = Dict{Int,Dict{Int,Vector{ComplexF64}}}()

    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------
    # Leaf initialization
    # ------------------------------------------------------------
    for Oleaf in treeO[LO]
        obsvals = values(testT, Oleaf)
        getsubdict!(coefficients, NS)[Oleaf] = Vector{ComplexF64}(undef, size(P[Oleaf])[2])
        @views mul!(getsubdict!(coefficients, NS)[Oleaf], transpose(P[Oleaf]), v[obsvals])
    end

    # ------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------
    for l in 1:(L - 1)
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)

        if !source_is_frozen && !obs_is_frozen
            for Overt in treeO[LO - l]
                for Svert in treeS[l]
                    coeff_S = getsubdict!(coefficients, Svert)

                    # 1. Sum up all projections from Ochild FIRST
                    first_child = true
                    for Ochild in children(testT, Overt)
                        contrib = transpose(R[Svert][Ochild]) * coeff_S[Ochild]
                        if first_child
                            coeff_S[Overt] = contrib
                            first_child = false
                        else
                            coeff_S[Overt] .+= contrib
                        end
                    end

                    # 2. THEN split the fully assembled sum to Schild
                    offset = 0
                    for Schild in children(trialT, Svert)
                        if l < LS - 1 #&& l < LO - 1
                            getsubdict!(coefficients, Schild)[Overt] = coeff_S[Overt][(offset + 1):(offset + size(
                                R[Schild][Overt]
                            )[1])]
                            offset += size(R[Schild][Overt])[1]
                        else
                            getsubdict!(coefficients, Schild)[Overt] = coeff_S[Overt][(offset + 1):(offset + size(
                                Q[Schild]
                            )[1])]
                            offset += size(Q[Schild])[1]
                        end
                    end
                end
            end

        elseif source_is_frozen && !obs_is_frozen
            for Overt in treeO[LO - l]
                for Svert in treeS[1]
                    coeff_S = getsubdict!(coefficients, Svert)

                    # 1. Sum up all projections from Ochild FIRST
                    first_child = true
                    for Ochild in children(testT, Overt)
                        contrib = transpose(R[Svert][Ochild]) * coeff_S[Ochild]
                        if first_child
                            coeff_S[Overt] = contrib
                            first_child = false
                        else
                            coeff_S[Overt] .+= contrib
                        end
                    end
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Overt in treeO[LO]
                for Svert in treeS[l]
                    coeff_S = getsubdict!(coefficients, Svert)
                    contrib = transpose(R[Svert][Overt]) * coeff_S[Overt]
                    coeff_S[Overt] = contrib

                    # 2. THEN split the fully assembled sum to Schild
                    offset = 0
                    for Schild in children(trialT, Svert)
                        if l < LS - 1 #&& l < LO - 1
                            getsubdict!(coefficients, Schild)[Overt] = coeff_S[Overt][(offset + 1):(offset + size(
                                R[Schild][Overt]
                            )[1])]
                            offset += size(R[Schild][Overt])[1]
                        else
                            getsubdict!(coefficients, Schild)[Overt] = coeff_S[Overt][(offset + 1):(offset + size(
                                Q[Schild]
                            )[1])]
                            offset += size(Q[Schild])[1]
                        end
                    end
                end
            end
        else
            break
        end
    end

    # ------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------
    rootvals = values(trialT, H2Trees.root(trialT))
    result = zeros(ComplexF64, length(rootvals))
    for Sleaf in treeS[LS]
        inds = values(trialT, Sleaf)
        dest = @view result[inds]
        mul!(dest, transpose(Q[Sleaf]), coefficients[Sleaf][NO])
    end
    return result
end

"""
    apply_butterflyh2_adjoint(H2Blocktree, Butterfly::BF, v)

Compute `adjoint(B)*v` (conjugate transpose).
Same as transpose but uses adjoint on blocks.
"""
function apply_butterflyh2_adjoint(H2Blocktree, Butterfly::BF, v::Vector{ComplexF64})
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NO = Butterfly.NO
    NS = Butterfly.NS

    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)
    coefficients = Dict{Int,Dict{Int,Vector{ComplexF64}}}()

    source_is_frozen = true
    obs_is_frozen = true

    # ------------------------------------------------------------
    # Leaf initialization
    # ------------------------------------------------------------
    for Oleaf in treeO[LO]
        obsvals = values(testT, Oleaf)
        getsubdict!(coefficients, NS)[Oleaf] = Vector{ComplexF64}(undef, size(P[Oleaf])[2])
        @views mul!(getsubdict!(coefficients, NS)[Oleaf], adjoint(P[Oleaf]), v[obsvals])
    end
    frozenoffsetS = 0
    frozenoffsetO = 0
    if LS > LO
        frozenoffsetO = LS - LO
    elseif LO > LS
        frozenoffsetS = LO - LS
    end
    # ------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------
    for l in 1:(L - 1)
        l > (LO - LS) && (source_is_frozen = false)
        l > (LS - LO) && (obs_is_frozen = false)
        if !source_is_frozen && !obs_is_frozen
            for Overt in treeO[LO - l + frozenoffsetO]
                for Svert in treeS[l - frozenoffsetS]
                    coeff_S = getsubdict!(coefficients, Svert)

                    # 1. Sum up all projections from Ochild FIRST
                    first_child = true
                    for Ochild in children(testT, Overt)
                        contrib = adjoint(R[Svert][Ochild]) * coeff_S[Ochild]
                        if first_child
                            coeff_S[Overt] = contrib
                            first_child = false
                        else
                            coeff_S[Overt] .+= contrib
                        end
                    end

                    # 2. THEN split the fully assembled sum to Schild
                    offset = 0
                    for Schild in children(trialT, Svert)
                        if l - frozenoffsetS < LS - 1 #&& l - frozenoffsetO < LO - 1
                            getsubdict!(coefficients, Schild)[Overt] = coeff_S[Overt][(offset + 1):(offset + size(
                                R[Schild][Overt]
                            )[1])]
                            offset += size(R[Schild][Overt])[1]
                        else
                            getsubdict!(coefficients, Schild)[Overt] = coeff_S[Overt][(offset + 1):(offset + size(
                                Q[Schild]
                            )[1])]
                            offset += size(Q[Schild])[1]
                        end
                    end
                end
            end

        elseif source_is_frozen && !obs_is_frozen
            for Overt in treeO[LO - l]
                for Svert in treeS[1]
                    coeff_S = getsubdict!(coefficients, Svert)

                    # 1. Sum up all projections from Ochild FIRST
                    first_child = true
                    for Ochild in children(testT, Overt)
                        contrib = adjoint(R[Svert][Ochild]) * coeff_S[Ochild]
                        if first_child
                            coeff_S[Overt] = contrib
                            first_child = false
                        else
                            coeff_S[Overt] .+= contrib
                        end
                    end
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Overt in treeO[LO]
                for Svert in treeS[l]
                    coeff_S = getsubdict!(coefficients, Svert)
                    contrib = adjoint(R[Svert][Overt]) * coeff_S[Overt]
                    coeff_S[Overt] = contrib

                    # 2. THEN split the fully assembled sum to Schild
                    offset = 0
                    for Schild in children(trialT, Svert)
                        if l < LS - 1
                            getsubdict!(coefficients, Schild)[Overt] = coeff_S[Overt][(offset + 1):(offset + size(
                                R[Schild][Overt]
                            )[1])]
                            offset += size(R[Schild][Overt])[1]
                        else
                            getsubdict!(coefficients, Schild)[Overt] = coeff_S[Overt][(offset + 1):(offset + size(
                                Q[Schild]
                            )[1])]
                            offset += size(Q[Schild])[1]
                        end
                    end
                end
            end
        else
            break
        end
    end

    # ------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------
    rootvals = values(trialT, H2Trees.root(trialT))
    result = zeros(ComplexF64, length(rootvals))
    for Sleaf in treeS[LS]
        inds = values(trialT, Sleaf)
        dest = @view result[inds]
        mul!(dest, adjoint(Q[Sleaf]), coefficients[Sleaf][NO])
    end
    return result
end
