function (Butterfly::BF)(v::Vector{ComplexF64}, H2Blocktree)
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
                # ---- downward projection (observer side) ----
                for Overt in treeO[l]
                    for Ochild in children(testT, Overt)
                        coeff_S[Ochild] = Vector{ComplexF64}(undef, 0)
                        first = true
                        for Schild in children(trialT, Svert)
                            coeff_temp = Vector{ComplexF64}(
                                undef, size(R[Schild][Ochild])[1]
                            )
                            @views mul!(
                                coeff_temp, R[Schild][Ochild], coefficients[Schild][Overt]
                            )

                            if first
                                coeff_S[Ochild] = coeff_temp
                                first = false
                            else
                                coeff_S[Ochild] += coeff_temp
                            end
                        end
                    end
                end
            end

        elseif source_is_frozen && !obs_is_frozen
            for Svert in treeS[1]
                coeff_S = getsubdict!(coefficients, Svert)

                for Overt in treeO[l]
                    for Ochild in children(Overt, testT)
                        coeff_S[Ochild] = Vector{ComplexF64}(undef, 0)
                        first = true
                        for Schild in children(trialT, Svert)
                            coeff_temp = Vector{ComplexF64}(
                                undef, size(R[Schild][Ochild])[1]
                            )
                            @views mul!(
                                coeff_temp, R[Schild][Ochild], coefficients[Schild][Overt]
                            )

                            if first
                                coeff_S[Ochild] = coeff_temp
                                first = false
                            else
                                coeff_S[Ochild] += coeff_temp
                            end
                        end
                    end
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Svert in treeS[LS - l]
                coeff_S = getsubdict!(coefficients, Svert)

                # ---- frozen observer projection ----
                for Overt in treeO[LO]
                    coeff_S[Overt] = Vector{ComplexF64}(undef, 0)
                    first = true
                    for Schild in children(trialT, Svert)
                        coeff_temp = Vector{ComplexF64}(undef, size(R[Schild][Overt])[1])
                        @views mul!(
                            coeff_temp, R[Schild][Overt], coefficients[Schild][Overt]
                        )

                        if first
                            coeff_S[Overt] = coeff_temp
                            first = false
                        else
                            coeff_S[Overt] += coeff_temp
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
