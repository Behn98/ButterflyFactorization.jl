#=
if !source_is_frozen && !obs_is_frozen
            for Overt in treeO[LO - l]
                for Ochild in children(testT, Overt)
                    #R_temp = Vector{Matrix{ComplexF64}}()
                    for Svert in treeS[l]
                        coeff_S = getsubdict!(coefficients, Svert)
                        #R_temp = push!(R_temp, transpose(R[Svert][Ochild]))
                        coeff_S[Overt] = adjoint(R[Svert][Ochild]) * coeff_S[Ochild]
                        offset = 0
                        for Schild in children(trialT, Svert)
                            if l < LS - 1 && l < LO - 1
                                #@show coefficients[Svert][Overt]
                                #@show R[Schild][Overt]
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
            end

=#
