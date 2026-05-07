@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::BF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    result = apply_BF(Butterfly, x)
    copyto!(y, result)
    return nothing
end

function apply_BF(Butterfly::BF, v::Vector{ComplexF64})
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NO = Butterfly.NO
    NS = Butterfly.NS
    PermQ = Butterfly.PermQ
    PermP = Butterfly.PermP
    coefficients = Dict{Int,Dict{Tuple{Int,Int},Vector{ComplexF64}}}()
    H2Blocktree = Butterfly.tree
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values

    # ------------------------------------------------------------
    # Leaf initialization
    # ------------------------------------------------------------
    for Sleaf in keys(Q)
        srcvals = PermQ[Sleaf]  # Get the permuted source indices for this leaf
        getsubdict!(coefficients, 0)[NO, Sleaf] = Vector{ComplexF64}(
            undef, size(Q[Sleaf])[1]
        )
        @views mul!(coefficients[0][NO, Sleaf], Q[Sleaf], v[srcvals])
    end

    # Step 2: Sequentially apply R factors
    for l in eachindex(R)
        for row in keys(R[l])
            first = true
            for col in keys(R[l][row])
                if first
                    getsubdict!(coefficients, l)[row] = Vector{ComplexF64}(
                        undef, size(R[l][row][col])[1]
                    )
                    @views mul!(
                        coefficients[l][row], R[l][row][col], coefficients[l - 1][col]
                    )
                    first = false
                else
                    coeff_temp = Vector{ComplexF64}(undef, size(R[l][row][col])[1])
                    @views mul!(coeff_temp, R[l][row][col], coefficients[l - 1][col])
                    coefficients[l][row] += coeff_temp
                end
            end
        end
    end

    # Step 3: Apply P to the result from the last R factor
    # ------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------
    rootvals = values(testT, H2Trees.root(testT))
    result = zeros(ComplexF64, length(rootvals))
    for Oleaf in keys(P)
        inds = PermP[Oleaf]  # Get the permuted observer indices for this leaf
        dest = @view result[inds]
        mul!(dest, P[Oleaf], coefficients[length(R)][(Oleaf, NS)])
    end
    return result
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::BF_Mats, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    result = applyBF_Mats(Butterfly, x)
    copyto!(y, result)
    return nothing
end

function applyBF_Mats(t::BF_Mats, v::Vector{ComplexF64})
    y = v[t.PermQ]  #permute input vector according to Q blocks
    y = t.Q * y
    for R_block in t.R
        y = R_block * y
    end
    y = t.P * y
    y_out = zeros(ComplexF64, length(v))
    y_out[t.PermP] = y  #permute output vector according to P blocks
    return y_out
end

function applyBF_Mats_adjoint(t::BF_Mats, v::Vector{ComplexF64})
    # Gather input using the observer permutation
    y = v[t.PermP]

    # Adjoint of P
    y = t.P' * y

    # Adjoint of R blocks in reverse order
    for R_block in Iterators.reverse(t.R)
        y = R_block' * y
    end

    # Adjoint of Q
    y = t.Q' * y

    # Scatter output to the correct geometric source coordinates
    y_out = zeros(ComplexF64, length(v))
    y_out[t.PermQ] = y

    return y_out
end
