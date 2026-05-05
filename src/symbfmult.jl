@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::BF3, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    result = apply_BF3(Butterfly, x)
    copyto!(y, result)
    return nothing
end

function apply_BF3(Butterfly::BF3, v::Vector{ComplexF64})
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NO = Butterfly.NO
    NS = Butterfly.NS
    coefficients = Dict{Int,Dict{Int,Vector{ComplexF64}}}()
    H2Blocktree = Butterfly.tree
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values

    # ------------------------------------------------------------
    # Leaf initialization
    # ------------------------------------------------------------
    for Sleaf in keys(Q)
        srcvals = values(trialT, Sleaf)
        getsubdict!(coefficients, NO)[Sleaf] = Vector{ComplexF64}(undef, size(Q[Sleaf])[1])
        @views mul!(coefficients[NO][Sleaf], Q[Sleaf], v[srcvals])
    end

    # Step 2: Sequentially apply R factors
    for l in eachindex(R)
        for (obs_child, src_node) in keys(R[l])
            first = true
            for (obs_node, src_child) in keys(R[l][(obs_child, src_node)])
                if first
                    getsubdict!(coefficients, obs_child)[src_node] = Vector{ComplexF64}(
                        undef, size(R[l][(obs_child, src_node)][(obs_node, src_child)])[1]
                    )
                    @views mul!(
                        coefficients[obs_child][src_node],
                        R[l][(obs_child, src_node)][(obs_node, src_child)],
                        coefficients[obs_node][src_child],
                    )
                    first = false
                else
                    coeff_temp = Vector{ComplexF64}(
                        undef, size(R[l][(obs_child, src_node)][(obs_node, src_child)])[1]
                    )
                    @views mul!(
                        coeff_temp,
                        R[l][(obs_child, src_node)][(obs_node, src_child)],
                        coefficients[obs_node][src_child],
                    )
                    getsubdict!(coefficients, obs_child)[src_node] += coeff_temp
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
        inds = values(testT, Oleaf)
        dest = @view result[inds]
        mul!(dest, P[Oleaf], coefficients[Oleaf][NS])
    end
    return result
end
