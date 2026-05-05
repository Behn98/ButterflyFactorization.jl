@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::BF3, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    result = apply_BF3(Butterfly, x)
    copyto!(y, result)
    return nothing
end

@inline function getsubdict!(D::Dict{Int,Dict{Tuple{Int,Int},T}}, k::Int) where {T}
    get!(D, k) do
        Dict{Tuple{Int,Int},T}()
    end
end

function apply_BF3(Butterfly::BF3, v::Vector{ComplexF64})
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NO = Butterfly.NO
    NS = Butterfly.NS
    coefficients = Dict{Int,Dict{Tuple{Int,Int},Vector{ComplexF64}}}()
    H2Blocktree = Butterfly.tree
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values

    # ------------------------------------------------------------
    # Leaf initialization
    # ------------------------------------------------------------
    for Sleaf in keys(Q)
        srcvals = values(trialT, Sleaf)
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
        inds = values(testT, Oleaf)
        dest = @view result[inds]
        mul!(dest, P[Oleaf], coefficients[length(R)][(Oleaf, NS)])
    end
    return result
end
