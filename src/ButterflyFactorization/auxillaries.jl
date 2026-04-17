function blockdiag(blocks::AbstractMatrix...)
    isempty(blocks) && return zeros(0, 0)

    T = promote_type(map(eltype, blocks)...)

    rows = sum(size(b, 1) for b in blocks)
    cols = sum(size(b, 2) for b in blocks)

    M = zeros(T, rows, cols)

    r = 1
    c = 1
    for B in blocks
        nr, nc = size(B)
        M[r:(r + nr - 1), c:(c + nc - 1)] .= B
        r += nr
        c += nc
    end

    return M
end

function sparse_blockdiag(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Convert blocks to sparse to utilize SparseArrays.blockdiag
    sparse_blocks = map(sparse, blocks)
    return SparseArrays.blockdiag(sparse_blocks...)
end

function sparse_vcat(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Convert blocks to sparse and vertically concatenate
    sparse_blocks = map(sparse, blocks)
    return vcat(sparse_blocks...)
end

function blocksparse_blockdiag(blocks...)
    isempty(blocks) && return BlockSparseMatrix(
        Matrix{ComplexF64}[], UnitRange{Int}[], UnitRange{Int}[], (0, 0)
    )

    # Helper to get indices whether it's a BlockSparseMatrix or a regular Matrix
    get_rowidx(b) = hasproperty(b, :rowindices) ? b.rowindices : [1:size(b, 1)]
    get_colidx(b) = hasproperty(b, :colindices) ? b.colindices : [1:size(b, 2)]
    get_blocks(b) = hasproperty(b, :blocks) ? b.blocks : [b]

    if length(blocks) == 1
        b = blocks[1]
        return BlockSparseMatrix(get_blocks(b), get_rowidx(b), get_colidx(b), size(b))
    elseif length(blocks) > 2
        return blocksparse_blockdiag(
            blocksparse_blockdiag(blocks[1], blocks[2]), blocks[3:end]...
        )
    end

    s1 = size(blocks[1])
    s2 = size(blocks[2])

    rowindices = vcat(get_rowidx(blocks[1]), [vs .+ s1[1] for vs in get_rowidx(blocks[2])])
    colindices = vcat(get_colidx(blocks[1]), [vs .+ s1[2] for vs in get_colidx(blocks[2])])

    combined_blocks = vcat(get_blocks(blocks[1]), get_blocks(blocks[2]))

    return BlockSparseMatrix(
        combined_blocks, rowindices, colindices, (s1[1] + s2[1], s1[2] + s2[2])
    )
end

function blocksparse_vcat(blocks...)
    isempty(blocks) && return BlockSparseMatrix(
        Matrix{ComplexF64}[], UnitRange{Int}[], UnitRange{Int}[], (0, 0)
    )

    get_rowidx(b) = hasproperty(b, :rowindices) ? b.rowindices : [1:size(b, 1)]
    get_colidx(b) = hasproperty(b, :colindices) ? b.colindices : [1:size(b, 2)]
    get_blocks(b) = hasproperty(b, :blocks) ? b.blocks : [b]

    if length(blocks) == 1
        b = blocks[1]
        return BlockSparseMatrix(get_blocks(b), get_rowidx(b), get_colidx(b), size(b))
    elseif length(blocks) > 2
        return blocksparse_vcat(blocksparse_vcat(blocks[1], blocks[2]), blocks[3:end]...)
    end

    s1 = size(blocks[1])
    s2 = size(blocks[2])
    @assert s1[2] == s2[2] "All blocks must have the same number of columns for vertical concatenation."

    rowindices = vcat(get_rowidx(blocks[1]), [vs .+ s1[1] for vs in get_rowidx(blocks[2])])
    colindices = get_colidx(blocks[1])

    combined_blocks = vcat(get_blocks(blocks[1]), get_blocks(blocks[2]))

    return BlockSparseMatrix(
        combined_blocks, rowindices, colindices, (s1[1] + s2[1], s1[2])
    )
end

function Base.adjoint(t::BF_Mats)
    return BF_Mats(
        t.P',                                                      # Q becomes P'
        AbstractMatrix{ComplexF64}[r' for r in Iterators.reverse(t.R)], # Reverse and map R
        t.Q',                                                      # P becomes Q'
        t.NO,                                                      # NS and NO swap roles
        t.NS,
        t.k,
        t.τ,
        t.PermQ,                                                   # Permutations swap roles
        t.PermP,
    )
end

function Base.transpose(t::BF_Mats)
    return BF_Mats(
        transpose(t.P),
        AbstractMatrix{ComplexF64}[transpose(r) for r in Iterators.reverse(t.R)],
        transpose(t.Q),
        t.NO,
        t.NS,
        t.k,
        t.τ,
        t.PermQ,
        t.PermP,
    )
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

function h2treelevels(tree::H2Trees.TwoNTree, root::Int64)
    isleaf = H2Trees.isleaf
    getchildren = H2Trees.children

    levels = Vector{Vector{Int}}()
    current = [root]

    while !isempty(current)
        push!(levels, current)
        next = Int[]

        for node in current
            if !isleaf(tree, node)
                append!(next, getchildren(tree, node))
            end
        end

        current = next
    end

    return levels
end

abstract type SpaceOrderingStyle end
struct PermuteSpaceInPlace <: SpaceOrderingStyle end
function (::PermuteSpaceInPlace)(tree, testspace, trialspace)
    testperm = permutation(testtree(tree))
    permute!(testspace, testperm)

    if testspace === trialspace && testtree(tree) === trialtree(tree)
        return nothing
    elseif !(testspace === trialspace) && !(testtree(tree) === trialtree(tree))
        trialperm = permutation(trialtree(tree))
        permute!(trialspace, trialperm)
        return nothing
    else
        @warn "Risky territory: Permuting trialtree not trialspace."
        trialperm = permutation(trialtree(tree))
        return nothing
    end
end
struct PreserveSpaceOrder <: SpaceOrderingStyle end
function (::PreserveSpaceOrder)(tree, testspace, trialspace)
    return nothing
end
