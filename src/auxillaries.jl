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

function Base.adjoint(B::BF)
    R_adj = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()

    for nodeS in keys(B.R)
        for nodeO in keys(B.R[nodeS])
            if !haskey(R_adj, nodeO)
                R_adj[nodeO] = Dict{Int,Matrix{ComplexF64}}()
            end
            R_adj[nodeO][nodeS] = adjoint(B.R[nodeS][nodeO])
        end
    end

    Q_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.Q)
        Q_adj[k] = adjoint(B.Q[k])
    end

    P_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.P)
        P_adj[k] = adjoint(B.P[k])
    end

    return BF(P_adj, R_adj, Q_adj, B.NO, B.NS, B.k, B.τ, B.sym)
end

function Base.transpose(B::BF)
    R_tr = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()

    for nodeS in keys(B.R)
        for nodeO in keys(B.R[nodeS])
            if !haskey(R_adj, nodeO)
                R_tr[nodeO] = Dict{Int,Matrix{ComplexF64}}()
            end
            R_tr[nodeO][nodeS] = transpose(B.R[nodeS][nodeO])
        end
    end

    Q_tr = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.Q)
        Q_tr[k] = transpose(B.Q[k])
    end

    P_tr = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.P)
        P_tr[k] = transpose(B.P[k])
    end

    return BF(P_tr, R_tr, Q_tr, B.NO, B.NS, B.k, B.τ, B.sym)
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

permute(space, perm) = permute!(copy(space), perm)

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
#=
function AdaptiveCrossApproximation.permutation(tree::H2Trees.H2ClusterTree)
    perm = zeros(Int, H2Trees.numberofvalues(tree))
    n = 1
    for leaf in H2Trees.leaves(tree)
        perm[n:(n + length(H2Trees.values(tree, leaf)) - 1)] = H2Trees.values(tree, leaf)
        tree.nodes[leaf].data.values .= n:(n + length(H2Trees.values(tree, leaf)) - 1)
        n += length(H2Trees.values(tree, leaf))
    end
    return perm
end
=#
