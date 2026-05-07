"""
The auxillaries file contains helper functions that are used across multiple parts in the
ButterflyFactorization package. The blockdiag function Constructs a block diagonal matrix
from the given matrices. Each input matrix is placed on the diagonal, and the off-diagonal
blocks are filled with zeros. The resulting matrix has dimensions equal to the sum of the
dimensions of the input matrices.
"""

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

"""
Constructs a block diagonal sparse matrix from the given matrices. Each input matrix is
placed on the diagonal, and the off-diagonal blocks are filled with zeros. The resulting
sparse matrix has dimensions equal to the sum of the dimensions of the input matrices.
"""

function sparse_blockdiag(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Convert blocks to sparse to utilize SparseArrays.blockdiag
    sparse_blocks = map(sparse, blocks)
    return SparseArrays.blockdiag(sparse_blocks...)
end

"""
Vertically concatenates the given matrices. Each input matrix is stacked on top of the next,
resulting in a matrix with the same number of columns as the input matrices and a number of
rows equal to the sum of the rows of the input matrices.
"""

function sparse_vcat(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Convert blocks to sparse and vertically concatenate
    sparse_blocks = map(sparse, blocks)
    return vcat(sparse_blocks...)
end

"""
This Function constructs a block diagonal matrix from the given blocks, which can be either
regular matrices or BlockSparseMatrix instances. It handles the combination of row and
column indices appropriately to maintain the structure of the resulting BlockSparseMatrix.
"""

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

"""
This function vertically concatenates the given blocks, which can be either regular matrices
or BlockSparseMatrix instances. It ensures that the resulting BlockSparseMatrix maintains
the correct structure by appropriately combining the row indices while keeping the column
indices consistent across the blocks.
"""

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

"""
This function retrieves a sub-dictionary from a nested dictionary structure. If the
specified key does not exist in the outer dictionary, it initializes a new inner dictionary
at that key and returns it. This is useful for building up nested dictionaries without
having to check for the existence of keys at each level.
"""
@inline function getsubdict!(D::Dict{Int,Dict{Int,T}}, k::Int) where {T}
    get!(D, k) do
        Dict{Int,T}()
    end
end

"""
The same as the previous function but for a different type of nested dictionary structure
wherethe keys are tuples of integers. This allows for more complex indexing schemes in the
nested dictionaries
"""

@inline function getsubdict!(
    D::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},T}}, k::Tuple{Int,Int}
) where {T}
    get!(D, k) do
        Dict{Tuple{Int,Int},T}()
    end
end

"""
This simply finds all the rows in a nested dictionary structure where the inner dictionary
contains a specific column index. It iterates through the outer dictionary, checks if the
inner dictionary has the specified column index as a key, and collects the corresponding row
keys into a vector. This is useful for operations that need to identify which rows are
associated with a given column in the context of the ButterflyFactorization's R factors.
"""

function find_rows_for_column(R::Dict{T,Dict{T,U}}, col_idx::T) where {T,U}
    rows = Vector{T}()
    for (row, inner_dict) in R
        if haskey(inner_dict, col_idx)
            push!(rows, row)
        end
    end
    return rows
end

"""
This function computes the levels of a hierarchical tree structure (H2Trees.TwoNTree)
starting from a specified root node. It traverses the tree level by level, collecting the
nodes at each level into a vector of vectors. The function uses the isleaf and children
functions from the H2Trees package to determine if a node is a leaf and to retrieve its
children, respectively.
"""

function h2treelevels(tree::T, root::Int64) where {T}
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

"""
The Butterfly logic enforces, that the all physical DoFs are related alone to the Q and P
factors, and that the R factors only contain the "artificial" DoFs that are introduced by
the hierarchical structure of the trees. This function traverses the tree levels and creates
ghost nodes for any leaves that are present at a level above leaf level. This ensures that
the R factors only contain the artificial DoFs and that the physical DoFs are correctly
associated with the Q and P factors. Also it ensures that skeletons are computed first with
respect to the observer root before travelling downward,while also ensuring that they reach
the source root just at the same time as the leaf level of the observer tree. This is
important for the correct construction of the ButterflyFactorization, as it maintains the
necessary structure and relationships between the test and trial spaces as defined by the
Trees.
"""

function traverseandpad(H2tree::T, root::Int64) where {T}
    isleaf = H2Trees.isleaf
    tree = h2treelevels(H2tree, root)
    for l in 2:(length(tree) - 1)
        for node in tree[l]
            if isleaf(H2tree, node)
                push!(tree[l + 1], node)
            end
        end
    end
    return tree
end

"""
Here we define two styles for ordering the spaces in the H2Trees. The PermuteSpaceInPlace
style permutes the test and trial spaces in place according to the permutation derived from
the tree structure, while the PreserveSpaceOrder style leaves the spaces unchanged. The
choice of style can affect the performance of the ButterflyFactorization, as certain
orderings may lead to more efficient computations. The functions associated with each style
take the tree and the test and trial spaces as input and perform the necessary permutations
or leave them as is.
"""

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

function permutation(tree::H2Trees.H2ClusterTree)
    perm = zeros(Int, H2Trees.numberofvalues(tree))
    n = 1
    for leaf in H2Trees.leaves(tree)
        perm[n:(n + length(H2Trees.values(tree, leaf)) - 1)] = H2Trees.values(tree, leaf)
        tree.nodes[leaf].data.values .= n:(n + length(H2Trees.values(tree, leaf)) - 1)
        n += length(H2Trees.values(tree, leaf))
    end
    return perm
end
