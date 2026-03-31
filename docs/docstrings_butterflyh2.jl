"""
    estimate_rank_3d(k, c_s::SVector, c_o::SVector, a_s::Float64, a_o::Float64, ε::Float64; C=1.0, Cε=3.0, Rmin=5)

Estimate the rank required for low-rank approximation in 3D space based on geometric and tolerance parameters.

# Arguments

  - `k`: Wave number or frequency parameter
  - `c_s::SVector`: Center of source box
  - `c_o::SVector`: Center of observation box
  - `a_s::Float64`: Half-size of source box
  - `a_o::Float64`: Half-size of observation box
  - `ε::Float64`: Tolerance for accuracy
  - `C::Float64`: Geometric constant (default: 1.0)
  - `Cε::Float64`: Tolerance scaling constant (default: 3.0)
  - `Rmin::Int`: Minimum rank (default: 5)

# Returns

  - `Int`: Estimated rank
"""
function estimate_rank_3d(
    k,
    c_s::SVector,
    c_o::SVector,
    a_s::Float64,
    a_o::Float64,
    ε::Float64;
    C=1.0,
    Cε=3.0,
    Rmin=5,
)
    # implementation...
end

"""
    subroutine_getskelh2(farassembler, src_index::Vector{Int}, obs_index::Vector{Int}, c_s::SVector, c_o::SVector, a_s::Float64, a_o::Float64, ε::Float64, k_est::Float64)

Compute skeleton indices and coefficients for H2 matrix approximation using pivoted QR factorization.

# Arguments

  - `farassembler`: Function to assemble matrix blocks
  - `src_index::Vector{Int}`: Source node indices
  - `obs_index::Vector{Int}`: Observation node indices
  - `c_s::SVector`: Source box center
  - `c_o::SVector`: Observation box center
  - `a_s::Float64`: Source box half-size
  - `a_o::Float64`: Observation box half-size
  - `ε::Float64`: QR tolerance
  - `k_est::Float64`: Wave number estimate

# Returns

  - `Tuple`: (q_ks, k, r) - coefficients, skeleton indices, and rank
"""
function subroutine_getskelh2(
    farassembler,
    src_index::Vector{Int},
    obs_index::Vector{Int},
    c_s::SVector,
    c_o::SVector,
    a_s::Float64,
    a_o::Float64,
    ε::Float64,
    k_est::Float64,
)
    # implementation...
end

"""
    subroutine_BF_approx_treeh2(farassembler, H2Blocktree, NO::Int, NS::Int, k::Float64, τ::Float64)

Construct butterfly factorization by traversing H2 block tree and computing Q, R, K, and P blocks.

# Arguments

  - `farassembler`: Matrix assembly function
  - `H2Blocktree`: H2 block tree structure
  - `NO::Int`: Observation node id
  - `NS::Int`: Source node id
  - `k::Float64`: Wave number
  - `τ::Float64`: Tolerance

# Returns

  - `BF`: Butterfly factorization structure
"""
function subroutine_BF_approx_treeh2(
    farassembler, H2Blocktree, NO::Int, NS::Int, k::Float64, τ::Float64
)
    # implementation...
end

"""
    apply_butterflyh2(H2Blocktree, Butterfly::BF, v::Vector{ComplexF64})

Apply butterfly factorization to vector v via hierarchical traversal.

# Arguments

  - `H2Blocktree`: H2 block tree
  - `Butterfly::BF`: Butterfly factorization
  - `v::Vector{ComplexF64}`: Input vector

# Returns

  - `Vector{ComplexF64}`: Result vector
"""
function apply_butterflyh2(H2Blocktree, Butterfly::BF, v::Vector{ComplexF64})
    # implementation...
end

"""
    assemble_Bfly(farassembler, H2Blocktree, k, τ, α)

Assemble complete butterfly approximation with near and far-field interactions.

# Arguments

  - `farassembler`: Matrix assembly function
  - `H2Blocktree`: H2 block tree structure
  - `k`: Wave number
  - `τ`: Tolerance
  - `α`: Admissibility parameter

# Returns

  - `BFapprox`: Complete butterfly approximation structure
"""
function assemble_Bfly(farassembler, H2Blocktree, k, τ, α)
    # implementation...
end

"""
    apply_Butterfly(Butter::BFapprox, v::AbstractVector{ComplexF64})

Apply complete butterfly approximation to vector, combining near and far-field contributions.

# Arguments

  - `Butter::BFapprox`: Butterfly approximation
  - `v::AbstractVector{ComplexF64}`: Input vector

# Returns

  - `Vector{ComplexF64}`: Result vector
"""
function apply_Butterfly(Butter::BFapprox, v::AbstractVector{ComplexF64})
    # implementation...
end

"""
    struct PetrovGalerkinBF{T,NearInteractionsType} <: LinearMaps.LinearMap{T}

A Petrov-Galerkin Butterfly Factorization representation of a linear operator.

# Fields

  - `nearinteractions::NearInteractionsType`: Block sparse matrix for near-field interactions
  - `dim::Tuple{Int,Int}`: Dimensions (rows, cols) of the operator
  - `tree::H2Trees.BlockTree`: H2 block tree structure partitioning the domain
  - `farinteractions::Dict{Int64,Vector{Int64}}`: Far-field interactions mapping observer nodes to source nodes
  - `BFs::Vector{BF}`: Butterfly factorization blocks for far-field interactions
"""
struct PetrovGalerkinBF{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    # ...existing code...
end

"""
    PetrovGalerkinBF(operator, testspace, trialspace, tree::BlockTree, k::Float64;
                     tol=1e-3, ntasks=Threads.nthreads(), α=2)

Construct a Petrov-Galerkin Butterfly Factorization from an operator and function spaces.

# Arguments

  - `operator`: Kernel operator to approximate
  - `testspace`: Test function space (observer side)
  - `trialspace`: Trial function space (source side)
  - `tree::BlockTree`: H2 block tree for hierarchical partitioning
  - `k::Float64`: Wave number for rank estimation
  - `tol::Float64`: Tolerance for low-rank approximation (default: 1e-3)
  - `ntasks::Int`: Number of threads for assembly (default: nthreads())
  - `α::Float64`: Separation parameter for near/far field (default: 2)

# Returns

`PetrovGalerkinBF` object representing the factorized operator
"""
function PetrovGalerkinBF(
    operator,
    testspace,
    trialspace,
    tree::BlockTree,
    k::Float64;
    tol=1e-3,
    ntasks=Threads.nthreads(),
    α=2,
)
    # ...existing code...
end

"""
    Base.size(A::PetrovGalerkinBF, dim=nothing) -> Union{Tuple{Int,Int}, Int}

Return the dimensions of the Petrov-Galerkin butterfly operator.

# Arguments

  - `A::PetrovGalerkinBF`: Butterfly operator
  - `dim::Union{Nothing, Int}`: Dimension to query (1 or 2), or nothing for both

# Returns

  - If `dim=nothing`: `Tuple{Int,Int}` of (rows, cols)
  - If `dim=1` or `dim=2`: `Int` size of that dimension
"""
function Base.size(A::ButterflyFactorization.PetrovGalerkinBF, dim=nothing)
    # ...existing code...
end

"""
    LinearAlgebra.mul!(y::AbstractVecOrMat, A::PetrovGalerkinBF, x::AbstractVector) -> AbstractVecOrMat

Apply the Butterfly factorization to vector `x` and accumulate result in `y`.

Computes: `y += A * x` by summing near-field and far-field contributions.

# Arguments

  - `y::AbstractVecOrMat`: Output vector (modified in-place)
  - `A::PetrovGalerkinBF`: Butterfly factorization operator
  - `x::AbstractVector`: Input vector

# Returns

Modified vector `y`
"""
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::ButterflyFactorization.PetrovGalerkinBF, x::AbstractVector{T}
) where {T}
    # ...existing code...
end

"""
    struct isFarFunctor

Functor for determining admissibility of far-field interactions based on geometric separation.

# Fields

  - `α::Float64`: Separation parameter controlling the admissibility criterion

# Description

Implements a geometric admissibility test for H2-matrices. Two clusters are admissible
(can be treated as far-field) if they are sufficiently separated relative to their sizes.
Uses both minimum distance and bounding sphere radius checks.
"""
struct isFarFunctor
    # ...existing code...
end

"""
    (t::isFarFunctor)(srctree, tsttree, snode, onode) -> Bool

Check if source and observer nodes are admissible for far-field treatment.

# Arguments

  - `srctree`: Trial tree (source side) from H2Trees
  - `tsttree`: Test tree (observer side) from H2Trees
  - `snode::Int`: Source node identifier
  - `onode::Int`: Observer node identifier

# Returns

`true` if nodes are admissible (far-field), `false` otherwise (near-field)

# Description

Two checks are performed:

 1. Separation check: distance between centers minus bounding sphere radii

 2. Bounding box check: minimum Euclidean distance between boxes

    # ...existing code...

Both must exceed `α * W` where W is the maximum halfsize of the two nodes.
"""
function (t::isFarFunctor)(srctree, tsttree, snode, onode)
    # ...existing code...
end

"""
    process_nodes!(srctree, tsttree, node_o, node_s, admissible, farinteractions, nearsv, nearov)

Recursively partition source and observer domains into near/far field interactions.

# Arguments

  - `srctree`: Trial tree (source side)
  - `tsttree`: Test tree (observer side)
  - `node_o::Int`: Current observer node
  - `node_s::Int`: Current source node
  - `admissible::isFarFunctor`: Admissibility functor
  - `farinteractions::Dict`: Maps observer node → source nodes (modified in-place)
  - `nearsv::Vector`: Near-field source node indices (modified in-place)
  - `nearov::Vector`: Near-field observer node indices (modified in-place)

# Description

Recursively traverses the H2-tree structure:

  - If nodes are admissible: adds to far-field interactions
  - If both nodes are leaves: adds to near-field interactions
  - Otherwise: refines by splitting the larger node and recursing
"""
function process_nodes!(
    srctree,
    tsttree,
    node_o,
    node_s,
    admissible::isFarFunctor,
    farinteractions,
    nearsv,
    nearov,
)
    # ...existing code...
end

"""
    nearandfar2(tree::H2Trees.BlockTree, α) -> (nearov, nearsv, farinteractions)

Partition H2-tree into near-field and far-field interactions (vector-based version).

# Arguments

  - `tree::H2Trees.BlockTree`: H2 block tree structure
  - `α::Float64`: Separation parameter for admissibility criterion

# Returns

Tuple of three elements:

  - `nearov::Vector{Vector{Int}}`: Observer node indices for near-field blocks
  - `nearsv::Vector{Vector{Int}}`: Source node indices for near-field blocks
  - `farinteractions::Dict{Int64,Vector{Int64}}`: Observer node → source nodes (far-field)

# Description

Performs hierarchical partitioning starting from tree roots. Returns near-field    # ...existing code...
interactions as parallel vectors and far-field as a dictionary.
"""
function nearandfar2(tree::H2Trees.BlockTree, α)
    # ...existing code...
end

"""
    nearandfar(tree::H2Trees.BlockTree, α) -> (nearinteractions, farinteractions)

Partition H2-tree into near-field and far-field interactions (dictionary-based version).

# Arguments

  - `tree::H2Trees.BlockTree`: H2 block tree structure
  - `α::Float64`: Separation parameter for admissibility criterion

# Returns

Tuple of two dictionaries:

  - `nearinteractions::Dict{Int64,Vector{Int64}}`: Observer node → source nodes (near-field)
  - `farinteractions::Dict{Int64,Vector{Int64}}`: Observer node → source nodes (far-field)

# Description

Performs hierarchical partitioning starting from tree roots. Both near-field and
far-field interactions are returned as dictionaries mapping observer nodes to vectors    # ...existing code...
of source node identifiers.
"""
function nearandfar(tree::H2Trees.BlockTree, α)
    # ...existing code...
end
