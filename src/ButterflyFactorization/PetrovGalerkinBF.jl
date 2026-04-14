struct PetrovGalerkinBF_mats{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    # Here come all other fields needed for the ButterflyFactorization
    dim::Tuple{Int,Int}
    #tree::H2Trees.BlockTree                            #not necessary for matrix version but for the dicts version
    farinteractions::Dict{Int64,Vector{Int64}}           #observernodeid --> sourcenodeid
    BFs::Vector{BF_Mats}                    #BF --> dictionary version BF_Mats matrix version
    function PetrovGalerkinBF_mats{T}(
        nearinteractions,
        #tree,
        farinteractions,
        BFs,
        dim,
    ) where {T}
        return new{T,typeof(nearinteractions)}(
            nearinteractions,
            dim,
            # Here come all other fields needed for the ButterflyFactorization
            #tree,#::H2Trees.BlockTree
            farinteractions,#::Dict{Int64,Vector{Int64}}           #observernodeid --> sourcenodeid
            BFs,#::Vector{BF}
        )
    end
end

struct PetrovGalerkinBF{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    # Here come all other fields needed for the ButterflyFactorization
    dim::Tuple{Int,Int}
    tree::H2Trees.BlockTree                            #not necessary for matrix version but for the dicts version
    farinteractions::Dict{Int64,Vector{Int64}}           #observernodeid --> sourcenodeid
    BFs::Vector{BF}                    #BF --> dictionary version BF_Mats matrix version
    function PetrovGalerkinBF{T}(
        nearinteractions, tree, farinteractions, BFs, dim
    ) where {T}
        return new{T,typeof(nearinteractions)}(
            nearinteractions,
            dim,
            # Here come all other fields needed for the ButterflyFactorization
            tree,#::H2Trees.BlockTree
            farinteractions,#::Dict{Int64,Vector{Int64}}           #observernodeid --> sourcenodeid
            BFs,#::Vector{BF}
        )
    end
end

function PetrovGalerkinBF_mats(
    operator,
    testspace,
    trialspace,
    tree::BlockTree,
    k::Float64,
    Compressor::Abstractcompressor;
    tol=1e-3,
    ntasks=Threads.nthreads(),
    α=2,
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)   #quadstrat=nearquadstrat
    #values, nearvalues = nearinteractions(tree; isnear=isnear)
    values, nearvalues, farints = nearandfar2(tree, α)

    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @show ntasks length(values)
    @tasks for i in eachindex(values)
        @set ntasks = 1#ntasks
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix)) #size(nearmatrix)
    fly = Vector{BF_Mats}()
    for (NO, source_nodes) in farints
        for NS in source_nodes
            push!(
                fly,
                subroutine_BF_approx_treeh2_mats(
                    nearmatrix, tree, NO, NS, k, tol, Compressor
                ),
            )
            #end
        end
    end

    return PetrovGalerkinBF_mats{eltype(operator)}(  #BEAST.scalartype(operator)
        nears,
        #tree,
        farints,
        fly,        # Here come all other fields needed for the ButterflyFactorization
        size(nearmatrix),
    )
end

function PetrovGalerkinBF(
    operator,
    testspace,
    trialspace,
    tree::BlockTree,
    k::Float64,
    tol=1e-3,
    ntasks=Threads.nthreads(),
    α=2,
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)   #quadstrat=nearquadstrat
    #values, nearvalues = nearinteractions(tree; isnear=isnear)
    values, nearvalues, farints = nearandfar2(tree, α)

    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @show ntasks length(values)
    @tasks for i in eachindex(values)
        @set ntasks = 1#ntasks
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix)) #size(nearmatrix)
    fly = Vector{BF}()
    for (NO, source_nodes) in farints
        for NS in source_nodes
            push!(fly, subroutine_BF_approx_treeh2(nearmatrix, tree, NO, NS, k, tol))
            #end
        end
    end

    return PetrovGalerkinBF{eltype(operator)}(  #BEAST.scalartype(operator)
        nears,
        #tree,
        farints,
        fly,        # Here come all other fields needed for the ButterflyFactorization
        size(nearmatrix),
    )
end

function Base.size(A::ButterflyFactorization.PetrovGalerkinBF, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(A::ButterflyFactorization.PetrovGalerkinBF_mats, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(
    A::Adjoint{T}, dim=nothing
) where {T<:ButterflyFactorization.PetrovGalerkinBF}
    if dim === nothing
        return reverse(A.dim[1], A.dim[2])
    elseif dim == 1
        return h2mat.lmap.dim[2]
    elseif dim == 2
        return h2mat.lmap.dim[1]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(
    A::Adjoint{T}, dim=nothing
) where {T<:ButterflyFactorization.PetrovGalerkinBF_mats}
    if dim === nothing
        return reverse(A.dim[1], A.dim[2])
    elseif dim == 1
        return h2mat.lmap.dim[2]
    elseif dim == 2
        return h2mat.lmap.dim[1]
    else
        error("dim must be either 1 or 2")
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::ButterflyFactorization.PetrovGalerkinBF, x::AbstractVector{T}
) where {T}     #PetrovGalerkinBF(T) instead of x::AbstractVector{T}?
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y += A.nearinteractions * x
    i = 1
    for (NO, source_nodes) in A.farinteractions
        for NS in source_nodes
            y += apply_butterflyh2(A.tree, A.BFs[i], x)
            i += 1
        end
    end

    return y
end #mul should not return anything.... change to @views y

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::ButterflyFactorization.PetrovGalerkinBF_mats,
    x::AbstractVector{T},
) where {T}     #PetrovGalerkinBF(T) instead of x::AbstractVector{T}?
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y += A.nearinteractions * x
    i = 1
    for (NO, source_nodes) in A.farinteractions
        for NS in source_nodes
            y += applyBF_Mats(A.BFs[i], x)
            i += 1
        end
    end

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorization.PetrovGalerkinBF{T}},
    x::AbstractVector,
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y += transpose(At.lmap.nearinteractions) * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorization.PetrovGalerkinBF{T}},
    x::AbstractVector,
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y += adjoint(At.lmap.nearinteractions) * x

    return y
end
