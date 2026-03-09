struct PetrovGalerkinBF{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    # Here come all other fields needed for the ButterflyFactorization
    dim::Tuple{Int,Int}
    tree::H2Trees.BlockTree
    farinteractions::Dict{Int64,Vector{Int64}}           #observernodeid --> sourcenodeid
    BFs::Vector{BF} #Dict{Int,Dict{Int,BF}}
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
    #=
    fly = Dict{Int,Dict{Int,BF}}()                  #observer node id --> source node id --> BF
    #i = 1
    for (NO, source_nodes) in farints
        flyNO = getsubdict!(fly, NO) #=do nothing, just initialize the subdict for this observer node=#
        for NS in source_nodes
            flyNO[NS] = subroutine_BF_approx_treeh2(nearmatrix, tree, NO, NS, k, tol)
            #i += 1
        end
    end
    =#
    fly = Vector{BF}()
    for (NO, source_nodes) in farints
        for NS in source_nodes
            #if !isemtpy(H2Trees.values(H2Trees.testtree(tree), NO)) &
            #!isemtpy(H2Trees.values(H2Trees.trialtree(tree), NS))
            push!(fly, subroutine_BF_approx_treeh2(nearmatrix, tree, NO, NS, k, tol))
            #end
        end
    end

    return PetrovGalerkinBF{eltype(operator)}(  #BEAST.scalartype(operator)
        nears,
        tree,
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

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::ButterflyFactorization.PetrovGalerkinBF, x::AbstractVector{T}
) where {T}     #PetrovGalerkinBF(T) instead of x::AbstractVector{T}?
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y += A.nearinteractions * x
    #=
    for (NO, source_nodes) in A.farinteractions
        BFs_NO = getsubdict!(A.BFs, NO)
        for NS in source_nodes
            y += apply_butterflyh2(A.tree, BFs_NO[NS], x)
            #i += 1
        end
    end
    =#
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
