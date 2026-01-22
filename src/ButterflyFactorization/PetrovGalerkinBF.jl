struct PetrovGalerkinBF{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    # Here come all other fields needed for the ButterflyFactorization
    dim::Tuple{Int,Int}

    function PetrovGalerkinBF{T}(nearinteractions) where {T}
        return new{T,typeof(nearinteractions)}(
            nearinteractions,
            # Here come all other fields needed for the ButterflyFactorization
            dim,
        )
    end
end

function PetrovGalerkinBF(
    operator, testspace, trialspace, tree::BlockTree; tol=1e-3, ntasks=Threads.nthreads()
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)
    farmatrix = AbstractKernelMatrix(operator, testspace, trialspace)
    values, nearvalues = nearinteractions(tree; isnear=isnear)

    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @tasks for i in eachindex(values)
        @set ntasks = ntasks
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))

    return PetrovGalerkinBF{eltype(operator)}(
        nears,
        # Here come all other fields needed for the ButterflyFactorization
        size(nearmatrix),
    )
end

function Base.size(A::PetrovGalerkinBF, dim=nothing)
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

function Base.size(A::Adjoint{T}, dim=nothing) where {T<:PetrovGalerkinBF}
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
    y::AbstractVecOrMat, A::PetrovGalerkinBF{T}, x::AbstractVector
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y += A.nearinteractions * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinBF{T}},
    x::AbstractVector,
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y += transpose(At.lmap.nearinteractions) * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinBF{T}},
    x::AbstractVector,
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y += adjoint(At.lmap.nearinteractions) * x

    return y
end
