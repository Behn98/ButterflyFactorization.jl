
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::ButterflyFactorization.PetrovGalerkinBF, x::AbstractVector{T}
) where {T}
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
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::ButterflyFactorization.PetrovGalerkinBF_mats,
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y += A.nearinteractions * x
    i = 1
    for i in eachindex(A.BFs)
        y += applyBF_Mats(A.BFs[i], x)
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorization.PetrovGalerkinBF_mats},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y += transpose(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y += applyBF_Mats(transpose(At.lmap.BFs[i]), x)
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorization.PetrovGalerkinBF_mats},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y += adjoint(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y += applyBF_Mats(At.lmap.BFs[i]', x)
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorization.PetrovGalerkinBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y += transpose(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y += apply_butterflyh2_transpose(At.lmap.tree, At.lmap.BFs[i], x)
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorization.PetrovGalerkinBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y += adjoint(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y += apply_butterflyh2_adjoint(At.lmap.tree, At.lmap.BFs[i], x)
    end
    return y
end
