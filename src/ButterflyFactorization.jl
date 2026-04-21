module ButterflyFactorization

using BlockSparseMatrices
using H2Trees
using LinearAlgebra
using LinearMaps
using StaticArrays
using Random
using OhMyThreads
using LowRankApprox
using SparseArrays

include("kernelmatrix/abstractkernelmatrix.jl")
include("kernelmatrix/beastkernelmatrix.jl")

include("nearinteractions.jl")
include("ButterflyFactorization/Compressors.jl")
include("ButterflyFactorization/subroutines.jl")
include("ButterflyFactorization/auxillaries.jl")
include("ButterflyFactorization/PetrovGalerkinBF.jl")
include("ButterflyFactorization/algebraic_recomp.jl")

end # module ButterflyFactorization
