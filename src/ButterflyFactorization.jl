module ButterflyFactorization

using BlockSparseMatrices
using H2Trees
using LinearAlgebra
using LinearMaps
using StaticArrays
using Random
using OhMyThreads
using LowRankApprox

include("kernelmatrix/abstractkernelmatrix.jl")
include("kernelmatrix/beastkernelmatrix.jl")

include("nearinteractions.jl")
include("ButterflyFactorization/subroutines.jl")
include("ButterflyFactorization/PetrovGalerkinBF.jl")

end # module ButterflyFactorization
