module ButterflyFactorization

using BlockSparseMatrices
using H2Trees
using LinearAlgebra
using LinearMaps
using OhMyThreads

include("kernelmatrix/abstractkernelmatrix.jl")
include("kernelmatrix/beastkernelmatrix.jl")

include("nearinteractions.jl")

include("ButterflyFactorization/PetrovGalerkinBF.jl")

end # module ButterflyFactorization
