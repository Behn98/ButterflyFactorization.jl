module ButterflyFactorization

using BlockSparseMatrices
using H2Trees
using LinearAlgebra
using LinearMaps
using OhMyThreads
using Random

include("kernelmatrix/abstractkernelmatrix.jl")
include("kernelmatrix/beastkernelmatrix.jl")
include("farinteractions.jl")
include("ButterflyFactorization/subroutines.jl")
include("nearinteractions.jl")

include("ButterflyFactorization/PetrovGalerkinBF.jl")

end # module ButterflyFactorization
