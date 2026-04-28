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
include("compressors.jl")
include("subroutines.jl")
include("symstruct.jl")
include("bfmult.jl")
include("symbfmult.jl")
include("auxillaries.jl")
include("ButterflyFactorization/petrovgalerkinbf.jl")
include("Butterflyalgebra/matrixvector.jl")
include("Butterflyalgebra/matrixmatrix.jl")
include("Butterflyalgebra/algebraicrecomp.jl")
include("Butterflyalgebra/bfaddition.jl")

end # module ButterflyFactorization
