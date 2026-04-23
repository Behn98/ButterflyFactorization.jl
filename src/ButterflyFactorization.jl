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
include("auxillaries.jl")
include("symstruct.jl")
include("bfmult.jl")
include("symbfmult.jl")
include("ButterflyFactorization/petrovgalerkinbf.jl")
include("Butterflyalgebra/matrixvector.jl")
include("Butterflyalgebra/matrixmatrix.jl")
include("Butterflyalgebra/algebraicrecomp.jl")

end # module ButterflyFactorization
