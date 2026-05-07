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

#Helper funcitons
include("auxillaries.jl")

#Kernelmatrix import
include("kernelmatrix/abstractkernelmatrix.jl")
include("kernelmatrix/beastkernelmatrix.jl")

#Butterfly algebra --> any Block related functions
include("Butterflyalgebra/bfstructs.jl")
include("Butterflyalgebra/bfdims.jl")
include("Butterflyalgebra/bfvector.jl")
include("Butterflyalgebra/bfmatrix.jl")
include("Butterflyalgebra/algrecomp.jl")
include("Butterflyalgebra/bfbfadd.jl")
include("Butterflyalgebra/bfbfmul.jl")

#Tree traversale and Butterfly construction
include("intlists.jl")
include("compressors.jl")
include("subroutines.jl")

#Full Matrix Assembly
include("ButterflyFactorization/petrovgalerkinbf.jl")

include("matrixalgebra/dims.jl")
include("matrixalgebra/matrixadjtr.jl")
include("matrixalgebra/matrixvector.jl")
include("matrixalgebra/matrixmatrix.jl")

end
