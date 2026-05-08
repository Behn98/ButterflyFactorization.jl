@testitem "Testing Full Matrix Assembly" begin
    using Test
    using H2Trees
    using CompScienceMeshes
    using BEAST
    using ButterflyFactorization
    using StaticArrays
    using LinearAlgebra
    using ParallelKMeans

    #========================================================================
    =========================================================================
                            Geometry and Operators
    =========================================================================
    =========================================================================#
    lambda = 1.0
    k = 2 * pi / lambda
    x = meshsphere(1.0, lambda / 10)
    op = Maxwell3D.singlelayer(; wavenumber=k)
    T = raviartthomas(x)
    length(T)

    ##
    #========================================================================
    =========================================================================
                    Tree construction  and Kernelmatrix assembly
    =========================================================================
    =========================================================================#

    tree1 = TwoNTree(T, T, lambda / 10)     #testspace, trialspace

    #========================================================================
    =========================================================================
                        Assembly of Matrices and Vectors
    =========================================================================
    =========================================================================#

    @time A1 = assemble(op, T, T)

    x_t = randn(ComplexF64, length(T))

    x_s1 = A1 * x_t

    #========================================================================
    =========================================================================
                            Buttefly routines calling
    =========================================================================
    =========================================================================#

    @time Bfly1 = ButterflyFactorization.PetrovGalerkinBF(op, T, T, tree1, k; tol=1e-3, α=2)
    @time Bfly2 = ButterflyFactorization.PetrovGalerkinBF_mats(
        op, T, T, tree1, k; tol=1e-3, α=2
    )

    x_test = zeros(ComplexF64, length(T))

    @views mul!(x_test, Bfly1, x_t)

    @test norm(x_s1 - x_test) / norm(x_s1) < 1e-3

    @test Base.summarysize(A1) > Base.summarysize(Bfly1)

    @views mul!(x_test, Bfly2, x_t)

    @test norm(x_s1 - x_test) / norm(x_s1) < 1e-3
end
