using H2Trees
using CompScienceMeshes
using BEAST
using ButterflyFactorization
using Butterfly
using StaticArrays
using LinearAlgebra
using LowRankApprox
using ProfileView, Profile
##
x = meshsphere(1.0, 0.1)
y = translate(meshsphere(1.0, 0.1), SVector(3.0, 0.0, 0.0))
lambda = 1.0
k = 2 * pi / lambda
op = Maxwell3D.singlelayer(; wavenumber=k)
T = raviartthomas(x)
U = raviartthomas(y)
tree3 = TwoNTree(T, U, 0.1)
x = randn(ComplexF64, length(T))
@views farasm = BEAST.blockassembler(op, T, U)
@views function farassembler1(Z, tdata, sdata)
    @views store(v, m, n) = (Z[m, n] += v)
    return farasm(tdata, sdata, store)
end
@time Bfly = subroutine_BF_approx_treeh2(farassembler1, tree3, 1, 1, k, 10^(-1));
ProfileView.view()
#@time subroutine_BF_approx_treeh2_nodict(farassembler1, tree3, 1, 1, k, 10^(-4));
@time y_bfly = apply_butterflyh2(tree3, Bfly, x)
#y_bflynd = apply_butterflyh2_nodict(farassembler1, tree3, 1, 1, k, 10^(-4), x)
@time A1 = assemble(op, T, U);
@time y_exact = A1 * x
relerr = norm(y_exact - y_bfly) / norm(y_exact)
#qr = pqr(A1; rtol=10^-10)

tree4 = TwoNTree(T, T, 0.1)
A2 = assemble(op, T, T)
@views farasm2 = BEAST.blockassembler(op, T, T)
@views function farassembler2(Z, tdata, sdata)
    @views store(v, m, n) = (Z[m, n] += v)
    return farasm2(tdata, sdata, store)
end
@time y_exact2 = A2 * x
BFapprox = Butterfly.assemble_Bfly(farassembler2, tree4, k, 10^-4, 2)
@code_warntype y = apply_Butterfly(BFapprox, x)
Base.summarysize(BFapprox)
Base.summarysize(A2)
relerr = norm(y_exact2 - y) / norm(y_exact2)
norm(y)
norm(y_exact2)
