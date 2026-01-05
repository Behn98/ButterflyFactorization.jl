
function ButterflyFactorization.AbstractKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
)
    return ButterflyFactorization.BEASTKernelMatrix{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; quadstrat=quadstrat)
    )
end

struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    @views f.matrix[m, n] += v
    return nothing
end

function (blk::ButterflyFactorization.BEASTKernelMatrix)(matrixblock, tdata, sdata)
    blk.nearassembler(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end
