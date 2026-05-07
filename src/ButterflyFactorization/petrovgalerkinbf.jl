function PetrovGalerkinBF_mats(
    operator,
    testspace,
    trialspace,
    tree::BlockTree,
    k::Float64;
    Compressor=ButterflyFactorization.PartialQR(),
    tol=1e-3,
    ntasks=Threads.nthreads(),
    α=2,
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)   #quadstrat=nearquadstrat
    #values, nearvalues = nearinteractions(tree; isnear=isnear)
    values, nearvalues, farints = nearandfar2(tree, α)

    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @show ntasks length(values)
    @tasks for i in eachindex(values)
        @set ntasks = 1#ntasks
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix)) #size(nearmatrix)
    fly = Vector{BF_Mats}()
    for (NO, source_nodes) in farints
        for NS in source_nodes
            push!(
                fly,
                subroutine_BF_mats(nearmatrix, tree, NO, NS, k, tol; Compressor=Compressor),
            )
            #end
        end
    end

    return PetrovGalerkinBF_mats{eltype(operator)}(  #BEAST.scalartype(operator)
        nears,
        #tree,
        farints,
        fly,        # Here come all other fields needed for the ButterflyFactorization
        size(nearmatrix),
    )
end

function PetrovGalerkinBF(
    operator,
    testspace,
    trialspace,
    tree::BlockTree,
    k::Float64;
    Compressor=ButterflyFactorization.PartialQR(),
    tol=1e-3,
    ntasks=Threads.nthreads(),
    α=2,
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)   #quadstrat=nearquadstrat
    #values, nearvalues = nearinteractions(tree; isnear=isnear)
    values, nearvalues, farints = nearandfar2(tree, α)

    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @show ntasks length(values)
    @tasks for i in eachindex(values)
        @set ntasks = 1#ntasks
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix)) #size(nearmatrix)
    fly = Vector{BF}()
    for (NO, source_nodes) in farints
        for NS in source_nodes
            push!(
                fly, subroutine_BF(nearmatrix, tree, NO, NS, k, tol; Compressor=Compressor)
            )
        end
    end

    return PetrovGalerkinBF{eltype(operator)}(  #BEAST.scalartype(operator)
        nears,
        tree,
        farints,
        fly,        # Here come all other fields needed for the ButterflyFactorization
        size(nearmatrix),
    )
end
