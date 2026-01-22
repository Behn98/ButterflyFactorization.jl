import H2Trees: isleaf, testtree, trialtree, root, children

struct isNearFunctor
    α::Float64
    isNearFunctor(α) = new(α)
end

function (t::isNearFunctor)(srctree, tsttree, snode, onode)
    ocenter = H2Trees.center(tsttree, onode)
    olength = H2Trees.halfsize(tsttree, onode)
    scenter = H2Trees.center(srctree, snode)
    slength = H2Trees.halfsize(srctree, snode)
    W = min(H2Trees.halfsize(srctree, snode), H2Trees.halfsize(tsttree, onode))
    ro = (sqrt(3) / 2) * olength
    rs = (sqrt(3) / 2) * slength
    if norm(scenter - ocenter) - (ro + rs) > t.α * W
        return true
    else
        mind = 0.0
        length = (slength + olength) / 2
        for i in 1:3
            mind += max(0.0, abs(ocenter[i] - scenter[i]) - length)^2
        end
        mind = sqrt(mind)
        if mind > t.α * W
            return true
        else
            return false
        end
    end
end

function process_nodes!(
    srctree,
    tsttree,
    node_o,
    node_s,
    admissible::isNearFunctor,
    farinteractions,
    nearinteractions,
)
    if admissible(srctree, tsttree, node_s, node_o)
        push!(get!(farinteractions, node_o, Int64[]), node_s)
        return nothing
    end
    if isleaf(tsttree, node_o) && isleaf(srctree, node_s)
        push!(get!(nearinteractions, node_o, Int64[]), node_s)
        return nothing
    end
    # split the larger node
    if H2Trees.halfsize(tsttree, node_o) >= H2Trees.halfsize(srctree, node_s)
        for child_o in collect(children(tsttree, node_o))
            process_nodes!(
                srctree,
                tsttree,
                child_o,
                node_s,
                admissible,
                farinteractions,
                nearinteractions,
            )
        end
    else
        for child_s in collect(children(srctree, node_s))
            process_nodes!(
                srctree,
                tsttree,
                node_o,
                child_s,
                admissible,
                farinteractions,
                nearinteractions,
            )
        end
    end
end

function nearandfar(tree::H2Trees.BlockTree, α)
    admissible = isNearFunctor(α)
    srctree = trialtree(tree)
    tsttree = testtree(tree)
    node_o = root(tsttree)
    node_s = root(srctree)
    nearinteractions = Dict{Int64,Vector{Int64}}()          #observernodeid --> sourcenodeid
    farinteractions = Dict{Int64,Vector{Int64}}()           #observernodeid --> sourcenodeid
    process_nodes!(
        srctree, tsttree, node_o, node_s, admissible, farinteractions, nearinteractions
    )
    return nearinteractions, farinteractions
end
