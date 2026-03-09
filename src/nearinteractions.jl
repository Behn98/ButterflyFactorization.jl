import H2Trees: isleaf, testtree, trialtree, root, children
#=
function nears!(
    tree,
    values::Vector{V},
    nearvalues::Vector{V},
    tnode::Int,
    snodes::V;
    isnear=H2Trees.isnear,
) where {V<:Vector{Int}}
    nearnodes = Int[]
    childnearnodes = Int[]
    for snode in snodes
        if isnear(testtree(tree), trialtree(tree), tnode, snode)
            if isleaf(testtree(tree), tnode) || isleaf(trialtree(tree), snode)
                push!(nearnodes, snode)
            else
                append!(childnearnodes, collect(children(trialtree(tree), snode)))
            end
        end
    end
    if nearnodes != []
        push!(nearvalues, H2Trees.values(trialtree(tree), nearnodes))
        push!(values, H2Trees.values(testtree(tree), tnode))
    end
    if childnearnodes != []
        for child in children(testtree(tree), tnode)
            nears!(tree, values, nearvalues, child, childnearnodes; isnear=isnear)
        end
    end
end

function nearinteractions(tree::H2Trees.BlockTree; isnear=H2Trees.isnear)
    !isnear(testtree(tree), trialtree(tree), root(testtree(tree)), root(trialtree(tree))) &&
        return Vector{Int}(), Vector{Int}[]
    values = Vector{Int}[]
    nearvalues = Vector{Int}[]
    nears!(
        tree,
        values,
        nearvalues,
        root(testtree(tree)),
        [root(trialtree(tree))];
        isnear=isnear,
    )
    return values, nearvalues
end
=#
struct isFarFunctor
    α::Float64
    isFarFunctor(α) = new(α)
end

function (t::isFarFunctor)(srctree, tsttree, snode, onode)
    ocenter = H2Trees.center(tsttree, onode)
    olength = H2Trees.halfsize(tsttree, onode)
    scenter = H2Trees.center(srctree, snode)
    slength = H2Trees.halfsize(srctree, snode)
    W = max(H2Trees.halfsize(srctree, snode), H2Trees.halfsize(tsttree, onode))
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
    admissible::isFarFunctor,
    farinteractions,
    nearsv,
    nearov,
)
    if admissible(srctree, tsttree, node_s, node_o)
        push!(get!(farinteractions, node_o, Int64[]), node_s)
        return nothing
    elseif isleaf(tsttree, node_o) && isleaf(srctree, node_s)
        push!(nearsv, H2Trees.values(srctree, node_s))
        push!(nearov, H2Trees.values(tsttree, node_o))
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
                nearsv,
                nearov,
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
                nearsv,
                nearov,
            )
        end
    end
end

function nearandfar2(tree::H2Trees.BlockTree, α)
    admissible = isFarFunctor(α)
    srctree = trialtree(tree)
    tsttree = testtree(tree)
    node_o = root(tsttree)
    node_s = root(srctree)
    nearsv = Vector{Int}[]
    nearov = Vector{Int}[]
    #nearinteractions = Dict{Int64,Vector{Int64}}()          #observernodeid --> sourcenodeid
    farinteractions = Dict{Int64,Vector{Int64}}()           #observernodeid --> sourcenodeid
    process_nodes!(
        srctree, tsttree, node_o, node_s, admissible, farinteractions, nearsv, nearov
    )
    return nearov, nearsv, farinteractions
end

function process_nodes!(
    srctree,
    tsttree,
    node_o,
    node_s,
    admissible::isFarFunctor,
    farinteractions,
    nearinteractions,
)
    if admissible(srctree, tsttree, node_s, node_o)
        push!(get!(farinteractions, node_o, Int64[]), node_s)
        return nothing
    elseif isleaf(tsttree, node_o) && isleaf(srctree, node_s)
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
    admissible = isFarFunctor(α)
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
