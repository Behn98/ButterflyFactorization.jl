function estimate_rank_3d(k, c_s, c_o, a_s, a_o, ε; C=1.0, Cε=3.0, Rmin=5)

    # Center separation
    d = norm(c_s .- c_o)

    # Minimum separation (avoid singular or near-field cases)
    dmin = max(d - 0.5 * (a_s + a_o), 1e-12)

    # Geometric directional rank estimate
    R_geom = C * k * (a_s * a_o) / dmin

    # Tolerance-dependent padding
    R_tol = Cε * log(1 / ε)

    # Final rank
    R = ceil(Int, R_geom + R_tol)

    return max(R, Rmin)
end

function subroutine_getskelh2(
    AKM::AbstractKernelMatrix, src_index, obs_index, c_s, c_o, a_s, a_o, ε, k
)
    #S_otilde = []
    isempty(src_index) && return zeros(ComplexF64, 0, 0), Int[], 0
    isempty(obs_index) && return zeros(ComplexF64, 0, 0), Int[], 0
    #n_otilde = max(1, Int(3 * round(length(obs_index) / 4)))       old
    n_otilde = estimate_rank_3d(k, c_s, c_o, a_s, a_o, ε; C=1.0, Cε=3.0, Rmin=5)         #experimental

    row = zeros(Int64, n_otilde)
    for i in 1:n_otilde                         #adjust n_otilde depending on the obsindex length!
        row[i] = obs_index[rand(1:length(obs_index))]            #pick randomly from obs index here!
    end
    z_otilde_s = zeros(ComplexF64, n_otilde, length(src_index))
    col = src_index
    AKM(z_otilde_s, row, col)
    #Rank revealing QR
    qrp = qr(z_otilde_s, Val(true))
    q_qr = qrp.Q
    r_qr = qrp.R
    p_qr = qrp.p
    #--> extract Q_Ks. K and R
    r = 0
    error = ε
    while error >= ε && r < min(size(r_qr)...)
        r += 1
        error = abs(r_qr[r, r]) / abs(r_qr[1, 1])
    end
    k = src_index[p_qr[1:r]]
    r_11 = r_qr[1:r, 1:r]
    q_1 = q_qr[:, 1:r]
    q_ks = inv(r_11) * adjoint(q_1) * z_otilde_s
    return q_ks, k, r
end

function subroutine_BF_approx_treeh2(AKM::AbstractKernelMatrix, H2Blocktree, NO, NS, k, τ)
    Q = Dict{Int,Matrix{ComplexF64}}()                #sourceid --> Q-block
    R = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()     #(sourceid, observerid) --> R-block
    P = Dict{Int,Matrix{ComplexF64}}()                #observerid --> P-block
    K = Dict{Tuple{Int,Int},Vector{Int}}()         #(sourceid, observerid) --> skeleton
    U = Dict{Tuple{Int,Int},Vector{Int}}()         #(sourceid, observerid) --> skeleton
    εrank = Dict{Tuple{Int,Int},Int}()             #(sourceid, observerid) --> ε-rank
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)
    leaves = H2Trees.leaves
    values = H2Trees.values
    center = H2Trees.center
    halfsize = H2Trees.halfsize
    getchildren = H2Trees.children
    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)
    L = length(treeS)             #obstree and sourcetree should be of equal height
    #LO = length(treeO)
    for leave in treeS[L]
        srcindex = values(trialT, leave)
        obsindex = values(testT, NO)
        c_s = center(trialT, leave)
        c_o = center(testT, NO)
        a_s = halfsize(trialT, leave)
        a_o = halfsize(testT, NO)
        q_ks, k_l, r_l = subroutine_getskelh2(
            AKM, srcindex, obsindex, c_s, c_o, a_s, a_o, τ, k
        )
        Q[leave] = q_ks
        K[leave, NO] = k_l
        εrank[leave, NO] = r_l
    end
    for l in 1:(L - 1)
        for Svert in treeS[L - l]
            temp = Int64[]
            for i in treeO[l]
                for Schild in collect(getchildren(trialT, Svert))
                    append!(temp, K[Schild, i])
                end
                U[Svert, i] = temp
                temp = Int64[]
            end
        end
        #current = 1
        for Overt in treeO[l]
            for Ochild in collect(getchildren(testT, Overt))
                for Svert in treeS[L - l]
                    #if !isempty(Svert.U[current]) && !isempty(Ochild.subcluster)
                    obsindex = values(testT, Ochild)
                    c_s = center(trialT, Svert)
                    c_o = center(testT, Ochild)
                    a_s = halfsize(trialT, Svert)
                    a_o = halfsize(testT, Ochild)
                    q_ks, k_l, r_l = subroutine_getskelh2(
                        AKM, U[Svert, Overt], obsindex, c_s, c_o, a_s, a_o, τ, k
                    ) #=srcfns, obsfns,=#
                    R[Svert, Ochild] = q_ks
                    K[Svert, Ochild] = k_l
                    εrank[Svert, Ochild] = r_l
                    #end
                end
            end
            #current += 1
        end
    end
    for Oleave in treeO[L]
        col = K[NS, Oleave]
        row = values(testT, Oleave)
        p = zeros(ComplexF64, length(row), length(col))
        AKM(p, row, col)
        P[Oleave] = p
    end
    return Q, R, P
end

function apply_butterflyh2(K::AbstractKernelMatrix, H2Blocktree, NO, NS, k, τ, v)             # N - number of children
    Q, R, P = subroutine_BF_approx_treeh2(K, H2Blocktree, NO, NS, k, τ)
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)
    #leaves = H2Trees.leaves
    values = H2Trees.values
    getchildren = H2Trees.children
    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)
    coefficients = Dict{Tuple{Int,Int},Vector{ComplexF64}}()
    L = length(treeS)             #obstree and sourcetree should be of equal height
    for leave in treeS[L]
        temp = v[values(trialT, leave)]
        coefficients[leave, NO] = Q[leave] * temp
    end
    for l in 1:(L - 1)
        for Svert in treeS[L - l]
            temp = ComplexF64[]
            for i in treeO[l]
                for Schild in collect(getchildren(trialT, Svert))
                    append!(temp, coefficients[Schild, i])
                end
                coefficients[Svert, i] = temp
                temp = ComplexF64[]
            end
            for i in treeO[l]
                for j in collect(getchildren(testT, i))
                    coefficients[(Svert, j)] = R[Svert, j] * coefficients[Svert, i]
                end
            end
        end
    end
    result = zeros(ComplexF64, maximum(values(testT, NO)) - minimum(values(testT, NO)) + 1) #delete this line if result is handed over
    mini = minimum(values(testT, NO))
    for oleave in treeO[L]
        result[values(testT, oleave) .- (mini - 1)] = P[oleave] * coefficients[NS, oleave]
        #if the result vector is already given: result[values(testT, oleave)] = P[oleave] * coefficients[NS, oleave]
    end
    return result
end

function h2treelevels(tree, root)
    isleaf = H2Trees.isleaf
    getchildren = H2Trees.children

    levels = Vector{Vector{Int}}()
    current = [root]

    while !isempty(current)
        push!(levels, current)
        next = Int[]

        for node in current
            if !isleaf(tree, node)
                append!(next, getchildren(tree, node))
            end
        end

        current = next
    end

    return levels
end
