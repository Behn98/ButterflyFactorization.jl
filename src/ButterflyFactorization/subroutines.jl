struct BF
    Q::Dict{Int,Matrix{ComplexF64}}
    #Q = Vector{Matrix{ComplexF64}}()          #any source leaf has a Q-block related to the root of the obs sub tree
    #any Q block has the dimension of R×number of source indices in the corresponding leaf
    R::Dict{Int,Dict{Int,Matrix{ComplexF64}}}
    #R = Vector{Matrix{ComplexF64}}()      #any source node on any level l∈[1:L] has exactly as many R-blocks as there are nodes on the corresponding obs level L-l+1
    # where level 0 is the root of the tree
    P::Dict{Int,Matrix{ComplexF64}}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    BF(Q, R, P, NS, NO, k, τ) = new(Q, R, P, NS, NO, k, τ)
end

struct BF_Mats
    Q::AbstractMatrix{ComplexF64}
    R::Vector{AbstractMatrix{ComplexF64}}      #any source node on any level l∈[1:L] has exactly as many R-blocks as there are nodes on the corresponding obs level L-l+1
    # where level 0 is the root of the tree
    P::AbstractMatrix{ComplexF64}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    PermP::Vector{Int}          #permutation of source indices for P blocks, needed for correct assembly of R blocks
    PermQ::Vector{Int}          #permutation of source indices for Q blocks, needed for correct assembly of R blocks
    BF_Mats(Q, R, P, NS, NO, k, τ, PermP, PermQ) = new(Q, R, P, NS, NO, k, τ, PermP, PermQ)
end

struct BFapprox
    tree::H2Trees.BlockTree
    nearinteractions::Dict{Int64,Vector{Int64}}          #observernodeid --> sourcenodeid
    farinteractions::Dict{Int64,Vector{Int64}}           #observernodeid --> sourcenodeid
    BFs::Vector{BF}
    NI::Vector{Matrix{ComplexF64}}
    BFapprox(tree, nearints, farints, BFs, NI) = new(tree, nearints, farints, BFs, NI)
end

abstract type Abstractcompressor end

struct PartialQR <: Abstractcompressor
    PartialQR() = new()
end

function (t::PartialQR)(
    farassembler, src_index::Vector{Int}, obs_index::Vector{Int}, n_otilde::Int, ε::Float64
)
    n_obs = length(obs_index)
    n_src = length(src_index)
    n_otilde = min(n_otilde, n_obs)

    # --- random row sampling (type stable) ---
    idx = randperm(n_obs)
    row = @view obs_index[idx[1:n_otilde]]
    col = src_index  # full view, no copy

    # --- assemble Z ---
    Z = zeros(ComplexF64, n_otilde, n_src)
    farassembler(Z, row, col)

    # --- pivoted QR (LAPACK-backed) ---
    Fqr = pqr(Z; rtol=ε)

    Q = Fqr[1]
    R = Fqr[2]
    P = Fqr[3]

    r = size(Q, 2)

    # --- views to avoid allocations ---
    Q1 = @view Q[:, 1:r]
    R11 = UpperTriangular(@view R[1:r, 1:r])

    # --- compute q_ks without inv ---
    # tmp = Q1' * Z
    tmp = Matrix{ComplexF64}(undef, r, n_src)
    mul!(tmp, adjoint(Q1), Z)

    # q_ks = R11 \ tmp
    ldiv!(R11, tmp)

    k = src_index[P[1:r]]

    return tmp, k, r
end

function estimate_rank_3d(
    k,
    c_s::SVector,
    c_o::SVector,
    a_s::Float64,
    a_o::Float64,
    ε::Float64,
    ;
    C=1.0,
    Cε=3.0,
    Rmin=5,
)

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
    farassembler,
    src_index::Vector{Int},
    obs_index::Vector{Int},
    c_s::SVector,
    c_o::SVector,
    a_s::Float64,
    a_o::Float64,
    ε::Float64,
    k_est::Float64,
)

    # --- rank estimation ---
    n_otilde = estimate_rank_3d(k_est, c_s, c_o, a_s, a_o, ε; C=1.0, Cε=3.0, Rmin=3)

    n_obs = length(obs_index)
    n_src = length(src_index)
    n_otilde = min(n_otilde, n_obs)

    # --- random row sampling (type stable) ---
    idx = randperm(n_obs)
    row = @view obs_index[idx[1:n_otilde]]
    col = src_index  # full view, no copy

    # --- assemble Z ---
    Z = zeros(ComplexF64, n_otilde, n_src)
    farassembler(Z, row, col)

    # --- pivoted QR (LAPACK-backed) ---
    Fqr = pqr(Z; rtol=ε)

    Q = Fqr[1]
    R = Fqr[2]
    P = Fqr[3]

    r = size(Q, 2)

    # --- views to avoid allocations ---
    Q1 = @view Q[:, 1:r]
    R11 = UpperTriangular(@view R[1:r, 1:r])

    # --- compute q_ks without inv ---
    # tmp = Q1' * Z
    tmp = Matrix{ComplexF64}(undef, r, n_src)
    mul!(tmp, adjoint(Q1), Z)

    # q_ks = R11 \ tmp
    ldiv!(R11, tmp)

    k = src_index[P[1:r]]

    return tmp, k, r
end

@inline function getsubdict!(D::Dict{Int,Dict{Int,T}}, k::Int) where {T}
    get!(D, k) do
        Dict{Int,T}()
    end
end

function subroutine_BF_approx_treeh2(
    farassembler, H2Blocktree, NO::Int, NS::Int, k::Float64, τ::Float64
)

    # --- containers ---
    Q = Dict{Int,Matrix{ComplexF64}}()
    #Q = Vector{Matrix{ComplexF64}}()          #any source leaf has a Q-block related to the root of the obs sub tree
    #any Q block has the dimension of R×number of source indices in the corresponding leaf
    R = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
    #R = Vector{Matrix{ComplexF64}}()      #any source node on any level l∈[1:L] has exactly as many R-blocks as there are nodes on the corresponding obs level L-l+1
    # where level 0 is the root of the tree
    K = Dict{Int,Dict{Int,Vector{Int}}}()
    #K = Vector{Vector{Int}}()          #any source node on any level l∈[1:L] has exactly as many skeletons K as there are nodes on the corresponding obs level L-l+1
    # where level 0 is the root of the tree
    U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions

    # ε-rank kept for future use
    # εrank = Dict{Int, Dict{Int, Int}}()

    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    center = H2Trees.center
    halfsize = H2Trees.halfsize
    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    for Sleaf in treeS[LS]
        srcindex = values(trialT, Sleaf)
        obsindex = values(testT, NO)
        #isempty(srcindex) && continue
        c_s = center(trialT, Sleaf)
        c_o = center(testT, NO)
        a_s = halfsize(trialT, Sleaf)
        a_o = halfsize(testT, NO)

        q_ks, k_l, r_l = subroutine_getskelh2(
            farassembler, srcindex, obsindex, c_s, c_o, a_s, a_o, τ, k
        )

        Q[Sleaf] = q_ks
        #push!(Q, q_ks)
        #push!(K, k_l)
        getsubdict!(K, Sleaf)[NO] = k_l
        #getsubdict!(εrank, Sleaf)[NO] = r_l
    end

    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------------
    for l in 1:(L - 1)
        #K_new = Vector{Vector{Int}}()
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)

        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        if !source_is_frozen
            for Svert in treeS[LS - l]
                U_S = getsubdict!(U, Svert)

                for Overt in treeO[min(l, LO)]
                    temp = Int[]
                    # optional future optimization:
                    # sizehint!(temp, estimated_size)

                    for Schild in children(trialT, Svert)
                        Ks = getsubdict!(K, Schild)
                        ks = get(Ks, Overt, nothing)
                        #ks === nothing && continue
                        append!(temp, ks)
                    end

                    U_S[Overt] = temp
                end
            end
        end

        # --------------------------------------------------------------
        # Compute R blocks
        # --------------------------------------------------------------
        if !source_is_frozen && !obs_is_frozen
            rowsizeR = 0
            for Overt in treeO[l]
                for Ochild in children(testT, Overt)
                    obsindex = values(testT, Ochild)
                    isempty(obsindex) && continue
                    c_o = center(testT, Ochild)
                    a_o = halfsize(testT, Ochild)
                    for Svert in treeS[LS - l]
                        srcindex = U[Svert][Overt]
                        #isempty(srcindex) && continue
                        c_s = center(trialT, Svert)
                        a_s = halfsize(trialT, Svert)
                        #@show srcindex
                        #@show obsindex
                        #@show Svert
                        #@show Ochild
                        #@show obsindex
                        #@show srcindex
                        #@show collect(children(trialT, Svert))
                        #@show values(trialT, Svert)
                        q_ks, k_l, r_l = subroutine_getskelh2(
                            farassembler, srcindex, obsindex, c_s, c_o, a_s, a_o, τ, k
                        )
                        rowsizeR += size(q_ks, 1)
                        getsubdict!(R, Svert)[Ochild] = q_ks
                        #push!(R, q_ks)
                        getsubdict!(K, Svert)[Ochild] = k_l
                        #push!(K_new, k_l)
                        #getsubdict!(εrank, Svert)[Ochild] = r_l
                    end
                end
            end
            @show l
            @show rowsizeR

        elseif source_is_frozen && !obs_is_frozen
            @show source_is_frozen
            for Overt in treeO[l]
                for Ochild in children(testT, Overt)
                    obsindex = values(testT, Ochild)
                    isempty(obsindex) && continue
                    for Svert in treeS[1]
                        #inner = get(K, Svert, nothing)
                        #col = inner === nothing ? nothing : get(inner, Overt, nothing)
                        #col === nothing && continue
                        col = K[Svert][Overt]

                        Z = zeros(ComplexF64, length(obsindex), length(col))
                        farassembler(Z, obsindex, col)

                        getsubdict!(R, Svert)[Ochild] = Z
                        getsubdict!(K, Svert)[Ochild] = col
                    end
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Overt in treeO[LO]
                obsindex = values(testT, Overt)
                isempty(obsindex) && continue
                for Svert in treeS[LS - l]
                    #inner = get(U, Svert, nothing)
                    #srcindex = inner === nothing ? nothing : get(inner, Overt, nothing)
                    #srcindex === nothing && continue
                    srcindex = U[Svert][Overt]

                    c_s = center(trialT, Svert)
                    c_o = center(testT, Overt)
                    a_s = halfsize(trialT, Svert)
                    a_o = halfsize(testT, Overt)

                    q_ks, k_l, r_l = subroutine_getskelh2(
                        farassembler, srcindex, obsindex, c_s, c_o, a_s, a_o, τ, k
                    )

                    getsubdict!(R, Svert)[Overt] = q_ks
                    getsubdict!(K, Svert)[Overt] = k_l
                    #getsubdict!(εrank, Svert)[Overt] = r_l
                end
            end

        else
            break
        end
        #K = K_new
    end

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------
    P = Dict{Int,Matrix{ComplexF64}}()
    #P = Vector{Matrix{ComplexF64}}()
    for Oleaf in treeO[LO]
        #inner = get(K, NS, nothing)
        #col = inner === nothing ? nothing : get(inner, Oleaf, nothing)
        #col === nothing && continue
        col = K[NS][Oleaf]
        row = values(testT, Oleaf)
        #isempty(row) && continue
        Z = zeros(ComplexF64, length(row), length(col))
        farassembler(Z, row, col)

        P[Oleaf] = Z
        #push!(P, Z)
    end
    result = BF(Q, R, P, NS, NO, τ, k)
    return result
end

function blockdiag(blocks::AbstractMatrix...)
    isempty(blocks) && return zeros(0, 0)

    T = promote_type(map(eltype, blocks)...)

    rows = sum(size(b, 1) for b in blocks)
    cols = sum(size(b, 2) for b in blocks)

    M = zeros(T, rows, cols)

    r = 1
    c = 1
    for B in blocks
        nr, nc = size(B)
        M[r:(r + nr - 1), c:(c + nc - 1)] .= B
        r += nr
        c += nc
    end

    return M
end

function sparse_blockdiag(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Convert blocks to sparse to utilize SparseArrays.blockdiag
    sparse_blocks = map(sparse, blocks)
    return SparseArrays.blockdiag(sparse_blocks...)
end

function sparse_vcat(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Convert blocks to sparse and vertically concatenate
    sparse_blocks = map(sparse, blocks)
    return vcat(sparse_blocks...)
end

function blocksparse_blockdiag(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Assemble as a standard sparse matrix first, then convert to BlockSparseMatrix
    sp_mat = SparseArrays.blockdiag(map(sparse, blocks)...)
    return BlockSparseMatrix(sp_mat)
end

function blocksparse_vcat(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Vertically concatenate sparse matrices, then convert to BlockSparseMatrix
    sp_mat = vcat(map(sparse, blocks)...)
    return BlockSparseMatrix(sp_mat)
end

function subroutine_BF_approx_treeh2_mats(
    farassembler,
    H2Blocktree,
    NO::Int,
    NS::Int,
    k::Float64,
    τ::Float64,
    Compressor::Abstractcompressor,
)

    # --- containers ---
    #Q = Dict{Int,Matrix{ComplexF64}}()
    Q = Matrix{ComplexF64}(undef, 0, 0)          #any source leaf has a Q-block related to the root of the obs sub tree
    #any Q block has the dimension of R×number of source indices in the corresponding leaf
    #R = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
    R = Vector{AbstractMatrix{ComplexF64}}()      #any source node on any level l∈[1:L] has exactly as many R-blocks as there are nodes on the corresponding obs level L-l+1
    # where level 0 is the root of the tree
    #P = Dict{Int,Matrix{ComplexF64}}()
    P = Matrix{ComplexF64}(undef, 0, 0)

    K = Dict{Int,Dict{Int,Vector{Int}}}()
    #K = Vector{Vector{Int}}()          #any source node on any level l∈[1:L] has exactly as many skeletons K as there are nodes on the corresponding obs level L-l+1
    # where level 0 is the root of the tree
    U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions

    PermQ = Vector{Int}()          #permutation of source indices for Q blocks, needed for correct assembly of R blocks
    PermP = Vector{Int}()          #permutation of source indices for P blocks, needed for correct assembly of R blocks
    # ε-rank kept for future use
    # εrank = Dict{Int, Dict{Int, Int}}()

    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    center = H2Trees.center
    halfsize = H2Trees.halfsize
    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    for Sleaf in treeS[LS]
        srcindex = values(trialT, Sleaf)
        push!(PermQ, srcindex...)
        obsindex = values(testT, NO)
        #isempty(srcindex) && continue
        c_s = center(trialT, Sleaf)
        c_o = center(testT, NO)
        a_s = halfsize(trialT, Sleaf)
        a_o = halfsize(testT, NO)
        n_otilde = estimate_rank_3d(k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3)
        q_ks, k_l, r_l = Compressor(farassembler, srcindex, obsindex, n_otilde, τ)
        Q = sparse_blockdiag(Q, q_ks)               #SPARSITY: blocksparse_ or sparse_
        #Q[Sleaf] = q_ks
        #push!(Q, q_ks)
        #push!(K, k_l)
        getsubdict!(K, Sleaf)[NO] = k_l
        #getsubdict!(εrank, Sleaf)[NO] = r_l
    end

    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------------
    for l in 1:(L - 1)
        #K_new = Vector{Vector{Int}}()
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)

        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        if !source_is_frozen
            for Svert in treeS[LS - l]
                U_S = getsubdict!(U, Svert)

                for Overt in treeO[min(l, LO)]
                    temp = Int[]
                    # optional future optimization:
                    # sizehint!(temp, estimated_size)

                    for Schild in children(trialT, Svert)
                        Ks = getsubdict!(K, Schild)
                        ks = get(Ks, Overt, nothing)
                        #ks === nothing && continue
                        append!(temp, ks)
                    end

                    U_S[Overt] = temp
                end
            end
        end

        # --------------------------------------------------------------
        # Compute R blocks
        # --------------------------------------------------------------
        if !source_is_frozen && !obs_is_frozen
            rowsizeR = 0
            R_temp1 = Matrix{ComplexF64}(undef, 0, 0)
            for Overt in treeO[l]
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()
                for Ochild in children(testT, Overt)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                    obsindex = values(testT, Ochild)
                    #isempty(obsindex) && continue
                    c_o = center(testT, Ochild)
                    a_o = halfsize(testT, Ochild)
                    for Svert in treeS[LS - l]
                        srcindex = U[Svert][Overt]
                        #isempty(srcindex) && continue
                        c_s = center(trialT, Svert)
                        a_s = halfsize(trialT, Svert)

                        n_otilde = estimate_rank_3d(
                            k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3
                        )
                        q_ks, k_l, r_l = Compressor(
                            farassembler, srcindex, obsindex, n_otilde, τ
                        )
                        R_temp3 = sparse_blockdiag(R_temp3, q_ks)   #SPARSITY: blocksparse_ or sparse_
                        rowsizeR += size(q_ks, 1)
                        #getsubdict!(R, Svert)[Ochild] = q_ks

                        #push!(R, q_ks)
                        getsubdict!(K, Svert)[Ochild] = k_l
                        #push!(K_new, k_l)
                        #getsubdict!(εrank, Svert)[Ochild] = r_l
                    end
                    push!(R_temp2, R_temp3)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                end
                R_temp1 = sparse_blockdiag(R_temp1, sparse_vcat(R_temp2...))    #SPARSITY: blocksparse_ or sparse_
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()
            end
            @show l
            @show rowsizeR
            push!(R, R_temp1)

        elseif source_is_frozen && !obs_is_frozen
            @show source_is_frozen
            for Overt in treeO[l]
                for Ochild in children(testT, Overt)
                    obsindex = values(testT, Ochild)
                    isempty(obsindex) && continue
                    for Svert in treeS[1]
                        #inner = get(K, Svert, nothing)
                        #col = inner === nothing ? nothing : get(inner, Overt, nothing)
                        #col === nothing && continue
                        col = K[Svert][Overt]

                        Z = zeros(ComplexF64, length(obsindex), length(col))
                        farassembler(Z, obsindex, col)

                        getsubdict!(R, Svert)[Ochild] = Z
                        getsubdict!(K, Svert)[Ochild] = col
                    end
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Overt in treeO[LO]
                obsindex = values(testT, Overt)
                isempty(obsindex) && continue
                for Svert in treeS[LS - l]
                    #inner = get(U, Svert, nothing)
                    #srcindex = inner === nothing ? nothing : get(inner, Overt, nothing)
                    #srcindex === nothing && continue
                    srcindex = U[Svert][Overt]

                    c_s = center(trialT, Svert)
                    c_o = center(testT, Overt)
                    a_s = halfsize(trialT, Svert)
                    a_o = halfsize(testT, Overt)

                    n_otilde = estimate_rank_3d(
                        k, c_s, c_o, a_s, a_o, τ; C=1.0, Cε=3.0, Rmin=3
                    )
                    q_ks, k_l, r_l = Compressor(
                        farassembler, srcindex, obsindex, n_otilde, τ
                    )

                    getsubdict!(R, Svert)[Overt] = q_ks
                    getsubdict!(K, Svert)[Overt] = k_l
                    #getsubdict!(εrank, Svert)[Overt] = r_l
                end
            end

        else
            break
        end
        #K = K_new
    end

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------

    for Oleaf in treeO[LO]
        #inner = get(K, NS, nothing)
        #col = inner === nothing ? nothing : get(inner, Oleaf, nothing)
        #col === nothing && continue
        col = K[NS][Oleaf]
        row = values(testT, Oleaf)
        push!(PermP, row...)
        #isempty(row) && continue
        Z = zeros(ComplexF64, length(row), length(col))
        farassembler(Z, row, col)
        P = sparse_blockdiag(P, Z)              #SPARSITY: blocksparse_ or sparse_
        #P[Oleaf] = Z
        #push!(P, Z)
    end
    return BF_Mats(Q, R, P, NS, NO, τ, k, PermP, PermQ)
end

function applyBF_Mats(t::BF_Mats, v::Vector{ComplexF64})
    y = v[t.PermQ]  #permute input vector according to Q blocks
    y = t.Q * y
    for R_block in t.R
        y = R_block * y
    end
    y = t.P * y
    y_out = zeros(ComplexF64, length(v))
    y_out[t.PermP] = y  #permute output vector according to P blocks
    return y_out
end

function apply_butterflyh2(H2Blocktree, Butterfly::BF, v::Vector{ComplexF64})
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NO = Butterfly.NO
    NS = Butterfly.NS

    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    children = H2Trees.children

    treeS = h2treelevels(trialT, NS)
    treeO = h2treelevels(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO

    # coefficients[Svert][Overt] = Vector
    coefficients = Dict{Int,Dict{Int,Vector{ComplexF64}}}()

    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------
    # Leaf initialization
    # ------------------------------------------------------------
    for Sleaf in treeS[LS]
        srcvals = values(trialT, Sleaf)
        #isempty(srcvals) && continue
        #getsubdict!(coefficients, Sleaf)[NO] = Q[Sleaf] * v[srcvals]
        getsubdict!(coefficients, Sleaf)[NO] = Vector{ComplexF64}(undef, size(Q[Sleaf])[1])
        @views mul!(getsubdict!(coefficients, Sleaf)[NO], Q[Sleaf], v[srcvals])
    end

    # ------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------
    for l in 1:(L - 1)
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)

        if !source_is_frozen && !obs_is_frozen
            for Svert in treeS[LS - l]
                coeff_S = getsubdict!(coefficients, Svert)

                # ---- upward aggregation (source side) ----
                for Overt in treeO[l]
                    temp = ComplexF64[]

                    for Schild in children(trialT, Svert)
                        #inner = get(coefficients, Schild, nothing)
                        #coeffs_child =
                        #    inner === nothing ? nothing : get(inner, Overt, nothing)
                        #coeffs_child === nothing && continue
                        coeffs_child = coefficients[Schild][Overt]
                        append!(temp, coeffs_child)
                    end

                    coeff_S[Overt] = temp
                end

                # ---- downward projection (observer side) ----
                for Overt in treeO[l]
                    coeff_SO = coeff_S[Overt]

                    for Ochild in children(testT, Overt)
                        #inner = get(R, Svert, nothing)
                        #R_check = inner === nothing ? nothing : get(inner, Ochild, nothing)
                        #R_check === nothing && continue
                        R_check = R[Svert][Ochild]
                        coeff_S[Ochild] = Vector{ComplexF64}(undef, size(R_check)[1])
                        @views mul!(coeff_S[Ochild], R_check, coeff_SO)
                        #coeff_S[Ochild] = R[Svert][Ochild] * coeff_SO
                    end
                end
            end

        elseif source_is_frozen && !obs_is_frozen
            #@show l
            for Svert in treeS[1]
                coeff_S = getsubdict!(coefficients, Svert)

                for Overt in treeO[LS]
                    coeff_SO = coeff_S[Overt]

                    for Ochild in h2treelevels(testT, Overt)[l - LS + 2]
                        #inner = get(R, Svert, nothing)
                        #R_check = inner === nothing ? nothing : get(inner, Ochild, nothing)
                        #R_check === nothing && continue
                        R_check = R[Svert][Ochild]
                        coeff_S[Ochild] = Vector{ComplexF64}(undef, size(R_check)[1])
                        @views mul!(coeff_S[Ochild], R_check, coeff_SO)
                        #coeff_S[Ochild] = R[Svert][Ochild] * coeff_SO
                    end
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Svert in treeS[LS - l]
                coeff_S = getsubdict!(coefficients, Svert)

                # ---- upward aggregation ----
                for Overt in treeO[LO]
                    temp = ComplexF64[]

                    for Schild in children(trialT, Svert)
                        #inner = get(coefficients, Schild, nothing)
                        #coeffs_child =
                        #    inner === nothing ? nothing : get(inner, Overt, nothing)
                        #coeffs_child === nothing && continue
                        coeffs_child = coefficients[Schild][Overt]
                        append!(temp, coeffs_child)
                    end

                    coeff_S[Overt] = temp
                end

                # ---- frozen observer projection ----
                for Overt in treeO[LO]
                    #inner = get(R, Svert, nothing)
                    #R_check = inner === nothing ? nothing : get(inner, Overt, nothing)
                    #R_check === nothing && continue
                    R_check = R[Svert][Overt]
                    temp = coeff_S[Overt]
                    coeff_S[Overt] = Vector{ComplexF64}(undef, size(R_check)[1])
                    @views mul!(coeff_S[Overt], R_check, temp)
                    #coeff_S[Overt] = R[Svert][Overt] * coeff_S[Overt]
                end
            end

        else
            break
        end
    end

    # ------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------
    rootvals = values(testT, H2Trees.root(testT))
    #result = Vector{ComplexF64}(undef, length(rootvals))
    result = zeros(ComplexF64, length(rootvals))
    if LS >= LO
        for Oleaf in treeO[LO]
            #result[values(testT, Oleaf)] = P[Oleaf] * coefficients[NS][Oleaf]
            inds = values(testT, Oleaf)
            #isempty(inds) && continue
            dest = @view result[inds]
            mul!(dest, P[Oleaf], coefficients[NS][Oleaf])
        end
    else
        for Oleaf in treeO[LO]
            inds = values(testT, Oleaf)
            #isempty(inds) && continue
            result[inds] = coefficients[NS][Oleaf]
        end
    end

    return result
end

function nearinteraction(farassembler, H2Blocktree, LO, LS)
    row = H2Trees.values(H2Trees.testtree(H2Blocktree), LO)
    col = H2Trees.values(H2Trees.trialtree(H2Blocktree), LS)
    p = zeros(ComplexF64, length(row), length(col))
    farassembler(p, row, col)
    return p
end

function assemble_Bfly(farassembler, H2Blocktree, k, τ, α)
    nearints, farints = nearandfar(H2Blocktree, α)
    # --------------------------------------------------
    # 2. Near-field contributions
    # --------------------------------------------------
    NI = Vector{Matrix{ComplexF64}}()
    #i = 1
    for (NO, source_nodes) in nearints
        for LS in source_nodes
            push!(NI, nearinteraction(farassembler, H2Blocktree, NO, LS))
            #i += 1
        end
    end
    # --------------------------------------------------
    # 3. Far-field (butterfly) contributions
    # --------------------------------------------------
    fly = Vector{BF}()
    #i = 1
    for (NO, source_nodes) in farints
        for NS in source_nodes
            push!(fly, subroutine_BF_approx_treeh2(farassembler, H2Blocktree, NO, NS, k, τ))
            #i += 1
        end
    end
    Butter = BFapprox(H2Blocktree, nearints, farints, fly, NI)
    return Butter
end

function apply_Butterfly(Butter::BFapprox, v::AbstractVector{ComplexF64})
    # --------------------------------------------------
    # 1. Get near / far interaction maps
    # --------------------------------------------------
    nearints, farints = Butter.nearinteractions, Butter.farinteractions
    tsttree = H2Trees.testtree(Butter.tree)
    srctree = H2Trees.trialtree(Butter.tree)

    y = zeros(ComplexF64, length(v))
    # --------------------------------------------------
    # 2. Near-field contributions
    # --------------------------------------------------
    i = 1
    for (NO, source_nodes) in nearints
        row_idx = H2Trees.values(tsttree, NO)

        for LS in source_nodes
            col_idx = H2Trees.values(srctree, LS)
            y[row_idx] += Butter.NI[i] * v[col_idx]
            i += 1
        end
    end

    # --------------------------------------------------
    # 3. Far-field (butterfly) contributions
    # --------------------------------------------------
    i = 1
    for (NO, source_nodes) in farints
        for NS in source_nodes
            y += apply_butterflyh2(Butter.tree, Butter.BFs[i], v)
            i += 1
        end
    end

    return y
end

function h2treelevels(tree::H2Trees.TwoNTree, root::Int64)
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
