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
