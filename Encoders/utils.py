import numpy as np
from PDESystems.pfr_pde import PfrIpoptEnv


def flatten_primals_fields(cA, cB, cC, Tr, F, Tj):
    """
    Flatten primals in a fixed order:
      [cA(:), cB(:), cC(:), Tr(:), F(:), Tj(:)]
    where cA,cB,cC,Tr have shape (N, K+1),
          F has shape (K+1,),
          Tj has shape (n_zones, K+1).
    """
    return np.concatenate([
        cA.ravel(order="C"),
        cB.ravel(order="C"),
        cC.ravel(order="C"),
        Tr.ravel(order="C"),
        F.ravel(order="C"),
        Tj.ravel(order="C"),
    ])


def unflatten_primals_fields(vec, N, K, n_zones):
    """
    Inverse of flatten_primals_fields.
    Given a 1D vec, recover (cA,cB,cC,Tr,F,Tj) with shapes:
      - cA,cB,cC,Tr: (N, K+1)
      - F: (K+1, )
      - Tj: (n_zones, K+1)
    """
    vec = np.asarray(vec, dtype=float).ravel()
    n_t = K + 1

    n_state_block = N * n_t

    idx = 0
    cA = vec[idx:idx + n_state_block].reshape(N, n_t)
    idx += n_state_block
    cB = vec[idx:idx + n_state_block].reshape(N, n_t)
    idx += n_state_block
    cC = vec[idx:idx + n_state_block].reshape(N, n_t)
    idx += n_state_block
    Tr = vec[idx:idx + n_state_block].reshape(N, n_t)
    idx += n_state_block

    F = vec[idx:idx + n_t]
    idx += n_t

    n_tj = n_zones * n_t
    Tj = vec[idx:idx + n_tj].reshape(n_zones, n_t)
    idx += n_tj

    # Just to make sure if the size of the vector completely has been used
    assert idx == vec.size, "unflatten_primals_fields: dimension mismatch."

    return cA, cB, cC, Tr, F, Tj


def flatten_primals_structured(env: PfrIpoptEnv) -> np.ndarray:
    """
    Build a primal vector in the SAME layout used by
    flatten_primals_fields / unflatten_primals_fields:
      [cA(:), cB(:), cC(:), Tr(:), F(:), Tj(:)]
    """
    m = env.m
    N, K, Z = env.N, env.K, env.n_zones
    n_t = K + 1

    # allocate arrays
    cA = np.zeros((N, n_t))
    cB = np.zeros((N, n_t))
    cC = np.zeros((N, n_t))
    Tr = np.zeros((N, n_t))

    # states on grid (i,k)
    for i in m.I:
        ii = int(i)
        for k in m.K:
            kk = int(k)
            cA[ii, kk] = float(m.cA[i, k].value)
            cB[ii, kk] = float(m.cB[i, k].value)
            cC[ii, kk] = float(m.cC[i, k].value)
            Tr[ii, kk] = float(m.Tr[i, k].value)

    # F(k)
    F = np.zeros(n_t)
    for k in m.K:
        kk = int(k)
        F[kk] = float(m.F[k].value)

    # Tj(zone, k)
    Tj = np.zeros((Z, n_t))
    for zi in m.Z:
        z_idx = int(zi) - 1
        for k in m.K:
            kk = int(k)
            Tj[z_idx, kk] = float(m.Tj[zi, k].value)

    return flatten_primals_fields(cA, cB, cC, Tr, F, Tj)


def interpolate_primals_coarse_to_fine(
        x_coarse_vec,
        coarse_env,
        fine_env,
):
    """
    Take a coarse-grid primals vector x_coarse_vec (from PCA, AE, ...),
    decode it, and interpolate to the fine env's grid.

    Assumes:
      - same time grid: K_coarse == K_fine
      - same n_zones: coarse_env.n_zones == fine_env.n_zones
      - same dt, same L
    """
    N_c, K_c, Z_c = coarse_env.N, coarse_env.K, coarse_env.n_zones
    N_f, K_f, Z_f = fine_env.N, fine_env.K, fine_env.n_zones

    assert K_c == K_f, "For now, require same number of time steps."
    assert Z_c == Z_f, "For now, require same number of jacket zones."

    # 1) unpack coarse fields
    cA_c, cB_c, cC_c, Tr_c, F_c, Tj_c = unflatten_primals_fields(
        x_coarse_vec, N_c, K_c, Z_c
    )

    z_c = coarse_env.z_nodes  # shape (N_c,)
    z_f = fine_env.z_nodes  # shape (N_f,)
    n_t = K_c + 1

    # 2) allocate fine arrays
    cA_f = np.zeros((N_f, n_t))
    cB_f = np.zeros((N_f, n_t))
    cC_f = np.zeros((N_f, n_t))
    Tr_f = np.zeros((N_f, n_t))

    # 3) interpolate along z for each time slice
    for k in range(n_t):
        cA_f[:, k] = np.interp(z_f, z_c, cA_c[:, k])
        cB_f[:, k] = np.interp(z_f, z_c, cB_c[:, k])
        cC_f[:, k] = np.interp(z_f, z_c, cC_c[:, k])
        Tr_f[:, k] = np.interp(z_f, z_c, Tr_c[:, k])

    # 4) controls: just reuse (no spatial dependence)
    F_f = F_c.copy()
    Tj_f = Tj_c.copy()  # same zones, same time grid

    # 5) flatten into fine-vector in same [cA,cB,cC,Tr,F,Tj] order
    x_fine_vec = flatten_primals_fields(cA_f, cB_f, cC_f, Tr_f, F_f, Tj_f)
    return x_fine_vec


def nearest_pca_code(feat, feats_data, Z_data):
    """
    Given a feature vector feat (shape (3,)),
    find nearest neighbor in feats_data and return its latent code z.
    """
    feat = np.asarray(feat, dtype=float).ravel()
    diffs = feats_data - feat
    d2 = np.sum(diffs ** 2, axis=1)
    idx_nn = int(np.argmin(d2))
    z_code = Z_data[idx_nn, :]  # 1D latent vector
    return z_code, idx_nn