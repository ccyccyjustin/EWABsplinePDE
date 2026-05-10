import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.linalg import lstsq
import scipy.linalg as la


def project_to_bspline(y, x, s=0, k=3):
    tck = splrep(x, y, s=s, k=k)
    y_fit = splev(x, tck)
    return y_fit


def form_omega(a, b, int_knots, deriv_order=2, deg=3):
    """
    Computes the roughness penalty matrix Omega using Simpson's rule.
    Uses the functional scipy.interpolate.splev interface.
    """
    q = int(deriv_order)
    p = int(deg)

    if q < 0 or q > p:
        raise ValueError(f"deriv_order q={q} must satisfy 0 <= q <= deg p={p}")

    # 1. Construct the full knot vector
    all_knots = np.r_[np.repeat(a, p + 1), int_knots, np.repeat(b, p + 1)]
    n_basis = len(all_knots) - p - 1

    # 2. Quadrature Setup (Simpson's Rule)
    intervals = np.diff(all_knots)
    n_int = len(intervals)

    # Evaluation points (xtilde) and Simpson weights (wts)
    xtilde = np.repeat(all_knots[:-1], 3) + np.tile([0, 0.5, 1], n_int) * np.repeat(intervals, 3)
    wts = np.repeat(intervals, 3) * np.tile([1 / 6, 4 / 6, 1 / 6], n_int)

    # 3. Construct Design Matrix of Derivatives using splev
    Bdd = np.zeros((len(xtilde), n_basis))
    for i in range(n_basis):
        # Create a one-hot coefficient vector for the i-th basis function
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        # Evaluate the q-th derivative
        Bdd[:, i] = splev(xtilde, (all_knots, coeffs, p), der=q) # TODO: Change to PDE Loss

    # 4. Weighted Inner Product (Omega = Bdd^T * W * Bdd)
    # This computes integral[ B_i^(q)(x) * B_j^(q)(x) dx ]
    return (Bdd.T * wts) @ Bdd # TODO: Normalization needed?

def orthogonalize_basis(B, omega):
    """
    Stable basis orthogonalization. Returns U, V, s such that:
    - U = B @ V
    - U.T @ U = Identity
    - V.T @ omega @ V = diag(s)
    """
    G = B.T @ B
    n_col = G.shape[1]
    # Small ridge for numerical stability
    eps = np.sqrt(np.finfo(float).eps) * np.trace(G) / n_col

    # Cholesky decomposition of Grammian: G = R.T @ R
    R = la.cholesky(G + eps * np.eye(n_col))

    # Solve R.T @ M @ R = omega  => M = Rinv.T @ omega @ Rinv
    # Use solve_triangular for Rinv terms
    Rinv = la.solve_triangular(R, np.eye(n_col))
    M = Rinv.T @ omega @ Rinv

    # Eigendecomposition of M
    values, vectors = la.eigh(M)
    s = np.maximum(values, 0)
    V = Rinv @ vectors
    U = B @ V

    return {"U": U, "V": V, "s": s}


def gsj_est(y):
    """
    Gasser-Sroka-Jennrich (GSJ) Noise Variance Estimator.
    Computes a differencing estimator for sigma^2.
    """
    y = np.asarray(y)
    n = len(y)
    # GSJ coefficients for a 3-point window
    # c = [0.4745, -0.9489, 0.4745] (approx)
    # Formally: c_i are weights that minimize variance while annihilating linear trends
    c = np.array([1 / np.sqrt(6), -2 / np.sqrt(6), 1 / np.sqrt(6)])

    # Convolve to get differences
    res = np.convolve(y, c, mode='valid')
    sigma_sq = np.sum(res ** 2) / (n - 2)
    return np.sqrt(sigma_sq)


def compute_sure(y, fhatps, lam_vec, s, sig2):
    """
    Vectorized computation of Stein's Unbiased Risk Estimate (SURE).
    """
    # 1. Residual Sum of Squares (RSS) for every lambda
    # fhatps is (n, n_lambda), y is (n,)
    # We compute col-wise: sum((y - fhat_j)^2)
    residuals = y[:, np.newaxis] - fhatps
    rss = np.sum(residuals ** 2, axis=0)

    # 2. Effective Degrees of Freedom (df) for every lambda
    # s is (p,), lam_vec is (n_lambda,)
    # df_j = sum_k( 1 / (1 + s_k * lam_j) )
    shrinkage_matrix = 1.0 / (1.0 + s[:, np.newaxis] * lam_vec[np.newaxis, :])
    df = np.sum(shrinkage_matrix, axis=0)

    # 3. SURE calculation
    return rss + 2.0 * sig2 * df


def ewa_weights(sure, prior, gamma):
    """
    Log-sum-exp stabilized EWA weights.
    """
    # Use log space to prevent underflow/overflow
    log_w = -sure / gamma + np.log(prior)
    log_w -= np.max(log_w)  # Stabilization
    w = np.exp(log_w)
    return w / np.sum(w)


def fit_ewa_helper(U, y, s, lam_vec, prior, gamma_c, sig2_est):
    """
    1. ridge shrinkage
    2. SURE
    3. weights & EWA
    Highly optimized for large N (rows of U).
    """
    n, p = U.shape
    n_lambda = len(lam_vec)
    n_gamma = len(gamma_c)

    # 1. Project y into the orthogonal space: U'y (Size: p)
    Uty = U.T @ y

    # 2. Compute Ridge Path (fhatps)
    # shrinkage_matrix is (p, n_lambda)
    shrinkage_matrix = 1.0 / (1.0 + s[:, np.newaxis] * lam_vec[np.newaxis, :])

    # coefficients in U-space: (p, n_lambda)
    # We multiply Uty (p,) across columns of shrinkage_matrix
    alpha_hat = shrinkage_matrix * Uty[:, np.newaxis]

    # fhatps is (n, n_lambda). Memory Warning: 10M x 100 is ~8GB
    fhatps = U @ alpha_hat

    # 3. Compute SURE for all lambdas
    sure_est = compute_sure(y, fhatps, lam_vec, s, sig2_est)

    # 4. EWA weights and fitted values for each gamma_c
    weights = np.zeros((n_gamma, n_lambda))
    fhat_ewa = np.zeros((n, n_gamma))

    for g in range(n_gamma):
        gamma = gamma_c[g] * sig2_est
        w = ewa_weights(sure_est, prior, gamma)
        weights[g, :] = w
        # Combine fitted values using EWA weights: (n, n_lambda) @ (n_lambda,)
        fhat_ewa[:, g] = fhatps @ w

    return {
        "fhatps": fhatps,
        "lam_used": lam_vec,
        "sure_est": sure_est,
        "weights": weights,
        "fhat_ewa": fhat_ewa
    }

def fit_ewa(y, ftrue, U, V, s, lam, prior, # B_s, a, b, int_knots deg=3, deriv_order=2
            sig_eps_est, gamma_mult=8):
    """
    Implementation of the EWA spline fitting logic.
    """

    # 2. Precompute projections in orthogonal space
    gamma_mult = np.atleast_1d(gamma_mult)
    fit_res = fit_ewa_helper(U, y, s, lam, prior, gamma_mult, sig2_est=sig_eps_est ** 2)
    fhat_ewa_mat = fit_res['fhat_ewa']

    # Calculate MSE if ftrue is provided (matching R's placeholder logic)
    mse_est = {}
    for i, gm in enumerate(gamma_mult):
        mse_est[f"EWA_sigEst_g{gm}"] = np.mean((fhat_ewa_mat[:, i] - ftrue) ** 2)

    # Find best lambda per gamma_mult
    all_weights_np = fit_res['weights'] #np.array(all_weights)
    #print(f'all_weights:{all_weights_np.shape}')

    # Returns a scalar if only one gamma_mult is used, otherwise a list/array
    best_indices = np.argmax(all_weights_np, axis=1)
    assert len(best_indices) == 1, "Mutiple lam_best found!"
    lam_best = lam[best_indices[0]]

    return {
        "mse_est": mse_est, # OK!
        "fhatps": fit_res['fhatps'], # OK!
        "lam_used": lam, # OK!
        "lam_best": lam_best,  # OK!
        "sure_est": fit_res['sure_est'], # OK!
        "weights": all_weights_np.flatten(), # OK!
    }