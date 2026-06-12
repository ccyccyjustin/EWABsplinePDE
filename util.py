import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.linalg import lstsq
import scipy.linalg as la
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.stats import norm, multivariate_normal


def parallel(func, iterable, n_jobs=-1, disable_par=False, **kwargs):
    if disable_par:
        results = {i: func(i, **kwargs) for i in tqdm(iterable)}
        return results

    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(i, **kwargs) for i in tqdm(iterable)
    )
    results = {k: v for k, v in zip(iterable, results)}
    return results


def project_to_bspline(y, x, s=0, k=3):
    tck = splrep(x, y, s=s, k=k)
    y_fit = splev(x, tck)
    return y_fit


import numpy as np
from scipy.interpolate import BSpline

import numpy as np
from scipy.interpolate import BSpline


def compute_spline_integral_gl(all_knots, u, v, deg, x_eval=None):
    """
    Optimized vector-constructed calculation of the 1D spline derivative integral.
    Handles non-uniform knot sequences exactly using vectorized Gauss-Legendre quadrature.

    Parameters:
    -----------
    all_knots : array-like
        The complete knot vector (including padded boundary knots).
    u, v : int
        Derivative orders.
    deg : int
        Spline polynomial degree.
    x_eval : array-like, optional
        Custom evaluation points or interval boundaries. If None, integrates over
        the entire internal knot domain.
    """
    # 1. Setup full padded knot vector
    n_basis = len(all_knots) - deg - 1

    # 2. Extract unique knot boundaries to identify individual spans
    internal_knots = all_knots[deg:-deg]

    if x_eval is None:
        unique_knots = internal_knots
    else:
        x_eval = np.asarray(x_eval)
        # Restrict internal knots to the range of x_eval to avoid creating empty outer spans
        x_min, x_max = x_eval.min(), x_eval.max()
        relevant_knots = internal_knots[(internal_knots >= x_min) & (internal_knots <= x_max)]

        # The union ensures we respect both x_eval intervals and spline knot boundaries
        unique_knots = np.unique(np.concatenate([x_eval, relevant_knots]))

    x_L = unique_knots[:-1]
    x_R = unique_knots[1:]

    # 3. Generate baseline Gauss-Legendre quadrature points on [-1, 1]
    n_quad = max(1, deg + 1)
    quad_x, quad_w = np.polynomial.legendre.leggauss(n_quad)

    # 4. Vectorized scaling transformation to local non-uniform intervals
    widths = x_R - x_L
    midpoints = x_R + x_L

    x_pts = 0.5 * widths[:, None] * quad_x[None, :] + 0.5 * midpoints[:, None]
    w_pts = 0.5 * widths[:, None] * quad_w[None, :]

    # Flatten arrays to perform a single unified evaluation pass
    x_flat = x_pts.ravel()
    w_flat = w_pts.ravel()

    # 5. Continuous evaluation of all basis functions simultaneously
    c = np.eye(n_basis)
    spl = BSpline(all_knots, c, deg)  # TODO:, extrapolate=False)

    vals_u = spl(x_flat, nu=u)
    vals_v = vals_u if u == v else spl(x_flat, nu=v)

    # Clean up any NaN limits produced precisely at the right domain boundary 'b'
    # vals_u = np.nan_to_num(vals_u)
    # vals_v = np.nan_to_num(vals_v)

    # 6. Contract across the flattened quadrature dimensions
    omega = np.einsum('k,ki,kj->ij', w_flat, vals_u, vals_v)
    return omega

def compute_spline_integral_simp(all_knots, u=2, v=2, deg=3):
    """
    Computes the integral of the product of the u-th and v-th derivatives
    of B-spline basis functions: integral[ B_i^(u)(x) * B_j^(v)(x) dx ]
    """
    p = int(deg)
    u, v = int(u), int(v)

    if max(u, v) > p:
        raise ValueError(f"Derivative orders ({u}, {v}) cannot exceed degree {p}")

    # 1. Construct full knot vector
    n_basis = len(all_knots) - p - 1

    # 2. Quadrature Setup (Filter zero-length intervals)
    # Generate quadrature points and Simpson weights
    intervals = np.diff(all_knots)
    xtilde = np.repeat(all_knots[:-1], 3) + np.tile([0, 0.5, 1], len(intervals)) * np.repeat(intervals, 3)
    wts = np.repeat(intervals, 3) * np.tile([1 / 6, 4 / 6, 1 / 6], len(intervals))

    # 3. Construct Design Matrices for both derivative orders
    # Bu: (n_quad_points x n_basis) matrix of u-th derivatives
    # Bv: (n_quad_points x n_basis) matrix of v-th derivatives
    Bu = np.zeros((len(xtilde), n_basis))
    Bv = np.zeros((len(xtilde), n_basis))

    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        tck = (all_knots, coeffs, p)

        Bu[:, i] = splev(xtilde, tck, der=u)
        Bv[:, i] = splev(xtilde, tck, der=v)

    # 4. Weighted Inner Product
    # Equivalent to: Omega_ij = sum_k [ Bu_ik * Bv_jk * wts_k ]
    # Using matrix notation: Bu^T @ diag(W) @ Bv
    Omega_uv = (Bu.T * wts) @ Bv
    return Omega_uv

def orthogonalize_basis_robust(B, omega, tol=1e-10):
    """
    Robust simultaneous diagonalization handling singular/near-singular metric G = B^T @ B.
    Uses subspace projection to explicitly remove the null space before diagonalizing.
    """
    # 1. Form the overlap matrix (Metric)
    G = B.T @ B

    # 2. Diagonalize G to identify its principal components and null space
    # Since G is symmetric positive semi-definite, standard eigh is safe
    evals_G, evecs_G = np.linalg.eigh(G)

    # 3. Filter out the near-singular (unstable) subspace
    # We scale the tolerance by the largest eigenvalue to make it scale-invariant
    threshold = tol * np.max(evals_G)
    valid_mask = evals_G > threshold

    evals_G_r = evals_G[valid_mask]
    evecs_G_r = evecs_G[:, valid_mask]
    rank = len(evals_G_r)

    # 4. Construct the whitening transformation matrix (T) for the stable subspace
    # T projects into a space where the metric is the Identity matrix
    T = evecs_G_r @ np.diag(1.0 / np.sqrt(evals_G_r))

    # 5. Project the penalty matrix 'omega' into this whitened, well-conditioned subspace
    omega_proj = T.T @ omega @ T

    # 6. Solve the Standard (not Generalized) Eigenvalue Problem in the projected space
    s, Z = np.linalg.eigh(omega_proj)

    # 7. Map the eigenvectors back to the original coefficient space
    V = T @ Z

    # 8. Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(s)[::-1]
    s = np.maximum(s[idx], 0)
    V = V[:, idx]

    # 9. Construct the orthonormal basis U in the original space
    U = B @ V
    return {'U': U, 'V': V, 's': s, 'rank': rank}

def orthogonalize_basis(B, omega):
    """
    Optimized simultaneous diagonalization using Generalized Eigenvalue Decomposition.
    - U = B @ V
    - U.T @ U = Identity
    - V.T @ omega @ V = diag(s)
    """
    # 1. Form the overlap matrix (Metric) S = B.T @ B
    # Complexity: O(m * n^2) - This is usually the bottleneck
    G = B.T @ B
    # L = np.linalg.cholesky(G @ G.T)
    # print('Stability', np.linalg.det(L))

    # 3. Add regularization to the diagonal of G in-place to save memory
    n_col = G.shape[1]
    eps = np.sqrt(np.finfo(float).eps) * np.trace(G) / n_col
    np.fill_diagonal(G, G.diagonal() + eps)

    # 2. Solve the Generalized Eigenvalue Problem: omega @ V = G @ V @ diag(s)
    # eigh with 'b' argument returns V such that V.T @ G @ V = I
    # This automatically satisfies (B @ V).T @ (B @ V) = I
    s, V = eigh(omega, G)

    # 3. Sort eigenvalues in descending order
    idx = np.argsort(s)[::-1]
    s = s[idx]
    V = V[:, idx]
    s = np.maximum(s, 0)

    # 4. Construct the orthonormal basis U in the original space
    U = B @ V
    return {'U': U, 'V': V, 's': s}


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


def compute_sure(y, fhatps, shrinkage_matrix, sig2):
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


def fit_ewa_helper(U, y_fit, s, lam_vec, prior, gamma_c, sig2_est):
    """
    1. ridge shrinkage
    2. SURE
    3. weights & EWA
    Highly optimized for large N (rows of U).
    fhat_ps = U @ ((1.0 / (1.0 + s[:, np.newaxis] * lam_vec[np.newaxis, :])) * (U.T @ y)[:, np.newaxis ])
    fhat_ewa = fhat_ewa @ weight
    """
    # 1. Project y into the orthogonal space: U'y (Size: p)
    Uty = U.T @ y_fit

    # 2. Compute Ridge Path (fhatps)
    # shrinkage_matrix is (p, n_lambda)
    shrinkage_matrix = 1.0 / (1.0 + s[:, np.newaxis] * lam_vec[np.newaxis, :])

    # coefficients in U-space: (p, n_lambda)
    # We multiply Uty (p,) across columns of shrinkage_matrix
    alpha_hat = shrinkage_matrix * Uty[:, np.newaxis]

    # fhatps is (n, n_lambda). Memory Warning: 10M x 100 is ~8GB
    fhatps = U @ alpha_hat

    # 3. Compute SURE for all lambdas
    sure_est = compute_sure(y_fit, fhatps, shrinkage_matrix, sig2_est)

    # 4. EWA weights and fitted values for each gamma_c
    gamma = gamma_c * sig2_est
    weights = ewa_weights(sure_est, prior, gamma)

    shrink_weight = (shrinkage_matrix @ weights.T).flatten()
    A = (U * shrink_weight) @ U.T
    fhat_ewa = A @ y_fit
    theta_hat = Uty * shrink_weight

    return {
        "fhatps": fhatps,
        "lam_used": lam_vec,
        "sure_est": sure_est,
        "weights": weights,
        "fhat_ewa": fhat_ewa, # A @ Y_fit
        'theta_hat': theta_hat,
        'hat_matrix': A,
        'shrink_weight': shrink_weight,
        'Uty': Uty
    }

def fit_ewa(y, ftrue, U, V, s, lam, prior, # B_s, a, b, int_knots deg=3, deriv_order=2
            sig_eps_est, gamma_mult=8):
    """
    Implementation of the EWA spline fitting logic.
    """

    # 2. Precompute projections in orthogonal space
    fit_res = fit_ewa_helper(U, y, s, lam, prior, gamma_mult, sig2_est=sig_eps_est ** 2)

    # Calculate MSE if ftrue is provided (matching R's placeholder logic)
    mse_est = {}
    fhat_ewa = fit_res['fhat_ewa']
    mse_est[f"EWA_sigEst_g{gamma_mult}"] = np.mean((fhat_ewa - ftrue) ** 2)

    # Find best lambda per gamma_mult
    best_idx = np.argmax(fit_res['weights'])
    lam_best = lam[best_idx]

    beta = V @ fit_res['theta_hat']
    return {
        "mse_est": mse_est, # OK!
        "lam_best": lam_best,  # OK!
        "beta": beta,
        **fit_res
    }

def canberra_similarity(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Avoid division by zero for matching zeros
    denominator = np.abs(x) + np.abs(y)
    numerator = np.abs(x - y)

    # Wherever both elements are zero, the difference/distance contribution is 0
    with np.errstate(invalid='ignore'):
        terms = np.where(denominator == 0, 0.0, numerator / denominator)

    return 1.0 - np.mean(terms)

def mvn_pairplot(mu, sigma, title='MVN Density Pairplot',
                 labels=None, num_grid_points=100, one_figsize=2.2, **kwargs):
    """
    Generates a lower-triangular corner pair plot displaying the EXACT
    analytical probability density functions of a Multivariate Normal
    distribution of any dimension (no simulations or random samples used).

    Parameters:
    -----------
    mu : array-like
        The mean vector of length D.
    sigma : array-like
        The D x D covariance matrix.
    labels : list of str, optional
        Names for the parameters. Defaults to ['$X_0$', '$X_1$', ...]
    num_grid_points : int
        Resolution of the analytical grid evaluation (default 100).
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    dim = len(mu)

    if sigma.shape != (dim, dim):
        raise ValueError("Covariance matrix shape must match mean vector length.")

    if labels is None:
        labels = [f'$X_{i}$' for i in range(dim)]

    # 1. Determine the analytical evaluation limits for each axis
    # We span +/- 3.5 standard deviations around each parameter's mean
    stdevs = np.sqrt(np.diagonal(sigma))
    mins = mu - 3.5 * stdevs
    maxs = mu + 3.5 * stdevs

    # 2. Set up the plotting grid matrix
    fig, axes = plt.subplots(dim, dim, figsize=(one_figsize * dim, one_figsize * dim), squeeze=False)

    for i in range(dim):
        for j in range(dim):
            ax = axes[i, j]

            # --- Case A: Upper Triangle (Unused) ---
            if j > i:
                ax.axis('off')
                continue

            # --- Case B: Diagonal (Exact Univariate Marginal PDF) ---
            if i == j:
                x_eval = np.linspace(mins[i], maxs[i], num_grid_points * 2)
                # Exact analytical 1D Gaussian density formula
                pdf_eval = norm.pdf(x_eval, loc=mu[i], scale=stdevs[i])

                ax.plot(x_eval, pdf_eval, color='midnightblue', linewidth=2)
                ax.fill_between(x_eval, pdf_eval, color='midnightblue', alpha=0.3)

                # Format diagonal axes (hide Y-axis as it's just raw density magnitude)
                ax.set_xlim(mins[i], maxs[i])
                ax.get_yaxis().set_visible(False)
                ax.grid(True, linestyle=':', alpha=0.5)

            # --- Case C: Lower Off-Diagonal (Exact Bivariate Marginal PDF) ---
            else:
                # Generate 2D evaluation grid
                x_eval = np.linspace(mins[j], maxs[j], num_grid_points)
                y_eval = np.linspace(mins[i], maxs[i], num_grid_points)
                X, Y = np.meshgrid(x_eval, y_eval)
                pos = np.dstack((X, Y))

                # Extract exact sub-mean and sub-covariance matrix analytically
                mu_sub = np.array([mu[j], mu[i]])
                sigma_sub = np.array([
                    [sigma[j, j], sigma[j, i]],
                    [sigma[i, j], sigma[i, i]]
                ])

                # Exact analytical 2D Gaussian density calculation
                mvn_sub = multivariate_normal(mean=mu_sub, cov=sigma_sub)
                Z = mvn_sub.pdf(pos)

                # Plot the smooth exact continuous density contours
                ax.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.85)
                ax.contour(X, Y, Z, levels=5, colors='midnightblue', linewidths=0.5, alpha=0.5)

                # Place a crosshair at the exact true mean point
                ax.scatter(mu[j], mu[i], color='crimson', marker='x', s=40, zorder=5)

                # Force alignment across columns and rows
                ax.set_xlim(mins[j], maxs[j])
                ax.set_ylim(mins[i], maxs[i])
                ax.grid(True, linestyle=':', alpha=0.5)

            # --- Label and Tick Cleanliness Management ---
            # Explicitly set outer labels
            if i == dim - 1:
                ax.set_xlabel(labels[j], fontsize=11, labelpad=10)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=11, labelpad=10)

            # Clean up inner ticks to prevent cluttered text overlaps
            if i != dim - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)

    plt.suptitle(title,
                 y=0.99, fontsize=13, fontweight='bold', color='midnightblue')
    plt.tight_layout()
    return fig, axes

#
# def fit_ewa_helper2(U, y_fit, s, lam_vec, prior, gamma_c, sig2_est):
#     """
#     Optimized SURE computation.
#     Completely avoids generating the (p, n_lambda) shrinkage matrix
#     and the (N, n_lambda) prediction matrix.
#     """
#     # Ensure 1D arrays to prevent accidental 2D broadcasting behaviors
#     y_fit = y_fit.ravel()
#
#     # 1. Project y into the orthogonal space: O(N * p)
#     # This is the ONLY time we interact with the massive N dimension.
#     Uty = (U.T @ y_fit).ravel()
#
#     # 2. Precompute constants independent of lambda
#     Uty_sq = Uty ** 2
#
#     # Compute base RSS: ||y_perp||^2 = ||y||^2 - ||U^Ty||^2
#     # max(0, ...) is used to safeguard against tiny negative values from float imprecision
#     rss_base = max(0.0, np.sum(y_fit ** 2) - np.sum(Uty_sq))
#
#     n_lam = len(lam_vec)
#     sure_est = np.zeros(n_lam)
#
#     # 3. Compute SURE for each lambda using a low-memory loop: O(p * n_lambda)
#     for j, lam in enumerate(lam_vec):
#         # Shrinkage vector for current lambda ONLY (Size: p)
#         # S_j = 1 / (1 + s * lam)
#         shrinkage_j = 1.0 / (1.0 + s * lam)
#
#         # (1 - S_j)
#         shrink_inv_j = 1.0 - shrinkage_j
#
#         # RSS_j = ||y_perp||^2 + sum( (Uty * (1 - S_j))^2 )
#         rss_j = rss_base + np.sum(Uty_sq * (shrink_inv_j ** 2))
#
#         # Effective degrees of freedom: sum( S_j )
#         df_j = np.sum(shrinkage_j)
#
#         # Compute final SURE for this lambda
#         sure_est[j] = rss_j + 2.0 * sig2_est * df_j
#
#     return sure_est


def tensor_prod(*vectors):
    n = len(vectors)
    lengths = [len(v) for v in vectors]
    dtype = np.result_type(*vectors)

    # 1. Allocate a single contiguous block of memory of shape (L1, L2, ..., Ln, n)
    out = np.empty((*lengths, n), dtype=dtype)

    # 2. Dynamically reshape each vector into a 1D slice along its specific axis
    for i, v in enumerate(vectors):
        # Create a shape tuple filled with 1s: [1, 1, ..., 1]
        broadcast_shape = [1] * n
        # Inject the vector's true length into its designated axis
        broadcast_shape[i] = lengths[i]

        # Pass the reshaped vector to out[..., i] to let C-level loops handle broadcasting
        out[..., i] = v.reshape(broadcast_shape)

    # 3. Flatten the spatial axes into a single axis, leaving the last dimension intact
    return out.reshape(-1, n)