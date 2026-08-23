import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, BSpline
from scipy.linalg import lstsq
import scipy.linalg as la
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.linalg import eigh
from scipy.stats import norm, multivariate_normal
import seaborn as sns
from matplotlib.colors import Normalize, CenteredNorm
from matplotlib.cm import ScalarMappable
import pandas as pd

CMAP = sns.diverging_palette(20, 220, as_cmap=True)


def parallel(func, iterable, n_jobs=-1, disable_par=False, **kwargs):
    if disable_par:
        results = {i: func(i, **kwargs) for i in tqdm(iterable)}
        return results

    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(i, **kwargs) for i in tqdm(iterable)
    )
    results = {k: v for k, v in zip(iterable, results)}
    return results


def perturb_diag(A):
    A2 = A.copy()
    n_col = A.shape[1]
    eps = np.sqrt(np.finfo(float).eps) * np.trace(A) / n_col
    np.fill_diagonal(A2, A.diagonal() + eps)
    return A2, eps


def perturb_B_svd(B, eps):
    # Decompose the original B
    U, s, Vt = np.linalg.svd(B, full_matrices=False)

    # Add epsilon to the squared singular values, then take the square root
    s2 = np.sqrt(s ** 2 + eps)

    # Reconstruct B2
    B2 = (U * s2) @ Vt
    return B2


def orthogonalize_basis_fast(G_perturb, omega, k=None):
    """
    Lets LAPACK handle the GEVP using the precomputed G matrix.
    This entirely avoids Python-level dense matrix multiplication overhead.
    """

    # 1. Solve GEVP directly (LAPACK's dsygvd handles Cholesky and symmetry in C/Fortran)
    if k is not None:
        N = G_perturb.shape[0]
        subset_by_index = [N - k, N - 1]
    else:
        subset_by_index = None

    # s_raw, V = eigh(omega, G_perturb, subset_by_index=subset_by_index)
    #
    # # 2. Sort eigenvalues and eigenvectors in descending order
    # idx = np.argsort(s_raw)[::-1]
    # s_raw = s_raw[idx]
    # # s_raw = np.maximum(s_raw, 0)
    # V = V[:, idx]

    omega_norm = np.trace(omega) / omega.shape[0]
    G_norm = np.trace(G_perturb) / G_perturb.shape[0] # Makes V close to 1
    s_raw, V = eigh(omega / omega_norm, G_perturb / G_norm, subset_by_index=subset_by_index)

    idx2 = np.argsort(s_raw)[::-1]
    s_raw = s_raw[idx2]
    # s_raw = np.maximum(s_raw, 0)
    V = V[:, idx2]
    s_raw = s_raw * omega_norm / G_norm
    V = V / np.sqrt(G_norm)

    # 3. Construct the orthonormal basis U
    return {'V': V, 's_raw': s_raw}

def gsj_est(y):

    """
    TODO!!!!!
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


from scipy.spatial import cKDTree


def gsj_est_2d_scattered(X, y):
    """
    Nearest-neighbor variance estimator for unstructured 2D/3D data.
    X: array of shape (N, 2)
    y: array of shape (N,)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    # Build a spatial tree to quickly find nearest neighbors
    tree = cKDTree(X)

    # k=2 because the 1st nearest neighbor is the point itself (distance 0)
    _, indices = tree.query(X, k=2)
    neighbor_idx = indices[:, 1]

    # Difference between point and its closest neighbor.
    # Division by sqrt(2) normalizes the variance (since Var(e_1 - e_2) = 2*sigma^2)
    res = (y - y[neighbor_idx]) / np.sqrt(2)

    # Calculate variance
    sigma_sq = np.sum(res ** 2) / n
    return np.sqrt(sigma_sq)


def compute_sure(y, fhatps, shrinkage_matrix, sig2):
    """
    Vectorized computation of Stein's Unbiased Risk Estimate (SURE).
    """
    # TODO: Ensure fhatps = shrinkage_matrix @ y + b
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
    prior=1
    log_w = -sure / gamma + np.log(prior)
    log_w -= np.max(log_w)  # Stabilization
    w = np.exp(log_w)
    return w / np.sum(w)


def fit_ewa_helper(U, y_fit, s, lam_vec, prior, gamma_c, sig2_est, force_proj=None):
    """
    1. ridge shrinkage
    2. SURE
    3. weights & EWA
    Highly optimized for large N (rows of U).
    fhat_ps = U @ ((1.0 / (1.0 + s[:, np.newaxis] * lam_vec[np.newaxis, :])) * (U.T @ y)[:, np.newaxis ])
    fhat_ewa = fhat_ewa @ weight
    """
    # 1. Project y into the orthogonal space: U'y (Size: p)
    # 2. Compute Ridge Path (fhatps)
    # shrinkage_matrix is (p, n_lambda)
    assert (lam_vec.min() >= 0) and (lam_vec.max() <= 1)
    Uty = U.T @ y_fit
    shrinkage_matrix = 1.0 / (1.0 + (s[:, np.newaxis] - 1) * lam_vec[np.newaxis, :])
    assert shrinkage_matrix.min() > 0

    if force_proj is not None:
        num = Uty[:, np.newaxis] + lam_vec[np.newaxis, :] * (force_proj - Uty)[:, np.newaxis]
    else:
        num = (1 - lam_vec)[np.newaxis, :] * Uty[:, np.newaxis]

    alpha_hat = shrinkage_matrix * num
    # fhatps is (n, n_lambda). Memory Warning: 10M x 100 is ~8GB
    fhatps = U @ alpha_hat # y_pred

    # 4. EWA weights and fitted values for each gamma_c
    sure_est = compute_sure(y_fit, fhatps, shrinkage_matrix, sig2_est)
    gamma = gamma_c * sig2_est
    weights = ewa_weights(sure_est, prior, gamma)

    shrink_weight = (shrinkage_matrix @ weights.T).flatten()
    #A = (U * shrink_weight) @ U.T
    trace_hat = np.sum(shrink_weight)
    # fhat_ewa = (U * shrink_weight) @ Uty # A @ y_fit
    # theta_hat = Uty * shrink_weight
    fhat_ewa = fhatps @ weights
    alpha_ewa = alpha_hat @ weights

    c_diag = shrinkage_matrix @ (lam_vec * weights)
    M_diag = shrink_weight - c_diag

    return {
        "fhatps": fhatps,
        "lam_used": lam_vec,
        "sure_est": sure_est,
        "weights": weights,
        "fhat_ewa": fhat_ewa, # A @ Y_fit
        'theta_hat': alpha_ewa,
        'trace_hat': trace_hat,
        'shrink_weight': shrink_weight,
        'Uty': Uty,
        'M_diag_alpha': M_diag,
    }

def fit_ewa(y, ftrue, U, V, s, lam, prior, # B_s, a, b, int_knots deg=3, deriv_order=2
            sig_eps_est, gamma_mult=8, force_proj=None):
    """
    Implementation of the EWA spline fitting logic.
    """

    # 2. Precompute projections in orthogonal space
    fit_res = fit_ewa_helper(U, y, s, lam, prior, gamma_mult, sig2_est=sig_eps_est ** 2, force_proj=force_proj)

    # Calculate MSE if ftrue is provided (matching R's placeholder logic)
    mse_est = {}
    fhat_ewa = fit_res['fhat_ewa']
    mse_est[f"EWA_sigEst_g{gamma_mult}"] = np.mean((fhat_ewa - ftrue) ** 2)

    # Find best lambda per gamma_mult
    best_idx = np.argmax(fit_res['weights'])
    lam_best = lam[best_idx]

    beta = V @ fit_res['theta_hat']
    rss = np.sum((fhat_ewa - y) ** 2)
    sigma_est = np.sqrt(rss / (len(y) - fit_res['trace_hat']))

    return {
        "mse_est": mse_est, # OK!
        "lam_best": lam_best,  # OK!
        "beta": beta,
        'sigma_est': sigma_est,
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


def contour_plot(X, Y, Z, fig=None, ax=None, vmin=None, vmax=None, cbar=True, **kwargs):
    # 3. Plot the 2D Z array directly
    # Note: x and y tick labels can be specified using xticklabels and yticklabels
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.contourf(X, Y, Z, cmap=CMAP, norm=norm)
    if cbar:
        sm = map_color(vmin, vmax)
        fig.colorbar(sm, ax=ax)
    return fig, ax

def norm_color(vmin, vmax):
    return Normalize(vmin=vmin, vmax=vmax)

def map_color(vmin, vmax):
    norm = norm_color(vmin=vmin, vmax=vmax)
    return ScalarMappable(norm, cmap=CMAP)


def tensor_diagnostics(F_tensor, tol=1e-12):
    """
    Computes statistics for every conditional slice of an N-dimensional tensor.
    For a shape (3,5,7,9), this computes exactly 24 sets of statistics.
    """
    ndim = F_tensor.ndim
    shape = F_tensor.shape

    # Store the 24 statistics in a dictionary mapped by (Axis, Index)
    tstats = {}

    for axis in range(ndim):
        axis_length = shape[axis]
        for index in range(axis_length):

            sub_tensor = np.take(F_tensor, indices=index, axis=axis)

            # 2. Compute the requested statistics on this specific slice
            # L2 Norm (Energy of the cross-section)
            energy = np.linalg.norm(sub_tensor)
            max_amp = np.max(np.abs(sub_tensor))
            sparsity = np.sum(np.abs(sub_tensor) < tol) / sub_tensor.size

            # Store it
            tstats[(axis, index)] = {
                'energy': energy,
                'max_amp': max_amp,
                'sparsity': sparsity
            }
        # 3. Axis-level diagnostics
        # energy_variance = np.var(axis_energies) / (np.mean(axis_energies) ** 2 + 1e-16)
        # print(f"  -> Energy Coefficient of Variation across Dim {axis}: {energy_variance:.4f}")
    tstats = pd.DataFrame(tstats).T.rename_axis(['dim', 'idx'])
    return tstats

def format_latex_name(op_key, axis_names = ['x', 'y', 'z', 'w']):
    """
    Translates the raw operator tuples directly into a LaTeX math string.
    Example: ((0, 0, 2), (1, 2), (0, 1, 1, 1, 2)) -> "$u_{xxz} u_{yz} u_{xyyyz}$"
    """

    # Fallback for string names (e.g., 'u_xx')
    assert isinstance(op_key, (list, tuple))
    # Handle the base case 'u' (empty tuple)
    if len(op_key) == 0:
        return "$u$"

    # Wrap an isolated factor like (0,0,2) into ((0,0,2),) to standardize the loop

    if isinstance(op_key[0], int):
        op_key = (op_key,)

    terms = []
    for term_axes in op_key:
        if len(term_axes) == 0:
            terms.append("u")
        else:
            # Direct 1-to-1 mapping of the integer to the axis letter
            assert all(i < len(axis_names) for i in term_axes), "Axis index exceeds available axis names."
            subscript = "".join([axis_names[i] for i in term_axes])
            terms.append(f"u_{{{subscript}}}")

    return "$" + " ".join(terms) + "$"