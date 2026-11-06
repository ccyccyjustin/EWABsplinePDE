import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from joblib import Parallel, delayed
from tqdm import tqdm

from scipy.linalg import eigh
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from scipy.stats import norm, multivariate_normal
from scipy.spatial import cKDTree
from scipy.optimize import lsq_linear

import time
import inspect
from functools import wraps
import sys

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CMAP = sns.diverging_palette(20, 220, as_cmap=True)


def parallel(func, iterable, n_jobs=-1, disable_par=False, **kwargs):
    def _worker_task(func, *args, **kwargs):
        """Executes inside each worker process to silence logs at the source."""
        logging.disable(logging.INFO)
        return func(*args, **kwargs)

    if disable_par:
        results = {i: func(i, **kwargs) for i in tqdm(iterable)}
        return results

    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker_task)(func, i, **kwargs) for i in tqdm(iterable)
    )
    results = {k: v for k, v in zip(iterable, results)}
    return results


def orthogonalize_basis_fast(G_perturb, omega):
    """
    Lets LAPACK handle the GEVP using the precomputed G matrix.
    This entirely avoids Python-level dense matrix multiplication overhead.
    """

    # 1. Solve GEVP directly (LAPACK's dsygvd handles Cholesky and symmetry in C/Fortran)
    if sp.issparse(omega):
        omega = omega.toarray()
    if sp.issparse(G_perturb):
        G_perturb = G_perturb.toarray()

    s_raw, V = eigh(omega, G_perturb)
    # sum_s = np.trace(np.linalg.solve(G_perturb, omega))
    # print(f'Eigvals explained: {s_raw.sum() / sum_s}')

    idx2 = np.argsort(s_raw)
    s_raw = s_raw[idx2]
    V = V[:, idx2]
    return s_raw, V


def compute_smallest_eigenvectors(penalty_matrix, gram, k, sigma=-1e-4, tol=1e-9):
    evals, evecs = spla.eigsh(
        A=penalty_matrix,
        k=k,
        M=gram,
        sigma=sigma,
        which='LM',
        tol=tol
    )
    return evals, evecs


def compute_smallest_eigenvectors_cg(penalty_matrix, gram, k):
    N = penalty_matrix.shape[0]

    from scipy.sparse.linalg import spilu, LinearOperator, lobpcg
    # 1. Provide an Incomplete LU / Cholesky Preconditioner
    # An ILU with drop_tol=1e-3 computes in ~0.05s and cuts condition number by 1000x
    ilu = spilu((penalty_matrix + 1.0 * gram), drop_tol=1e-3)
    M_precond = LinearOperator((N, N), matvec=lambda x: ilu.solve(x))

    # 2. Smooth Initialization (CRITICAL)
    # Generate random vectors, but smooth them through the preconditioner once
    rng = np.random.default_rng(42)
    X_raw = rng.standard_normal((N, k))
    X_smooth = ilu.solve(X_raw) # Filters out knot-scale noise instantly

    # 3. Run LOBPCG with preconditioner and higher iteration ceiling
    evals, evecs = lobpcg(
        A=penalty_matrix,
        X=X_smooth,
        B=gram,
        M=M_precond, # Preconditioner makes CG steps effective
        largest=False,
        tol=1e-4,
        maxiter=400
    )
    # eigenfunctions = scipy.linalg.eigh(penalty_matrix, self.gram)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


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


def compute_sure(y, fhatps, df, sig2):
    """
    Vectorized computation of Stein's Unbiased Risk Estimate (SURE).
    """
    # TODO: Ensure fhatps = shrinkage_matrix @ y + b
    # 1. Residual Sum of Squares (RSS) for every lambda
    # fhatps is (n, n_lambda), y is (n,)
    # We compute col-wise: sum((y - fhat_j)^2)
    residuals = y[:, np.newaxis] - fhatps
    rss = np.sum(residuals ** 2, axis=0)
    return rss + 2.0 * sig2 * df


def compute_sure_spectral(s, d, force_proj, y_fit, lam_vec, n, weight):
    w_tilde = 0.0 if force_proj is None else force_proj
    delta = (s * d) - w_tilde

    rss_spectral = n * np.sum(((lam_vec[np.newaxis, :] * delta[:, np.newaxis]) * weight) ** 2, axis=0)

    # Orthogonal complement: ||y_perp||^2 = ||y||^2 - n * ||d||^2
    y_perp_sq = np.sum(y_fit ** 2) - n * np.sum(d ** 2)
    rss_exact = rss_spectral + y_perp_sq

    # Exact Trace (Effective Degrees of Freedom)
    omega_matrix = (1.0 - lam_vec)[np.newaxis, :] * n * weight
    trace_exact = np.sum(omega_matrix, axis=0)
    return rss_exact, trace_exact


def ewa_weights(sure, prior=1.0, gamma=8.0):
    """
    Log-sum-exp stabilized EWA weights respecting the input prior distribution.
    """
    assert gamma >= 1e-10, "EWA gamma too small"
    log_prior = np.log(np.maximum(prior, 1e-300))
    log_w = -sure / gamma + log_prior
    log_w -= np.max(log_w)  # Stabilization
    w = np.exp(log_w)
    return w / np.sum(w)


def fit_ewa_helper(y_fit, force_proj, U, V, s,
                   lam_vec, prior,
                   gamma_mult, sigma_est
                ):
    """
    1. ridge shrinkage
    2. SURE
    3. weights & EWA
    Highly optimized for large N (rows of U).
    """
    # 1. Project y into the orthogonal space: U'y (Size: p)
    # 2. Compute Ridge Path (fhatps)
    # shrinkage_matrix is (p, n_lambda)
    assert (lam_vec.min() >= 0) and (lam_vec.max() <= 1)
    sig2 = sigma_est ** 2

    n = len(y_fit)
    d = U.T @ y_fit / n

    # Compute alpha
    denom = (1.0 - lam_vec)[np.newaxis, :] * n + lam_vec[np.newaxis, :] * s[:, np.newaxis]
    shrinkage_matrix = 1.0 / denom
    assert shrinkage_matrix.min() > 0
    assert np.all(s[:-1] <= s[1:])

    if force_proj is not None:
        num = (1.0 - lam_vec)[np.newaxis, :] * (n * d[:, np.newaxis]) + lam_vec[np.newaxis, :] * force_proj[:, np.newaxis]
    else:
        num = (1.0 - lam_vec)[np.newaxis, :] * (n * d[:, np.newaxis])

    alpha_hat = shrinkage_matrix * num

    # Compute y_fit and df
    omega_matrix = shrinkage_matrix * (1 - lam_vec) * n

    # 4. EWA weights and fitted values for each gamma_c
    # compute_sure(y=y_fit, fhatps=fhatps, df=df, sig2=sig2)
    rss_exact, trace_exact = compute_sure_spectral(s, d, force_proj, y_fit, lam_vec, n, shrinkage_matrix)
    sure_est = rss_exact + 2.0 * sig2 * trace_exact
    gamma = gamma_mult * sig2
    weights = ewa_weights(sure_est, prior, gamma)

    alpha_ewa = alpha_hat @ weights
    beta_ewa = V @ alpha_ewa
    fhat_ewa = U @ alpha_ewa

    shrink_weight = (omega_matrix @ weights.T).flatten()
    trace_hat = np.sum(shrink_weight)

    return {
        'gamma': gamma,
        "lam_used": lam_vec,
        "sure_est": sure_est,
        "weights": weights,
        "fhat_ewa": fhat_ewa, # A @ Y_fit
        'alpha_hat': alpha_ewa,
        'trace_hat': trace_hat,
        'shrink_weight': shrink_weight,
        'beta': beta_ewa,
        'omega_matrix': omega_matrix,
        'shrinkage_matrix': shrinkage_matrix
    }

def fit_ewa(y, ftrue, force_proj,
            U, V, s,
            lam_vec, prior,
            sigma_est, gamma_mult=8):
    """
    Implementation of the EWA spline fitting logic.
    """

    # 2. Precompute projections in orthogonal space
    fit_res = fit_ewa_helper(y, force_proj,
                             U, V, s,
                             lam_vec, prior,
                             gamma_mult, sigma_est)

    # Calculate MSE if ftrue is provided (matching R's placeholder logic)
    mse_est = {}
    fhat_ewa = fit_res['fhat_ewa']
    mse_est[f"EWA_sigEst_g{gamma_mult}"] = np.mean((fhat_ewa - ftrue) ** 2)

    # Find best lambda per gamma_mult
    best_idx = np.argmax(fit_res['weights'])
    lam_best = lam_vec[best_idx]

    # beta = V @ fit_res['theta_hat']
    rss = np.sum((fhat_ewa - y) ** 2)
    sigma_est = np.sqrt(rss / (len(y) - fit_res['trace_hat']))

    return {
        "mse_est": mse_est,
        "lam_best": lam_best,
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


def gaussian_crps(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Computes the analytical Continuous Ranked Probability Score (CRPS)
    for Gaussian predictive distributions N(mu, sigma^2).
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (y_true - mu) / sigma_safe
    crps = sigma_safe * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def mvn_pairplot(mu, sigma, true_vals=None, title='MVN Density Pairplot',
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
    if true_vals is not None:
        assert len(true_vals) == dim
        assert np.isfinite(true_vals).all()

    if sigma.shape != (dim, dim):
        raise ValueError("Covariance matrix shape must match mean vector length.")

    if labels is None:
        labels = [f'$X_{i}$' for i in range(dim)]

    # 1. Determine the analytical evaluation limits for each axis
    # We span +/- 3.5 standard deviations around each parameter's mean
    stdevs = np.sqrt(np.diagonal(sigma))
    mins = mu - 3.5 * stdevs
    maxs = mu + 3.5 * stdevs
    if true_vals is not None:
        true_vals = np.asarray(true_vals, dtype=float).ravel()
        mins = np.minimum(mins, true_vals - 1.5 * stdevs)
        maxs = np.maximum(maxs, true_vals + 1.5 * stdevs)

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
                pdf_eval = norm.pdf(x_eval, loc=mu[i], scale=stdevs[i])
                ax.plot(x_eval, pdf_eval, color='midnightblue', linewidth=2)
                ax.fill_between(x_eval, pdf_eval, color='midnightblue', alpha=0.3)
                ax.axvline(mu[i], color='darkorange', ls='--', lw=1.5, label='Estimate')
                if true_vals is not None:
                    ax.axvline(true_vals[i], color='crimson', ls='-', lw=1.8, label='True $\\theta^*$')

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
                mvn_sub = multivariate_normal(mean=mu_sub, cov=sigma_sub, allow_singular=True)
                Z = mvn_sub.pdf(pos)

                # Plot the smooth exact continuous density contours
                ax.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.85)
                ax.contour(X, Y, Z, levels=5, colors='midnightblue', linewidths=0.5, alpha=0.5)
                ax.scatter(mu[j], mu[i], color='crimson', marker='x', s=40, zorder=5)
                if true_vals is not None:
                    ax.scatter(true_vals[j], true_vals[i], color='crimson', marker='x', s=50, zorder=6)

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
            if i != dim - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)

    plt.suptitle(title, y=0.99, fontsize=13, fontweight='bold', color='midnightblue')
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
    Z_plot = np.ma.masked_invalid(Z)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.contourf(X, Y, Z_plot, cmap=CMAP, norm=norm)
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


def estimate_aniso(X, y, n_neighbor=None, max_iter=100):
    """
    Estimates the expected values of beta_k^2 and s^2 under the local
    nearest-neighbor regression model.

    Time complexity:
      - Nearest Neighbors: O(N * log(N))
      - Data Preparation: O(N * d)
      - Pooled Regression: O(N * d) per iteration
    """
    assert (X >= 0).all() and (X <= 1).all()
    N, d = X.shape

    if n_neighbor is None:
        n_neighbor = int(max(min(30, N / 10 ** d), 3))

    # 1. Find nearest neighbors
    # We query n_neighbor + 1 because a point is its own nearest neighbor (dist=0)
    tree = cKDTree(X)
    _, indices = tree.query(X, k=n_neighbor + 1)
    neighbor_indices = indices[:, 1:]

    # 2. Build regression differences in strictly O(N*d) using C-level broadcasting
    # X[:, None, :] reshapes to (N, 1, d), allowing seamless subtraction against (N, n_neighbor, d)
    dx2 = (X[:, None, :] - X[neighbor_indices]) ** 2
    dy2 = (y[:, None] - y[neighbor_indices]) ** 2

    # Flatten arrays to construct the global design matrix
    dx2_flat = dx2.reshape(-1, d)
    dy2_flat = dy2.ravel()

    # 3. Formulate the linear system A * theta = b
    num_equations = dx2_flat.shape[0]
    A = np.empty((num_equations, d + 1))
    A[:, 0] = 1.0
    A[:, 1:] = dx2_flat

    result = lsq_linear(
        A,
        dy2_flat,
        bounds=(0, np.inf),
        method='trf',
        lsq_solver='lsmr',
        max_iter=max_iter
    )

    theta = result.x
    grad2 = theta[1:]

    # 1. Expectation of intercept is 2 * s^2
    # 2. Variance of residual is 8 * s^4 + 8 s^2 * E(Squared 1d-Distance in neighborhood) ||grad||^2
    s2_intercept = theta[0] / 2

    rss = np.sum((dy2_flat - A @ theta) ** 2)
    df = N * n_neighbor - (d + 1)
    mse = rss / df

    s2_mse = np.sqrt(mse / 8)
    s2 = max(s2_intercept, s2_mse)
    # s2 = s2_intercept
    y_smooth = np.mean(y[neighbor_indices], axis=1)
    return s2, grad2, y_smooth

def time_method(func):
    """The actual timer wrapper for a single function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        if end - start > 0.1:
        # func.__qualname__ will print 'ClassName.method_name'
            logger.info(f"[{func.__qualname__}] took {end - start:.6f} seconds")
        return result
    return wrapper

def time_all_methods(cls):
    """Class decorator to apply the timer to all methods."""
    # vars(cls) ensures we only wrap methods explicitly defined in THIS class,
    # preventing us from accidentally wrapping inherited methods from 'object'
    for attr_name, attr_value in vars(cls).items():
        if inspect.isfunction(attr_value):
            setattr(cls, attr_name, time_method(attr_value))
    return cls

# --- Put this at the BOTTOM of your file ---
current_module = sys.modules[__name__]

# Find all functions defined in this file and wrap them
for name, obj in inspect.getmembers(current_module, inspect.isfunction):
    # Only wrap functions actually defined in this module (not imported ones)
    if obj.__module__ == __name__ and name != 'time_it':
        setattr(current_module, name, time_method(obj))