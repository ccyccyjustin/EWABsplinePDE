import numpy as np
import pandas as pd

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


def orthogonalize_basis(G, omega):
    """
    Lets LAPACK handle the GEVP using the precomputed G matrix.
    """

    # 1. Solve GEVP directly (LAPACK's dsygvd handles Cholesky and symmetry in C/Fortran)
    if sp.issparse(omega):
        omega = omega.toarray()
    if sp.issparse(G):
        G = G.toarray()

    s_raw, V = eigh(omega, G)

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


def gsj_est(X, y):
    """
    Gasser-Sroka-Jennrich (GSJ) Noise Variance Estimator for unstructured 2D/3D data.
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
        num = ((1.0 - lam_vec)[np.newaxis, :] * (n * d[:, np.newaxis])
               + lam_vec[np.newaxis, :] * force_proj[:, np.newaxis])
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


current_module = sys.modules[__name__]

# Find all functions defined in this file and wrap them
for name, obj in inspect.getmembers(current_module, inspect.isfunction):
    # Only wrap functions actually defined in this module (not imported ones)
    if obj.__module__ == __name__ and name != 'time_it':
        setattr(current_module, name, time_method(obj))