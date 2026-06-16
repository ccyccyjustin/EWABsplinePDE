import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, t
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import util
from util import orthogonalize_basis, gsj_est, fit_ewa, compute_spline_integral_gl, orthogonalize_basis_fast
import pandas as pd
from scipy.interpolate import BSpline, CubicSpline
from scipy.optimize import minimize_scalar, differential_evolution
import scipy
import seaborn as sns
from matplotlib.colors import Normalize, CenteredNorm
from matplotlib.cm import ScalarMappable

CMAP = sns.diverging_palette(20, 220, as_cmap=True)


def create_knots(x, df=None, knots=None, degree=3, intercept=False, boundary_knots=None):
    """
    Python implementation of R's splines::bs()
    """
    assert np.ndim(x) == 1, "support 1d now..."
    x = np.asarray(x)

    # 1. Determine Boundary Knots
    if boundary_knots is None:
        boundary_knots = [np.min(x), np.max(x)]
    else:
        boundary_knots = np.sort(boundary_knots)

    # 2. Determine Internal Knots
    if knots is None:
        if df is None:
            # Default behavior if neither df nor knots provided
            int_knots = []
        else:
            # Calculate internal knots based on degrees of freedom (quantiles)
            # R logic: n_internal_knots = df - degree - (1 if intercept else 0)
            n_int_knots = df - degree - (1 if intercept else 0)
            if n_int_knots < 0:
                raise ValueError(f"df={df} is too small for degree={degree}")

            # Place knots at quantiles within the boundary
            qs = np.linspace(0, 100, n_int_knots + 2)[1:-1]
            int_knots = np.percentile(x, qs)
    else:
        int_knots = np.sort(knots)

    # 3. Construct Augmented Knot Vector
    # We repeat the boundary knots (degree + 1) times at both ends
    all_knots = np.r_[np.repeat(boundary_knots[0], degree + 1), int_knots, np.repeat(boundary_knots[1], degree + 1)]
    return all_knots

def bs(x, all_knots, degree, intercept=False):
    # 4. Generate Basis Matrix
    basis_matrix = BSpline.design_matrix(x, all_knots, degree).toarray()

    # 5. Handle Intercept
    # R's bs() by default (intercept=FALSE) drops the first basis function
    if not intercept:
        basis_matrix = basis_matrix[:, 1:]
    return basis_matrix


def knots_data_count(X: np.ndarray, all_knots: list, deg: int):
    knot_counts = []
    for dim, knots_1d in enumerate(all_knots):
        knot_counts_1d = np.zeros(len(knots_1d) - 1 - 2 * deg)
        for i, a_i in enumerate(knots_1d[deg:-(1 + deg)]):
            b_i = knots_1d[deg + i + 1]
            knot_counts_1d[i] = ((X[:, dim] >= a_i) & (X[:, dim] < b_i)).sum()
        knot_counts.append(knot_counts_1d)
    return knot_counts

def get_deriv_order(deriv_str, variables):
    """
    Example:
    1. '' -> u
    2. '0' -> u_x
    3. '1' -> u_y
    4. '0,0' -> u_xx
    :param deriv_str:
    :param variables:
    :return:
    """
    dim = len(variables)

    # 1. Map both variable names AND valid string digits to their indices
    var_map = {v: i for i, v in enumerate(variables)}
    var_map.update({str(i): i for i in range(dim)})

    deriv_splits = []
    for token in deriv_str.replace(' ', '').split(','):
        if not token:
            continue

        if token in var_map:
            deriv_splits.append(var_map[token])
        else:
            error_val = int(token) if token.isdigit() else token
            raise ValueError(f'{error_val} not accepted when there are {dim} dimensions with variables {variables}')

    return deriv_splits

class EWABSplinePDE(BaseEstimator, RegressorMixin):
    def __init__(self, dim, deg=3, variables=None,
                 deriv_coef={}, action_coef={}, force=None, trainable=None,
                 max_iter=100, min_pct_chg=1e-4,
                 domain=None, num_int_knots=None, quantile_based=False, total_grid_size=1000,
                 sigma=None, gamma_mult=8, prior_type="light", norm_omega=False,
                 alpha=0.05, log_lamb=(-30, 20), log_lamb_step=0.01, tol=1e-10):
        """
        :param deg: degree of the 1-d Spline
        :param deriv_coef: should be a dict: for example u_xx + a^2 u_yy = 0 -> {'0,0': 1, '1,1': a**2}
        :param trainable: if not provided, assume everything except the highest deriv in 1st variable is trainable.
        :param max_iter:
        :param min_pct_chg:
        :param domain:
        :param num_int_knots:
        :param quantile_based:
        :param total_grid_size:
        :param sigma:
        :param gamma_mult:
        :param prior_type:
        :param norm_omega:
        :param alpha:
        :param log_lamb:
        :param log_lamb_step:
        :param tol:
        """
        self.deg = deg
        self.dim = dim

        if variables is not None:
            assert len(variables) == dim, "No. Variable does not match dim"
            self.variables = variables
        else:
            self.variables = list(range(dim))

        assert (len(deriv_coef) == 0) or (len(action_coef) == 0)
        if len(deriv_coef) > 0:
            deriv_keys = [get_deriv_order(k, self.variables) for k in deriv_coef.keys()]
            deriv_coef = sorted(zip(deriv_keys, deriv_coef.values()))
            self.coef_keys, self.deriv_coef_vals = zip(*deriv_coef)
            self.mode = 'PDE'
        if len(action_coef) > 0:
            action_keys = [tuple(get_deriv_order(p, self.variables) for p in pair) for pair in action_coef.keys()]
            action_coef = sorted(zip(action_keys, action_coef.values()))
            self.coef_keys, self.action_vals = zip(*action_coef)
            self.action_vals = np.array(self.action_vals)
            self.mode = 'ACTION'

        if trainable is not None:
            trainable_list = []
            for k in trainable:
                if isinstance(k, tuple):
                    k_tuple = tuple(get_deriv_order(ki, self.variables) for ki in k)
                    trainable_list.append(k_tuple)
                else:
                    trainable_list.append(get_deriv_order(k, self.variables))
            t_str = list(map(str, trainable_list))
            k_str = list(map(str, self.coef_keys))
            assert set(t_str) <= set(k_str), "Make sure trainable params are defined in deriv_coef."
            self.trainable_idx = [i for i, d in enumerate(k_str) if d in t_str]
        else:
            deriv_order = list(map(len, self.coef_keys))
            self.trainable_idx = [i for i, order in enumerate(deriv_order) if order < max(deriv_order)]

        if force is not None:
            self.force = force
        else:
            self.force = lambda s: np.zeros(s.shape[0])

        self.max_iter = max_iter
        self.min_pct_chg = min_pct_chg

        self.domain = domain
        self.num_int_knots = num_int_knots
        self.grid_size = int(total_grid_size ** (1 / self.dim))
        self.quantile_based = quantile_based
        self.sigma = sigma
        self.gamma_mult = gamma_mult
        self.prior_type = prior_type
        self.norm_omega = norm_omega
        self.alpha = alpha
        self.log_lamb = log_lamb
        self.log_lamb_step = log_lamb_step
        self.tol = tol

        self.omega_dict = {}
        self.integral_1d_cache = {}

    def generate_knots(self, X):
        n_samples = X.shape[0]
        dim = X.shape[1]

        if self.num_int_knots is None:
            # Rule of thumb for knot density
            n_knots = int(15 * n_samples ** (1 / (2 * self.deg + 3)))
        else:
            n_knots = self.num_int_knots

        self.n_knots_1d = int(n_knots ** (1 / dim))

        if self.quantile_based:
            self.int_knots_ = [np.percentile(X[:, i], np.linspace(0, 100, self.n_knots_1d + 2)[1:-1])
                               for i in range(X.shape[1])]
        else:
            self.int_knots_ = [np.linspace(*d, self.n_knots_1d + 2)[1:-1] for d in self.domain]

        self.step_size = [np.diff(knt).mean() for knt in self.int_knots_]

        self.all_knots = [create_knots(X[:, i], knots=self.int_knots_[i], degree=self.deg,
                                       intercept=True, boundary_knots=self.domain[i])
                          for i in range(X.shape[1])]  # Could have different #knots

        self.n_knots = np.array(list(map(len, self.all_knots)))
        self.knot_counts = knots_data_count(X, self.all_knots, self.deg)

    def _build_tensor_basis(self, X):
        """Constructs the tensor product of 1D B-splines."""
        B_list = [bs(X[:, i], all_knots=self.all_knots[i], degree=self.deg, intercept=True)
                  for i in range(self.dim)]
        self.n_splines = np.array(list(map(lambda s: s.shape[1], B_list)))

        B_tensor = B_list[0]
        for i in range(1, self.dim):
            # Row-wise Kronecker product (Khatri-Rao)
            B_tensor = (B_tensor[:, :, None] * B_list[i][:, None, :]).reshape(X.shape[0], -1)
        return B_tensor

    def compute_omega(self, deriv_keys, deriv_vals, x_eval=None):
        """
        First Compute u^th, and v^th order partial derivs.
        Then compute integral of their products
        Return Final weighted sum
        :param deriv_vals:
        :param x_eval:
        :return:
        """
        omega_final = None

        for u, c_u in enumerate(deriv_vals):
            d_u = np.array(deriv_keys[u])
            for v, c_v in enumerate(deriv_vals[:u + 1]):
                d_v = np.array(deriv_keys[v])

                if (u, v) not in self.omega_dict:
                    omega_uv = None
                    for p in range(self.dim):
                        d_up = (d_u == p).sum()
                        d_vp = (d_v == p).sum()

                        cache_key_1d = (p, d_up, d_vp)
                        if cache_key_1d not in self.integral_1d_cache:
                            self.integral_1d_cache[cache_key_1d] = compute_spline_integral_gl(
                                self.all_knots[p], u=d_up, v=d_vp, deg=self.deg, x_eval=x_eval
                            )

                        omega_1d = self.integral_1d_cache[cache_key_1d]

                        if omega_uv is None:
                            omega_uv = omega_1d
                        else:
                            omega_uv = np.kron(omega_uv, omega_1d)

                    self.omega_dict[u, v] = omega_uv
                    self.omega_dict[v, u] = omega_uv.T

                c = c_u * c_v
                if omega_final is None:
                    # First allocation: copy the scaled matrix
                    omega_final = self.omega_dict[u, v] * c
                else:
                    # Subsequent allocations: add in-place
                    omega_final += self.omega_dict[u, v] * c

                    # Add the symmetric (v, u) term if off-diagonal
                if u != v:
                    omega_final += self.omega_dict[v, u] * c
        return omega_final

    def compute_omega_std(self, P_dict, deriv_coef_vals):
        n_splines = np.prod(self.n_splines)
        omega_std = np.zeros((n_splines, n_splines))
        for u, c_u in enumerate(deriv_coef_vals):
            for v, c_v in enumerate(deriv_coef_vals[:u + 1]):
                c = c_u * c_v
                omega_std += P_dict[(u, v)] * c
                if u != v:
                    omega_std += P_dict[(v, u)] * c
        return omega_std

    def compute_k_action(self, action_keys, action_vals, x_eval=None):
        """
        Computes the Variational Physics Penalty Matrix (K_action) from the Weak Form.
        Executes in O(Q) time with zero cross-term combinatorial explosion.

        :param action_keys_left: List of parsed deriv keys for the left operator (e.g., [[0], [1], []])
        :param action_keys_right: List of parsed deriv keys for the right operator (e.g., [[0], [1], []])
        :param action_vals: List of coefficients for each bilinear pair
        """
        K_action_final = None

        for i, c in enumerate(action_vals):
            d_u = np.array(action_keys[i][0])
            d_v = np.array(action_keys[i][1])

            # Use a sorted tuple cache key because integral(u_x * u) == integral(u * u_x)
            cache_key = (i, i)

            # --- 1. Check the Global n-Dimensional Cache ---
            if cache_key not in self.omega_dict:
                K_uv = None
                for p in range(self.dim):
                    d_up = (d_u == p).sum()
                    d_vp = (d_v == p).sum()

                    # --- 2. Check the Local 1D Micro-Cache ---
                    cache_key_1d = (p, d_up, d_vp)
                    if cache_key_1d not in self.integral_1d_cache:
                        self.integral_1d_cache[cache_key_1d] = compute_spline_integral_gl(
                            self.all_knots[p], u=d_up, v=d_vp, deg=self.deg, x_eval=x_eval
                        )

                    K_1d = self.integral_1d_cache[cache_key_1d]

                    # --- 3. Assemble the Kronecker Product ---
                    if K_uv is None:
                        K_uv = K_1d
                    else:
                        K_uv = np.kron(K_uv, K_1d)

                self.omega_dict[cache_key] = K_uv

            # --- 4. In-Place Memory Accumulation ---
            if K_action_final is None:
                # First allocation
                K_action_final = self.omega_dict[cache_key] * c
            else:
                # Subsequent allocations
                K_action_final += self.omega_dict[cache_key] * c

        # --- 5. Enforce Global Symmetry ---
        # The physical action must yield a strictly symmetric bilinear form.
        K_action_final = (K_action_final + K_action_final.T) / 2.0
        return K_action_final

    def fit(self, X, y):
        """
        1. Knot
        2. Create Basis and Calculate D_q
        3. Write SVD -> (U, V, s) (U refers to the A matrix in that paper)
        4. Fit EWA on (U, y, s)
        :param X:
        :param y:
        :return:
        """
        import time

        X, y = check_X_y(X, y, accept_sparse=False)
        assert X.shape[1] == self.dim
        n_samples = X.shape[0]
        self.X_fit_, self.y_fit_ = X, y
        self.fit_n = n_samples

        self.a_, self.b_ = X.min(axis=0), X.max(axis=0)
        if self.domain is None:
            self.domain = list(zip(self.a_, self.b_))

        # 1. Knot Setup
        self.generate_knots(X)

        self.grids = [np.linspace(*d, self.grid_size) for d in self.domain]
        self.gridpoints = util.tensor_prod(*self.grids)
        self.B_grid = self._build_tensor_basis(self.gridpoints)

        # 2. Lambda & Prior Vectorization
        self.sigma_est = gsj_est(y) if self.sigma is None else self.sigma
        log_lams = np.arange(self.log_lamb[0], self.log_lamb[1], self.log_lamb_step)
        lam_grid = np.exp(np.sort(log_lams)[::-1])

        if self.prior_type == "light":
            p_dist = gamma(a=0.3, scale=1 / 0.3)
            prior_vals = p_dist.cdf(lam_grid)
        else:
            prior_vals = 2 * t.cdf(25 * lam_grid, df=3) - 1

        prior = np.diff(np.r_[0, prior_vals[::-1]])[::-1]
        prior, lam_grid = prior[prior > self.tol], lam_grid[prior > self.tol]

        if self.norm_omega:
            highest_deriv = len(self.deriv_coef_vals) - 1
            lam_grid /= self.step_size[0] ** (highest_deriv * 2)

        # 3. Basis & Orthogonalization
        B_s = self._build_tensor_basis(X)
        G = util.compute_G(B_s)

        if self.mode == 'PDE':
            coef_vals = np.array(self.deriv_coef_vals, dtype=float)
        else:
            coef_vals = np.array(self.action_vals, dtype=float)

        for i in range(self.max_iter):
            if self.mode == 'PDE':
                omega = self.compute_omega(list(self.coef_keys), coef_vals) # The integral should be just evaluation at knots
            else:
                omega = self.compute_k_action(self.coef_keys, coef_vals)

            total_var = np.trace(np.linalg.solve(G, omega))

            # TODO: looks deriv will blow up when more knots
            # if self.norm_omega:
            #     highest_deriv = len(self.deriv_coef_vals) - 1
            #     omega *= (self.step_size[0] ** (highest_deriv * 2))
                # print(omega.max(axis=1))
            # orth = orthogonalize_basis(B_s, omega)

            t0 = time.time()
            orth = orthogonalize_basis_fast(G, omega, mode=self.mode) # This is slow
            # k = np.argmax(orth['s'].cumsum() >= total_var * 0.99)
            # print(f'{k} PC for 99% Var')
            t1 = time.time()
            print(i, t1 - t0)

            # B_integral = (self.B_grid.T @ self.B_grid) / self.grid_size
            # orth_grid = orthogonalize_basis(self.B_grid / np.sqrt(self.grid_size), omega)
            self.V_, self.s_ = orth['V'], orth['s']
            self.U_ = B_s @ self.V_

            # 4. EWA Logic
            res = fit_ewa(
                y=y, ftrue=y, U=self.U_, V=self.V_, s=self.s_,
                lam=lam_grid, prior=prior,
                sig_eps_est=self.sigma_est, gamma_mult=self.gamma_mult,
            )

            #self.res = res
            if self.mode == 'PDE':
                gram_matrices = self.update_gram_matrix(self.omega_dict, res['beta'])
            else:
                gram_matrices = self.update_action_gram_matrix(self.omega_dict, res['beta'])
            coef_mean, coef_var = self.get_coef_dist(gram_matrices, coef_vals)

            old_theta = coef_vals[self.trainable_idx]
            pct_chg = np.abs(old_theta - coef_mean).sum() / np.abs(old_theta).sum()
            coef_vals[self.trainable_idx] = coef_mean
            print(coef_mean)

            if pct_chg < self.min_pct_chg:
                break

        self.res = res
        self.deriv_coef_vals = coef_vals
        self.coef_mean = coef_mean
        self.coef_var = coef_var
        # 5. Extract Coefficients
        # We only need the mean weights across the aggregation
        self.weights = pd.Series(res['weights'], index=lam_grid).sort_index()
        self.sigma_regr_adj = np.sqrt(np.sum((res['fhat_ewa'] - y)**2) / (len(y) - np.trace(res['hat_matrix'])))
        self.sigma_regr = np.sqrt(np.mean((res['fhat_ewa'] - y)**2))
        return self

    def update_gram_matrix(self, omega_dict, beta):
        m = len(self.coef_keys)
        gram_matrices = np.zeros((m, m))
        for key, omega_uv in omega_dict.items():
            gram_matrices[key] = beta.T @ omega_uv @ beta
        return gram_matrices

    def update_action_gram_matrix(self, omega_dict, beta):
        """
        Builds the Gram matrix for theta by minimizing the Weak-Form Galerkin residual.
        Instead of integrating cross-terms, it projects beta through the independent
        action matrices and measures their vector alignments.
        """
        m = len(self.coef_keys)
        V = np.zeros((len(beta), m))

        for key, K_q in omega_dict.items():
            assert key[0] == key[1]
            K_q = omega_dict[key]
            K_q = (K_q + K_q.T) / 2.0
            V[:, key[0]] = K_q @ beta

        # --- TRAP 2 FIX: BOUNDARY FLUX ISOLATION ---
        # Create an interior mask for the 1D basis functions
        mask_1d = [np.zeros(n) for n in self.n_splines]
        for i, n in enumerate(self.n_splines):
            mask_1d[i][self.deg: n - self.deg] = 1.0

        # Expand the mask to the full N-dimensional tensor product grid
        interior_mask = mask_1d[0]
        for i in range(1, self.dim):
            interior_mask = np.kron(interior_mask, mask_1d[i])

        interior_mask = interior_mask.astype(bool)

        # Filter out the boundary residual vectors
        V = V[interior_mask, :]

        # The new Gram matrix is the inner product of the residual vectors
        gram_matrix = V.T @ V
        return gram_matrix

    def get_coef_dist(self, G, deriv_coef):
        T = self.trainable_idx
        Tc = [i for i, _ in enumerate(self.coef_keys) if i not in self.trainable_idx]
        coef_Tc = deriv_coef[Tc]
        coef_mean = -np.linalg.solve(G[T][:, T], G[T][:, Tc] @ coef_Tc)
        coef_var = np.linalg.inv(G[T][:, T])
        return coef_mean, coef_var


    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False)
        assert X.shape[1] == self.dim

        # Project new X into the orthogonalized basis
        B_new = self._build_tensor_basis(X)
        U_new = B_new @ self.V_
        y_hat = U_new @ self.res['theta_hat']

        # np.diag(U_new @ U_new.T) + 1
        variances = np.sum((U_new ** 2) * (self.res['shrink_weight'] ** 2)[None, :], axis=1) + 1 #TODO: **2?
        se = self.sigma_regr * np.sqrt(variances)
        interval = norm.ppf(1 - self.alpha / 2) * se
        multi_idx = pd.MultiIndex.from_arrays(X.T, names=self.variables)
        preds = pd.DataFrame({'fhat': y_hat, 'se': se,
                             'ci_lower': y_hat - interval, 'ci_upper': y_hat + interval},
                             index=multi_idx)
        return preds

    def plot_fit(self, x_test=None, y_test=None, show_ci=True):
        check_is_fitted(self)
        x_grid = self.gridpoints
        fit_preds = self.predict(x_grid)

        if x_test is None:
            x_scatter, y_scatter = self.X_fit_, self.y_fit_
        else:
            x_scatter, y_scatter = x_test, y_test

        if self.dim == 1:
            fig, ax = plt.subplots(figsize=(10, 5))

            if show_ci:
                ax.fill_between(x_grid[:, 0], fit_preds['ci_lower'], fit_preds['ci_upper'],
                                color='C0', alpha=0.15, label=f'{1 - self.alpha:.0%} CI (during Fit)')
            ax.plot(x_grid[:, 0], fit_preds['fhat'], color='C0', lw=2, label='EWA Prediction')
            ax.scatter(x_scatter[:, 0], y_scatter, s=10, color='black', alpha=0.5, label='Data')

            deriv_coef = list(map(lambda x: float(f"{x:.4g}"), self.deriv_coef_vals))
            title = (
                f"EWA fit [n={len(self.X_fit_)}, deg={self.deg}, deriv_coef={deriv_coef}, "
                f"nk={self.n_knots_1d}, sigma={self.sigma_regr:.3f}, "
                f"{100 * (1 - self.alpha):.0f}% CI]"
            )
            ax.set(title=title, xlabel="X", ylabel="y")
            ax.legend()
            plt.show()

        elif self.dim == 2:
            X = x_grid[:, 0].reshape(self.grid_size, self.grid_size)
            Y = x_grid[:, 1].reshape(self.grid_size, self.grid_size)
            Z = fit_preds['fhat'].values.reshape(self.grid_size, self.grid_size)

            # 3. Plot the 2D Z array directly
            # Note: x and y tick labels can be specified using xticklabels and yticklabels
            fig, ax = plt.subplots(figsize=(6, 5))
            vmax = np.abs(y_scatter).max()
            norm = Normalize(vmin=-vmax, vmax=vmax)
            c = ax.contourf(X, Y, Z, levels=50, cmap=CMAP, norm=norm)

            # Add scatter
            ax.scatter(x_scatter[:, 0], x_scatter[:, 1], c=y_scatter,
                       norm=norm, cmap=CMAP, edgecolors='black', s=20, label='Data')

            fig.colorbar(c, ax=ax)
            deriv_coef = list(map(lambda x: float(f"{x:.4g}"), self.deriv_coef_vals))
            deriv_coef = dict(zip(list(map(str, self.coef_keys)), deriv_coef))
            title = (
                f"2D EWA fit [n={len(self.X_fit_)}, deg={self.deg}, deriv_coef={deriv_coef}, "
                f"nk={self.n_knots_1d}, sigma={self.sigma_regr:.3f}, "
                f"{100 * (1 - self.alpha):.0f}% CI]"
            )
            ax.set(title=title, xlabel="X1", ylabel="X2")
            plt.show()

        else:
            print(f"Plotting not supported for dimension {self.dim} > 2.")

    @property
    def basis_functions(self):
        basis = self.B_grid @ self.V_
        basis_wgt = self.res['theta_hat']

        basis_decomp = basis * basis_wgt[None, :]
        feature_importance = (basis_decomp ** 2).sum(axis=0)
        feature_importance = feature_importance / feature_importance.sum()
        multi_idx = pd.MultiIndex.from_arrays(self.gridpoints.T, names=self.variables)

        basis_decomp = pd.DataFrame(basis_decomp, index=multi_idx, columns=feature_importance) #.round(4)
        basis_decomp = basis_decomp.sort_index(axis=1, ascending=False)
        return basis_decomp

    def plot_basis_functions(self, thres=0.999, max_subplots=12, **kwargs):
        """
        Plots the top basis functions that explain the given variance threshold.
        Handles 1D (line plots), 2D (heatmaps), and 3D (2D cross-section heatmaps).

        :param thres: Cumulative variance threshold to determine how many bases to plot.
        :param max_subplots: Cap the number of subplots to prevent memory/rendering blowouts.
        """
        check_is_fitted(self)
        df2 = self.basis_functions.copy()

        # Calculate how many basis functions are needed to reach the threshold
        exp_var = df2.columns.values.cumsum()
        n_col = np.argmax(exp_var > thres) + 1
        n_col = min(n_col, max_subplots)  # Safety cap
        df2.columns = df2.columns.map('{:.0%}'.format)

        # Format column names to percentage strings
        top_bases = df2.iloc[:, :n_col]
        global_vmax = np.abs(top_bases.values).max()
        common_norm = Normalize(vmin=-global_vmax, vmax=global_vmax)

        # --- 1D Case: All bases on a single line plot ---
        if self.dim == 1:
            top_bases = df2.iloc[:, :n_col]
            ax = top_bases.plot(title=f'Top {n_col} Basis Functions ({exp_var[n_col - 1] * 100:.1f}% Var)', **kwargs)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[0:n_col], labels[0:n_col])
            plt.show()

        # --- 2D Case: Grid of Heatmaps ---
        elif self.dim == 2:
            grid_shape = (len(self.grids[0]), len(self.grids[1]))
            X1, X2 = np.meshgrid(self.grids[0], self.grids[1], indexing='ij')

            # Setup subplot grid (e.g., 3 columns max per row)
            cols = min(n_col, 3)
            rows = int(np.ceil(n_col / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows),
                                     squeeze=False, layout='constrained')

            for idx in range(n_col):
                r, c = divmod(idx, cols)
                ax = axes[r, c]

                # Reshape the 1D column back into the 2D grid
                Z = top_bases.iloc[:, idx].values.reshape(grid_shape)

                # Use coolwarm to easily see positive (red) and negative (blue) basis weights
                ax.contourf(X1, X2, Z, levels=30, cmap=CMAP, norm=common_norm)
                ax.set(title=f"Basis {idx + 1} [Var: {df2.columns[idx]}]", xlabel="X1", ylabel="X2")

            # Hide any empty subplots if n_col doesn't perfectly fill the grid
            for idx in range(n_col, rows * cols):
                r, c = divmod(idx, cols)
                axes[r, c].axis('off')

            sm = ScalarMappable(norm=common_norm, cmap=CMAP)
            fig.colorbar(sm, ax=axes.ravel().tolist(), aspect=40, pad=0.03, label='Basis Weight Magnitude')
            fig.suptitle(f'Top {n_col} 2D Basis Functions ({exp_var[n_col - 1] * 100:.1f}% Variance)', fontsize=14)
            plt.show()

        # --- 3D Case: Grid of Heatmaps (Sliced at middle of Z-axis) ---
        elif self.dim == 3:
            grid_shape = (len(self.grids[0]), len(self.grids[1]), len(self.grids[2]))

            # Find the middle slice of the 3rd dimension
            z_mid = len(self.grids[2]) // 2
            z_val = self.grids[2][z_mid]

            cols = min(n_col, 3)
            rows = int(np.ceil(n_col / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

            for idx in range(n_col):
                r, c = divmod(idx, cols)
                ax = axes[r, c]

                # Reshape to 3D, then extract the 2D cross-section
                Z_full = top_bases.iloc[:, idx].values.reshape(grid_shape)
                Z_slice = Z_full[:, :, z_mid]

                cf = ax.contourf(self.grids[0], self.grids[1], Z_slice, levels=30, cmap=CMAP)
                fig.colorbar(cf, ax=ax)
                ax.set(title=f"Basis {idx + 1} [Var: {df2.columns[idx]}]", xlabel="X1", ylabel="X2")

            for idx in range(n_col, rows * cols):
                r, c = divmod(idx, cols)
                axes[r, c].axis('off')

            fig.suptitle(f'Top {n_col} 3D Bases (Central Slice X3 ≈ {z_val:.2g})', fontsize=14)
            plt.tight_layout()
            plt.show()

        else:
            print(f"Plotting basis functions not supported for dimension {self.dim} > 3.")

    def plot_coef_dist(self, **kwargs):
        labels = [rf'$\theta_{{{i}}}$' for i in self.trainable_idx]
        util.mvn_pairplot(self.coef_mean, self.coef_var, labels=labels,
                          title=r'Density of Physical $\theta$ given $\hat \beta$', **kwargs)

    def __repr__(self):
        status = "Fitted" if hasattr(self, 'theta_hat_') else "Unfitted"
        fit_info = {k: self.__getattribute__(k)
                    for k in ['fit_n', 'domain', 'deg', 'deriv_coef', 'n_knots',
                              'sigma_regr', 'gamma_mult', 'prior_type', 'alpha',
                              ]}
        return f"EWABSpline({status}, {fit_info})"


class GeneralizedLocalPolynomialRegressor(object):
    def __init__(self, x, y, degree=1, reg=1e-6):
        """
        Fits Local Polynomial Regression of arbitrary degree with a Gaussian Kernel.
        Fully vectorized to precompute structural spatial moments for O(1) loop iteration.
        """
        idx = np.argsort(x)
        self.x = np.array(x[idx], dtype=float)
        self.y = np.array(y[idx], dtype=float)
        self.degree = degree
        self.reg = reg
        self.n = len(self.x)

        # Internal cache for static spatial weights across optimization runs
        self._spatial_cache = {}

        # =====================================================================
        # PRECOMPUTATION: Static Spatial Tensors
        # =====================================================================
        self.D = self.x[None, :] - self.x[:, None]
        self.D2 = self.D ** 2

        k_S = np.arange(2 * self.degree + 1)
        self.D_powers = self.D[:, :, None] ** k_S[None, None, :]

        k_T = np.arange(self.degree + 1)
        self.DY_powers = (self.D[:, :, None] ** k_T[None, None, :]) * self.y[None, :, None]

        self.pq_indices = np.add.outer(np.arange(self.degree + 1), np.arange(self.degree + 1))
        self.reg_matrix = np.eye(self.degree + 1) * self.reg

    def _get_spatial_weights(self, spatial_weighting, edge_trim_fraction):
        """
        Caches and retrieves the Riemann spacing (dx) and valid data masks.
        Prevents redundant O(N) calculations during heavy optimization loops.
        """
        cache_key = (spatial_weighting, edge_trim_fraction)
        if cache_key in self._spatial_cache:
            return self._spatial_cache[cache_key]

        # 1. Quantile-Based Boundary Masking
        q_low, q_high = np.quantile(self.x, [edge_trim_fraction, 1.0 - edge_trim_fraction])
        interior_mask = (self.x >= q_low) & (self.x <= q_high)

        if not np.any(interior_mask):
            raise ValueError("No Data inside Interior, Please check data or trim fraction!")

        valid_x = self.x[interior_mask]
        n_valid = len(valid_x)

        # 2. Spatial Weighting via Riemann Spacing
        if spatial_weighting and n_valid > 1:
            dx = np.zeros_like(valid_x)
            dx[1:-1] = (valid_x[2:] - valid_x[:-2]) / 2.0
            dx[0] = valid_x[1] - valid_x[0]
            dx[-1] = valid_x[-1] - valid_x[-2]
        else:
            dx = np.ones_like(valid_x)

        # Store mask, spacing vector, number of valid elements, and the pre-computed sum of dx
        self._spatial_cache[cache_key] = (interior_mask, dx, n_valid, np.sum(dx))
        return self._spatial_cache[cache_key]

    def fit_predict(self, h):
        """Fully vectorized LPR evaluation and Hat matrix extraction."""
        # Unpack h if it comes from a scipy array vector, and protect against div by zero
        h = max(float(np.squeeze(h)), 1e-12)

        W = np.exp(-self.D2 / (2 * h ** 2))

        S = np.einsum('ij,ijk->ik', W, self.D_powers)
        T = np.einsum('ij,ijk->ik', W, self.DY_powers)

        XTWX = S[:, self.pq_indices] + self.reg_matrix

        try:
            inv_XTWX = np.linalg.inv(XTWX)
            beta = np.einsum('nij,nj->ni', inv_XTWX, T)
            y_hat = beta[:, 0]
            H_ii = inv_XTWX[:, 0, 0]
        except np.linalg.LinAlgError:
            y_hat = self.y.copy()
            H_ii = np.ones(self.n)

        return y_hat, H_ii

    @property
    def h_bounds(self):
        domain_length = np.abs(self.D).max()
        return (domain_length / self.n / (2 * np.sqrt(np.pi)), domain_length / self.degree)

    def loocv_score(self, h, spatial_weighting=True, edge_trim_fraction=0.0):
        h = float(np.squeeze(h))
        if h <= 0:
            return np.inf

        y_hat, H_ii = self.fit_predict(h)

        denom = np.clip(np.abs(1.0 - H_ii), 1e-4, 1.0)
        squared_residuals = ((self.y - y_hat) / denom) ** 2

        # Fetch O(1) Cached Spatial logic
        interior_mask, dx, _, sum_dx = self._get_spatial_weights(spatial_weighting, edge_trim_fraction)

        valid_squared_res = squared_residuals[interior_mask]

        if spatial_weighting and len(dx) > 1:
            return np.sum(valid_squared_res * dx) / sum_dx
        return np.mean(valid_squared_res)

    def white_noise_score(self, h, max_lag=1, spatial_weighting=True, edge_trim_fraction=0.0):
        h = float(np.squeeze(h))
        if h <= 0:
            return np.inf

        y_hat, _ = self.fit_predict(h)
        residuals = self.y - y_hat

        # Fetch O(1) Cached Spatial logic
        interior_mask, dx, n_valid, sum_dx = self._get_spatial_weights(spatial_weighting, edge_trim_fraction)

        if n_valid <= max_lag:
            return np.inf

        valid_res = residuals[interior_mask]

        # Spatially weighted mean and variance
        mean_res = np.sum(valid_res * dx) / sum_dx
        res_centered = valid_res - mean_res
        var_res = np.sum((res_centered ** 2) * dx) / sum_dx

        if var_res == 0:
            return np.inf

        rho_sq_sum = 0.0
        for lag in range(1, max_lag + 1):
            if n_valid <= lag:
                break

            res_left = res_centered[:-lag]
            res_right = res_centered[lag:]
            w_pair = (dx[:-lag] + dx[lag:]) / 2.0

            cov_lag = np.sum(res_left * res_right * w_pair) / np.sum(w_pair)
            rho_lag = cov_lag / var_res
            rho_sq_sum += rho_lag ** 2

        return rho_sq_sum

    @staticmethod
    def optimize_h(score_func, opt=None, args=(), h_bounds=None, num_intervals=1):
        opt = opt or 'differential_evolution'
        assert opt in ['minimize_scalar', 'differential_evolution']
        min_h, max_h = h_bounds

        interval_edges = np.linspace(min_h, max_h, num_intervals + 1)
        best_h, best_score = None, np.inf

        # Internal memoization protects against evaluating shared edges multiple times
        memo = {}

        def memo_score(h_val):
            h_clean = float(np.squeeze(h_val))
            if h_clean not in memo:
                memo[h_clean] = score_func(h_clean, *args)
            return memo[h_clean]

        for i in range(num_intervals):
            sub_low, sub_high = interval_edges[i], interval_edges[i + 1]
            bounds = (sub_low, sub_high)

            optimizer = getattr(scipy.optimize, opt)

            # Setup correct kwargs depending on the optimizer choice
            if opt == 'differential_evolution':
                kwargs = {'strategy': 'currenttobest1bin', 'bounds': [bounds]}
            else:
                # Required: standard minimize_scalar strictly demands method='bounded' to use bounds
                kwargs = {'bounds': bounds, 'method': 'bounded'}

            res = optimizer(lambda x: memo_score(x), **kwargs)

            if res.success:
                current_score, current_h = res.fun, float(np.squeeze(res.x))
            else:
                scores_low = memo_score(sub_low)
                scores_high = memo_score(sub_high)

                if scores_low < scores_high:
                    current_score, current_h = scores_low, sub_low
                else:
                    current_score, current_h = scores_high, sub_high
                print(
                    f'Opt Failed! Using Boundaries ({sub_low:.4g}, {sub_high:.4g}) -> ({scores_low:.4g}, {scores_high:.4g})')

            if current_score < best_score:
                best_score = current_score
                best_h = current_h

        if best_h is None:
            raise ValueError('Optimization completely failed. Check hyperparameter logic.')

        return best_h

    def optimize_h_loocv(self, opt=None, num_intervals=1, h_bounds=None, spatial=True, trim=0.0):
        return self.optimize_h(self.loocv_score, opt=opt, num_intervals=num_intervals,
                               args=(spatial, trim), h_bounds=h_bounds or self.h_bounds)

    def optimize_h_white_noise(self, opt=None, num_intervals=1, h_bounds=None, max_lag=5, spatial=True, trim=0.0):
        return self.optimize_h(self.white_noise_score, opt=opt, num_intervals=num_intervals,
                               args=(max_lag, spatial, trim), h_bounds=h_bounds or self.h_bounds)


class ContinuousTimeACF:
    def __init__(self, t, x):
        """
        Initializes the continuous-time framework for non-uniform data.
        """
        # Ensure time is strictly sorted
        idx = np.argsort(t)
        self.t = np.array(t)[idx]
        self.x = np.array(x)[idx]
        self.T_total = self.t[-1] - self.t[0]

        # 1. Compute the Continuous Mean using Riemann integration (Trapezoidal)
        # We don't use np.mean(x) because clustered points would bias the mean.
        self.mean_x = np.trapz(self.x, self.t) / self.T_total
        self.x_centered = self.x - self.mean_x

        # 2. Fit a Continuous Cubic Spline to the uneven data
        self.spline_x = CubicSpline(self.t, self.x_centered)

        # 3. Extract the exact analytical derivative (Velocity: X_dot)
        self.spline_x_dot = self.spline_x.derivative()

    def _integrate_expectation(self, func_A, func_B, tau, num_points=2000):
        """
        Helper method to compute the continuous expected value: E[A(t) B(t+tau)]
        """
        if tau >= self.T_total:
            return 0.0

        t_start = self.t[0]
        t_end = self.t[-1] - tau

        # Create a dense, perfectly uniform integration grid over the valid window
        t_grid = np.linspace(t_start, t_end, num_points)

        # Evaluate the continuous functions
        A_t = func_A(t_grid)
        B_t_plus_tau = func_B(t_grid + tau)

        # Integrate using Trapezoidal rule and normalize by the overlap window length
        integral = np.trapz(A_t * B_t_plus_tau, t_grid)
        return integral / (t_end - t_start)

    def acf(self, tau, num_points=2000):
        """ Computes R_XX(tau) = E[X(t) X(t+tau)] """
        return self._integrate_expectation(self.spline_x, self.spline_x, tau, num_points)

    def acf_derivative(self, tau, num_points=2000):
        """ Computes d/d_tau R_XX(tau) = E[X(t) X_dot(t+tau)] """
        return self._integrate_expectation(self.spline_x, self.spline_x_dot, tau, num_points)

    def curvature_at_zero(self, num_points=2000):
        """ Computes R''_XX(0) = -E[X_dot(t)^2] """
        t_grid = np.linspace(self.t[0], self.t[-1], num_points)
        x_dot_t = self.spline_x_dot(t_grid)

        integral = np.trapz(x_dot_t ** 2, t_grid)
        return -integral / self.T_total

    def compute_curves(self, max_tau, num_taus=100, num_integral_points=2000):
        """ Batch computes the ACF and its derivative across an array of lags """
        taus = np.linspace(0, max_tau, num_taus)
        r_xx = np.array([self.acf(tau, num_integral_points) for tau in taus])
        dr_xx = np.array([self.acf_derivative(tau, num_integral_points) for tau in taus])
        return taus, r_xx, dr_xx


if __name__ == '__main__':
    print(1)