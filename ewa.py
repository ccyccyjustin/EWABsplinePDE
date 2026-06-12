import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, t
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import util
from util import orthogonalize_basis, gsj_est, fit_ewa, compute_spline_integral_gl
import pandas as pd
from scipy.interpolate import BSpline, CubicSpline
from scipy.optimize import minimize_scalar, differential_evolution
import scipy

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


class EWABSplinePDE(BaseEstimator, RegressorMixin):
    def __init__(self, deg=3, deriv_coef=[0., 0., 1.], trainable=None,
                 max_iter=100, min_pct_chg=1e-8,
                 domain=None, num_int_knots=None, quantile_based=False, grid_size=1000,
                 sigma=None, gamma_mult=8, prior_type="light", norm_omega=False,
                 alpha=0.05, log_lamb=(-30, 20), log_lamb_step=0.01, tol=1e-10):
        self.deg = deg
        self.deriv_coef = np.array(deriv_coef)
        if trainable is None:
            self.trainable = list(range(len(deriv_coef)))
        else:
            self.trainable = trainable
        self.max_iter = max_iter
        self.min_pct_chg = min_pct_chg

        self.domain = domain
        self.num_int_knots = num_int_knots
        self.grid_size = grid_size
        self.quantile_based = quantile_based
        self.sigma = sigma
        self.gamma_mult = gamma_mult
        self.prior_type = prior_type
        self.norm_omega = norm_omega
        self.alpha = alpha
        self.log_lamb = log_lamb
        self.log_lamb_step = log_lamb_step
        self.tol = tol

    def generate_knots(self, X):
        n_samples = X.shape[0]
        dim = X.shape[1]

        if self.num_int_knots is None:
            # Rule of thumb for knot density
            self.n_knots = int(15 * n_samples ** (1 / (2 * self.deg + 3)))
        else:
            self.n_knots = self.num_int_knots

        self.n_knots = int(self.n_knots ** (1 / dim))

        if self.quantile_based:
            self.int_knots_ = [np.percentile(X[:, i], np.linspace(0, 100, self.n_knots + 2)[1:-1])
                               for i in range(X.shape[1])]
        else:
            self.int_knots_ = [np.linspace(*d, self.n_knots + 2)[1:-1] for d in self.domain]

        self.step_size = [np.diff(knt).mean() for knt in self.int_knots_]

        self.all_knots = [create_knots(X[:, i], knots=self.int_knots_[i], degree=self.deg,
                                       intercept=True, boundary_knots=self.domain[i])
                          for i in range(X.shape[1])]  # Could have different #knots

        self.knot_counts = knots_data_count(X, self.all_knots, self.deg)

    def compute_omega(self, deriv_coef, x_eval=None):
        self.omega_dict = {}
        omega = []
        for u, c_u in enumerate(deriv_coef):
            # if abs(c_u) < 1e-6:
            #     continue
            for v, c_v in enumerate(deriv_coef[:u + 1]):
                # if abs(c_v) < 1e-6:
                #     continue

                if (u, v) in self.omega_dict:
                    omega_uv = self.omega_dict[u, v]
                else:
                    omega_uv = compute_spline_integral_gl(self.all_knots[0], u=u, v=v, deg=self.deg, x_eval=x_eval)#self.all_grids[0])
                    self.omega_dict[u, v] = omega_uv
                    self.omega_dict[v, u] = omega_uv.T

                c = c_u * c_v
                omega.append(omega_uv * c)

                if u != v:
                    omega.append(omega_uv.T * c)

        omega = np.add.reduce(omega) #Shape: (kn, kn)
        return omega

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
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = X.shape[0]
        self.X_fit_, self.y_fit_ = X, y
        self.fit_n = n_samples

        self.a_, self.b_ = X.min(axis=0), X.max(axis=0)
        if self.domain is None:
            self.domain = list(zip(self.a_, self.b_))

        # 1. Knot Setup
        self.generate_knots(X)

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
            highest_deriv = len(self.deriv_coef) - 1
            lam_grid /= self.step_size[0] ** (highest_deriv * 2)

        # 3. Basis & Orthogonalization
        B_s = [bs(X[:, i], all_knots=self.all_knots[i], degree=self.deg, intercept=True)
               for i in range(X.shape[1])][0] #TODO: Hack for now...

        deriv_coef = self.deriv_coef.copy()

        for i in range(30):
            omega = self.compute_omega(deriv_coef) # The integral should be just evaluation at knots

            # TODO: looks deriv will blow up when more knots
            if self.norm_omega:
                highest_deriv = len(self.deriv_coef) - 1
                omega *= (self.step_size[0] ** (highest_deriv * 2))
                # print(omega.max(axis=1))
            orth = orthogonalize_basis(B_s, omega)

            self.grids = [np.linspace(*d, self.grid_size) for d in self.domain]
            self.B_grid = [bs(self.grids[i], all_knots=self.all_knots[i], degree=self.deg, intercept=True)
                   for i in range(X.shape[1])][0] #TODO: Hack for now...
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
            gram_matrices = self.update_gram_matrix(self.omega_dict, res['beta'])
            coef_mean, coef_var = self.get_coef_dist(gram_matrices, deriv_coef)

            old_theta = deriv_coef[self.trainable]
            pct_chg = np.abs(old_theta - coef_mean).sum() / np.abs(old_theta).sum()
            deriv_coef[self.trainable] = coef_mean
            # print(coef_mean, coef_var)
            if pct_chg < self.min_pct_chg:
                break

        self.res = res
        self.deriv_coef = deriv_coef
        self.coef_mean = coef_mean
        self.coef_var = coef_var
        # 5. Extract Coefficients
        # We only need the mean weights across the aggregation
        self.weights = pd.Series(res['weights'], index=lam_grid).sort_index()
        self.sigma_regr_adj = np.sqrt(np.sum((res['fhat_ewa'] - y)**2) / (len(y) - np.trace(res['hat_matrix'])))
        self.sigma_regr = np.sqrt(np.mean((res['fhat_ewa'] - y)**2))
        return self

    def update_gram_matrix(self, omega_dict, beta):
        m = len(self.deriv_coef)
        gram_matrices = np.zeros((m, m))
        for (u, v), omega_uv in omega_dict.items():
            gram_matrices[u, v] = beta.T @ omega_uv @ beta
        return gram_matrices

    def get_coef_dist(self, G, deriv_coef):
        T = self.trainable
        Tc = [i for i, _ in enumerate(self.deriv_coef) if i not in self.trainable]
        coef_Tc = deriv_coef[Tc]
        coef_mean = -np.linalg.solve(G[T][:, T], G[T][:, Tc] @ coef_Tc)
        coef_var = np.linalg.inv(G[T][:, T])
        return coef_mean, coef_var


    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False)
        assert np.ndim(X) == 1

        # Project new X into the orthogonalized basis
        B_new = bs(X, all_knots=self.all_knots[0], degree=self.deg, intercept=True)#TODO: 2d
        U_new = B_new @ self.V_
        y_hat = U_new @ self.res['theta_hat']

        # np.diag(U_new @ U_new.T) + 1
        variances = np.sum((U_new ** 2) * (self.res['shrink_weight'] ** 2)[None, :], axis=1) + 1 #TODO: **2?
        se = self.sigma_regr * np.sqrt(variances)
        interval = norm.ppf(1 - self.alpha / 2) * se
        preds = pd.DataFrame({'fhat': y_hat, 'se': se,
                             'ci_lower': y_hat - interval, 'ci_upper': y_hat + interval},
                             index=X)
        return preds

    def plot_fit(self, x_test=None, y_test=None, show_ci=True):
        check_is_fitted(self)
        x_grid = self.grids[0]
        fit_preds = self.predict(x_grid)

        if x_test is None:
            x_scatter, y_scatter = self.X_fit_, self.y_fit_
        else:
            x_scatter, y_scatter = x_test, y_test

        fig, ax = plt.subplots(figsize=(10, 5))
        if show_ci:
            ax.fill_between(x_grid, fit_preds['ci_lower'], fit_preds['ci_upper'],
                            color='C0', alpha=0.15, label=f'{1 - self.alpha:.0%} CI (during Fit)')
        ax.plot(x_grid, fit_preds['fhat'], color='C0', lw=2, label='EWA Prediction')
        ax.scatter(x_scatter, y_scatter, s=10, color='black', alpha=0.5, label='Data')

        deriv_coef = list(map(lambda x: float(f"{x:.4g}"), self.deriv_coef))
        title = (
            f"EWA fit [n={len(self.X_fit_)}, deg={self.deg}, deriv_coef={deriv_coef}, "
            f"nk={self.n_knots}, sigma={self.sigma_regr:.3f}, "
            f"{100 * (1 - self.alpha):.0f}% CI]"
        )
        ax.set(title=title, xlabel="X", ylabel="y")
        ax.legend()
        plt.show()

    @property
    def basis_functions(self):
        basis = self.B_grid @ self.V_
        basis_wgt = self.res['theta_hat']

        basis_decomp = basis * basis_wgt[None, :]
        feature_importance = (basis_decomp ** 2).sum(axis=0)
        feature_importance = feature_importance / feature_importance.sum()
        basis_decomp = pd.DataFrame(basis_decomp, index=self.grids[0], columns=feature_importance.round(4))
        basis_decomp = basis_decomp.sort_index(axis=1, ascending=False)
        return basis_decomp

    def plot_basis_functions(self, thres=0.999, **kwargs):
        df2 = self.basis_functions.copy()
        exp_var = df2.columns.values.cumsum()
        n_col = np.argmax(exp_var > thres) + 1
        df2.columns = df2.columns.map('{:.0%}'.format)

        ax = df2.plot(title=f'{thres * 100:.2f}% Variance Explained by Basis Functions', **kwargs)
        handles, labels = ax.get_legend_handles_labels()

        ax.legend(handles[0:n_col], labels[0:n_col])
        plt.show()

    def plot_coef_dist(self, **kwargs):
        labels = [rf'$\theta_{{{i}}}$' for i in self.trainable]
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