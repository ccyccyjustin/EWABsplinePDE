import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scipy.stats import gamma, norm, t
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import util
from itertools import product
from typing import Optional, Dict, List, Tuple
from Bsplines import (create_knots, create_spline_basis, knots_data_count,
                      compute_spline_integral_gl,
                      build_custom_tensor_basis, build_custom_sparse_tensor_basis)

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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

    return tuple(int((np.array(deriv_splits) == p).sum()) for p in range(dim))

@util.time_all_methods
class EWABSplinePDE(BaseEstimator, RegressorMixin):
    """
    Exponentially Weighted Aggregation (EWA) B-Spline Estimator with Physics-Informed Penalties.
    """

    def __init__(self,
                 dim: int,
                 deg: int = 3,
                 variables: Optional[List[str]] = None,
                 weak_form: bool = False,
                 bias_corrected: bool = False,
                 deriv_coef: Dict[str, float] = None,
                 force: callable = None,
                 boundary_conditions: dict = {},
                 trainable: Optional[List[str]] = None,
                 max_iter: int = 100,
                 min_param_pct_chg: float = 1e-4,
                 domain: Optional[List[Tuple[float, float]]] = None,
                 num_int_knots: Optional[int] = None,
                 quantile_based: bool = False,
                 total_grid_size: int = 1000,
                 sigma: Optional[float] = None,
                 gamma_mult: float = 8.0,
                 prior_type: str = "light",
                 alpha: float = 0.05,
                 lam_size: int = 5000,
                 tol: float = 1e-8,
                 sparse: bool = True,
                 max_eigfunc: int = None
                 ):

        self.dim = dim
        self.deg = deg
        self.variables = variables or [str(i) for i in range(dim)]

        self.weak_form = weak_form
        self.bias_corrected = bias_corrected
        self.deriv_coef = deriv_coef or {}
        self.force = force or (lambda s: np.zeros(s.shape[0]))
        self.boundary_conditions = boundary_conditions
        self.trainable = trainable

        self.max_iter = max_iter
        self.min_param_pct_chg = min_param_pct_chg
        self.domain = domain
        self.num_int_knots = num_int_knots
        self.quantile_based = quantile_based
        self.total_grid_size = total_grid_size

        self.sigma = sigma
        self.gamma_mult = gamma_mult
        self.prior_type = prior_type
        self.alpha = alpha
        self.lam_size = lam_size
        self.tol = tol

        # Internal Caches
        self._omega_dict = {}
        self._integral_1d_cache = {}
        self.spline_objs = None
        self.sparse = sparse
        self.max_eigfunc = max_eigfunc

    def _parse_physics_configuration(self):
        """Parses and validates PDE/Action coefficients into internal mathematical formats."""
        if self.deriv_coef:
            self.deriv_map = {get_deriv_order(k, self.variables): k for k in self.deriv_coef.keys()}
            sorted_coef = sorted(zip(self.deriv_map.keys(), self.deriv_coef.values()))
            self._coef_keys, self._initial_coef_vals = zip(*sorted_coef)
            self.mode = 'PDE'
        else:
            raise ValueError("Must provide physics constraints via deriv_coef or action_coef.")

        self.coef_keys = [self.format_deriv_orders(k) for k in self._coef_keys]
        self._initial_coef_vals = np.array(self._initial_coef_vals, dtype=float)
        self._zero_deriv = tuple([0] * self.dim)
        self._determine_trainable_indices()

    def parse_derivative_order(self, op_key):
        """
        Parses a nonlinear PDE term into a list of derivative order vectors.
        From a tuple of (dx_i) to a structured count of derivatives in each dimension
        Example: ((0, 0, 2), (1,2)) -> [ [2, 0, 1], [0, 1, 1] ]
        """
        factors = []

        # Handle the base case where the operator is just 'u' (no derivatives)
        if len(op_key) == 0:
            return [np.zeros(self.dim, dtype=int)]

        # Wrap single factors so an isolated (0,0,2) becomes ((0,0,2),)
        if isinstance(op_key[0], int):
            op_key = (op_key,)

        for term_axes in op_key:
            order = np.zeros(self.dim, dtype=int)
            for axis in term_axes:
                assert axis < self.dim
                order[axis] += 1
            factors.append(order)

        return factors

    def format_deriv_orders(self, deriv_order: tuple) -> str:
        """
        Maps a list of derivative order tuples to their string representations.
        Relies on self.variables (e.g., ['x', 'y']).

        Example input:  [(0, 0), (2, 0), (0, 2), (1, 1)]
        Example output: ['u', 'u_xx', 'u_yy', 'u_xy']
        """

        if sum(deriv_order) == 0:
            return "$u$"
        else:
            # Multiply each variable name by its derivative degree and concatenate
            subscript = "".join(self.variables[i] * degree for i, degree in enumerate(deriv_order))
            return f"$u_{{{subscript}}}$"

    def _determine_trainable_indices(self):
        """Determines which physics parameters are actively optimized."""
        if self.trainable is not None:
            trainable_list = []
            for k in self.trainable:
                if isinstance(k, tuple):
                    k_tuple = tuple(get_deriv_order(ki, self.variables) for ki in k)
                else:
                    k_tuple = get_deriv_order(k, self.variables)
                # k_tuple = tuple(int((np.array(k_tuple) == p).sum()) for p in range(self.dim))
                trainable_list.append(k_tuple)
            t_str = list(map(str, trainable_list))
            k_str = list(map(str, self._coef_keys))
            assert set(t_str) <= set(k_str), "Make sure trainable params are defined in deriv_coef."
            self._trainable_idx = [i for i, d in enumerate(k_str) if d in t_str]
        else:
            deriv_order = list(map(len, self._coef_keys))
            self._trainable_idx = [i for i, order in enumerate(deriv_order) if order < max(deriv_order)]

    def design_knots(self, X, y, gamma=3.0, **kwargs):
        from scipy.special import bernoulli, factorial
        N = X.shape[0]
        domain_size = self.domain[:, 1] - self.domain[:, 0]

        # Estimate the grad norm of the scaled function
        X_scaled = (X - self.domain[:, 0]) / domain_size
        s2, grad2, y_smooth = util.estimate_aniso(X_scaled, y, **kwargs)

        # Estimate Data Frequency and enery norm up to 1st order
        V = np.prod(domain_size)
        I_0 = np.mean(y_smooth ** 2) * V
        I_1 = grad2 / (domain_size ** 2) * V
        knot_ratio = np.sqrt(I_1) * domain_size #np.sqrt(grad2 / V)

        q0, q = 2, 2 * (self.deg + 1)
        data_snr = (N / V / s2) ** (1 / (q + self.dim))
        spline_coef = q0 * np.abs(bernoulli(q0)[-1]) / factorial(q0)
        spline_knots_factor = data_snr * (spline_coef / np.prod(knot_ratio)) ** (1 / (q0 + self.dim))
        spline_knots = knot_ratio * spline_knots_factor

        omega = np.sqrt(I_1 / I_0)
        spline_knots_freq = (gamma ** (1/self.dim)) * (domain_size * omega) / np.pi
        spline_knots = np.maximum(spline_knots, spline_knots_freq)

        logger.info(f'Estimated sigma^2: {np.sqrt(s2): .4f}, '
              f'1st order norm: {np.sqrt(I_1)}, '
              f'Frequency (Hertz): {omega / 2 / np.pi}, '
              f'allocating splines knots: {spline_knots}')
        return spline_knots

    def generate_knots(self, X, y):
        n_samples = X.shape[0]
        dim = X.shape[1]
        self.a_, self.b_ = X.min(axis=0), X.max(axis=0)
        self.domain = self.domain or list(zip(self.a_, self.b_))
        self.domain = np.array(self.domain)

        if self.num_int_knots is None:
            n_knots = self.design_knots(X, y).astype(int)
        else:
            self.num_int_knots = min(self.num_int_knots, n_samples // 3) # Avoid Singularity...
            n_knots = np.repeat(self.num_int_knots ** (1 / dim), self.dim).astype(int)

        if self.sparse is None:
            self.sparse = np.prod(n_knots) > 3000

        if self.quantile_based:
            self.int_knots_ = [np.percentile(X[:, i], np.linspace(0, 100, n_knots[i] + 2)[1:-1])
                               for i in range(X.shape[1])]
        else:
            self.int_knots_ = [np.linspace(*d, n_knots[i] + 2 + 3*i)[1:-1]
                               for i, d in enumerate(self.domain)] #TODO!!!! remove + 3d

        self.step_size = [np.diff(knt).mean() for knt in self.int_knots_]

        self.all_knots = [create_knots(int_knots=self.int_knots_[i], degree=self.deg, boundary_knots=self.domain[i])
                          for i in range(X.shape[1])]  # Could have different #knots

        self.n_knots = np.array(list(map(len, self.all_knots))) # n_int_knots_1d + (self.deg + 1) * 2
        self.knot_counts = knots_data_count(X, self.all_knots, self.deg)

    def get_spline_locations(self, d):
        """Calculates the Greville Abscissae (knot averages) for the x-axis."""
        knots = self.all_knots[d]
        p = self.deg
        n = self.n_splines[d]
        # The physical center of the i-th B-spline is the average of its p subsequent knots
        return np.array([np.mean(knots[i + 1 : i + 1 + p]) for i in range(n)])

    def _build_tensor_basis(self, X):
        """Constructs the tensor product of 1D B-splines.
        B_list: [(N, M_i)] N is no. data, M_i is no. Spline per dim

        Suppose 3D: (B_list[0][x] * B_list[1][y] * B_list[2][z]) -> B_tensor[x*(M_1*M_2)+y*(M_2)+z]
        """
        if self.spline_objs is None:
            self.spline_objs = [create_spline_basis(all_knots=self.all_knots[i], degree=self.deg)
                      for i in range(self.dim)]
            self.n_splines = np.array([s.c.shape[0] for s in self.spline_objs])
            self.total_n_splines = np.prod(self.n_splines)

        if self.sparse:
            B_tensor = build_custom_sparse_tensor_basis(X, self.n_splines, self.all_knots,
                                                    self.deg, self.dim, list(range(self.dim)))
        else:
            B_tensor = build_custom_tensor_basis(X, self.spline_objs, self.dim, list(range(self.dim)))
        return B_tensor

    def predict_derivative(self, X, deriv_orders: Tuple[int, ...]):
        """
        Calculates the partial derivatives of the fitted EWA B-spline surface.

        Parameters:
        -----------
        X : np.ndarray
            Evaluation points, shape (N, dim).
        deriv_orders : Tuple[int, ...]
            The order of the derivative for each dimension. Length must equal self.dim.
            E.g., in 2D, to get f_xx pass (2, 0). To get f_yy pass (0, 2). To get f_xy pass (1, 1).
        return_ci : bool
            Whether to return a DataFrame containing standard errors and confidence intervals.

        Returns:
        --------
        np.ndarray or pd.DataFrame
            The evaluated derivatives at points X, optionally with CI bands.
        """
        assert len(deriv_orders) == self.dim, f"deriv_orders must have length {self.dim}"

        B_list = []

        for i in range(self.dim):
            nu = deriv_orders[i]
            B_1d = self.spline_objs[i](X[:, i], nu=nu)
            B_list.append(B_1d)

        # Construct the tensor product basis (Khatri-Rao product) exactly like _build_tensor_basis
        B_tensor = B_list[0]
        for i in range(1, self.dim):
            B_tensor = (B_tensor[:, :, None] * B_list[i][:, None, :]).reshape(X.shape[0], -1)

        # Base prediction using optimal EWA coefficients in the original B-spline space
        y_hat_deriv = B_tensor @ self.beta_full_
        return y_hat_deriv

    def _initialize_basis(self, X: np.ndarray, y):
        """Generates knots, grids, and projects the design matrix."""
        # 1. Generate Knots
        self.generate_knots(X, y)  # (Keep your existing robust knot generation logic)

        # 2. Data Basis & Gram Matrix
        self.B_data_ = self._build_tensor_basis(X)
        self.G_ = (self.B_data_.T @ self.B_data_)

        # 3. Setup Evaluation Grids
        min_grid_size = int(self.total_grid_size ** (1 / self.dim))
        self.grid_size_ = [max(self.n_knots[d] * 3, min_grid_size)  for d in range(self.dim)]
        self.grids_ = [np.linspace(*d, self.grid_size_[dim]) for (d, dim) in zip(self.domain, range(self.dim))]
        self.gridpoints_ = util.tensor_prod(*self.grids_)
        self.B_grid_ = self._build_tensor_basis(self.gridpoints_)

    def _setup_priors_and_lambda(self, X, y: np.ndarray, tail_clip=0.01):
        """Constructs the discrete lambda grid and SURE prior distributions."""
        #self.sigma_est_ = util.gsj_est(y) if self.sigma is None else self.sigma
        self.sigma_est_ = util.gsj_est_2d_scattered(X, y) if self.sigma is None else self.sigma
        lam_grid = np.linspace(0, 1, self.lam_size)[1:-1]
        prior = np.ones(len(lam_grid))
        prior = prior / prior.sum()

        # log_lams = np.arange(self.log_lamb[0], self.log_lamb[1], self.log_lamb_step)
        # lam_grid = np.exp(log_lams)
        #
        # if self.prior_type == "light":
        #     prior_vals = gamma(a=0.3, scale=1 / 0.3).cdf(lam_grid)
        # else:
        #     prior_vals = 2 * t.cdf(25 * lam_grid, df=3) - 1
        #
        # prior = np.diff(np.r_[0, prior_vals])
        # lq, uq = np.argmax(prior_vals >= tail_clip), np.argmax(prior_vals >= 1-tail_clip)
        # r_lq, r_uq = 1 - prior[lq] / tail_clip, 1 - prior[uq] / tail_clip
        # prior[:lq] = prior[lq] * (r_lq ** np.arange(lq))[::-1]
        # prior[uq:] = prior[uq] * (r_uq ** np.arange(len(log_lams) - uq))
        # prior = prior / prior.sum()
        #
        # #TODO: scale with n_sample?
        #
        self.prior_ = prior
        self.lam_grid_ = lam_grid

    def compute_K_uv(self, d_u, d_v):
        K_uv = None
        for p in range(self.dim):
            d_up = d_u[p]
            d_vp = d_v[p]
            # --- 2. Check the Local 1D Micro-Cache ---
            cache_key_1d = (p, d_up, d_vp)
            if cache_key_1d not in self._integral_1d_cache:
                self._integral_1d_cache[cache_key_1d] = compute_spline_integral_gl(
                    self.spline_objs[p], u=d_up, v=d_vp, sparse=self.sparse
                )

            K_1d = self._integral_1d_cache[cache_key_1d]

            # --- 3. Assemble the Kronecker Product ---
            if K_uv is None:
                K_uv = K_1d
            else:
                if self.sparse:
                    K_uv = sp.kron(K_uv, K_1d, format='csr')
                else:
                    K_uv = np.kron(K_uv, K_1d)
        return K_uv

    def _weak_deriv_splits(self, deriv_order: Tuple[int, ...]) -> List[
        Tuple[Tuple[int, ...], Tuple[int, ...], float]]:
        """
        Applies multi-dimensional integration by parts on interior test splines (v|_{dOmega} = 0):
          - Pure 2nd order (e.g. (2, 0)): -> [((1, 0), (1, 0), -1.0)]
          - Mixed 2nd order (e.g. (1, 1)): -> [((1, 0), (0, 1), -0.5), ((0, 1), (1, 0), -0.5)]
          - Order <= 1 (e.g. (1, 0) or (0, 0)): -> [(self._zero_deriv, deriv_order, +1.0)]
        """
        if sum(deriv_order) == 2:
            dims = [d for d, count in enumerate(deriv_order) for _ in range(count)]
            d1, d2 = dims[0], dims[1]
            if d1 == d2:
                td = tuple(1 if d == d1 else 0 for d in range(self.dim))
                return [(td, td, -1.0)]
            else:
                da = tuple(1 if d == d1 else 0 for d in range(self.dim))
                db = tuple(1 if d == d2 else 0 for d in range(self.dim))
                return [(da, db, -0.5), (db, da, -0.5)]
        return [(self._zero_deriv, deriv_order, 1.0)]

    def compute_Weak_K_u(self, k_u: Tuple[int, ...]) -> np.ndarray:
        """Assembles the weak bilinear matrix K_u of shape (M_total, M_total) for operator k_u."""
        K_full = np.zeros((self.total_n_splines, self.total_n_splines), dtype=float)
        for test_d, trial_d, sign in self._weak_deriv_splits(k_u):
            K_part = self.compute_K_uv(test_d, trial_d)
            K_full += sign * K_part
        return K_full

    def compute_omega(self, deriv_keys):
        """
        First Compute u^th, and v^th order partial derivs.
        Then compute integral of their products
        Return Final weighted sum
        :param deriv_vals:
        :param x_eval:
        :return:
        """

        for u, k_u in enumerate(deriv_keys):
            for v, k_v in enumerate(deriv_keys[:u + 1]):
                if (k_u, k_v) not in self._omega_dict:
                    omega_uv = self.compute_K_uv(k_u, k_v)
                    self._omega_dict[k_u, k_v] = omega_uv
                    self._omega_dict[k_v, k_u] = omega_uv.T
        return

    def compute_smooth_omega(self, deriv_keys, x_eval=None):
        """
        First Compute u^th, and v^th order partial derivs.
        Then compute integral of their products
        Return Final weighted sum
        :param deriv_vals:
        :param x_eval:
        :return:
        """
        idx_I = self.idx_I
        gram_int = self.gram[np.ix_(idx_I, idx_I)]
        if self.sparse:
            gram_int = gram_int.toarray()
        self.gram_int_inv = np.linalg.inv(gram_int)
        H_0_dict = {}

        for u, k_u in enumerate(deriv_keys):
            if u not in H_0_dict:
                H_0_dict[u] = self.compute_Weak_K_u(k_u)
            H_0u = H_0_dict[u]

            for v, k_v in enumerate(deriv_keys[:u + 1]):
                H_0v = H_0_dict[v]
                omega_uv = H_0u.T[:, idx_I] @ self.gram_int_inv @ H_0v[idx_I, :]
                self._omega_dict[k_u, k_v] = omega_uv
                self._omega_dict[k_v, k_u] = omega_uv.T
        return

    @staticmethod
    def _project_function(func, B_matrix, eval_points, func_name=None):
        """
        Common projection function evaluating an analytical function and finding B-spline coefficients via least squares.
        """
        y_val = func(eval_points).ravel()
        # Project onto the B-spline basis: B * w = y
        if sp.issparse(B_matrix):
            res = spla.lsmr(B_matrix, y_val, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, maxiter=500)
            w = res[0]
        else:
            w, _, _, _ = np.linalg.lstsq(B_matrix, y_val, rcond=None)

        # Validate approximation fidelity
        g_approx = B_matrix @ w
        mse = np.mean((y_val - g_approx) ** 2)
        var_g = np.var(y_val) if np.var(y_val) > 0 else 1.0
        rel_error = mse / var_g

        if func_name is None:
            func_name = func.__name__
        logger.info(f"{func_name} Projection Relative MSE: {rel_error:.2e}")
        if rel_error > 0.05:
            logger.warning("B-spline basis may be too sparse!")
        return w

    def _project_forcing_function(self):
        """
        Projects the continuous forcing function g(x) onto the B-spline basis to find w.
        Actively reports the approximation fidelity.
        """
        return self._project_function(self.force, self.B_grid_, self.gridpoints_, func_name='Force')

    def precompute_forcing_components(self):
        """
        Computes the forcing load vector F_g = H * w using exact 1D Kronecker separation,
        eliminating the need for D-dimensional numerical integration.
        """
        w = self._project_forcing_function()
        gram = self.compute_K_uv(self._zero_deriv, self._zero_deriv)
        self.gram = gram

        if self.weak_form:
            self.compute_smooth_omega(list(self._coef_keys))
        else:
            self.compute_omega(list(self._coef_keys))

        self.F_g_dict = {}
        for i, u in enumerate(self._coef_keys):
            H_u = self.compute_K_uv(self._zero_deriv, u)
            if self.weak_form:
                H_u = self.gram[:, self.idx_I] @ self.gram_int_inv @ H_u[self.idx_I, :]
            self.F_g_dict[i] = H_u.T @ w

        self.force_norm = w.T @ self.gram @ w

    def get_omega_theoretical_scale(self, alpha, beta):
        """
        Class method wrapper to compute the theoretical Galerkin scale S
        dynamically using the model's internal knot spacing.
        """
        # Calculate the average mesh spacing 'h' for each dimension
        h_vec = np.zeros(self.dim)
        for d in range(self.dim):
            knots = self.all_knots[d]
            # Number of intervals = (total basis functions) - degree
            num_intervals = self.n_splines[d] - self.deg
            h_vec[d] = (knots[-1] - knots[0]) / num_intervals

        h_vec = np.asarray(h_vec, dtype=float)
        alpha = np.asarray(alpha, dtype=int)
        beta = np.asarray(beta, dtype=int)

        # 1. Compute the exponent for each dimension
        powers = 1 - alpha - beta
        scale_factor = np.prod(h_vec ** powers)
        return float(scale_factor)

    def _build_penalty_and_force(self, coef_vals):
        M = self.total_n_splines
        if self.sparse and not self.weak_form:
            penalty_matrix = sp.csr_matrix((M, M), dtype=np.float64)
        else:
            penalty_matrix = np.zeros((M, M))
        forcing_vector = np.zeros(self.G_.shape[0])

        for i, k_i in enumerate(self._coef_keys):
            forcing_vector += coef_vals[i] * self.F_g_dict[i]
            for j, k_j in enumerate(self._coef_keys):
                penalty_matrix += coef_vals[i] * coef_vals[j] * self._omega_dict[(k_i, k_j)]

        assert np.abs(penalty_matrix - penalty_matrix.T).max() < self.tol, (
            np.abs(penalty_matrix - penalty_matrix.T).max(), coef_vals)
        penalty_matrix = (penalty_matrix + penalty_matrix.T) / 2.0

        # --- 2. Hard Constrained Reduced Space Projection ---
        if len(self.boundary_conditions) > 0:
            T_mat = self.T_matrix_
            if not sp.issparse(penalty_matrix) and sp.issparse(T_mat):
                T_mat = T_mat.toarray()
            A_reduced = T_mat.T @ penalty_matrix @ T_mat
            v_reduced = T_mat.T @ (forcing_vector - np.array(penalty_matrix @ self.d_vec_).ravel())
            return A_reduced, v_reduced

        return penalty_matrix, forcing_vector

    def compute_theta_system_pde(self, omega_dict, beta, theta,
                                 bias_corrected=False, sigma=None):
        """
        :param omega_dict:
        :param beta:
        :return: Q, r, G
        """
        m = len(self._coef_keys)
        n = len(beta)

        Gram = np.zeros((m, m))
        for i, k_u in enumerate(self._coef_keys):
            for j, k_v in enumerate(self._coef_keys):
                omega_uv = self._omega_dict[(k_u, k_v)]
                if hasattr(omega_uv, "toarray"):
                    omega_uv = omega_uv.toarray()
                omega_uv = np.asarray(omega_uv)
                G = beta.T @ omega_uv @ beta
                # if bias_corrected:
                #     bias = np.sum(self.Vw * (omega_uv @ self.Vw)) #np.sum(diag_K_uv * M_squared)
                #     bias = bias * (sigma ** 2 / self.n_train)
                #     G = G - bias
                Gram[i, j] = G

        proj_vec = np.zeros(m)
        for i, _ in enumerate(self._coef_keys):
            proj_vec[i] = beta.T @ self.F_g_dict[i]

        Jacobian = np.zeros((n, m))
        for i, k_i in enumerate(self._coef_keys):
            Omega_i = np.zeros((n, n))
            for j, k_j in enumerate(self._coef_keys):
                Omega_i += theta[j] * (omega_dict[k_i, k_j] + omega_dict[k_j, k_i])
            Jacobian[:, i] = Omega_i @ beta - self.F_g_dict[i]

        return Gram, proj_vec, Jacobian

    def compute_physics_loss(self, theta, gram_matrices, v_vec):
        """
        Computes the Unbiased PDE Loss: L(theta | beta) = 1/2 * theta^T * G_unbiased * theta - theta^T * v
        Reuses the pre-computed system matrices and applies the EIV trace correction.
        """
        # 1. Fetch empirical matrices directly from the existing pipeline
        pde_loss = np.dot(theta, gram_matrices @ theta) - 2 * np.dot(theta, v_vec) + self.force_norm
        return float(pde_loss)

    def get_coef_dist(self, G, deriv_coef, v_vec, relaxation=1.0):
        """
        Solves the local OLS system for the physics parameters, utilizing
        Successive Over-Relaxation (SOR) to break BCD orthogonal deadlocks.
        """
        T = self._trainable_idx
        Tc = [i for i, _ in enumerate(self._coef_keys) if i not in self._trainable_idx]
        theta_new = deriv_coef.copy()

        # Check G_TT
        G_TT = G[np.ix_(T, T)]
        self.G_TT = G_TT
        if np.linalg.eigvals(G_TT).min() < self.tol:
            logger.warning('theta eq. not Positive!')
        if np.linalg.cond(G_TT) > 1 / self.tol:
            logger.warning('theta eq. not Invertible!')
        GTT_inv = np.linalg.inv(G_TT)

        # 1. Calculate the pure OLS target
        rhs = v_vec[T]
        if len(Tc) > 0:
            coef_Tc = deriv_coef[Tc]
            G_TTc = G[np.ix_(T, Tc)]
            rhs = rhs - G_TTc @ coef_Tc
        c_calc = GTT_inv @ rhs

        # 2. Apply Momentum / Over-Relaxation to accelerate the tail convergence
        theta_new[T] = deriv_coef[T] + relaxation * (c_calc - deriv_coef[T])
        return theta_new

    def get_coef_dist_corrected(self, G, deriv_coef, v_vec, A, B, relaxation=1.0):
        """
        Solves the symmetric, PSD Godambe information system:
            (A^T @ B_inv @ A + mu * I) @ Delta_theta_T = -A^T @ B_inv @ Psi
        where:
            A: Bread matrix (J_TT + M_ind)
            B: Meat matrix (score covariance)
            Psi: G_TT @ theta_T + G_TTc @ theta_Tc - v_T
        """
        T = self._trainable_idx
        Tc = [i for i, _ in enumerate(self._coef_keys) if i not in T]
        theta_new = deriv_coef.copy()
        theta_T_old = deriv_coef[T]

        G_TT = G[np.ix_(T, T)]
        self.G_TT = G_TT

        # 1. Compute estimating equation residual Psi(theta^(t))
        Psi = G_TT @ theta_T_old - v_vec[T]
        if len(Tc) > 0:
            Psi += G[np.ix_(T, Tc)] @ deriv_coef[Tc]

        # 2. Invert Meat matrix B with Tikhonov safeguard
        B_inv = np.linalg.inv(B)

        # 3. Assemble symmetric, PSD system matrix S and RHS gradient
        S = A.T @ B_inv @ A
        rhs_grad = -A.T @ (B_inv @ Psi)
        delta_theta = np.linalg.solve(S, rhs_grad)

        # 6. Successive Over-Relaxation (SOR) update
        theta_new[T] = theta_T_old + relaxation * delta_theta
        return theta_new

    def project_bc(self, bc_func, d, x_star):
        dims_sub = [j for j in range(self.dim) if j != d]
        # --- 1. Project boundary conditions on the N-1 dimension sub-grid ---
        if self.dim > 1:
            grids_sub = [self.grids_[j] if j != d else np.array([x_star]) for j in range(self.dim)]
            eval_points = util.tensor_prod(*grids_sub)
            if self.sparse:
                B_sub = build_custom_sparse_tensor_basis(eval_points, self.n_splines, self.all_knots,
                                                            self.deg, self.dim, dims_sub)
            else:
                B_sub = build_custom_tensor_basis(eval_points, self.spline_objs, self.dim, dims_sub)
            c_sub = self._project_function(bc_func, B_sub, eval_points,
                                           func_name=f'Boundary Condition at dim {self.variables[d]}={x_star}')
        else:
            eval_points = np.array([[x_star]])
            c_sub = bc_func(eval_points).ravel()
        return c_sub

    def locate_boundary_idx(self):
        """
        Locates the flat indices of boundary splines (idx_B_) and interior splines (idx_I_)
        for the tensor-product B-spline grid.
        A spline is interior iff in all dimensions d, 1 <= i_d <= n_splines[d] - 2.

        Returns:
        --------
        idx_B : np.ndarray
            Flat indices of boundary B-splines.
        idx_I : np.ndarray
            Flat indices of interior B-splines.
        """
        is_interior = np.zeros(self.n_splines, dtype=bool)
        interior_slices = tuple(slice(1, -1) for _ in range(self.dim))
        is_interior[interior_slices] = True

        self.idx_I = np.flatnonzero(is_interior)
        self.idx_B = np.flatnonzero(~is_interior)
        assert len(self.idx_B) == len(np.unique(self.idx_B))
        assert len(self.idx_I) == np.prod(self.n_splines - 2)
        return self.idx_B, self.idx_I

    def _parse_boundary_key(self, bound_str: str) -> Tuple[int, str]:
        """Parses strings like '0_min' or 'x_max' into integer axis and side."""
        d_str, side = bound_str.split('_')
        d = self.variables.index(d_str) if d_str in self.variables else int(d_str)
        assert side in ['min', 'max'], "Side must be 'min' or 'max'."
        return d, side

    def add_boundary_conditions(self, bc_dict: Dict[Tuple[str, int], callable]):
        """
        Incorporates Generalized boundary conditions (Dirichlet, Neumann, etc.) into the tensor B-splines.

        Parameters:
        -----------
        bc_dict : Dict[Tuple[str, int], callable]
            Dictionary with keys as (boundary_string, p).
            e.g., ('0_min', 0) for Dirichlet on dimension 0 minimum.
            e.g., ('1_max', 1) for Neumann on dimension 1 maximum.

        Returns:
        --------
        T_matrix : np.ndarray
            The affine elimination matrix isolating interior variables.
        d_vec : np.ndarray
            The boundary constants offset vector.
        """
        M_total = self.total_n_splines

        if len(bc_dict) == 0:
            self.idx_B_BC = np.array([], dtype=int)
            self.idx_I_BC = np.arange(M_total, dtype=int)
            self.T_matrix_ = (
                sp.eye(M_total, format="csr", dtype=np.float64)
                if self.sparse
                else np.eye(M_total)
            )
            self.d_vec_ = np.zeros(M_total)
            return self.T_matrix_, self.d_vec_

        is_dependent = np.zeros(self.n_splines, dtype=bool)
        rows_A, cols_A, vals_A = [], [], []
        c_eqs, dep_indices = [], []
        eq_counter = 0
        sorted_items = sorted(bc_dict.items(), key=lambda item: self._parse_boundary_key(item[0][0]))

        for (bound_str, p), bc_func in sorted_items:
            d, side = self._parse_boundary_key(bound_str)
            x_star = self.domain[d][0] if side == 'min' else self.domain[d][1]
            c_sub = self.project_bc(bc_func, d, x_star)
            weights = self.spline_objs[d](np.array([x_star]), nu=p)[0]
            assert len(weights) == self.n_splines[d]

            # --- 2. Compute the constraints on our regression coefficients (A * beta = c) ---
            sub_shape = [self.n_splines[j] for j in range(self.dim) if j != d]
            sub_ranges = product(*[range(s) for s in sub_shape])
            boundary_idx = 0 if side == 'min' else self.n_splines[d] - 1
            active_k_indices = np.nonzero(weights)[0]

            for i, sub_midx in enumerate(sub_ranges):
                bound_midx = sub_midx[:d] + (boundary_idx,) + sub_midx[d:]

                # Application of "Shell-Peeling"
                if not is_dependent[bound_midx]:
                    is_dependent[bound_midx] = True
                    dep_indices.append(np.ravel_multi_index(bound_midx, self.n_splines))

                    # Assemble the constraint row based ONLY on the sparse derivative stencil
                    for k in active_k_indices:
                        k_midx = sub_midx[:d] + (k,) + sub_midx[d:]
                        flat_k_idx = np.ravel_multi_index(k_midx, self.n_splines)
                        rows_A.append(eq_counter)
                        cols_A.append(flat_k_idx)
                        vals_A.append(weights[k])

                    c_eqs.append(c_sub[i])
                    eq_counter += 1

        # --- 3. Segment independent (interior) vs dependent (boundary) coefficients ---
        self.idx_B_BC = np.array(dep_indices, dtype=int)
        self.idx_I_BC = np.setdiff1d(np.arange(M_total), self.idx_B_BC)

        A_csr = sp.csr_matrix(
            (vals_A, (rows_A, cols_A)), shape=(eq_counter, M_total), dtype=float
        )
        c_full = np.array(c_eqs, dtype=float)

        if self.sparse:
            n_I = len(self.idx_I_BC)
            A_B = A_csr[:, self.idx_B_BC].tocsc()
            A_I = A_csr[:, self.idx_I_BC].tocsc()

            if A_B.shape[0] <= 2500:
                cond_num = np.linalg.cond(A_B.toarray())
                assert (
                        cond_num < 1e12
                ), f"A_B is ill-conditioned! Condition number: {cond_num:.2e}"

            invA_c = spla.spsolve(A_B, c_full)
            invA_AI = spla.spsolve(A_B, A_I)
            if not sp.issparse(invA_AI):
                invA_AI = sp.csc_matrix(invA_AI)

            invA_AI_coo = (-invA_AI).tocoo()
            T_rows = np.concatenate([self.idx_I_BC, self.idx_B_BC[invA_AI_coo.row]])
            T_cols = np.concatenate([np.arange(n_I, dtype=int), invA_AI_coo.col])
            T_vals = np.concatenate([np.ones(n_I, dtype=float), invA_AI_coo.data])

            self.T_matrix_ = sp.csr_matrix(
                (T_vals, (T_rows, T_cols)), shape=(M_total, n_I), dtype=float
            )

        else:
            A_full = A_csr.toarray()
            A_B, A_I = A_full[:, self.idx_B_BC], A_full[:, self.idx_I_BC]
            cond_num = np.linalg.cond(A_B)
            assert cond_num < 1e12, f"A_B is too ill-conditioned for stable inversion! Condition number: {cond_num:.2e}"

            invA_c = np.linalg.solve(A_B, c_full)
            invA_AI = np.linalg.solve(A_B, A_I)

            self.T_matrix_ = np.zeros((M_total, len(self.idx_I_BC)))
            self.T_matrix_[self.idx_I_BC, :] = np.eye(len(self.idx_I_BC))
            self.T_matrix_[self.idx_B_BC, :] = -invA_AI

        self.d_vec_ = np.zeros(M_total)
        self.d_vec_[self.idx_B_BC] = invA_c
        return self.T_matrix_, self.d_vec_

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the EWA Spline model to the data."""
        X, y = check_X_y(X, y, accept_sparse=False)
        n = len(y)
        self.X_fit_, self.y_fit_, self.n_train = X, y, n

        self._parse_physics_configuration()
        self._setup_priors_and_lambda(X, y)
        self._initialize_basis(X, y)
        self.locate_boundary_idx()
        self.precompute_forcing_components()
        self.add_boundary_conditions(self.boundary_conditions)

        # --- Project Data Loss into the Unconstrained Space ---
        if len(self.boundary_conditions) > 0:
            self.B_data_reduced_ = self.B_data_ @ self.T_matrix_
            self.G_reduced_ = self.T_matrix_.T @ self.G_ @ self.T_matrix_
            self.y_adj_ = y - (self.B_data_ @ self.d_vec_)
        else:
            self.B_data_reduced_ = self.B_data_
            self.G_reduced_ = self.G_
            self.y_adj_ = y

        coef_vals = self._initial_coef_vals
        self.fit_stats = {}
        last_iter = False
        # --- Block Coordinate Descent for Physics Parameters ---
        for iteration in range(self.max_iter):
            penalty_matrix, forcing_vector = self._build_penalty_and_force(coef_vals)
            G_norm = self.G_reduced_ / n
            m = G_norm.shape[0]

            if (self.max_eigfunc is not None) and (m > 3000):
                k = min(min(self.max_eigfunc, m - 1), 3000)
                evals, evecs = util.compute_smallest_eigenvectors(penalty_matrix, G_norm, k)
                d = evecs.T @ self.B_data_reduced_.T @ self.y_adj_ / n
                var_exp = np.cumsum(d ** 2)
                var_exp = (var_exp + self.sigma_est_ ** 2) / np.mean(self.y_adj_ ** 2)
                logger.info(f'{var_exp[-1]:.2%} Var Explained with {k} basis')
            else:
                evals, evecs = util.orthogonalize_basis_fast(G_norm, penalty_matrix)
                # evals = evals[:k]
                # evecs = evecs[:, :k]

            self.V_, self.s_raw_ = evecs, evals #G^{-1} = VV^T
            self.U_ = self.B_data_reduced_ @ self.V_

            # 2. Project Forcing Vector into Orthogonal Space
            self.force_proj = self.V_.T @ forcing_vector
            self.s_ = np.maximum(self.s_raw_, 0.0)
            logger.info('Eigvals: '+ ', '.join([f'{s:.4f}' for s in self.s_raw_[:8]]))

            # 3. Fit EWA Model
            self.ewa_res_ = util.fit_ewa(
                y=self.y_adj_, ftrue=self.y_adj_, U=self.U_, V=self.V_, s=self.s_,
                lam_vec=self.lam_grid_, prior=self.prior_,
                sigma_est=self.sigma_est_, gamma_mult=self.gamma_mult,
                force_proj=self.force_proj
            )

            self.sigma_est_ = self.ewa_res_['sigma_est']

            # --- NEW: 4. Construct the rigid physical surrogate (\tilde{\beta}) ---
            beta_I = self.ewa_res_['beta']
            if len(self.boundary_conditions) > 0:
                self.beta_full_ = self.T_matrix_ @ beta_I + self.d_vec_
                self.Vw = (self.T_matrix_ @ self.V_) * self.ewa_res_['shrink_weight']
            else:
                self.beta_full_ = beta_I
                self.Vw = self.V_ * self.ewa_res_['shrink_weight']

            if last_iter:
                break

            gram_matrices, proj_vec, J_phi = self.compute_theta_system_pde(self._omega_dict,
                                                                           self.beta_full_,
                                                                           coef_vals,
                                                                           sigma=self.sigma_est_,)

            if not self.bias_corrected:
                coef_mean = self.get_coef_dist(gram_matrices, coef_vals, proj_vec)
            else:
                A, B, _, _, _ = self.compute_asymptotic_variance(coef_vals, self.ewa_res_, self.y_adj_, self.force_proj,
                                                                                lam_vec=self.lam_grid_, U=self.U_, V=self.V_, s=self.s_)
                coef_mean = self.get_coef_dist_corrected(gram_matrices, coef_vals, proj_vec, A, B)

            w = self.ewa_res_['weights']
            lamb_mean = (self.lam_grid_ * w).sum()
            lamb_mle = self.lam_grid_[np.argmax(w)]
            lamb_lq = self.lam_grid_[np.argmax(np.cumsum(w) >= self.alpha / 2)]
            lamb_uq = self.lam_grid_[np.argmax(np.cumsum(w) >= 1 - self.alpha / 2)]

            physics_loss = self.compute_physics_loss(coef_mean, gram_matrices, proj_vec)
            coef_dict = dict(zip(self.coef_keys, coef_mean))

            logger.info(f"Iteration {iteration} | Coefficients: {coef_dict} | "
                        f"sigma_est: {self.sigma_est_:.4e} | physics_loss: {physics_loss:.4e} | "
                        #f"Penalty norm: {p_norm: .4e} | "
                        f"lamb_mean: {lamb_mean:.4e} | lamb_mle: {lamb_mle:.4e}") #| hess_T: {hess_T}

            self.fit_stats[iteration] = np.array([physics_loss, n * self.sigma_est_ ** 2,
                                                  lamb_lq, lamb_mean, lamb_uq])

            # 5. Check Convergence
            old_theta_T = coef_vals[self._trainable_idx]
            new_theta_T = coef_mean[self._trainable_idx]
            coef_pct_chg = np.abs(old_theta_T - new_theta_T).sum() / (np.abs(old_theta_T).sum() + 1e-8)

            coef_vals = coef_mean
            if coef_pct_chg < self.min_param_pct_chg:
                logger.info("Physics parameters converged.")
                last_iter = True

            # old_physics_loss = self.fit_stats.get(iteration - 1, [np.inf])[0]
            # loss_pct_decr = 1 - (physics_loss / old_physics_loss)
            #
            # if loss_pct_decr <= self.min_pde_loss_chg:
            #     logger.info("Physics parameters converged.")
            #     break

        # --- Finalize Fitted State ---
        self.deriv_coef = {self.deriv_map[self._coef_keys[i]]: v for i, v in enumerate(coef_vals)} #coef_vals
        self.coef_mean = coef_mean

        self.bread_matrix, self.meat_matrix, self.cov_beta_sqrt, _, _ = self.compute_asymptotic_variance(self.coef_mean, self.ewa_res_, self.y_adj_, self.force_proj,
                                                                                                   lam_vec=self.lam_grid_, U=self.U_, V=self.V_, s=self.s_)
        bread_inv = np.linalg.inv(self.bread_matrix)
        self.coef_var = bread_inv @ self.meat_matrix @ bread_inv.T
        self.fit_stats = pd.DataFrame(self.fit_stats, index=['pde_loss', 'fit_loss',
                                                             'w_lq', 'w_mean', 'w_uq']).T.rename_axis('iter')
        return self

    def compute_cov_beta(self, omega_matrix, weights, H_lam, V, r, sig2, n, gamma):
        E_D = np.diag(omega_matrix @ weights)  # (p, p)

        A_grid = H_lam * r[:, np.newaxis]  # (p, n_lambda)
        B_grid = (H_lam ** 2) * r[:, np.newaxis]  # (p, n_lambda)
        B_centered = B_grid - (B_grid @ weights)[:, np.newaxis]  # (p, n_lambda)
        Cov_w = A_grid @ (B_centered * weights[np.newaxis, :]).T  # (p, p)

        Xi = E_D + (2.0 * n / gamma) * Cov_w  # (p, p)

        has_bc = len(self.boundary_conditions) > 0
        TV = (self.T_matrix_ @ V) if has_bc else V
        cov_beta_sqrt = np.sqrt(sig2 / n) * (TV @ Xi)
        return Xi, cov_beta_sqrt

    def compute_pde_projected_operators(
            self,
            theta,
            V: np.ndarray,
            d: np.ndarray,
            omega_dict,
            F_g_dict,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Projects ambient PDE penalty Hessians and forcing gradients into the GEVP

        basis V, and computes the projected OLS parameter residual matrix J_OLS:

            1. H_matrices[k]   = V^T @ ambient_H_matrices[k] @ V
            2. force_proj_grad = V^T @ ambient_force_grad
            3. J_OLS[:, k]     = H_matrices[k] @ d - force_proj_grad[:, k]
        """
        p = self.total_n_splines
        p_red = V.shape[0]
        K = len(self._trainable_idx)

        has_bc = len(self.boundary_conditions) > 0
        TV = (self.T_matrix_ @ V) if has_bc else V
        d_vec = self.d_vec_ if has_bc else np.zeros(p)

        H_matrices = []
        force_proj_grad = np.zeros((p_red, K))
        for i, t_i in enumerate(self._trainable_idx):
            k_i = self._coef_keys[t_i]
            H_k = np.zeros((p, p))
            for j, k_j in enumerate(self._coef_keys):
                H_k += theta[j] * (omega_dict[(k_i, k_j)] + omega_dict[(k_j, k_i)])

            H_proj = TV.T @ H_k @ TV #V.T @ H_k @ V
            f_proj = TV.T @ (F_g_dict[t_i] - np.array(H_k @ d_vec).flatten())

            H_matrices.append(H_proj)
            force_proj_grad[:, i] = f_proj

        # 4. Assemble projected OLS residual matrix J_OLS:
        # J_OLS[:, k] = \widetilde{\mathcal{H}}_k @ d - \tilde{f}_k = V^T (H_k @ beta_OLS - f_k)
        J_OLS = np.zeros((p_red, K))
        for k in range(K):
            J_OLS[:, k] = H_matrices[k] @ d - force_proj_grad[:, k]
        return J_OLS, H_matrices, force_proj_grad

    def compute_grad_beta_lambda(
            self,
            H_s: np.ndarray,
            H_lam: np.ndarray,
            r: np.ndarray,
            J_OLS: np.ndarray,
            H_matrices: list[np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        """Computes the gradient of candidate splines with respect to theta_k on lambda grid:

            d(alpha(lambda)) / d(theta_k) = - (h(lambda)/s) * J_OLS[:, k] +
            (h(lambda)/s) * (H_k @ (h(lambda)*r))

        -------
        grad_alpha : np.ndarray
            Modal candidate spline gradients, shape (p, n_lambda, K).
        grad_beta : np.ndarray or None
            Ambient candidate spline gradients, shape (p, n_lambda, K).
        """
        p, n_lam = H_lam.shape
        K = J_OLS.shape[1]

        H_r = H_lam * r[:, np.newaxis]  # (p, n_lambda)
        grad_alpha = np.zeros((p, n_lam, K))

        for k in range(K):
            term1 = -H_s * J_OLS[:, k, np.newaxis]
            term2 = H_s * np.array(H_matrices[k] @ H_r)
            grad_alpha[:, :, k] = term1 + term2
        return grad_alpha

    def compute_grad_sure(
            self,
            grad_alpha: np.ndarray,
            H_lam: np.ndarray,
            omega_matrix: np.ndarray,
            H_s: np.ndarray,
            r: np.ndarray,
            H_matrices: list[np.ndarray] | np.ndarray,
            n: int,
            sig2: float,
    ) -> np.ndarray:
        """Computes the SURE risk gradient matrix G_SURE across the candidate grid:

            d(SURE(lambda)) / d(theta_k) = -2n * (h(lambda)*r)^T (d alpha(lambda) /
            d theta_k)
                                           - 2 * sig2 * tr(W_df(lambda) * H_k)
        Returns
        -------
        G_SURE : np.ndarray
            SURE gradient matrix, shape (n_lambda, K).
        """
        K = grad_alpha.shape[2]
        H_r = H_lam * r[:, np.newaxis]  # (p, n_lambda)

        # Term 1 & 2: d(RSS) / d(theta_k) via contraction with grad_alpha
        # -2n * sum_p [ (h * r) * (d alpha / d theta_k) ]
        d_RSS = -2.0 * n * np.sum(H_r[:, :, np.newaxis] * grad_alpha, axis=0)  # (n_lambda, K)

        # Term 3: Complexity penalty trace term -2 * sig2 * tr(W_df @ H_k)
        W_df = omega_matrix * H_s  # (p, n_lambda)
        D_H = np.column_stack([np.diag(H_matrices[k]) for k in range(K)])  # (p, K)
        d_df = -2.0 * sig2 * (W_df.T @ D_H)  # (n_lambda, K)

        G_SURE = d_RSS + d_df
        return G_SURE

    def compute_grad_beta_ewa(
            self,
            grad_alpha: np.ndarray,
            G_SURE: np.ndarray,
            weights: np.ndarray,
            H_lam: np.ndarray,
            r: np.ndarray,
            gamma: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the profile EWA spline Jacobian combining direct model

        sensitivity and adaptive SURE model selection:

            J_EWA_spectral = E_w[ d alpha(lambda) / d theta ] - (H*r) @ J_w

        Returns
        -------
        J_EWA_spectral : np.ndarray
            Modal profile Jacobian (d alpha_EWA / d theta_T^T), shape (p, K).
        J_w : np.ndarray
            Gibbs weight Jacobian d(w) / d(theta_T)^T, shape (n_lambda, K).
        J_EWA : np.ndarray or None
            Ambient profile Jacobian d beta_EWA / d theta_T^T, shape (p, K).
        """
        # 1. Weight Jacobian J_w (centered Gibbs derivative)
        g_bar = weights @ G_SURE  # (K,)
        J_w = -(1.0 / gamma) * weights[:, np.newaxis] * (G_SURE - g_bar[np.newaxis, :])  # (n_lambda, K)

        # 2. Term 1: Direct sensitivity expectation E_w[ d alpha(lambda) / d theta ]
        J_direct = np.sum(grad_alpha * weights[np.newaxis, :, np.newaxis], axis=1)  # (p, K)

        # 3. Term 2: Adaptive selection feedback - (H * r) @ J_w
        H_r = H_lam * r[:, np.newaxis]  # (p, n_lambda)
        J_adapt = -H_r @ J_w  # (p, K)

        J_alpha_EWA = J_direct + J_adapt  # (p, K)
        return J_alpha_EWA

    def compute_asymptotic_variance(
            self,
            theta,
            fit_res: dict,
            y_fit: np.ndarray,
            force_proj: np.ndarray | None,
            lam_vec,
            U: np.ndarray,
            V: np.ndarray,
            s: np.ndarray,
    ):
        """
        cov_beta : np.ndarray
            Profile spline covariance Cov(beta_EWA), shape (p, p).
        """
        n = len(y_fit)
        T = self._trainable_idx
        K = len(T)

        sig2 = fit_res['sigma_est'] ** 2
        gamma = fit_res['gamma']
        weights = fit_res["weights"]
        shrinkage_matrix = fit_res['shrinkage_matrix']
        alpha_ewa = fit_res['alpha_hat']
        pos_mask = s > self.tol
        p = len(alpha_ewa)

        # 1. Spectral defect vector r (shape: p)
        d = U.T @ y_fit / n
        r = d.copy()
        if force_proj is not None:
            r[pos_mask] = d[pos_mask] - (force_proj[pos_mask] / s[pos_mask])

        # 2. Spectral shrinkage operators
        omega_matrix = fit_res["omega_matrix"].copy()
        omega_matrix[~pos_mask, :] = 1.0
        H_lam = 1.0 - omega_matrix  # I - D
        H_s = lam_vec[np.newaxis, :] * shrinkage_matrix
        H_s[~pos_mask, :] = 0.0 # TODO!!!!!

        Xi, cov_beta_sqrt = self.compute_cov_beta(omega_matrix, weights, H_lam, V, r, sig2, n, gamma)
        J_OLS, H_matrices, force_proj_grad = self.compute_pde_projected_operators(theta, V, d,
                                                                                  self._omega_dict, self.F_g_dict)

        grad_alpha = self.compute_grad_beta_lambda(H_s, H_lam, r, J_OLS, H_matrices)
        G_SURE = self.compute_grad_sure(grad_alpha, H_lam, omega_matrix, H_s, r, H_matrices, n, sig2)
        J_alpha_EWA = self.compute_grad_beta_ewa(grad_alpha, G_SURE, weights, H_lam, r, gamma)

        # --- Step 4: Metric Tensor J_TT, K_tilde, and Bread matrix A ---
        # J_TT (Metric Tensor G_TT)
        has_bc = len(self.boundary_conditions) > 0
        TV = (self.T_matrix_ @ V) if has_bc else V
        d_vec = self.d_vec_ if has_bc else np.zeros(p)
        beta = TV @ alpha_ewa + d_vec

        J_TT = np.zeros((K, K))
        for i, t_i in enumerate(T):
            for j, t_j in enumerate(T):
                k_u, k_v = self._coef_keys[t_i], self._coef_keys[t_j]
                omega_uv = self._omega_dict[(k_u, k_v)]
                if hasattr(omega_uv, "toarray"):
                    omega_uv = omega_uv.toarray()
                omega_uv = np.asarray(omega_uv)
                J_TT[i, j] = beta.T @ omega_uv @ beta

        # K_tilde (Projected Score Jacobian J_Phi @ V)
        K_tilde = np.zeros((K, p))
        for k in range(K):
            K_tilde[k, :] = H_matrices[k] @ alpha_ewa - force_proj_grad[:, k]

        A = J_TT + K_tilde @ J_alpha_EWA # Extra non-linear term! K_tilde @ J_alpha_EWA!
        K_Xi = K_tilde @ Xi  # (K, p_red)
        B = (sig2 / n) * (K_Xi @ K_Xi.T)  # (K, K)
        return A, B, cov_beta_sqrt, K_tilde, J_alpha_EWA

    def plot_fit_stats(self, logy=True, **kwargs):
        """Plots Training Losses and Lambda distribution evolution in a single figure."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # 1. Plot Losses (Top Subplot)
        loss_cols = self.fit_stats.columns.str.contains('loss')
        loss_df = self.fit_stats.loc[:, loss_cols]

        loss_df.plot(logy=logy, ax=axes[0], **kwargs)
        axes[0].set_title("Training Loss Evolution")
        axes[0].set_ylabel("Loss Magnitude")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend(loc='best')

        # 2. Plot Lambda Evolution (Bottom Subplot)
        axes[1].plot(self.fit_stats['w_mean'], color='darkblue', label='Expected $\lambda$')
        axes[1].fill_between(self.fit_stats.index,
                             self.fit_stats['w_lq'],
                             self.fit_stats['w_uq'],
                             color='blue', alpha=0.2, label=f'{1 - self.alpha:.0%} CI')

        # Set log scale for Lambda plot
        if logy:
            axes[1].set_yscale('log')

        axes[1].set_title("Evolution of $\lambda$ PDF")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Penalty $\lambda$ (log scale)")
        axes[1].legend(loc='best')
        axes[1].grid(True, linestyle='--', alpha=0.6, which="both")

        plt.tight_layout()
        plt.show()

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Predicts the continuous function evaluation at X alongside confidence intervals."""
        check_is_fitted(self, 'ewa_res_')
        X = check_array(X, ensure_2d=False)

        B_new = self._build_tensor_basis(X)
        y_hat = B_new @ self.beta_full_

        # CI of the fitted function itself
        BL = B_new @ self.cov_beta_sqrt
        se = np.sqrt(np.sum(BL**2, axis=1))
        interval = norm.ppf(1 - self.alpha / 2) * se

        idx = pd.MultiIndex.from_arrays(X.T, names=self.variables)
        return pd.DataFrame({
            'fhat': y_hat,
            'se': se,
            'ci_lower': y_hat - interval,
            'ci_upper': y_hat + interval
        }, index=idx)

    def plot_fit(self, x_test=None, y_test=None, true_coef_vals=None):
        """
        Plots diagnostic subplots for the fitted EWA surface:
        1. Scatter plot of Data Residuals (Y - X * beta_ewa) with CI Error bounds
        2. Heatmap of Learnt Pointwise PDE Loss
        3. Heatmap of True Pointwise PDE Loss (Optional)
        """
        check_is_fitted(self)
        x_grid = self.gridpoints_

        # Get continuous predictions on the grid for the CI Error Bounds
        fit_preds_grid = self.predict(x_grid)

        # --- 1. Compute Data Residuals ---
        if x_test is None:
            x_scatter, y_scatter = self.X_fit_, self.y_fit_
        else:
            x_scatter, y_scatter = x_test, y_test

        fit_preds_scatter = self.predict(x_scatter)
        data_residuals = y_scatter - fit_preds_scatter['fhat'].values

        # Define Error CI Bounds relative to the fitted surface
        err_ci_lower = fit_preds_grid['ci_lower'] - fit_preds_grid['fhat']
        err_ci_upper = fit_preds_grid['ci_upper'] - fit_preds_grid['fhat']

        # --- 2. Compute Pointwise PDE Loss (Learnt Parameters) ---
        pde_loss_learnt = -self.force(x_grid).ravel()
        for i, (k, v) in enumerate(self.deriv_coef.items()):
            order = self._coef_keys[i]
            pde_loss_learnt += v * self.predict_derivative(x_grid, order)

        # --- 3. Compute Pointwise PDE Loss (True Parameters) ---
        pde_loss_true = None
        if true_coef_vals is not None:
            pde_loss_true = -self.force(x_grid).ravel()

            # Handle if user passes a dictionary (e.g., {'0,0': 1.0}) or an aligned array
            if isinstance(true_coef_vals, dict):
                parsed_true_coefs = {get_deriv_order(k, self.variables): v for k, v in true_coef_vals.items()}
                for i, order in enumerate(self._coef_keys):
                    theta_true = parsed_true_coefs.get(order, 0.0)
                    pde_loss_true += theta_true * self.predict_derivative(x_grid, order)
            else:
                for i, order in enumerate(self._coef_keys):
                    pde_loss_true += true_coef_vals[i] * self.predict_derivative(x_grid, order)

        # --- 4. Plotting Setup ---
        n_subplots = 3 if pde_loss_true is not None else 2
        fig, axes = plt.subplots(1, n_subplots, figsize=(5.5 * n_subplots, 5), layout='constrained')

        # Ensure axes is iterable even if it's somehow collapsed
        if n_subplots == 2:
            axes = [axes[0], axes[1]]

        # --- 1D Case ---
        if self.dim == 1:
            # Subplot 1: Data Residuals with Shaded Error CI
            axes[0].fill_between(x_grid[:, 0], err_ci_lower, err_ci_upper, color='C0', alpha=0.15,
                                 label=f'{1 - self.alpha:.0%} CI')
            axes[0].scatter(x_scatter[:, 0], data_residuals, color='black', alpha=0.6, s=15, label='Residuals')
            axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
            axes[0].set(title="Data Residuals $(Y - \hat{Y})$", xlabel="X", ylabel="Residual")
            axes[0].legend(loc='best')

            # Subplot 2: Learnt PDE Loss
            axes[1].plot(x_grid[:, 0], pde_loss_learnt, color='tab:orange', lw=2)
            axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
            axes[1].set(title="Learnt PDE Pointwise Loss", xlabel="X", ylabel="PDE Residual")

            # Subplot 3: True PDE Loss
            if pde_loss_true is not None:
                axes[2].plot(x_grid[:, 0], pde_loss_true, color='tab:green', lw=2)
                axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
                axes[2].set(title="True PDE Pointwise Loss", xlabel="X", ylabel="PDE Residual")

        # --- 2D Case ---
        elif self.dim == 2:
            xlabel, ylabel = self.variables
            X_mesh = x_grid[:, 0].reshape(*self.grid_size_)
            Y_mesh = x_grid[:, 1].reshape(*self.grid_size_)

            # Subplot 1: Data Residual Scatter + CI Contours
            vmax_res = np.max(np.abs(data_residuals))
            if vmax_res == 0: vmax_res = 1e-8

            # Plot CI Contour overlays (since bounds are symmetric, plotting the upper margin maps the +/- envelope)
            err_ci_upper_mesh = err_ci_upper.values.reshape(*self.grid_size_)
            cs = axes[0].contour(X_mesh, Y_mesh, err_ci_upper_mesh, levels=5, colors='black', alpha=0.4,
                                 linestyles='dashed')
            axes[0].clabel(cs, inline=True, fontsize=8, fmt='CI: ±%.2f')

            # Plot scatter
            norm_res = util.norm_color(vmin=-vmax_res, vmax=vmax_res)
            sc = axes[0].scatter(x_scatter[:, 0], x_scatter[:, 1], c=data_residuals,
                                 norm=norm_res, cmap=util.CMAP, edgecolors='black', s=20)
            fig.colorbar(sc, ax=axes[0], shrink=0.8, pad=0.04)
            axes[0].set(title="Data Residuals $(Y - \hat{Y})$", xlabel=xlabel, ylabel=ylabel)

            # Subplot 2: Learnt PDE Loss Heatmap
            Z_learnt = pde_loss_learnt.reshape(*self.grid_size_)
            vmax_pde = np.max(np.abs(Z_learnt))
            if vmax_pde == 0: vmax_pde = 1e-8

            _, ax1 = util.contour_plot(X_mesh, Y_mesh, Z_learnt, ax=axes[1], vmin=-vmax_pde, vmax=vmax_pde, levels=30,
                                       cbar=False)
            sm_learnt = util.map_color(vmin=-vmax_pde, vmax=vmax_pde)
            fig.colorbar(sm_learnt, ax=axes[1], shrink=0.8, pad=0.04)
            axes[1].set(title="Learnt PDE Pointwise Loss", xlabel=xlabel, ylabel=ylabel)

            # Subplot 3: True PDE Loss Heatmap
            if pde_loss_true is not None:
                Z_true = pde_loss_true.reshape(*self.grid_size_)
                vmax_true = np.max(np.abs(Z_true))
                if vmax_true == 0: vmax_true = 1e-8

                _, ax2 = util.contour_plot(X_mesh, Y_mesh, Z_true, ax=axes[2], vmin=-vmax_true, vmax=vmax_true,
                                           levels=30, cbar=False)
                sm_true = util.map_color(vmin=-vmax_true, vmax=vmax_true)
                fig.colorbar(sm_true, ax=axes[2], shrink=0.8, pad=0.04)
                axes[2].set(title="True PDE Pointwise Loss", xlabel = xlabel, ylabel = ylabel)

        else:
            logger.warning(f"Plotting not supported for dimension {self.dim} > 2.")

        # --- 5. Common Formatting ---
        deriv_coef = {str(k): float(f"{v:.4g}") for k, v in zip(self.coef_keys, self.deriv_coef_vals_)}
        title = (
            f"EWA Fit Diagnostics | n={len(x_scatter)} | deg={self.deg}\n"
            f"Learnt Coefs: {deriv_coef}"
        )
        fig.suptitle(title, fontsize=14, y=1.05)
        plt.show()

    @property
    def basis_functions(self):
        # Expand reduced orthogonal vectors V_ back to full grid dimensions
        if len(self.boundary_conditions) > 0:
            basis = self.B_grid_ @ (self.T_matrix_ @ self.V_)
        else:
            basis = self.B_grid_ @ self.V_
        basis_wgt = self.ewa_res_['alpha_hat']

        basis_decomp = basis * basis_wgt[None, :]
        feature_importance = (basis_decomp ** 2).sum(axis=0)
        feature_importance = feature_importance / feature_importance.sum()
        multi_idx = pd.MultiIndex.from_arrays(self.gridpoints_.T, names=self.variables)

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
        #common_norm = Normalize(vmin=-global_vmax, vmax=global_vmax)

        # --- 1D Case: All bases on a single line plot ---
        if self.dim == 1:
            top_bases = df2.iloc[:, :n_col]
            ax = top_bases.plot(title=f'Top {n_col} Basis Functions ({exp_var[n_col - 1] * 100:.1f}% Var)', **kwargs)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[0:n_col], labels[0:n_col])
            plt.show()

        # --- 2D Case: Grid of Heatmaps ---
        elif self.dim == 2:
            xlabel, ylabel = self.variables
            grid_shape = (len(self.grids_[0]), len(self.grids_[1]))
            X1, X2 = np.meshgrid(self.grids_[0], self.grids_[1], indexing='ij')

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
                _, ax = util.contour_plot(X1, X2, Z, ax=ax,
                                            vmin=-global_vmax, vmax=global_vmax, levels=30, cbar=False)
                #ax.contourf(X1, X2, Z, levels=30, cmap=util.CMAP, norm=common_norm)
                ax.set(title=f"Basis {idx + 1} [Var: {df2.columns[idx]}]", xlabel = xlabel, ylabel = ylabel)

            # Hide any empty subplots if n_col doesn't perfectly fill the grid
            for idx in range(n_col, rows * cols):
                r, c = divmod(idx, cols)
                axes[r, c].axis('off')

            sm = util.map_color(vmin=-global_vmax, vmax=global_vmax)
            fig.colorbar(sm, ax=axes.ravel().tolist(), aspect=40, pad=0.03, label='Basis Weight Magnitude')
            fig.suptitle(f'Top {n_col} 2D Basis Functions ({exp_var[n_col - 1] * 100:.1f}% Variance)', fontsize=14)
            plt.show()

        # --- 3D Case: Grid of Heatmaps (Sliced at middle of Z-axis) ---
        elif self.dim == 3:
            grid_shape = (len(self.grids_[0]), len(self.grids_[1]), len(self.grids_[2]))

            # Find the middle slice of the 3rd dimension
            z_mid = len(self.grids_[2]) // 2
            z_val = self.grids_[2][z_mid]

            cols = min(n_col, 3)
            rows = int(np.ceil(n_col / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

            for idx in range(n_col):
                r, c = divmod(idx, cols)
                ax = axes[r, c]

                # Reshape to 3D, then extract the 2D cross-section
                Z_full = top_bases.iloc[:, idx].values.reshape(grid_shape)
                Z_slice = Z_full[:, :, z_mid]

                fig, ax = util.contour_plot(self.grids_[0], self.grids_[1], Z_slice, levels=30)
                ax.set(title=f"Basis {idx + 1} [Var: {df2.columns[idx]}]", xlabel="X1", ylabel="X2")

            for idx in range(n_col, rows * cols):
                r, c = divmod(idx, cols)
                axes[r, c].axis('off')

            fig.suptitle(f'Top {n_col} 3D Bases (Central Slice X3 ≈ {z_val:.2g})', fontsize=14)
            plt.tight_layout()
            plt.show()

        else:
            logger.warning(f"Plotting basis functions not supported for dimension {self.dim} > 3.")

    def plot_coef_dist(self, true_vals=None, **kwargs):
        labels = [rf'$\theta_{{{i}}}$' for i in self._trainable_idx]
        util.mvn_pairplot(self.coef_mean[self._trainable_idx], self.coef_var, labels=labels,
                          title=r'Density of Physical $\theta$ given $\hat \beta$', true_vals=true_vals, **kwargs)

    def plot_lambda_weight(self, show_sure: bool = True, figsize: tuple = (13, 5), save_path: str = None):
        """
        Visualizes the EWA posterior probability weights w(lambda) alongside
        the continuous SURE risk curve and theoretical asymptotic target.

        Parameters:
        -----------
        show_sure : bool, default=True
            Whether to plot the corresponding SURE(lambda) curve side-by-side.
        figsize : tuple, default=(13, 5)
            Figure dimensions.
        save_path : str, optional
            File path to save the figure if desired.
        """
        check_is_fitted(self, 'ewa_res_')

        lams = self.lam_grid_
        weights = self.ewa_res_['weights']
        sure = self.ewa_res_.get('sure_est', None)

        # 1. Summary Statistics of the Distribution
        lam_mle = float(lams[np.argmax(weights)])
        lam_mean = float(np.sum(lams * weights))

        cum_w = np.cumsum(weights)
        idx_lq = np.argmax(cum_w >= self.alpha / 2.0)
        idx_uq = np.argmax(cum_w >= 1.0 - self.alpha / 2.0)
        lam_lq, lam_uq = float(lams[idx_lq]), float(lams[idx_uq])

        lam_theo = getattr(self, 'lam_asymp_', None)

        # 2. Setup Figure
        n_plots = 2 if (show_sure and sure is not None) else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, layout='constrained')
        ax_w = axes[0] if n_plots == 2 else axes

        # --- PANEL 1: Posterior Weights w(lambda) ---
        ax_w.plot(lams, weights, color='#1f77b4', lw=2.2, label=r'Posterior $w(\lambda)$')
        ax_w.fill_between(lams, 0, weights, color='#1f77b4', alpha=0.25)

        # Shaded Credible Interval
        ax_w.fill_between(
            lams[idx_lq:idx_uq + 1],
            0,
            weights[idx_lq:idx_uq + 1],
            color='#aec7e8',
            alpha=0.5,
            label=f'{1 - self.alpha:.0%} Credible Interval'
        )

        # Markers for Critical Values
        ax_w.axvline(lam_mle, color='crimson', ls='--', lw=1.8, label=rf'$\lambda_{{\mathrm{{MLE}}}} = {lam_mle:.4f}$')
        ax_w.axvline(lam_mean, color='darkorange', ls=':', lw=2.0, label=rf'$\mathbb{{E}}[\lambda] = {lam_mean:.4f}$')

        if lam_theo is not None:
            ax_w.axvline(lam_theo, color='forestgreen', ls='-.', lw=1.8,
                         label=rf'$\lambda_{{\mathrm{{theo}}}} = {lam_theo:.4f}$')

        ax_w.set_title(r"EWA Posterior Weights Distribution $w(\lambda)$", fontsize=12, fontweight='bold')
        ax_w.set_xlabel(r"Regularization Parameter $\lambda$", fontsize=11)
        ax_w.set_ylabel("Posterior Probability Weight", fontsize=11)
        ax_w.set_xlim(0.0, 1.0)
        ax_w.set_ylim(bottom=0.0)
        ax_w.grid(True, ls='--', alpha=0.5)
        ax_w.legend(loc='best', frameon=True, fontsize=9)

        # --- PANEL 2: Objective SURE(lambda) Landscape ---
        if n_plots == 2:
            ax_s = axes[1]

            # Center SURE around its minimum for numerical readability
            ax_s.plot(lams, sure, color='#2ca02c', lw=2.0, label=r'$\mathrm{SURE}(\lambda) - \min$')
            ax_s.axvline(lam_mle, color='crimson', ls='--', lw=1.8, label=r'Argmin $\lambda^*$')

            if lam_theo is not None:
                ax_s.axvline(lam_theo, color='forestgreen', ls='-.', lw=1.8, label=r'Theoretical $\lambda^*$')

            ax_s.set_title(r"Objective Landscape $\mathrm{SURE}(\lambda)$", fontsize=12, fontweight='bold')
            ax_s.set_xlabel(r"Regularization Parameter $\lambda$", fontsize=11)
            ax_s.set_ylabel(r"Relative Risk $(\mathrm{SURE} - \min)$", fontsize=11)
            ax_s.set_xlim(0.0, 1.0)
            ax_s.grid(True, ls='--', alpha=0.5)
            ax_s.legend(loc='best', frameon=True, fontsize=9)

        fig.suptitle(
            f"Regularization Diagnostics | $n={self.n_train}$ | $\sigma_{{\mathrm{{est}}}}={self.sigma_est_:.3e}$",
            fontsize=13
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_eigenvalues(self, figsize=(12, 5)):
        """
        Plots the GEVP spectrum s_i and the corresponding shrinkage filter factors.

        Parameters
        ----------
        candidate_lambdas : list or array-like, optional
            Values of lambda to overlay on the filter factor subplot.
            Defaults to [0.1, 0.5, 0.9, self.lambda_ (if fitted)].
        figsize : tuple, default=(12, 5)
            Figure dimensions (width, height).
        """

        s = np.asarray(self.s_)
        k = len(s)
        mode_indices = np.arange(1, k + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # -------------------------------------------------------------
        # Subplot 1: GEVP Eigenvalue Spectrum Growth
        # -------------------------------------------------------------
        # Use symlog to handle zero/near-zero null space modes without crashing
        ax1.plot(mode_indices, s, 'o-', color='#1f77b4', markersize=3.5, linewidth=1.2, label=r'$s_i$ (Penalty Energy)')
        ax1.set_yscale('symlog', linthresh=1e-5)
        ax1.set_xlabel('Eigenmode Index ($i$)', fontsize=11)
        ax1.set_ylabel(r'Eigenvalue $s_i$ (Symlog Scale)', fontsize=11)
        ax1.set_title(f'GEVP Spectrum ($k={k}$ Computed Modes)', fontsize=12, fontweight='bold')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        # Annotate zero/harmonic modes
        near_zero_mask = np.abs(s) < 1e-4
        n_zeros = np.sum(near_zero_mask)
        ax1.axvline(n_zeros, color='crimson', linestyle=':', alpha=0.8)

        # -------------------------------------------------------------
        # Subplot 2: Eigenfunction Coefficients (alpha_hat)
        # -------------------------------------------------------------
        alpha_hat_abs = np.abs(self.ewa_res_['alpha_hat'])
        ax2.plot(mode_indices[:len(alpha_hat_abs)], alpha_hat_abs, color='#2ca02c',
                 linewidth=1.2, label=r'$\hat{\alpha}_{\mathrm{ewa}}$')

        ax2.set_xlabel('Eigenmode Index ($i$)', fontsize=11)
        ax2.set_ylabel(r'$|\hat{\alpha}_i|$', fontsize=11)
        ax2.set_title(r'Abs. Eigenfunction Coefficients $\hat{\alpha}_{\mathrm{ewa}}$', fontsize=12, fontweight='bold')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_derivatives(self, deriv_orders_list: List[Tuple[int, ...]], max_cols: int = 3,
                         true_deriv_func: callable = None, true_deriv_kwargs: dict = None, **kwargs):
        """
        Plots the requested partial derivatives over the evaluation grid.
        Handles 1D (line plots), 2D (heatmaps), and 3D (2D cross-section heatmaps).
        If `true_deriv_func` is provided, plots the Error (Predicted - True) directly below each prediction.

        Parameters:
        -----------
        deriv_orders_list : List[Tuple[int, ...]]
            A list of derivative order tuples to plot. E.g. [(2, 0), (0, 2), (1, 1)] for a 2D problem.
        max_cols : int
            Maximum number of columns in the subplot grid.
        true_deriv_func: callable, optional
            A function that takes `(x, **kwargs)` and returns a dictionary mapping derivative tuples to exact analytical evaluations.
        true_deriv_kwargs: dict, optional
            Physics parameters to pass to `true_deriv_func`. E.g., {'p_m': 1, 'p_n': 1, 'p_a1': 1.0, 'p_a2': 0.5}.
        """
        check_is_fitted(self, 'ewa_res_')

        # Evaluate true derivatives directly on the model's gridpoints
        if true_deriv_func is not None:
            t_kwargs = true_deriv_kwargs or {}
            true_derivs = true_deriv_func(self.gridpoints_, **t_kwargs)
            plot_error = True
        else:
            true_derivs = None
            plot_error = False

        n_plots = len(deriv_orders_list)
        cols = min(n_plots, max_cols)

        # Calculate rows needed for the primary plots, double them if plotting errors
        primary_rows = int(np.ceil(n_plots / cols))
        rows = primary_rows * 2 if plot_error else primary_rows

        # --- 1D Case: Grid of Line Plots ---
        if self.dim == 1:
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                                     squeeze=False, layout='constrained')

            for idx, order in enumerate(deriv_orders_list):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                ax_pred = axes[base_row, c]

                # Predict the derivative
                y_deriv = self.predict_derivative(self.gridpoints_, order)
                ax_pred.plot(self.grids_[0], y_deriv, lw=2, color='tab:orange')
                ax_pred.set(title=f"Derivative {order}", xlabel="X", ylabel="Value")
                ax_pred.grid(True, linestyle='--', alpha=0.6)

                if plot_error:
                    ax_err = axes[base_row + 1, c]
                    error_vals = y_deriv - true_derivs[order]
                    ax_err.plot(self.grids_[0], error_vals, lw=2, color='tab:red')
                    ax_err.set(title=f"Error {order}", xlabel="X", ylabel="Error (Pred - True)")
                    ax_err.grid(True, linestyle='--', alpha=0.6)

            # Hide empty subplots
            for idx in range(n_plots, primary_rows * cols):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                axes[base_row, c].axis('off')
                if plot_error:
                    axes[base_row + 1, c].axis('off')

            fig.suptitle("Fitted Surface Derivatives" + (" & Error Profiles" if plot_error else ""), fontsize=14)
            plt.show()

        # --- 2D Case: Grid of Heatmaps ---
        elif self.dim == 2:
            xlabel, ylabel = self.variables
            grid_shape = (len(self.grids_[0]), len(self.grids_[1]))
            X1, X2 = np.meshgrid(self.grids_[0], self.grids_[1], indexing='ij')

            fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                                     squeeze=False, layout='constrained')

            for idx, order in enumerate(deriv_orders_list):
                order_str = self.format_deriv_orders(order)
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                ax_pred = axes[base_row, c]

                # Extract and reshape derivative
                Z_flat = self.predict_derivative(self.gridpoints_, order)
                Z = Z_flat.reshape(grid_shape)

                vmax = np.max(np.abs(Z))
                if vmax == 0: vmax = 1e-8

                _, ax_pred = util.contour_plot(X1, X2, Z, ax=ax_pred, vmin=-vmax, vmax=vmax, levels=30, cbar=False)
                sm = util.map_color(vmin=-vmax, vmax=vmax)
                fig.colorbar(sm, ax=ax_pred, shrink=0.8, pad=0.04)
                ax_pred.set(title=f"Predicted {order_str}", xlabel = xlabel, ylabel = ylabel)

                if plot_error:
                    ax_err = axes[base_row + 1, c]
                    Z_true = true_derivs[order]
                    Z_err = (Z_flat - Z_true)

                    Z_true_norm = np.sqrt(np.nansum(Z_true ** 2))
                    Z_rel_err = Z_err / Z_true_norm
                    Z_rel_err = Z_rel_err.reshape(grid_shape)

                    # np.sqrt(np.sum(Z_err ** 2) / np.sum(Z_true ** 2))
                    Z_rel_err_norm = np.sqrt(np.nansum(Z_rel_err ** 2))

                    vmax_err = np.nanmax(np.abs(Z_rel_err))
                    if vmax_err == 0: vmax_err = 1e-8

                    _, ax_err = util.contour_plot(X1, X2, Z_rel_err, ax=ax_err, vmin=-vmax_err, vmax=vmax_err, levels=30,
                                                  cbar=False)
                    sm_err = util.map_color(vmin=-vmax_err, vmax=vmax_err)
                    fig.colorbar(sm_err, ax=ax_err, shrink=0.8, pad=0.04, format=mtick.StrMethodFormatter('{x:.0%}'))
                    ax_err.set(title=f"{order_str} Relative Error: {Z_rel_err_norm:.4%}", xlabel = xlabel, ylabel = ylabel)

            for idx in range(n_plots, primary_rows * cols):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                axes[base_row, c].axis('off')
                if plot_error:
                    axes[base_row + 1, c].axis('off')

            fig.suptitle("2D Fitted Surface Derivatives" + (" & Error Heatmaps" if plot_error else ""), fontsize=14)
            plt.show()

        # --- 3D Case: Grid of Heatmaps (Sliced at middle of Z-axis) ---
        elif self.dim == 3:
            grid_shape = (len(self.grids_[0]), len(self.grids_[1]), len(self.grids_[2]))

            z_mid = len(self.grids_[2]) // 2
            z_val = self.grids_[2][z_mid]

            fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                                     squeeze=False, layout='constrained')

            for idx, order in enumerate(deriv_orders_list):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                ax_pred = axes[base_row, c]

                # Reshape to 3D, extract 2D cross-section
                Z_flat = self.predict_derivative(self.gridpoints_, order)
                Z_full = Z_flat.reshape(grid_shape)
                Z_slice = Z_full[:, :, z_mid]

                vmax = np.max(np.abs(Z_slice))
                if vmax == 0: vmax = 1e-8

                _, ax_pred = util.contour_plot(self.grids_[0], self.grids_[1], Z_slice,
                                               ax=ax_pred, vmin=-vmax, vmax=vmax, levels=30, cbar=False)
                sm = util.map_color(vmin=-vmax, vmax=vmax)
                fig.colorbar(sm, ax=ax_pred, shrink=0.8, pad=0.04)
                ax_pred.set(title=f"Predicted {order}", xlabel="X1", ylabel="X2")

                if plot_error:
                    ax_err = axes[base_row + 1, c]

                    Z_err_full = (Z_flat - true_derivs[order]).reshape(grid_shape)
                    Z_err_slice = Z_err_full[:, :, z_mid]

                    vmax_err = np.max(np.abs(Z_err_slice))
                    if vmax_err == 0: vmax_err = 1e-8

                    _, ax_err = util.contour_plot(self.grids_[0], self.grids_[1], Z_err_slice,
                                                  ax=ax_err, vmin=-vmax_err, vmax=vmax_err, levels=30, cbar=False)
                    sm_err = util.map_color(vmin=-vmax_err, vmax=vmax_err)
                    fig.colorbar(sm_err, ax=ax_err, shrink=0.8, pad=0.04)
                    ax_err.set(title=f"Error {order}", xlabel="X1", ylabel="X2")

            for idx in range(n_plots, primary_rows * cols):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                axes[base_row, c].axis('off')
                if plot_error:
                    axes[base_row + 1, c].axis('off')

            fig.suptitle(f"3D Derivatives (Central Slice X3 ≈ {z_val:.2g})" + (" & Errors" if plot_error else ""),
                         fontsize=14)
            plt.show()

        else:
            logger.warning(f"Plotting derivatives not supported for dimension {self.dim} > 3.")

    def galerkin_force_diagnostics(self):
        op_dfs = {
            op_key: util.tensor_diagnostics(force_vec.reshape(*self.n_splines))
            for op_key, force_vec in self.F_g_dict.items()
        }

        # 2. Generate a single Subplot Grid
        fig, axes = plt.subplots(len(op_dfs), self.dim,
                                 figsize=(6 * self.dim, 4 * len(op_dfs)),
                                 squeeze=False)

        fig.suptitle("Galerkin Force Diagnostics", fontsize=15, fontweight='bold', y=1.02)

        for row, (op_key, stat_df) in enumerate(op_dfs.items()):

            op_name = self._coef_keys[op_key] if hasattr(self, '_coef_keys') and isinstance(op_key, int) else op_key

            for axis in range(self.dim):
                ax = axes[row, axis]

                slice_df = stat_df.loc[axis]
                x_locs = self.get_spline_locations(axis)

                # Lists to collect line handles for a unified legend
                lines, labels = [], []

                # --- 3a. Plot Primary Metrics (Energy, Max Amp) ---
                primary_metrics = [m for m in slice_df.columns if m != 'sparsity']
                for metric in primary_metrics:
                    line, = ax.plot(x_locs, slice_df[metric], 'o-', ms=4,
                                    label=metric.replace('_', ' ').title())
                    lines.append(line)
                    labels.append(line.get_label())

                # --- 3b. Plot Secondary Metric (Sparsity) on Twin Axis ---
                if 'sparsity' in slice_df.columns:
                    ax2 = ax.twinx()
                    # Use a distinct dashed style and a specific color (red) so it stands out
                    line2, = ax2.plot(x_locs, slice_df['sparsity'], 's--', color='tab:red', ms=4,
                                      label='Sparsity')
                    lines.append(line2)
                    labels.append(line2.get_label())

                    # Color-code the right y-axis to match the sparsity line
                    ax2.set_ylabel("Sparsity Ratio", color='tab:red', fontsize=10)
                    ax2.tick_params(axis='y', labelcolor='tab:red')
                    ax2.set_ylim(0, 1)

                # --- 4. Batch Formatting & Unified Legend ---
                ax.set(title=f"Operator: {op_name} | Axis: {axis}",
                       xlabel=f"Domain Location (Dim {axis})")

                if axis == 0:
                    ax.set_ylabel("Magnitude (Energy / Max Amp)")

                ax.grid(True, ls='--', alpha=0.6)

                # Pass the collected lines/labels to the primary axis legend to prevent overlapping boxes
                ax.legend(lines, labels, fontsize=9, loc='upper right')

                # Pin primary y-axis to 0 if all primary values are positive
                if slice_df[primary_metrics].min().min() >= 0:
                    ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()

    def galerkin_penalty_diagnostics(self):
        op_dfs = {}
        tensor_shape = tuple(self.n_splines) * 2

        for (idx_test, idx_trial), matrix in self._omega_dict.items():
            key_test = self._coef_keys[idx_test]
            key_trial = self._coef_keys[idx_trial]

            name_test = util.format_latex_name(key_test)
            name_trial = util.format_latex_name(key_trial)

            factors_test = self.parse_derivative_order(key_test)
            factors_trial = self.parse_derivative_order(key_trial)

            alpha_total = np.sum(factors_test, axis=0)
            beta_total = np.sum(factors_trial, axis=0)

            # --- Clean Reuse of Your Class Method ---
            # Note: adjust to self.calculate_galerkin_scale(alpha, beta) if that is what you named it
            scale_factor = self.get_omega_theoretical_scale(alpha_total, beta_total)

            logger.info(f"Normalizing ({name_test}, {name_trial}) | Scale Factor: {scale_factor:.2e}")

            # Normalize the matrix to O(1) baseline
            matrix_normalized = matrix / scale_factor

            op_dfs[(name_test, name_trial)] = util.tensor_diagnostics(matrix_normalized.reshape(*tensor_shape))

        # =================================================================
        # 3. Generate the Scaled Subplot Grid
        # =================================================================
        total_axes = 2 * self.dim
        fig, axes = plt.subplots(len(op_dfs), total_axes,
                                 figsize=(6 * total_axes, 4 * len(op_dfs)),
                                 squeeze=False)

        fig.suptitle("Mesh-Normalized Galerkin Penalty Diagnostics",
                     fontsize=16, fontweight='bold', y=1.02)

        for row, ((name_test, name_trial), stat_df) in enumerate(op_dfs.items()):
            for axis in range(total_axes):
                ax = axes[row, axis]

                space_name = "Test Space" if axis < self.dim else "Trial Space"
                phys_dim = axis % self.dim

                slice_df = stat_df.loc[axis]
                x_locs = self.get_spline_locations(phys_dim)
                lines, labels = [], []

                # --- Primary Metrics (Normalized Energy, Max Amp) ---
                primary_metrics = [m for m in slice_df.columns if m != 'sparsity']
                for metric in primary_metrics:
                    line, = ax.plot(x_locs, slice_df[metric], 'o-', ms=4,
                                    label=metric.replace('_', ' ').title())
                    lines.append(line)
                    labels.append(line.get_label())

                # --- Secondary Metric (Sparsity) ---
                if 'sparsity' in slice_df.columns:
                    ax2 = ax.twinx()
                    line2, = ax2.plot(x_locs, slice_df['sparsity'], 's--', color='tab:red', ms=4,
                                      label='Sparsity')
                    lines.append(line2)
                    labels.append(line2.get_label())

                    ax2.set_ylabel("Sparsity Ratio", color='tab:red', fontsize=10)
                    ax2.tick_params(axis='y', labelcolor='tab:red')
                    ax2.set_ylim(0, 1)

                # --- Labels and Unified Legend ---
                ax.set(title=f"{space_name} | Dim: {phys_dim}",
                       xlabel=f"Domain Location (Dim {phys_dim})")

                if axis == 0:
                    ax.set_ylabel(f"Pair: ({name_test}, {name_trial})\n\nNormalized Magnitude")

                ax.grid(True, ls='--', alpha=0.6)
                ax.legend(lines, labels, fontsize=9, loc='upper right')

                if primary_metrics and slice_df[primary_metrics].min().min() >= 0:
                    ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        status = "Fitted" if hasattr(self, 'theta_hat_') else "Unfitted"
        fit_info = {k: self.__getattribute__(k)
                    for k in ['n_train', 'domain', 'deg', 'deriv_coef', 'n_knots',
                              'sigma_est_', 'gamma_mult', 'alpha',
                              ]}
        return f"EWABSpline({status}, {fit_info})"


if __name__ == '__main__':
    print(1)