import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, t
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import util
import pandas as pd
from scipy.interpolate import BSpline, CubicSpline
from scipy.optimize import minimize_scalar, differential_evolution
import scipy
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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

    return tuple(deriv_splits)


class EWABSplinePDE(BaseEstimator, RegressorMixin):
    """
    Exponentially Weighted Aggregation (EWA) B-Spline Estimator with Physics-Informed Penalties.
    """

    def __init__(self,
                 dim: int,
                 deg: int = 3,
                 variables: Optional[List[str]] = None,
                 deriv_coef: Dict[str, float] = None,
                 action_coef: Dict[tuple, float] = None,
                 force: callable = None,
                 trainable: Optional[List[str]] = None,
                 max_iter: int = 1000,
                 min_pct_chg: float = 1e-4,
                 domain: Optional[List[Tuple[float, float]]] = None,
                 num_int_knots: Optional[int] = None,
                 quantile_based: bool = False,
                 total_grid_size: int = 1000,
                 sigma: Optional[float] = None,
                 gamma_mult: float = 8.0,
                 prior_type: str = "light",
                 norm_omega: bool = False,
                 alpha: float = 0.05,
                 log_lamb: Tuple[float, float] = (-30, 20),
                 log_lamb_step: float = 0.01,
                 tol: float = 1e-8):

        self.dim = dim
        self.deg = deg
        self.variables = variables or [str(i) for i in range(dim)]

        self.deriv_coef = deriv_coef or {}
        self.action_coef = action_coef or {}
        self.force = force or (lambda s: np.zeros(s.shape[0]))
        self.trainable = trainable

        self.max_iter = max_iter
        self.min_pct_chg = min_pct_chg
        self.domain = domain
        self.num_int_knots = num_int_knots
        self.quantile_based = quantile_based
        self.total_grid_size = total_grid_size

        self.sigma = sigma
        self.gamma_mult = gamma_mult
        self.prior_type = prior_type
        self.norm_omega = norm_omega
        self.alpha = alpha
        self.log_lamb = log_lamb
        self.log_lamb_step = log_lamb_step
        self.tol = tol

        # Internal Caches
        self._omega_dict = {}
        self._integral_1d_cache = {}

    def _parse_physics_configuration(self):
        """Parses and validates PDE/Action coefficients into internal mathematical formats."""
        assert not (self.deriv_coef and self.action_coef), "Provide either deriv_coef or action_coef, not both."

        if self.deriv_coef:
            deriv_keys = [get_deriv_order(k, self.variables) for k in self.deriv_coef.keys()]
            sorted_coef = sorted(zip(deriv_keys, self.deriv_coef.values()))
            self._coef_keys, self._initial_coef_vals = zip(*sorted_coef)
            self.mode = 'PDE'
        elif self.action_coef:
            action_keys = [tuple(get_deriv_order(p, self.variables) for p in pair) for pair in
                           self.action_coef.keys()]
            sorted_action = sorted(zip(action_keys, self.action_coef.values()))
            self._coef_keys, self._initial_coef_vals = zip(*sorted_action)
            self.mode = 'DUAL'
        else:
            raise ValueError("Must provide physics constraints via deriv_coef or action_coef.")

        self._initial_coef_vals = np.array(self._initial_coef_vals, dtype=float)
        self._determine_trainable_indices()

    def parse_derivative_order(self, op_key):
        """
        Parses a nonlinear PDE term into a list of derivative order vectors.
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

    def _determine_trainable_indices(self):
        """Determines which physics parameters are actively optimized."""
        if self.trainable is not None:
            trainable_list = []
            for k in self.trainable:
                if isinstance(k, tuple):
                    k_tuple = tuple(get_deriv_order(ki, self.variables) for ki in k)
                    trainable_list.append(k_tuple)
                else:
                    trainable_list.append(get_deriv_order(k, self.variables))
            t_str = list(map(str, trainable_list))
            k_str = list(map(str, self._coef_keys))
            assert set(t_str) <= set(k_str), "Make sure trainable params are defined in deriv_coef."
            self._trainable_idx = [i for i, d in enumerate(k_str) if d in t_str]
        else:
            deriv_order = list(map(len, self._coef_keys))
            self._trainable_idx = [i for i, order in enumerate(deriv_order) if order < max(deriv_order)]

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

    def get_spline_locations(self, d):
        """Calculates the Greville Abscissae (knot averages) for the x-axis."""
        knots = self.all_knots[d]
        p = self.deg
        n = self.n_splines[d]
        # The physical center of the i-th B-spline is the average of its p subsequent knots
        return np.array([np.mean(knots[i + 1 : i + 1 + p]) for i in range(n)])

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

    def _initialize_basis(self, X: np.ndarray):
        """Generates knots, grids, and projects the design matrix."""
        self.a_, self.b_ = X.min(axis=0), X.max(axis=0)
        self.domain_ = self.domain or list(zip(self.a_, self.b_))

        # 1. Generate Knots
        self.generate_knots(X)  # (Keep your existing robust knot generation logic)

        # 2. Setup Evaluation Grids
        self.grid_size_ = int(self.total_grid_size ** (1 / self.dim))
        self.grids_ = [np.linspace(*d, self.grid_size_) for d in self.domain_]
        self.gridpoints_ = util.tensor_prod(*self.grids_)
        self.B_grid_ = self._build_tensor_basis(self.gridpoints_)

        # 3. Data Basis & Gram Matrix
        self.B_data_ = self._build_tensor_basis(X)
        self.G_ = util.compute_G(self.B_data_)

    def _setup_priors_and_lambda(self, X, y: np.ndarray, tail_clip=0.01):
        """Constructs the discrete lambda grid and SURE prior distributions."""
        #self.sigma_est_ = util.gsj_est(y) if self.sigma is None else self.sigma
        self.sigma_est_ = util.gsj_est_2d_scattered(X, y) if self.sigma is None else self.sigma
        print(f"Estimated Sigma: {self.sigma_est_}")

        log_lams = np.arange(self.log_lamb[0], self.log_lamb[1], self.log_lamb_step)
        lam_grid = np.exp(np.sort(log_lams)[::-1])

        if self.prior_type == "light":
            prior_vals = gamma(a=0.3, scale=1 / 0.3).cdf(lam_grid)
        else:
            prior_vals = 2 * t.cdf(25 * lam_grid, df=3) - 1

        prior = np.diff(np.r_[0, prior_vals[::-1]])[::-1]
        uq, lq = np.argmax(prior_vals <= 1-tail_clip), np.argmax(prior_vals <= tail_clip)
        r_uq, r_lq = 1 - prior[uq] / tail_clip, 1 - prior[lq] / tail_clip
        prior[:uq] = prior[uq] * (r_uq ** np.arange(uq))[::-1]
        prior[lq:] = prior[lq] * (r_lq ** np.arange(len(log_lams) - lq))
        prior = prior / prior.sum()

        # prior_s = pd.Series(prior, index=lam_grid)
        # prior_s.plot(logx=True)
        #TODO: scale with n_sample?

        self.prior_ = prior
        self.lam_grid_ = lam_grid

    def compute_K_uv(self, d_u, d_v, x_eval=None):
        K_uv = None
        for p in range(self.dim):
            d_up = int((np.array(d_u) == p).sum())
            d_vp = int((np.array(d_v) == p).sum())
            # --- 2. Check the Local 1D Micro-Cache ---
            cache_key_1d = (p, d_up, d_vp)
            if cache_key_1d not in self._integral_1d_cache:
                self._integral_1d_cache[cache_key_1d] = util.compute_spline_integral_gl(
                    self.all_knots[p], u=d_up, v=d_vp, deg=self.deg, x_eval=x_eval
                )

            K_1d = self._integral_1d_cache[cache_key_1d]

            # --- 3. Assemble the Kronecker Product ---
            if K_uv is None:
                K_uv = K_1d
            else:
                K_uv = np.kron(K_uv, K_1d)
        print(f"{d_u, d_v}Computed")
        return K_uv

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
            for v, c_v in enumerate(deriv_vals[:u + 1]):
                if (u, v) not in self._omega_dict:
                    omega_uv = self.compute_K_uv(deriv_keys[u], deriv_keys[v], x_eval)
                    self._omega_dict[u, v] = omega_uv
                    self._omega_dict[v, u] = omega_uv.T

                c = c_u * c_v
                if omega_final is None:
                    # First allocation: copy the scaled matrix
                    omega_final = self._omega_dict[u, v] * c
                else:
                    # Subsequent allocations: add in-place
                    omega_final += self._omega_dict[u, v] * c

                    # Add the symmetric (v, u) term if off-diagonal
                if u != v:
                    omega_final += self._omega_dict[v, u] * c
        return omega_final

    def compute_k_action(self, action_keys, action_vals, x_eval=None):
        """
        Compute K_{action} from Bilinear Differential Operators

        :param action_keys_left: List of parsed deriv keys for the left operator (e.g., [[0], [1], []])
        :param action_keys_right: List of parsed deriv keys for the right operator (e.g., [[0], [1], []])
        :param action_vals: List of coefficients for each bilinear pair
        """
        K_action_final = None

        for i, c in enumerate(action_vals):
            # Each key_i is a bilinear deriv
            d_u, d_v = action_keys[i]

            # Use a sorted tuple cache key because integral(u_x * u) == integral(u * u_x)
            cache_key = (i, i)

            # --- 1. Check the Global n-Dimensional Cache ---
            if cache_key not in self._omega_dict:
                self._omega_dict[cache_key] = self.compute_K_uv(d_u, d_v, x_eval)

            # --- 4. In-Place Memory Accumulation ---
            if K_action_final is None:
                # First allocation
                K_action_final = self._omega_dict[cache_key] * c
            else:
                # Subsequent allocations
                K_action_final += self._omega_dict[cache_key] * c

        # --- 5. Enforce Global Symmetry ---
        # The physical action must yield a strictly symmetric bilinear form.
        # u_x_sqr = np.kron(self._integral_1d_cache[(0, 1, 1)], self._integral_1d_cache[(1, 0, 0)])
        # u_y_sqr = np.kron(self._integral_1d_cache[(0, 0, 0)], self._integral_1d_cache[(1, 1, 1)])
        diff_sym = np.abs(K_action_final - K_action_final.T).sum()
        assert diff_sym < self.tol, f"Weak Form Penalty Matrix not symmetric, diff={diff_sym}"
        K_action_final = (K_action_final + K_action_final.T) / 2
        return K_action_final

    def _project_forcing_function(self):
        """
        Projects the continuous forcing function g(x) onto the B-spline basis to find w.
        Actively reports the approximation fidelity.
        """
        # Evaluate exact forcing function on a dense grid
        g_val = self.force(self.gridpoints_).ravel()

        # Project onto the B-spline basis: B * w = g
        w, _, _, _ = np.linalg.lstsq(self.B_grid_, g_val, rcond=None)

        # Validate approximation fidelity
        g_approx = self.B_grid_ @ w
        mse = np.mean((g_val - g_approx) ** 2)
        var_g = np.var(g_val) if np.var(g_val) > 0 else 1.0
        rel_error = mse / var_g

        print(f"Forcing Term Projection Relative MSE: {rel_error:.2e}")
        if rel_error > 0.05:
            print("WARNING: B-spline basis may be too sparse to accurately capture the forcing function!")

        return w

    def _compute_dual_F_g_dict(self, K_components: dict, w: np.ndarray) -> dict:
        """
        Maps the DUAL weak-form forcing components into the unified structure.
        Computes f_i = K_i^T * w
        """
        dual_F_g_dict = {}
        for i, K_i in K_components.items():
            assert i[0] == i[1]
            v = K_i.T @ w
            dual_F_g_dict[i[0]] = v

        return dual_F_g_dict

    def _compute_dual_omega_dict(self, K_components: dict) -> dict:
        """
        Maps the DUAL weak-form penalty matrices into the unified structure.
        Computes \Omega_ij = K_i^T * H_0^{-1} * K_j
        """
        dual_omega_dict = {}

        for j, K_j in K_components.items():
            assert j[0] == j[1]
            # Perform the O(p^2) Cholesky solve once per physical parameter
            H0_inv_Kj = scipy.linalg.cho_solve(self.H_0_cho_, K_j)

            for i, K_i in K_components.items():
                assert i[0] == i[1]
                omega_ij = K_i.T @ H0_inv_Kj
                # Clean up floating point asymmetry
                # assert np.abs(omega_ij - omega_ij.T).sum() < self.tol
                dual_omega_dict[(i[0], j[0])] = omega_ij #(omega_ij + omega_ij.T) / 2.0

        return dual_omega_dict

    def precompute_forcing_components(self):
        """
        Computes the forcing load vector F_g = H * w using exact 1D Kronecker separation,
        eliminating the need for D-dimensional numerical integration.
        """
        w = self._project_forcing_function()

        if self.mode == 'DUAL':
            # Weak form structurally uses the raw basis test functions: B_i
            # H_0 = integral( B * B^T ) dx
            self.H_0_ = self.compute_K_uv(d_u=[], d_v=[])
            self.H_0_cho_ = scipy.linalg.cho_factor(self.H_0_)
            self.F_weak_ = self.H_0_ @ w

            # 2. Extract the raw K_i matrices into a clean dictionary
            for i, (d_u, d_v) in enumerate(self._coef_keys):
                if (i, i) not in self._omega_dict:
                # self._coef_keys stores the derivative tuples for action_coef
                # Generate the raw spatial matrix and store it temporarily
                    self._omega_dict[(i, i)] = self.compute_K_uv(d_u, d_v)

            # 3. Delegate to the standalone computational methods
            self.F_g_dict = self._compute_dual_F_g_dict(self._omega_dict, w)
            self._omega_dict = self._compute_dual_omega_dict(self._omega_dict)
            # self.force_norm = self.F_g_dict @

        else:
            # Strong form evaluates the explicit component projections D_k(B_i)
            # H_u = integral( D_u(B) * B^T ) dx
            self.compute_omega(list(self._coef_keys), self._initial_coef_vals)
            self.F_g_dict = {}
            for i, u in enumerate(self._coef_keys):
                H_u = self.compute_K_uv(u, [])
                self.F_g_dict[i] = H_u @ w

        self.force_norm = w.T @ self.compute_K_uv([], []) @ w / 2

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

    def _build_penalty_matrix(self, coef_vals: np.ndarray) -> np.ndarray:
        """Dispatches to the correct continuous integral calculation based on mode."""
        if self.mode == 'PDE':
            return self.compute_omega(list(self._coef_keys), coef_vals)
        return self.compute_k_action(self._coef_keys, coef_vals)

    def _build_penalty_and_force(self, coef_vals):
        penalty_matrix = np.zeros_like(self.G_)
        forcing_vector = np.zeros(self.G_.shape[0])

        for i in range(len(self._coef_keys)):
            forcing_vector += coef_vals[i] * self.F_g_dict[i]
            for j in range(len(self._coef_keys)):
                penalty_matrix += coef_vals[i] * coef_vals[j] * self._omega_dict[(i, j)]

        assert np.abs(penalty_matrix - penalty_matrix.T).max() < self.tol
        penalty_matrix = (penalty_matrix + penalty_matrix.T) / 2.0
        return penalty_matrix, forcing_vector

    # def _build_forcing_vector(self, coef_vals: np.ndarray) -> np.ndarray:
    #     """Constructs the projected forcing constraint vector."""
    #     """
    #     PDE: sum_k int (D_k(B) * g); g = B * w; for all diff. op. D_k in L
    #     Dual: int (B * g)
    #     """
    #     if self.mode == 'DUAL':
    #         return self.F_weak_
    #
    #     f_int = np.zeros(np.prod(self.n_splines))
    #     for u, c_u in zip(self._coef_keys, coef_vals):
    #         f_int += c_u * self.F_g_dict[u]
    #     return f_int


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

            print(f"Normalizing ({name_test}, {name_trial}) | Scale Factor: {scale_factor:.2e}")

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

    def compute_physics_loss(self, theta, gram_matrices, v_vec):
        """
        Computes the Unbiased PDE Loss: L(theta | beta) = 1/2 * theta^T * G_unbiased * theta - theta^T * v
        Reuses the pre-computed system matrices and applies the EIV trace correction.
        """
        # 1. Fetch empirical matrices directly from the existing pipeline
        pde_loss = 0.5 * np.dot(theta, gram_matrices @ theta) - np.dot(theta, v_vec) + self.force_norm
        return float(pde_loss)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the EWA Spline model to the data."""
        X, y = check_X_y(X, y, accept_sparse=False)
        n = len(y)
        self.X_fit_, self.y_fit_ = X, y

        self._parse_physics_configuration()
        self._initialize_basis(X)
        self._setup_priors_and_lambda(X, y)
        self.precompute_forcing_components()  # Populates self.F_g_dict_ or self.F_weak_
        #self.galerkin_force_diagnostics()
        #self.galerkin_penalty_diagnostics()

        coef_vals = self._initial_coef_vals.copy()
        # --- Block Coordinate Descent for Physics Parameters ---
        for iteration in range(self.max_iter):
            penalty_matrix, forcing_vector = self._build_penalty_and_force(coef_vals)

            # 1. Orthogonalization via GEVP
            orth = util.orthogonalize_basis_fast(self.G_, penalty_matrix)
            self.V_, self.s_raw_ = orth['V'], orth['s_raw'] #G^{-1} = VV^T
            self.U_ = self.B_data_ @ self.V_

            # 2. Project Forcing Vector into Orthogonal Space
            force_proj = self.V_.T @ forcing_vector
            s = np.maximum(self.s_raw_, 0)

            # 3. Fit EWA Model
            self.ewa_res_ = util.fit_ewa(
                y=y, ftrue=y, U=self.U_, V=self.V_, s=s,
                lam=self.lam_grid_, prior=self.prior_,
                sig_eps_est=self.sigma_est_, gamma_mult=self.gamma_mult,
                force_proj=force_proj
            )

            gram_matrices, v_vec = self.compute_theta_system_pde(self._omega_dict, self.ewa_res_['beta'])
            coef_mean, self.coef_var = self.get_coef_dist(gram_matrices, coef_vals, v_vec,
                                                          beta=self.ewa_res_['beta'],
                                                          V=self.V_,
                                                          shrink_weight=self.ewa_res_['shrink_weight'],
                                                          sigma_sq=self.sigma_est_ ** 2
                                                          )

            # 5. Check Convergence
            old_theta = coef_vals[self._trainable_idx]
            coef_pct_chg = np.abs(old_theta - coef_mean).sum() / (np.abs(old_theta).sum() + 1e-8)
            coef_vals[self._trainable_idx] = coef_mean

            if coef_pct_chg < self.min_pct_chg:
                logger.info("Physics parameters converged.")
                break

            w = self.ewa_res_['weights']
            rss = np.sum((self.ewa_res_['fhat_ewa'] - y) ** 2)
            self.sigma_est_ = np.sqrt(rss / (len(y) - self.ewa_res_['trace_hat']))

            w_mean = (self.lam_grid_ * w).sum()
            physics_loss = self.compute_physics_loss(coef_vals, gram_matrices, v_vec)
            total_loss = n * self.sigma_est_ ** 2 + w_mean * physics_loss

            logger.info(f"Iteration {iteration} | Coefficients: {coef_mean} | Change: {coef_pct_chg:.4e} | "
                        f"sigma_est: {self.sigma_est_:.4e} | physics_loss: {physics_loss:.4e} | "
                        f"Total loss: {total_loss: .4e} | " 
                        f"w_mean: {w_mean:.4e} | w_mle: {self.lam_grid_[np.argmax(w)]:.4e}")

        # --- Finalize Fitted State ---
        self.deriv_coef_vals_ = coef_vals
        self.coef_mean = coef_mean
        self.weights_ = pd.Series(self.ewa_res_['weights'], index=self.lam_grid_).sort_index()
        # self.weights_.plot(logx=True)
        return self

    def compute_theta_system_pde(self, omega_dict, beta):
        m = len(self._coef_keys)
        gram_matrices = np.zeros((m, m))
        for key, omega_uv in omega_dict.items():
            gram_matrices[key] = beta.T @ omega_uv @ beta

        v_vec = np.zeros(m)
        for i, u in enumerate(self._coef_keys):
            v_vec[i] = beta.T @ self.F_g_dict[i]

        return gram_matrices, v_vec

    def compute_theta_system_action(self, omega_dict, beta, V):
        """
        Builds the Gram matrix for theta by minimizing the Weak-Form Galerkin residual.
        """
        m = len(self._coef_keys)
        V_weak = np.zeros((len(beta), m))

        for key, K_q in omega_dict.items():
            assert key[0] == key[1]
            K_q = omega_dict[key]
            # Enforce symmetry to clean up float imprecision from integration
            assert np.abs(K_q - K_q.T).sum() < self.tol
            K_q = (K_q + K_q.T) / 2.0
            V_weak[:, key[0]] = K_q @ beta

        # --- EUCLIDEAN ALIGNMENT FOR NUMERICAL STABILITY ---
        # Applying V^T (which perfectly reconstructs G^-1) amplifies B-spline float noise by 10^8.
        # Because \theta are global scalars without spatial saddle-point topology,
        # we safely revert to the noise-resistant Euclidean inner product to compute stable gradients.

        C_weak = V.T @ V_weak
        C_force = V.T @ self.F_weak_
        gram_matrix = C_weak.T @ C_weak
        # gram_matrix = V_weak.T @ V_weak
        assert (np.abs(gram_matrix - gram_matrix.T) / np.abs(gram_matrix)).sum() < self.min_pct_chg
        gram_matrix = (gram_matrix + gram_matrix.T) / 2.0

        v_vec = C_weak.T @ C_force #V_weak.T @ self.F_weak_
        return gram_matrix, v_vec

    def get_coef_dist(self, G, deriv_coef, v_vec, beta, V, shrink_weight, sigma_sq, relaxation=1.0):
        """
        Computes Exact 2nd-Order Unbiased Physical Parameters and Variance.
        Evaluates the current stochastic trajectory (theta^(t)) to neutralize
        Errors-in-Variables (EIV) bias, Jensen's matrix ratio bias, and the
        empirical Jacobian plug-in inflation.
        """
        T = self._trainable_idx
        Tc = [i for i, _ in enumerate(self._coef_keys) if i not in self._trainable_idx]

        # ==========================================================
        # 1. 1st-Order Unbiased Base Matrices (EIV Correction)
        # ==========================================================
        G_unbiased = np.zeros_like(G)
        W2 = shrink_weight ** 2

        for k in range(G.shape[0]):
            for j in range(G.shape[1]):
                if (k, j) in self._omega_dict:
                    Omega_kj = self._omega_dict[(k, j)]
                elif (j, k) in self._omega_dict:
                    Omega_kj = self._omega_dict[(j, k)].T
                else:
                    G_unbiased[k, j] = G[k, j]
                    continue

                Omega_V = Omega_kj @ V
                bias = sigma_sq * np.dot(np.sum(V * Omega_V, axis=0), W2)
                G_unbiased[k, j] = G[k, j] - bias

        G_TT = G_unbiased[np.ix_(T, T)]
        G_TT_inv = np.linalg.inv(G_TT)

        # ==========================================================
        # 2. The Naive BCD Target (1st-Order Stationarity)
        # ==========================================================
        if len(Tc) == 0:
            theta_naive = G_TT_inv @ v_vec[T]
        else:
            G_TTc = G_unbiased[np.ix_(T, Tc)]
            coef_Tc = deriv_coef[Tc]
            theta_naive = G_TT_inv @ (v_vec[T] - G_TTc @ coef_Tc)

        # ==========================================================
        # 3. Empirical Jacobians at Stochastic State \theta^{(t)}
        # ==========================================================
        N_beta = len(beta)
        H = np.zeros((len(T), N_beta))
        grad_H = np.zeros((len(T), N_beta, N_beta))

        for idx, k in enumerate(T):
            # Base forcing function
            H_k = self.F_g_dict[k].copy()
            Sum_Omega_grad = np.zeros((N_beta, N_beta))

            for j in range(len(self._coef_keys)):
                if (k, j) in self._omega_dict:
                    Omega_kj = self._omega_dict[(k, j)]
                elif (j, k) in self._omega_dict:
                    Omega_kj = self._omega_dict[(j, k)].T
                else:
                    continue

                Omega_grad = Omega_kj + Omega_kj.T
                # Evaluated precisely at the current path state: deriv_coef[j]
                H_k -= deriv_coef[j] * (Omega_grad @ beta)
                Sum_Omega_grad += deriv_coef[j] * Omega_grad

            H[idx, :] = H_k
            grad_H[idx, :, :] = -Sum_Omega_grad

        # J_theta mapping: d(theta) / d(beta)
        J_theta = G_TT_inv @ H

        # ==========================================================
        # 4. The 2nd-Order Unbiased Estimator (Jensen's Shift)
        # ==========================================================
        b_ratio = np.zeros(len(T))
        for idx, k in enumerate(T):
            for i_idx, i in enumerate(T):
                if (k, i) in self._omega_dict:
                    Omega_ki = self._omega_dict[(k, i)]
                elif (i, k) in self._omega_dict:
                    Omega_ki = self._omega_dict[(i, k)].T
                else:
                    continue

                # Sensitivity of the Curvature Matrix: d(G_ki) / d(beta)
                J_G_ki = (Omega_ki + Omega_ki.T) @ beta

                # Exact Spectral Trace: tr(J_G_ki^T * Sigma_beta * J_theta_i^T)
                covariance_term = sigma_sq * np.sum((J_G_ki @ V) * W2 * (J_theta[i_idx, :] @ V))
                b_ratio[idx] += covariance_term

        # Construct the true mathematically unbiased target
        theta_unbiased = theta_naive + (G_TT_inv @ b_ratio)

        # ==========================================================
        # 5. Iterative Path Update
        # ==========================================================
        # Execute the Block Coordinate Descent relaxation towards the unbiased root
        coef_mean = deriv_coef[T] + relaxation * (theta_unbiased - deriv_coef[T])

        # ==========================================================
        # 6. Exact 2nd-Order Variance Geometry (Plug-in Corrected)
        # ==========================================================
        # Base Asymptotic Variance
        JV = J_theta @ V
        coef_var_base = sigma_sq * (JV * W2[None, :]) @ JV.T

        # Tensor projection for the exact 2nd-order geometrical inflation
        grad_J = np.tensordot(G_TT_inv, grad_H, axes=([1], [0]))
        var_correction = np.zeros((len(T), len(T)))
        W2_mat = W2[:, None] * W2[None, :]

        for i in range(len(T)):
            V_grad_J_i_V = V.T @ grad_J[i] @ V
            for j in range(len(T)):
                V_grad_J_j_V = V.T @ grad_J[j] @ V
                trace_val = np.sum(V_grad_J_i_V * V_grad_J_j_V * W2_mat)
                var_correction[i, j] = (sigma_sq ** 2) * trace_val

        # Subtract the deterministic empirical plug-in inflation
        coef_var = coef_var_base - var_correction

        return coef_mean, coef_var

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Predicts the continuous function evaluation at X alongside confidence intervals."""
        check_is_fitted(self, 'ewa_res_')
        X = check_array(X, ensure_2d=False)

        B_new = self._build_tensor_basis(X)
        U_new = B_new @ self.V_
        y_hat = U_new @ self.ewa_res_['theta_hat']

        # CI of the fitted function itself (excluding +1 observation noise)
        variances = np.sum((U_new ** 2) * (self.ewa_res_['shrink_weight'] ** 2)[None, :], axis=1) + 1
        se = self.sigma_est_ * np.sqrt(variances)
        interval = norm.ppf(1 - self.alpha / 2) * se

        idx = pd.MultiIndex.from_arrays(X.T, names=self.variables)
        return pd.DataFrame({
            'fhat': y_hat,
            'se': se,
            'ci_lower': y_hat - interval,
            'ci_upper': y_hat + interval
        }, index=idx)

    def plot_fit(self, x_test=None, y_test=None, show_ci=True):
        check_is_fitted(self)
        x_grid = self.gridpoints_
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

            deriv_coef = list(map(lambda x: float(f"{x:.4g}"), self.deriv_coef_vals_))
            title = (
                f"EWA fit [n={len(self.X_fit_)}, deg={self.deg}, deriv_coef={deriv_coef}, "
                f"nk={self.n_knots_1d}, sigma={self.sigma_est_:.3f}, "
                f"{100 * (1 - self.alpha):.0f}% CI]"
            )
            ax.set(title=title, xlabel="X", ylabel="y")
            ax.legend()
            plt.show()

        elif self.dim == 2:
            X = x_grid[:, 0].reshape(self.grid_size_, self.grid_size_)
            Y = x_grid[:, 1].reshape(self.grid_size_, self.grid_size_)
            Z = fit_preds['fhat'].values.reshape(self.grid_size_, self.grid_size_)
            vmax = np.abs(y_scatter).max()

            norm = util.norm_color(vmin=-vmax, vmax=vmax)
            fig, ax = util.contour_plot(X, Y, Z, vmin=-vmax, vmax=vmax)
            ax.scatter(x_scatter[:, 0], x_scatter[:, 1], c=y_scatter,
                       norm=norm, cmap=util.CMAP, edgecolors='black', s=20, label='Data')

            deriv_coef = list(map(lambda x: float(f"{x:.4g}"), self.deriv_coef_vals_))
            deriv_coef = dict(zip(list(map(str, self._coef_keys)), deriv_coef))
            title = (
                f"2D EWA fit [n={len(self.X_fit_)}, deg={self.deg}, deriv_coef={deriv_coef}, "
                f"nk={self.n_knots_1d}, sigma={self.sigma_est_:.3f}, "
                f"{100 * (1 - self.alpha):.0f}% CI]"
            )
            ax.set(title=title, xlabel="X1", ylabel="X2")
            plt.show()

        else:
            print(f"Plotting not supported for dimension {self.dim} > 2.")

    @property
    def basis_functions(self):
        basis = self.B_grid_ @ self.V_
        basis_wgt = self.ewa_res_['theta_hat']

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
                ax.set(title=f"Basis {idx + 1} [Var: {df2.columns[idx]}]", xlabel="X1", ylabel="X2")

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
            print(f"Plotting basis functions not supported for dimension {self.dim} > 3.")

    def plot_coef_dist(self, **kwargs):
        labels = [rf'$\theta_{{{i}}}$' for i in self._trainable_idx]
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