import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, t
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import util
import pandas as pd
import scipy
from typing import Optional, Dict, List, Tuple
import logging
import matplotlib.ticker as mtick
from Bsplines import create_knots, create_spline_basis, knots_data_count, bs, compute_spline_integral_gl
from itertools import product

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

    return tuple(deriv_splits)


class EWABSplinePDE(BaseEstimator, RegressorMixin):
    """
    Exponentially Weighted Aggregation (EWA) B-Spline Estimator with Physics-Informed Penalties.
    """

    def __init__(self,
                 dim: int,
                 deg: int = 3,
                 variables: Optional[List[str]] = None,
                 smooth: bool = False,
                 bias_corrected: bool = False,
                 deriv_coef: Dict[str, float] = None,
                 action_coef: Dict[tuple, float] = None,
                 force: callable = None,
                 boundary_conditions: dict = {},
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
                 tol: float = 1e-8,
                 ):

        self.dim = dim
        self.deg = deg
        self.variables = variables or [str(i) for i in range(dim)]

        self.smooth = smooth
        self.bias_corrected = bias_corrected
        self.deriv_coef = deriv_coef or {}
        self.action_coef = action_coef or {} #TODO: REMOVE, also self.mode
        self.force = force or (lambda s: np.zeros(s.shape[0]))
        self.boundary_conditions = boundary_conditions
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

        self._coef_keys = [tuple(int((np.array(k) == p).sum()) for p in range(self.dim)) for k in self._coef_keys]
        self._initial_coef_vals = np.array(self._initial_coef_vals, dtype=float)
        self._zero_deriv = tuple([0] * self.dim)
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
                else:
                    k_tuple = get_deriv_order(k, self.variables)
                k_tuple = tuple(int((np.array(k_tuple) == p).sum()) for p in range(self.dim))
                trainable_list.append(k_tuple)
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

        n_int_knots_1d = int(n_knots ** (1 / dim))

        if self.quantile_based:
            self.int_knots_ = [np.percentile(X[:, i], np.linspace(0, 100, n_int_knots_1d + 2)[1:-1])
                               for i in range(X.shape[1])]
        else:
            self.int_knots_ = [np.linspace(*d, n_int_knots_1d + 2 + 3*i)[1:-1]
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

    def _build_custom_tensor_basis(self, X, active_dims):
        """
        Generalization of tensor basis construction for subsets of dimensions.
        """

        # Evaluate 1D bases for the active dimensions
        assert X.shape[1] == self.dim
        B_list = [bs(X[:, d], self.spline_objs[d], intercept=True)
                  for d in active_dims]

        B_tensor = B_list[0]
        for i in range(1, len(B_list)):
            # Row-wise Kronecker product (Khatri-Rao)
            B_tensor = (B_tensor[:, :, None] * B_list[i][:, None, :]).reshape(X.shape[0], -1)

        return B_tensor

    def _build_tensor_basis(self, X):
        """Constructs the tensor product of 1D B-splines.
        B_list: [(N, M_i)] N is no. data, M_i is no. Spline per dim

        Suppose 3D: (B_list[0][x] * B_list[1][y] * B_list[2][z]) -> B_tensor[x*(M_1*M_2)+y*(M_2)+z]
        """
        self.spline_objs = [create_spline_basis(all_knots=self.all_knots[i], degree=self.deg)
                  for i in range(self.dim)]
        self.n_splines = np.array([s.c.shape[0] for s in self.spline_objs])

        B_tensor = self._build_custom_tensor_basis(X, list(range(self.dim)))
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
        self.G_ = self.B_data_.T @ self.B_data_
        self.G_, eps = util.perturb_diag(self.G_)
        self.B_data_perturb = util.perturb_B_svd(self.B_data_, eps)

    def _setup_priors_and_lambda(self, X, y: np.ndarray, tail_clip=0.01):
        """Constructs the discrete lambda grid and SURE prior distributions."""
        #self.sigma_est_ = util.gsj_est(y) if self.sigma is None else self.sigma
        self.sigma_est_ = util.gsj_est_2d_scattered(X, y) if self.sigma is None else self.sigma
        print(f"Estimated Sigma: {self.sigma_est_}")

        lam_grid = np.linspace(0, 1, 5000)[1:-1]
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
                    self.spline_objs[p], u=d_up, v=d_vp
                )

            K_1d = self._integral_1d_cache[cache_key_1d]

            # --- 3. Assemble the Kronecker Product ---
            if K_uv is None:
                K_uv = K_1d
            else:
                K_uv = np.kron(K_uv, K_1d)
        return K_uv

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
                if (u, v) not in self._omega_dict:
                    omega_uv = self.compute_K_uv(k_u, k_v)
                    self._omega_dict[u, v] = omega_uv
                    self._omega_dict[v, u] = omega_uv.T
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
        H_00 = self.compute_K_uv(d_u=self._zero_deriv, d_v=self._zero_deriv)
        H_00_inv = np.linalg.inv(H_00)
        H_0_dict = {}

        for u, k_u in enumerate(deriv_keys):
            if u not in H_0_dict:
                H_0_dict[u] = self.compute_K_uv(self._zero_deriv, k_u)
            H_0u = H_0_dict[u]

            for v, k_v in enumerate(deriv_keys[:u + 1]):
                H_0v = H_0_dict[v]
                omega_uv = H_0u.T @ H_00_inv @ H_0v
                self._omega_dict[u, v] = omega_uv
                self._omega_dict[v, u] = omega_uv.T
        return

    @staticmethod
    def _project_function(func, B_matrix, eval_points):
        """
        Common projection function evaluating an analytical function and finding B-spline coefficients via least squares.
        """
        y_val = func(eval_points).ravel()
        # Project onto the B-spline basis: B * w = y
        w, _, _, _ = np.linalg.lstsq(B_matrix, y_val, rcond=None)
        # Validate approximation fidelity
        g_approx = B_matrix @ w
        mse = np.mean((y_val - g_approx) ** 2)
        var_g = np.var(y_val) if np.var(y_val) > 0 else 1.0
        rel_error = mse / var_g

        print(f"{func} Projection Relative MSE: {rel_error:.2e}")
        if rel_error > 0.05:
            print("WARNING: B-spline basis may be too sparse!")
        return w

    def _project_forcing_function(self):
        """
        Projects the continuous forcing function g(x) onto the B-spline basis to find w.
        Actively reports the approximation fidelity.
        """
        return self._project_function(self.force, self.B_grid_, self.gridpoints_)

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
            self.H_0_ = self.compute_K_uv(d_u=self._zero_deriv, d_v=self._zero_deriv)
            self.H_0_cho_ = scipy.linalg.cho_factor(self.H_0_)

            # 2. Extract the raw K_i matrices into a clean dictionary
            for i, (d_u, d_v) in enumerate(self._coef_keys):
                if (i, i) not in self._omega_dict:
                    self._omega_dict[(i, i)] = self.compute_K_uv(d_u, d_v)

            # 3. Delegate to the standalone computational methods
            self.F_g_dict = self._compute_dual_F_g_dict(self._omega_dict, w)
            self._omega_dict = self._compute_dual_omega_dict(self._omega_dict)

        else:
            # Strong form evaluates the explicit component projections D_k(B_i)
            # H_u = integral( D_u(B) * B^T ) dx
            if self.smooth:
                self.compute_smooth_omega(list(self._coef_keys))
            else:
                self.compute_omega(list(self._coef_keys))
            self.F_g_dict = {}
            for i, u in enumerate(self._coef_keys):
                H_u = self.compute_K_uv(self._zero_deriv, u)
                self.F_g_dict[i] = H_u.T @ w

        self.force_norm = w.T @ self.compute_K_uv(self._zero_deriv, self._zero_deriv) @ w

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
        penalty_matrix = np.zeros_like(self.G_)
        forcing_vector = np.zeros(self.G_.shape[0])

        for i in range(len(self._coef_keys)):
            forcing_vector += coef_vals[i] * self.F_g_dict[i]
            for j in range(len(self._coef_keys)):
                penalty_matrix += coef_vals[i] * coef_vals[j] * self._omega_dict[(i, j)]

        assert np.abs(penalty_matrix - penalty_matrix.T).max() < self.tol
        penalty_matrix = (penalty_matrix + penalty_matrix.T) / 2.0

        # --- 2. Hard Constrained Reduced Space Projection ---
        if len(self.boundary_conditions) > 0:
            A_reduced = self.T_matrix_.T @ penalty_matrix @ self.T_matrix_
            v_reduced = self.T_matrix_.T @ (forcing_vector - penalty_matrix @ self.d_vec_)
            return A_reduced, v_reduced

        return penalty_matrix, forcing_vector

    def compute_theta_system_pde(self, omega_dict, beta, theta,
                                 bias_corrected=False, sigma=None,
                                 V=None, M_diag=None):
        """
        :param omega_dict:
        :param beta:
        :return: Q, r, G
        """
        m = len(self._coef_keys)
        n = len(beta)

        Gram = np.zeros((m, m))
        for key, omega_uv in omega_dict.items():
            G = beta.T @ omega_uv @ beta
            if bias_corrected:
                M_squared = M_diag ** 2
                if len(self.boundary_conditions) > 0:
                    V_proj = self.T_matrix_ @ V
                else:
                    V_proj = V
                K_uv = V_proj.T @ omega_uv @ V_proj
                bias = np.sum(np.diag(K_uv) * M_squared)
                bias = sigma ** 2 * bias
                print(f'Bias for {key}: {bias / G:.4%}')
                G = G - bias
            Gram[key] = G

        proj_vec = np.zeros(m)
        for i, _ in enumerate(self._coef_keys):
            proj_vec[i] = beta.T @ self.F_g_dict[i]

        Jacobian = np.zeros((n, m))
        for i, _ in enumerate(self._coef_keys):
            Omega_i = np.zeros((n, n))
            for j, _ in enumerate(self._coef_keys):
                Omega_i += theta[j] * (omega_dict[i, j] + omega_dict[j, i])
            Jacobian[:, i] = self.F_g_dict[i] - Omega_i @ beta

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
        theta_new = deriv_coef.copy()

        Tc = [i for i, _ in enumerate(self._coef_keys) if i not in self._trainable_idx]

        # 1. Calculate the pure OLS target
        if len(Tc) == 0:
            c_calc = np.linalg.solve(G[np.ix_(T, T)], v_vec[T])
        else:
            coef_Tc = deriv_coef[Tc]
            G_TT = G[np.ix_(T, T)]
            G_TTc = G[np.ix_(T, Tc)]
            if np.linalg.eigvals(G_TT).min() < self.tol:
                logger.warning('Trainable Gram not Positive!')
            if np.linalg.cond(G_TT) > 1 / self.tol:
                logger.warning('Trainable Gram not Invertible!')
            c_calc = np.linalg.solve(G_TT, v_vec[T] - G_TTc @ coef_Tc)

        # 2. Apply Momentum / Over-Relaxation to accelerate the tail convergence
        theta_new[T] = deriv_coef[T] + relaxation * (c_calc - deriv_coef[T])

        # 3. Variance estimation (Note: missing sigma^2 scaling)
        coef_var = np.linalg.inv(G[np.ix_(T, T)])
        return theta_new, coef_var

    def project_bc(self, bc_func, d, x_star):
        dims_sub = [j for j in range(self.dim) if j != d]
        # --- 1. Project boundary conditions on the N-1 dimension sub-grid ---
        if self.dim > 1:
            grids_sub = [self.grids_[j] if j != d else np.array([x_star]) for j in range(self.dim)]
            eval_points = util.tensor_prod(*grids_sub)
            B_sub = self._build_custom_tensor_basis(eval_points, dims_sub)
            c_sub = self._project_function(bc_func, B_sub, eval_points)
        else:
            eval_points = np.array([[x_star]])
            c_sub = bc_func(eval_points).ravel()
        return c_sub

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
        if len(bc_dict) == 0:
            return

        M_total = np.prod(self.n_splines)

        # 3D/ND occupancy grid mimicking the analytic shell-peeling to prevent redundant corner constraints
        is_dependent = np.zeros(self.n_splines, dtype=bool)

        A_eqs = []
        c_eqs = []
        dependent_flat_indices = []

        # Sort keys sequentially by dimension to guarantee lower block-triangular structure for invertibility
        def parse_boundary_string(bound_str):
            d_str, side = bound_str.split('_')
            # Safely map variables ('x', 'y') to indices (0, 1)
            d = self.variables.index(d_str) if d_str in self.variables else int(d_str)
            return d, side

        sorted_items = sorted(bc_dict.items(), key=lambda item: parse_boundary_string(item[0][0]))

        for (bound_str, p), bc_func in sorted_items:
            d_str, side = bound_str.split('_')
            d = int(d_str)
            assert side in ['min', 'max'], "Side must be 'min' or 'max'."

            # Extract derivative scalar evaluations of the 1D B-spline at the boundary
            x_star = self.domain_[d][0] if side == 'min' else self.domain_[d][1]
            c_sub = self.project_bc(bc_func, d, x_star)
            weights = self.spline_objs[d](np.array([x_star]), nu=p)[0]
            assert len(weights) == self.n_splines[d]

            # --- 2. Compute the constraints on our regression coefficients (A * beta = c) ---
            sub_shape = [self.n_splines[j] for j in range(self.dim) if j != d]
            sub_ranges = product(*[range(s) for s in sub_shape])
            boundary_idx = 0 if side == 'min' else self.n_splines[d] - 1

            # Pre-calculate active indices to skip iterating over zeros in the inner loop
            active_k_indices = np.nonzero(weights)[0]

            for i, sub_midx in enumerate(sub_ranges):
                # Cleanly insert the boundary index using native tuple slicing
                bound_midx = sub_midx[:d] + (boundary_idx,) + sub_midx[d:]

                # Application of "Shell-Peeling"
                if not is_dependent[bound_midx]:
                    is_dependent[bound_midx] = True
                    dependent_flat_indices.append(np.ravel_multi_index(bound_midx, self.n_splines))

                    # Assemble the constraint row based ONLY on the sparse derivative stencil
                    row = np.zeros(M_total)
                    for k in active_k_indices:
                        k_midx = sub_midx[:d] + (k,) + sub_midx[d:]
                        flat_k_idx = np.ravel_multi_index(k_midx, self.n_splines)
                        row[flat_k_idx] = weights[k]

                    A_eqs.append(row)
                    c_eqs.append(c_sub[i])

        # --- 3. Segment independent (interior) vs dependent (boundary) coefficients ---
        self.idx_B_ = np.array(dependent_flat_indices, dtype=int)
        self.idx_I_ = np.setdiff1d(np.arange(M_total), self.idx_B_)
        assert len(self.idx_B_) == len(np.unique(self.idx_B_))
        assert len(self.idx_I_) == np.prod(self.n_splines - 2)

        self.T_matrix_ = np.zeros((M_total, len(self.idx_I_)))
        self.d_vec_ = np.zeros(M_total)

        if len(A_eqs) > 0:
            A_full = np.array(A_eqs)
            c_full = np.array(c_eqs)

            A_B = A_full[:, self.idx_B_]
            A_I = A_full[:, self.idx_I_]

            cond_num = np.linalg.cond(A_B)
            assert cond_num < 1e12, f"A_B is too ill-conditioned for stable inversion! Condition number: {cond_num:.2e}"

            # Rigorous block-inversion maps boundary vars entirely to interior vars.
            # A_B is non-singular strictly because the shell-peeling prevents redundant rank deficiency.
            invA_c = np.linalg.solve(A_B, c_full)
            invA_AI = np.linalg.solve(A_B, A_I)

            self.T_matrix_[self.idx_I_, :] = np.eye(len(self.idx_I_))
            self.T_matrix_[self.idx_B_, :] = -invA_AI
            self.d_vec_[self.idx_B_] = invA_c
            # beta_full = self.T_matrix_ @ beta_I + self.d_vec_
        else:
            # Fallback if no valid BCs are passed (pure unconstrained)
            self.T_matrix_[self.idx_I_, :] = np.eye(len(self.idx_I_))
        # np.sort(np.linalg.eigvals(self.T_matrix_.T @ self.T_matrix_))
        return

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the EWA Spline model to the data."""
        X, y = check_X_y(X, y, accept_sparse=False)
        n = len(y)
        self.X_fit_, self.y_fit_ = X, y

        self._parse_physics_configuration()
        self._initialize_basis(X)
        self._setup_priors_and_lambda(X, y)
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

        # Perturb the reduced Gram matrix to guarantee numerical stability
        self.G_reduced_, eps = util.perturb_diag(self.G_reduced_)
        self.B_data_reduced_perturb = util.perturb_B_svd(self.B_data_reduced_, eps)

        coef_vals = self._initial_coef_vals.copy()
        self.fit_stats = {}
        # --- Block Coordinate Descent for Physics Parameters ---
        for iteration in range(self.max_iter):
            penalty_matrix, forcing_vector = self._build_penalty_and_force(coef_vals)
            p_norm = np.trace(penalty_matrix)

            # 1. Orthogonalization via GEVP
            orth = util.orthogonalize_basis_fast(self.G_reduced_, penalty_matrix)
            self.V_, self.s_raw_ = orth['V'], orth['s_raw'] #G^{-1} = VV^T
            self.U_ = self.B_data_reduced_perturb @ self.V_

            # 2. Project Forcing Vector into Orthogonal Space
            force_proj = self.V_.T @ forcing_vector
            s = np.maximum(self.s_raw_, 0)

            # 3. Fit EWA Model
            self.ewa_res_ = util.fit_ewa(
                y=self.y_adj_, ftrue=self.y_adj_, U=self.U_, V=self.V_, s=s,
                lam=self.lam_grid_, prior=self.prior_,
                sig_eps_est=self.sigma_est_, gamma_mult=self.gamma_mult,
                force_proj=force_proj
            )

            self.sigma_est_ = self.ewa_res_['sigma_est']

            # --- NEW: 4. Construct the rigid physical surrogate (\tilde{\beta}) ---
            beta_I = self.ewa_res_['beta']
            if len(self.boundary_conditions) > 0:
                self.beta_full_ = self.T_matrix_ @ beta_I + self.d_vec_
            else:
                self.beta_full_ = beta_I

            gram_matrices, proj_vec, jacob = self.compute_theta_system_pde(self._omega_dict,
                                                                           self.beta_full_,
                                                                           coef_vals,
                                                                           bias_corrected=self.bias_corrected,
                                                                           sigma=self.sigma_est_,
                                                                           V=self.V_,
                                                                           M_diag=self.ewa_res_['M_diag_alpha'])
            coef_mean, self.coef_var = self.get_coef_dist(gram_matrices, coef_vals, proj_vec)

            w = self.ewa_res_['weights']
            lamb_mean = (self.lam_grid_ * w).sum()
            lamb_mle = self.lam_grid_[np.argmax(w)]
            lamb_lq = self.lam_grid_[np.argmax(np.cumsum(w) >= self.alpha / 2)]
            lamb_uq = self.lam_grid_[np.argmax(np.cumsum(w) >= 1 - self.alpha / 2)]
            # self.weights_ = pd.Series(self.ewa_res_['weights'], index=self.lam_grid_).sort_index()

            # Gram, proj_vec, Jacobian = self.compute_theta_system_pde(self._omega_dict, self.ewa_res_['beta'], coef_vals)
            # grad = self.compute_grad(Gram, coef_vals, proj_vec)
            # hess = self.compute_hess(Gram, Jacobian, penalty_matrix, lamb_mean)
            # coef_mean = self.update_coef(coef_vals, grad, hess)
            # hess_T = np.linalg.eigvals(hess[np.ix_(self._trainable_idx, self._trainable_idx)])

            physics_loss = self.compute_physics_loss(coef_mean, gram_matrices, proj_vec)

            logger.info(f"Iteration {iteration} | Coefficients: {coef_mean} | "
                        f"sigma_est: {self.sigma_est_:.4e} | physics_loss: {physics_loss:.4e} | "
                        f"Penalty norm: {p_norm: .4e} | "
                        f"lamb_mean: {lamb_mean:.4e} ") #| hess_T: {hess_T}

            self.fit_stats[iteration] = np.array([physics_loss, n * self.sigma_est_ ** 2,
                                                  lamb_lq, lamb_mean, lamb_uq])

            # 5. Check Convergence
            old_theta_T = coef_vals[self._trainable_idx]
            new_theta_T = coef_mean[self._trainable_idx]
            coef_pct_chg = np.abs(old_theta_T - new_theta_T).sum() / (np.abs(old_theta_T).sum() + 1e-8)

            if coef_pct_chg < self.min_pct_chg:
                logger.info("Physics parameters converged.")
                break

            # old_physics_loss = self.fit_stats.get(iteration - 1, [np.inf])[0]
            # loss_pct_decr = 1 - (physics_loss / old_physics_loss)
            #
            # if loss_pct_decr <= self.min_pct_chg:
            #     print(loss_pct_decr)
            #     logger.info("Physics parameters converged.")
            #     break
            coef_vals = coef_mean

        # --- Finalize Fitted State ---
        self.deriv_coef_vals_ = coef_vals
        self.coef_mean = coef_mean
        self.fit_stats = pd.DataFrame(self.fit_stats, index=['pde_loss', 'fit_loss',
                                                             'w_lq', 'w_mean', 'w_uq']).T.rename_axis('iter')
        return self

    # def compute_grad(self, Gram, theta, proj_vec):
    #     grad = Gram @ theta - proj_vec
    #     T = self._trainable_idx
    #     Tc = [i for i, _ in enumerate(self._coef_keys) if i not in self._trainable_idx]
    #     grad_T = Gram[np.ix_(T, T)] @ theta[T] + Gram[np.ix_(T, Tc)] @ theta[Tc] - proj_vec[T]
    #     assert np.abs(grad_T - grad[T]).max() < 1e-6
    #     return grad
    #
    # def compute_hess(self, Gram, Jacobian, Omega, lamb):
    #     D = self.G_ + lamb * Omega
    #     hess = Gram - lamb * Jacobian.T @ np.linalg.solve(D, Jacobian)
    #     T = self._trainable_idx
    #     hess_TT = Gram[np.ix_(T, T)] - lamb * Jacobian[:, T].T @ np.linalg.solve(D, Jacobian[:, T])
    #     assert np.abs(hess_TT - hess[np.ix_(T, T)]).max() < 1e-6
    #     return hess
    #
    # def update_coef(self, theta_old, grad, hess):
    #     T = self._trainable_idx
    #     theta_new = theta_old.copy()
    #
    #     # Extract active blocks
    #     hess_TT = hess[np.ix_(T, T)]
    #     grad_T = grad[T]
    #
    #     # Solve for the step direction (Delta theta) directly
    #     step_T = np.linalg.solve(hess_TT, -grad_T)
    #
    #     # Update the active parameters
    #     theta_new[T] = theta_old[T] + step_T
    #     return theta_new
    #
    # def _profiled_objective(self, theta_T, y):
    #     """
    #     Callable objective function that simultaneously computes and caches
    #     the Loss, Exact Gradient, and Exact Hessian to avoid redundant EWA runs.
    #     """
    #     # 1. Protect against redundant calls from the solver
    #     if hasattr(self, '_cached_theta') and np.allclose(theta_T, self._cached_theta):
    #         return self._cached_loss, self._cached_grad
    #
    #     # 2. Reconstruct the full physical parameter vector
    #     coef_vals = self._initial_coef_vals.copy()
    #     coef_vals[self._trainable_idx] = theta_T
    #
    #     # 3. Build current penalty matrix and forcing vector
    #     penalty_matrix, forcing_vector = self._build_penalty_and_force(coef_vals)
    #
    #     # 4. Orthogonalization & Projection
    #     orth = util.orthogonalize_basis_fast(self.G_, penalty_matrix)
    #     self.V_, self.s_raw_ = orth['V'], orth['s_raw']
    #     self.U_ = self.B_data_ @ self.V_
    #     force_proj = self.V_.T @ forcing_vector
    #     s = np.maximum(self.s_raw_, 0)
    #
    #     # 5. Profile out beta by fitting the EWA Model
    #     self.ewa_res_ = util.fit_ewa(
    #         y=y, ftrue=y, U=self.U_, V=self.V_, s=s,
    #         lam=self.lam_grid_, prior=self.prior_,
    #         sig_eps_est=self.sigma_est_, gamma_mult=self.gamma_mult,
    #         force_proj=force_proj
    #     )
    #
    #     # 6. Extract expected parameters
    #     w = self.ewa_res_['weights']
    #     lamb_mean = (self.lam_grid_ * w).sum()
    #     beta = self.ewa_res_['beta']
    #     self.lamb_mean = lamb_mean
    #
    #     # 7. Evaluate exact components (We MUST compute the Jacobian here for the Hessian)
    #     Gram, proj_vec, Jacobian = self.compute_theta_system_pde(self._omega_dict, beta, coef_vals)
    #     physics_loss = self.compute_physics_loss(coef_vals, Gram, proj_vec)
    #
    #     # 8. Update exact noise variance
    #     rss = np.sum((self.ewa_res_['fhat_ewa'] - y) ** 2)
    #     self.sigma_est_ = np.sqrt(rss / (len(y) - self.ewa_res_['trace_hat']))
    #
    #     # 9. Compute Total Loss and Analytical Gradient
    #     total_loss = rss + lamb_mean * physics_loss
    #     grad_full = 2 * lamb_mean * (Gram @ coef_vals - proj_vec)
    #     grad_T = grad_full[self._trainable_idx]
    #
    #     # 10. Compute and Cache the Exact Hessian
    #     # Utilizing your existing compute_hess method
    #     hess_full = 2 * lamb_mean * self.compute_hess(Gram, Jacobian, penalty_matrix, lamb_mean)
    #     hess_T = hess_full[np.ix_(self._trainable_idx, self._trainable_idx)]
    #
    #     # 11. Save to Class State
    #     self._cached_theta = theta_T.copy()
    #     self._cached_loss = float(total_loss)
    #     self._cached_grad = grad_T.astype(np.float64)
    #     self._cached_hess = hess_T.astype(np.float64)
    #
    #     return self._cached_loss, self._cached_grad
    #
    # def _profiled_hessian(self, theta_T, y):
    #     """
    #     Retrieves the exact Hessian from the cache. The solver inherently
    #     evaluates the objective immediately before requesting the Hessian.
    #     """
    #     # Safety check: if the solver asks for a Hessian at a new point without
    #     # evaluating the objective first, force an update.
    #     if not (hasattr(self, '_cached_theta') and np.allclose(theta_T, self._cached_theta)):
    #         self._profiled_objective(theta_T, y)
    #
    #     return self._cached_hess
    #
    # def compute_asymp_var(self, Jacobian, Hessian, penalty_matrix, lamb):
    #     T = self._trainable_idx
    #     G_T = Jacobian[:, T]
    #     H_TT_inv = np.linalg.inv(Hessian)[np.ix_(T, T)]
    #
    #     D = self.G_ + lamb * penalty_matrix
    #     D_inv_GT = np.linalg.solve(D, G_T)
    #     sigma_sq = self.sigma_est_ ** 2
    #     M_T = D_inv_GT.T @ self.G_ @ D_inv_GT
    #     Sigma_T = sigma_sq * (H_TT_inv @ M_T @ H_TT_inv)
    #     return Sigma_T
    #
    # def _fit(self, X: np.ndarray, y: np.ndarray):
    #     """Fits the EWA Spline model to the data using L-BFGS-B optimization."""
    #     from scipy.optimize import minimize  # Ensure this is imported
    #
    #     X, y = check_X_y(X, y, accept_sparse=False)
    #     self.X_fit_, self.y_fit_ = X, y
    #
    #     self._parse_physics_configuration()
    #     self._initialize_basis(X)
    #     self._setup_priors_and_lambda(X, y)
    #     self.precompute_forcing_components()
    #
    #     # Start L-BFGS optimization
    #     initial_theta_T = self._initial_coef_vals[self._trainable_idx]
    #
    #     logger.info("Starting L-BFGS optimization for physics parameters...")
    #
    #     res = minimize(
    #         fun=self._profiled_objective,
    #         x0=initial_theta_T,
    #         args=(y,),
    #         method='trust-exact',#'L-BFGS-B',
    #         jac=True,
    #         hess=self._profiled_hessian,
    #         tol=self.tol,
    #         options={'maxiter': self.max_iter, 'disp': True}
    #     )
    #
    #     if res.success:
    #         logger.info(f"Physics parameters converged in {res.nit} iterations.")
    #     else:
    #         logger.warning(f"L-BFGS terminated early: {res.message}")
    #
    #     # --- Finalize Fitted State ---
    #     self.deriv_coef_vals_ = self._initial_coef_vals.copy()
    #     self.deriv_coef_vals_[self._trainable_idx] = res.x
    #     self.coef_mean = self.deriv_coef_vals_
    #
    #     # Run one final evaluation at the optimal parameters to lock in the final
    #     # V_, U_, and ewa_res_ variables so predict() has access to the final state.
    #     #final_loss, _ = self._profiled_objective(res.x, y)
    #     self._profiled_objective(res.x, y)
    #
    #     coef_vals = self.coef_mean
    #     penalty_matrix, forcing_vector = self._build_penalty_and_force(coef_vals)
    #     Gram, proj_vec, Jacobian = self.compute_theta_system_pde(self._omega_dict, self.ewa_res_['beta'], coef_vals)
    #     # grad = self.compute_grad(Gram, coef_vals, proj_vec)
    #     hess = self.compute_hess(Gram, Jacobian, penalty_matrix, self.lamb_mean)
    #     self.coef_var = self.compute_asymp_var(Jacobian, hess, penalty_matrix, self.lamb_mean)
    #     return self

    # def get_coef_dist(self, G, deriv_coef, v_vec, beta, V, shrink_weight, sigma_sq, relaxation=1.0):
    #     """
    #     Computes Exact 2nd-Order Unbiased Physical Parameters and Variance.
    #     Evaluates the current stochastic trajectory (theta^(t)) to neutralize
    #     Errors-in-Variables (EIV) bias, Jensen's matrix ratio bias, and the
    #     empirical Jacobian plug-in inflation.
    #     """
    #     T = self._trainable_idx
    #     Tc = [i for i, _ in enumerate(self._coef_keys) if i not in self._trainable_idx]
    #
    #     # ==========================================================
    #     # 1. 1st-Order Unbiased Base Matrices (EIV Correction)
    #     # ==========================================================
    #     G_unbiased = np.zeros_like(G)
    #     W2 = shrink_weight ** 2
    #
    #     for k in range(G.shape[0]):
    #         for j in range(G.shape[1]):
    #             if (k, j) in self._omega_dict:
    #                 Omega_kj = self._omega_dict[(k, j)]
    #             elif (j, k) in self._omega_dict:
    #                 Omega_kj = self._omega_dict[(j, k)].T
    #             else:
    #                 G_unbiased[k, j] = G[k, j]
    #                 continue
    #
    #             Omega_V = Omega_kj @ V
    #             bias = sigma_sq * np.dot(np.sum(V * Omega_V, axis=0), W2)
    #             G_unbiased[k, j] = G[k, j] - bias
    #
    #     G_TT = G_unbiased[np.ix_(T, T)]
    #     G_TT_inv = np.linalg.inv(G_TT)
    #
    #     # ==========================================================
    #     # 2. The Naive BCD Target (1st-Order Stationarity)
    #     # ==========================================================
    #     if len(Tc) == 0:
    #         theta_naive = G_TT_inv @ v_vec[T]
    #     else:
    #         G_TTc = G_unbiased[np.ix_(T, Tc)]
    #         coef_Tc = deriv_coef[Tc]
    #         theta_naive = G_TT_inv @ (v_vec[T] - G_TTc @ coef_Tc)
    #
    #     # ==========================================================
    #     # 3. Empirical Jacobians at Stochastic State \theta^{(t)}
    #     # ==========================================================
    #     N_beta = len(beta)
    #     H = np.zeros((len(T), N_beta))
    #     grad_H = np.zeros((len(T), N_beta, N_beta))
    #
    #     for idx, k in enumerate(T):
    #         # Base forcing function
    #         H_k = self.F_g_dict[k].copy()
    #         Sum_Omega_grad = np.zeros((N_beta, N_beta))
    #
    #         for j in range(len(self._coef_keys)):
    #             if (k, j) in self._omega_dict:
    #                 Omega_kj = self._omega_dict[(k, j)]
    #             elif (j, k) in self._omega_dict:
    #                 Omega_kj = self._omega_dict[(j, k)].T
    #             else:
    #                 continue
    #
    #             Omega_grad = Omega_kj + Omega_kj.T
    #             # Evaluated precisely at the current path state: deriv_coef[j]
    #             H_k -= deriv_coef[j] * (Omega_grad @ beta)
    #             Sum_Omega_grad += deriv_coef[j] * Omega_grad
    #
    #         H[idx, :] = H_k
    #         grad_H[idx, :, :] = -Sum_Omega_grad
    #
    #     # J_theta mapping: d(theta) / d(beta)
    #     J_theta = G_TT_inv @ H
    #
    #     # ==========================================================
    #     # 4. The 2nd-Order Unbiased Estimator (Jensen's Shift)
    #     # ==========================================================
    #     b_ratio = np.zeros(len(T))
    #     for idx, k in enumerate(T):
    #         for i_idx, i in enumerate(T):
    #             if (k, i) in self._omega_dict:
    #                 Omega_ki = self._omega_dict[(k, i)]
    #             elif (i, k) in self._omega_dict:
    #                 Omega_ki = self._omega_dict[(i, k)].T
    #             else:
    #                 continue
    #
    #             # Sensitivity of the Curvature Matrix: d(G_ki) / d(beta)
    #             J_G_ki = (Omega_ki + Omega_ki.T) @ beta
    #
    #             # Exact Spectral Trace: tr(J_G_ki^T * Sigma_beta * J_theta_i^T)
    #             covariance_term = sigma_sq * np.sum((J_G_ki @ V) * W2 * (J_theta[i_idx, :] @ V))
    #             b_ratio[idx] += covariance_term
    #
    #     # Construct the true mathematically unbiased target
    #     theta_unbiased = theta_naive + (G_TT_inv @ b_ratio)
    #
    #     # ==========================================================
    #     # 5. Iterative Path Update
    #     # ==========================================================
    #     # Execute the Block Coordinate Descent relaxation towards the unbiased root
    #     coef_mean = deriv_coef[T] + relaxation * (theta_unbiased - deriv_coef[T])
    #
    #     # ==========================================================
    #     # 6. Exact 2nd-Order Variance Geometry (Plug-in Corrected)
    #     # ==========================================================
    #     # Base Asymptotic Variance
    #     JV = J_theta @ V
    #     coef_var_base = sigma_sq * (JV * W2[None, :]) @ JV.T
    #
    #     # Tensor projection for the exact 2nd-order geometrical inflation
    #     grad_J = np.tensordot(G_TT_inv, grad_H, axes=([1], [0]))
    #     var_correction = np.zeros((len(T), len(T)))
    #     W2_mat = W2[:, None] * W2[None, :]
    #
    #     for i in range(len(T)):
    #         V_grad_J_i_V = V.T @ grad_J[i] @ V
    #         for j in range(len(T)):
    #             V_grad_J_j_V = V.T @ grad_J[j] @ V
    #             trace_val = np.sum(V_grad_J_i_V * V_grad_J_j_V * W2_mat)
    #             var_correction[i, j] = (sigma_sq ** 2) * trace_val
    #
    #     # Subtract the deterministic empirical plug-in inflation
    #     coef_var = coef_var_base - var_correction
    #
    #     return coef_mean, coef_var

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

        # Project test basis into the reduced space for correct variance propagation
        if len(self.boundary_conditions) > 0:
            B_new_reduced = B_new @ self.T_matrix_
        else:
            B_new_reduced = B_new

        U_new = B_new_reduced @ self.V_
        y_hat = B_new @ self.beta_full_

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
    #
    # def plot_fit(self, x_test=None, y_test=None, show_ci=True):
    #     check_is_fitted(self)
    #     x_grid = self.gridpoints_
    #     fit_preds = self.predict(x_grid)
    #
    #     if x_test is None:
    #         x_scatter, y_scatter = self.X_fit_, self.y_fit_
    #     else:
    #         x_scatter, y_scatter = x_test, y_test
    #
    #     if self.dim == 1:
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #
    #         if show_ci:
    #             ax.fill_between(x_grid[:, 0], fit_preds['ci_lower'], fit_preds['ci_upper'],
    #                             color='C0', alpha=0.15, label=f'{1 - self.alpha:.0%} CI (during Fit)')
    #         ax.plot(x_grid[:, 0], fit_preds['fhat'], color='C0', lw=2, label='EWA Prediction')
    #         ax.scatter(x_scatter[:, 0], y_scatter, s=10, color='black', alpha=0.5, label='Data')
    #
    #         deriv_coef = list(map(lambda x: float(f"{x:.4g}"), self.deriv_coef_vals_))
    #         title = (
    #             f"EWA fit [n={len(self.X_fit_)}, deg={self.deg}, deriv_coef={deriv_coef}, "
    #             f"nk={self.n_int_knots_1d}, sigma={self.sigma_est_:.3f}, "
    #             f"{100 * (1 - self.alpha):.0f}% CI]"
    #         )
    #         ax.set(title=title, xlabel="X", ylabel="y")
    #         ax.legend()
    #         plt.show()
    #
    #     elif self.dim == 2:
    #         X = x_grid[:, 0].reshape(self.grid_size_, self.grid_size_)
    #         Y = x_grid[:, 1].reshape(self.grid_size_, self.grid_size_)
    #         Z = fit_preds['fhat'].values.reshape(self.grid_size_, self.grid_size_)
    #         vmax = np.abs(y_scatter).max()
    #
    #         norm = util.norm_color(vmin=-vmax, vmax=vmax)
    #         fig, ax = util.contour_plot(X, Y, Z, vmin=-vmax, vmax=vmax)
    #         ax.scatter(x_scatter[:, 0], x_scatter[:, 1], c=y_scatter,
    #                    norm=norm, cmap=util.CMAP, edgecolors='black', s=20, label='Data')
    #
    #         deriv_coef = list(map(lambda x: float(f"{x:.4g}"), self.deriv_coef_vals_))
    #         deriv_coef = dict(zip(list(map(str, self._coef_keys)), deriv_coef))
    #         title = (
    #             f"2D EWA fit [n={len(self.X_fit_)}, deg={self.deg}, deriv_coef={deriv_coef}, "
    #             f"nk={self.n_int_knots_1d}, sigma={self.sigma_est_:.3f}, "
    #             f"{100 * (1 - self.alpha):.0f}% CI]"
    #         )
    #         ax.set(title=title, xlabel="X1", ylabel="X2")
    #         plt.show()
    #
    #     else:
    #         print(f"Plotting not supported for dimension {self.dim} > 2.")

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
        for i, order in enumerate(self._coef_keys):
            pde_loss_learnt += self.deriv_coef_vals_[i] * self.predict_derivative(x_grid, order)

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
            X_mesh = x_grid[:, 0].reshape(self.grid_size_, self.grid_size_)
            Y_mesh = x_grid[:, 1].reshape(self.grid_size_, self.grid_size_)

            # Subplot 1: Data Residual Scatter + CI Contours
            vmax_res = np.max(np.abs(data_residuals))
            if vmax_res == 0: vmax_res = 1e-8

            # Plot CI Contour overlays (since bounds are symmetric, plotting the upper margin maps the +/- envelope)
            err_ci_upper_mesh = err_ci_upper.values.reshape(self.grid_size_, self.grid_size_)
            cs = axes[0].contour(X_mesh, Y_mesh, err_ci_upper_mesh, levels=5, colors='black', alpha=0.4,
                                 linestyles='dashed')
            axes[0].clabel(cs, inline=True, fontsize=8, fmt='CI: ±%.2f')

            # Plot scatter
            norm_res = util.norm_color(vmin=-vmax_res, vmax=vmax_res)
            sc = axes[0].scatter(x_scatter[:, 0], x_scatter[:, 1], c=data_residuals,
                                 norm=norm_res, cmap=util.CMAP, edgecolors='black', s=20)
            fig.colorbar(sc, ax=axes[0], shrink=0.8, pad=0.04)
            axes[0].set(title="Data Residuals $(Y - \hat{Y})$", xlabel="X1", ylabel="X2")

            # Subplot 2: Learnt PDE Loss Heatmap
            Z_learnt = pde_loss_learnt.reshape(self.grid_size_, self.grid_size_)
            vmax_pde = np.max(np.abs(Z_learnt))
            if vmax_pde == 0: vmax_pde = 1e-8

            _, ax1 = util.contour_plot(X_mesh, Y_mesh, Z_learnt, ax=axes[1], vmin=-vmax_pde, vmax=vmax_pde, levels=30,
                                       cbar=False)
            sm_learnt = util.map_color(vmin=-vmax_pde, vmax=vmax_pde)
            fig.colorbar(sm_learnt, ax=axes[1], shrink=0.8, pad=0.04)
            axes[1].set(title="Learnt PDE Pointwise Loss", xlabel="X1", ylabel="X2")

            # Subplot 3: True PDE Loss Heatmap
            if pde_loss_true is not None:
                Z_true = pde_loss_true.reshape(self.grid_size_, self.grid_size_)
                vmax_true = np.max(np.abs(Z_true))
                if vmax_true == 0: vmax_true = 1e-8

                _, ax2 = util.contour_plot(X_mesh, Y_mesh, Z_true, ax=axes[2], vmin=-vmax_true, vmax=vmax_true,
                                           levels=30, cbar=False)
                sm_true = util.map_color(vmin=-vmax_true, vmax=vmax_true)
                fig.colorbar(sm_true, ax=axes[2], shrink=0.8, pad=0.04)
                axes[2].set(title="True PDE Pointwise Loss", xlabel="X1", ylabel="X2")

        else:
            print(f"Plotting not supported for dimension {self.dim} > 2.")

        # --- 5. Common Formatting ---
        deriv_coef = {str(k): float(f"{v:.4g}") for k, v in zip(self._coef_keys, self.deriv_coef_vals_)}
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
            grid_shape = (len(self.grids_[0]), len(self.grids_[1]))
            X1, X2 = np.meshgrid(self.grids_[0], self.grids_[1], indexing='ij')

            fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                                     squeeze=False, layout='constrained')

            for idx, order in enumerate(deriv_orders_list):
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
                ax_pred.set(title=f"Predicted {order}", xlabel="X1", ylabel="X2")

                if plot_error:
                    ax_err = axes[base_row + 1, c]
                    Z_true = true_derivs[order]
                    Z_err = (Z_flat - Z_true)

                    Z_true_norm = np.sqrt((Z_true ** 2).mean())
                    Z_rel_err = Z_err / Z_true_norm
                    Z_rel_err = Z_rel_err.reshape(grid_shape)

                    # np.sqrt(np.sum(Z_err ** 2) / np.sum(Z_true ** 2))
                    Z_rel_err_norm = np.sqrt((Z_rel_err ** 2).mean())

                    vmax_err = np.max(np.abs(Z_rel_err))
                    if vmax_err == 0: vmax_err = 1e-8

                    _, ax_err = util.contour_plot(X1, X2, Z_rel_err, ax=ax_err, vmin=-vmax_err, vmax=vmax_err, levels=30,
                                                  cbar=False)
                    sm_err = util.map_color(vmin=-vmax_err, vmax=vmax_err)
                    fig.colorbar(sm_err, ax=ax_err, shrink=0.8, pad=0.04, format=mtick.StrMethodFormatter('{x:.0%}'))
                    ax_err.set(title=f"{order} Relative Error: {Z_rel_err_norm:.4%}", xlabel="X1", ylabel="X2")

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
            print(f"Plotting derivatives not supported for dimension {self.dim} > 3.")

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

    def __repr__(self):
        status = "Fitted" if hasattr(self, 'theta_hat_') else "Unfitted"
        fit_info = {k: self.__getattribute__(k)
                    for k in ['fit_n', 'domain', 'deg', 'deriv_coef', 'n_knots',
                              'sigma_regr', 'gamma_mult', 'prior_type', 'alpha',
                              ]}
        return f"EWABSpline({status}, {fit_info})"


if __name__ == '__main__':
    print(1)