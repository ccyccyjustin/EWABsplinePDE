import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import util
from Bsplines import (
    build_custom_sparse_tensor_basis,
    build_custom_tensor_basis,
    create_knots,
    create_spline_basis,
    knots_data_count,
    tensor_prod,
)
from ewa_assembly import PDEAssemblyMixin
from ewa_bc import BoundaryConditionMixin
from ewa_inference import PDEInferenceMixin
from ewa_plot import EWAPlotMixin

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@util.time_all_methods
class EWABSplinePDE(
    PDEAssemblyMixin,
    BoundaryConditionMixin,
    PDEInferenceMixin,
    EWAPlotMixin,
    BaseEstimator,
    RegressorMixin,
):
    """
    Exponentially Weighted Aggregation (EWA) B-Spline Estimator with Physics-Informed Penalties.
    """

    def __init__(
        self,
        dim: int,
        deg: int = 3,
        variables: Optional[List[str]] = None,
        weak_form: bool = False,
        bias_corrected: bool = True,
        accelerate: bool = False,
        deriv_coef: Dict[str, float] = None,
        force: callable = None,
        boundary_conditions: dict = {},
        trainable: Optional[List[str]] = None,
        max_iter: int = 100,
        min_param_pct_chg: float = 1e-4,
        domain: Optional[List[Tuple[float, float]]] = None,
        min_samples_per_cell: int = 3,
        num_knots_1d_target: Optional[int] = 30,
        quantile_based: bool = False,
        total_grid_size: int = 1000,
        sigma: Optional[float] = None,
        gamma_mult: float = 8.0,
        prior_type: str = "light",
        alpha: float = 0.05,
        lam_size: int = 5000,
        tol: float = 1e-8,
        sparse: bool = True,
        max_eigfunc: int = None,
    ):
        self.dim = dim
        self.deg = deg
        self.variables = variables or [str(i) for i in range(dim)]

        self.weak_form = weak_form
        self.bias_corrected = bias_corrected
        self.accelerate = accelerate
        self.deriv_coef = deriv_coef or {}
        self.force = force or (lambda s: np.zeros(s.shape[0]))
        self.boundary_conditions = boundary_conditions
        self.trainable = trainable

        self.max_iter = max_iter
        self.min_param_pct_chg = min_param_pct_chg
        self.domain = domain
        self.min_samples_per_cell = min_samples_per_cell
        self.num_knots_1d_target = num_knots_1d_target
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

    # ------------------------------------------------------------------
    # Physics Configuration Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _get_deriv_order(deriv_str, variables):
        """
        Example:
        1. '' -> u
        2. '0' -> u_x
        3. '1' -> u_y
        4. '0,0' -> u_xx
        """
        dim = len(variables)

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
                raise ValueError(
                    f'{error_val} not accepted when there are {dim} dimensions with variables {variables}'
                )

        return tuple(int((np.array(deriv_splits) == p).sum()) for p in range(dim))

    def parse_derivative_order(self, op_key):
        """
        Parses a nonlinear PDE term into a list of derivative order vectors.
        Example: ((0, 0, 2), (1, 2)) -> [[2, 0, 1], [0, 1, 1]]
        """
        factors = []

        if len(op_key) == 0:
            return [np.zeros(self.dim, dtype=int)]

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
        Maps a derivative order tuple to its LaTeX string representation using self.variables.
        Example: (2, 0) -> '$u_{xx}$'
        """
        if sum(deriv_order) == 0:
            return "$u$"
        subscript = "".join(self.variables[i] * degree for i, degree in enumerate(deriv_order))
        return f"$u_{{{subscript}}}$"

    def _determine_trainable_indices(self):
        """Determines which physics parameters are actively optimized."""
        if self.trainable is not None:
            trainable_list = []
            for k in self.trainable:
                if isinstance(k, tuple):
                    k_tuple = tuple(self._get_deriv_order(ki, self.variables) for ki in k)
                else:
                    k_tuple = self._get_deriv_order(k, self.variables)
                trainable_list.append(k_tuple)
            t_str = list(map(str, trainable_list))
            k_str = list(map(str, self._coef_keys))
            assert set(t_str) <= set(k_str), "Make sure trainable params are defined in deriv_coef."
            self._trainable_idx = [i for i, d in enumerate(k_str) if d in t_str]
        else:
            deriv_order = list(map(len, self._coef_keys))
            self._trainable_idx = [i for i, order in enumerate(deriv_order) if order < max(deriv_order)]

    def _parse_physics_configuration(self):
        """Parses and validates PDE/Action coefficients into internal mathematical formats."""
        if self.deriv_coef:
            self.deriv_map = {self._get_deriv_order(k, self.variables): k for k in self.deriv_coef.keys()}
            sorted_coef = sorted(zip(self.deriv_map.keys(), self.deriv_coef.values()))
            self._coef_keys, self._initial_coef_vals = zip(*sorted_coef)
            self.mode = 'PDE'
        else:
            raise ValueError("Must provide physics constraints via deriv_coef or action_coef.")

        self.coef_keys = [self.format_deriv_orders(k) for k in self._coef_keys]
        self._initial_coef_vals = np.array(self._initial_coef_vals, dtype=float)
        self._zero_deriv = tuple([0] * self.dim)
        self._determine_trainable_indices()

    # ------------------------------------------------------------------
    # Knot, Basis & Prior Setup
    # ------------------------------------------------------------------
    def design_knots(self, X):
        n, d = X.shape
        p = self.deg
        K_1d_small_ss = (n / self.min_samples_per_cell) ** (1 / d)

        asym_order = 1 / (2 * (p + 1) + d)
        n_cutoff = self.min_samples_per_cell * (self.num_knots_1d_target ** d)
        scaling_const = self.num_knots_1d_target * (n_cutoff ** (-asym_order))
        K_1d_asym = scaling_const * (n ** asym_order)

        K_1d = min(K_1d_small_ss, K_1d_asym)
        K_1d = K_1d - (p + 1)
        K_1d = max(1, int(K_1d))

        n_knots = np.repeat(K_1d, self.dim).astype(int)
        self.num_int_knots = np.prod(n_knots)
        return n_knots

    def generate_knots(self, X, y):
        self.a_, self.b_ = X.min(axis=0), X.max(axis=0)
        self.domain = self.domain or list(zip(self.a_, self.b_))
        self.domain = np.array(self.domain)

        n_knots = self.design_knots(X)

        if self.sparse is None:
            self.sparse = np.prod(n_knots) > 3000

        if self.quantile_based:
            self.int_knots_ = [
                np.percentile(X[:, i], np.linspace(0, 100, n_knots[i] + 2)[1:-1])
                for i in range(X.shape[1])
            ]
        else:
            self.int_knots_ = [
                np.linspace(*d, n_knots[i] + 2)[1:-1]
                for i, d in enumerate(self.domain)
            ]

        self.step_size = [np.diff(knt).mean() for knt in self.int_knots_]
        self.all_knots = [
            create_knots(int_knots=self.int_knots_[i], degree=self.deg, boundary_knots=self.domain[i])
            for i in range(X.shape[1])
        ]
        self.n_knots = np.array(list(map(len, self.all_knots)))
        self.knot_counts = knots_data_count(X, self.all_knots, self.deg)

    def _build_tensor_basis(self, X):
        """Constructs the tensor product of 1D B-splines."""
        if self.spline_objs is None:
            self.spline_objs = [
                create_spline_basis(all_knots=self.all_knots[i], degree=self.deg)
                for i in range(self.dim)
            ]
            self.n_splines = np.array([s.c.shape[0] for s in self.spline_objs])
            self.total_n_splines = np.prod(self.n_splines)

        if self.sparse:
            return build_custom_sparse_tensor_basis(
                X, self.n_splines, self.all_knots, self.deg, self.dim, list(range(self.dim))
            )
        return build_custom_tensor_basis(X, self.spline_objs, self.dim, list(range(self.dim)))

    def _initialize_basis(self, X: np.ndarray, y):
        """Generates knots, grids, and projects the design matrix."""
        self.generate_knots(X, y)

        self.B_data_ = self._build_tensor_basis(X)
        self.G_ = self.B_data_.T @ self.B_data_

        min_grid_size = int(self.total_grid_size ** (1 / self.dim))
        self.grid_size_ = [max(self.n_knots[d] * 3, min_grid_size) for d in range(self.dim)]
        self.grids_ = [np.linspace(*d, self.grid_size_[dim]) for (d, dim) in zip(self.domain, range(self.dim))]
        self.gridpoints_ = tensor_prod(*self.grids_)
        self.B_grid_ = self._build_tensor_basis(self.gridpoints_)

    def _setup_priors_and_lambda(self, X, y: np.ndarray, tail_clip=0.01):
        """Constructs the discrete lambda grid and SURE prior distributions."""
        self.sigma_est_ = util.gsj_est(X, y) if self.sigma is None else self.sigma
        lam_grid = np.linspace(0, 1, self.lam_size)[1:-1]
        prior = np.ones(len(lam_grid))
        self.prior_ = prior / prior.sum()
        self.lam_grid_ = lam_grid

    # ------------------------------------------------------------------
    # Fit & Prediction
    # ------------------------------------------------------------------
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

        # Project Data Loss into the Unconstrained Space
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
        converged = False
        start_converge = False

        # Block Coordinate Descent for Physics Parameters
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
                evals, evecs = util.orthogonalize_basis(G_norm, penalty_matrix)

            self.V_, self.s_raw_ = evecs, evals
            self.U_ = self.B_data_reduced_ @ self.V_

            # Project Forcing Vector into Orthogonal Space
            self.force_proj = self.V_.T @ forcing_vector
            self.s_ = np.maximum(self.s_raw_, 0.0)
            logger.info('Eigvals: ' + ', '.join([f'{s:.4f}' for s in self.s_raw_[:8]]))

            # Fit EWA Model
            self.ewa_res_ = util.fit_ewa(
                y=self.y_adj_, ftrue=self.y_adj_, U=self.U_, V=self.V_, s=self.s_,
                lam_vec=self.lam_grid_, prior=self.prior_,
                sigma_est=self.sigma_est_, gamma_mult=self.gamma_mult,
                force_proj=self.force_proj,
            )
            self.sigma_est_ = self.ewa_res_['sigma_est']

            # Construct the rigid physical surrogate
            beta_I = self.ewa_res_['beta']
            if len(self.boundary_conditions) > 0:
                self.beta_full_ = self.T_matrix_ @ beta_I + self.d_vec_
                self.Vw = (self.T_matrix_ @ self.V_) * self.ewa_res_['shrink_weight']
            else:
                self.beta_full_ = beta_I
                self.Vw = self.V_ * self.ewa_res_['shrink_weight']

            gram_matrices, proj_vec, _ = self.compute_theta_system_pde(
                self._omega_dict,
                self.beta_full_,
                coef_vals,
                sigma=self.sigma_est_,
                bias_corrected=self.bias_corrected,
            )

            if converged:
                break

            if not (self.accelerate and start_converge):
                coef_mean = self.get_coef_dist(gram_matrices, coef_vals, proj_vec)
            else:
                A, B, _, _, _ = self.compute_asymptotic_variance(
                    coef_vals, self.ewa_res_, self.y_adj_, self.force_proj,
                    lam_vec=self.lam_grid_, U=self.U_, V=self.V_, s=self.s_,
                )
                coef_mean = self.get_coef_dist_corrected(gram_matrices, coef_vals, proj_vec, A, B)

            w = self.ewa_res_['weights']
            lamb_mean = (self.lam_grid_ * w).sum()
            lamb_mle = self.lam_grid_[np.argmax(w)]
            lamb_lq = self.lam_grid_[np.argmax(np.cumsum(w) >= self.alpha / 2)]
            lamb_uq = self.lam_grid_[np.argmax(np.cumsum(w) >= 1 - self.alpha / 2)]

            physics_loss = self.compute_physics_loss(coef_mean, gram_matrices, proj_vec)
            coef_dict = dict(zip(self.coef_keys, coef_mean))

            logger.info(
                f"Iteration {iteration} | Coefficients: {coef_dict} | "
                f"sigma_est: {self.sigma_est_:.4e} | physics_loss: {physics_loss:.4e} | "
                f"lamb_mean: {lamb_mean:.4e} | lamb_mle: {lamb_mle:.4e}"
            )

            self.fit_stats[iteration] = np.array([
                physics_loss, n * self.sigma_est_ ** 2, lamb_lq, lamb_mean, lamb_uq
            ])

            # Check Convergence
            old_theta_T = coef_vals[self._trainable_idx]
            new_theta_T = coef_mean[self._trainable_idx]
            coef_pct_chg = np.abs(old_theta_T - new_theta_T).sum() / (np.abs(old_theta_T).sum() + 1e-8)

            coef_vals = coef_mean
            if coef_pct_chg < 0.1:
                start_converge = True
            if coef_pct_chg < self.min_param_pct_chg:
                logger.info("Physics parameters converged.")
                converged = True

        # Finalize Fitted State
        self.deriv_coef = {self.deriv_map[self._coef_keys[i]]: v for i, v in enumerate(coef_vals)}
        self.coef_mean = coef_mean

        self.bread_matrix, self.meat_matrix, self.cov_beta_sqrt, _, _ = self.compute_asymptotic_variance(
            self.coef_mean, self.ewa_res_, self.y_adj_, self.force_proj,
            lam_vec=self.lam_grid_, U=self.U_, V=self.V_, s=self.s_,
        )
        bread_inv = np.linalg.inv(self.bread_matrix)
        self.coef_var = bread_inv @ self.meat_matrix @ bread_inv.T
        self.fit_stats = pd.DataFrame(
            self.fit_stats, index=['pde_loss', 'fit_loss', 'w_lq', 'w_mean', 'w_uq']
        ).T.rename_axis('iter')
        return self

    @property
    def basis_functions(self):
        if len(self.boundary_conditions) > 0:
            basis = self.B_grid_ @ (self.T_matrix_ @ self.V_)
        else:
            basis = self.B_grid_ @ self.V_
        basis_wgt = self.ewa_res_['alpha_hat']

        basis_decomp = basis * basis_wgt[None, :]
        feature_importance = (basis_decomp ** 2).sum(axis=0)
        feature_importance = feature_importance / feature_importance.sum()
        multi_idx = pd.MultiIndex.from_arrays(self.gridpoints_.T, names=self.variables)

        basis_decomp = pd.DataFrame(basis_decomp, index=multi_idx, columns=feature_importance)
        return basis_decomp.sort_index(axis=1, ascending=False)

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Predicts the continuous function evaluation at X alongside confidence intervals."""
        check_is_fitted(self, 'ewa_res_')
        X = check_array(X, ensure_2d=False)

        B_new = self._build_tensor_basis(X)
        y_hat = B_new @ self.beta_full_

        BL = B_new @ self.cov_beta_sqrt
        se = np.sqrt(np.sum(BL ** 2, axis=1))
        interval = norm.ppf(1 - self.alpha / 2) * se

        idx = pd.MultiIndex.from_arrays(X.T, names=self.variables)
        return pd.DataFrame(
            {
                'fhat': y_hat,
                'se': se,
                'ci_lower': y_hat - interval,
                'ci_upper': y_hat + interval,
            },
            index=idx,
        )

    def predict_derivative(self, X, deriv_orders: Tuple[int, ...]):
        """
        Calculates the partial derivatives of the fitted EWA B-spline surface.
        """
        assert len(deriv_orders) == self.dim, f"deriv_orders must have length {self.dim}"

        B_list = [self.spline_objs[i](X[:, i], nu=deriv_orders[i]) for i in range(self.dim)]
        B_tensor = B_list[0]
        for i in range(1, self.dim):
            B_tensor = (B_tensor[:, :, None] * B_list[i][:, None, :]).reshape(X.shape[0], -1)

        return B_tensor @ self.beta_full_

    def __repr__(self):
        status = "Fitted" if hasattr(self, 'ewa_res_') else "Unfitted"
        fit_info = {
            k: getattr(self, k, None)
            for k in ['n_train', 'domain', 'deg', 'deriv_coef', 'n_knots',
                      'sigma_est_', 'gamma_mult', 'alpha']
        }
        return f"EWABSpline({status}, {fit_info})"
