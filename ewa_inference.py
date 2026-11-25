import logging

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

import util

logger = logging.getLogger(__name__)


@util.time_all_methods
class PDEInferenceMixin:
    """PDE parameter estimation, bias correction, SURE gradients, and sandwich variance."""

    def compute_theta_system_pde(self, omega_dict, beta, theta,
                                 bias_corrected=False, sigma=None):
        """
        Builds the PDE parameter Gram matrix, forcing projection vector, and Jacobian.
        """
        m = len(self._coef_keys)
        n = len(beta)

        G_raw = np.zeros((m, m))
        G_bias = np.zeros((m, m))

        for i, k_u in enumerate(self._coef_keys):
            for j, k_v in enumerate(self._coef_keys):
                omega_uv = self._omega_dict[(k_u, k_v)]
                if hasattr(omega_uv, "toarray"):
                    omega_uv = omega_uv.toarray()
                omega_uv = np.asarray(omega_uv)
                g_val = beta.T @ omega_uv @ beta
                G_raw[i, j] = g_val

                if bias_corrected:
                    bias = np.sum(self.Vw * (omega_uv @ self.Vw))
                    bias = bias * (sigma ** 2 / self.n_train)
                    G_bias[i, j] = bias

        # Consistent generalized-eigenvalue damping guarantees:
        # G_raw_TT - eta * G_bias_TT >= eps_floor * G_raw_TT > 0
        if bias_corrected and len(self._trainable_idx) > 0:
            T = self._trainable_idx
            G_raw_TT = G_raw[np.ix_(T, T)]
            G_bias_TT = G_bias[np.ix_(T, T)]
            eps_floor = 0.05
            try:
                mu_max = float(np.max(eigh(G_bias_TT, G_raw_TT, eigvals_only=True)))
            except np.linalg.LinAlgError:
                mu_max = np.inf
            if mu_max <= 1.0 - eps_floor:
                eta = 1.0
            elif np.isfinite(mu_max) and mu_max > 0.0:
                eta = (1.0 - eps_floor) / mu_max
            else:
                eta = 0.0
            Gram = G_raw - eta * G_bias
        else:
            Gram = G_raw

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
        Computes the Unbiased PDE Loss: L(theta | beta) = theta^T * G_unbiased * theta - 2 * theta^T * v + ||g||^2
        """
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

    def get_coef_dist_corrected(
        self, G, deriv_coef, v_vec, A, B=None,
        relaxation=0.7, max_rel_step=0.5, lm_damping=1e-3,
    ):
        """
        Stabilized damped Newton update for the just-identified estimating equation:
            Psi(theta_T) = G_TT @ theta_T + G_TTc @ theta_Tc - v_T = 0
        using the coupled Jacobian A = J_TT + M_ind.

        Stability safeguards:
          1. Avoids inverting B (not needed for just-identified root-finding).
          2. Continuous Levenberg-Marquardt regularization on A if Re(eig(A)) <= tol.
          3. Relative step-size trust-region clipping to prevent Gibbs-weight overshoots.
          4. Damped relaxation (relaxation < 1) for alternating BCD stability.
        """
        T = self._trainable_idx
        Tc = [i for i, _ in enumerate(self._coef_keys) if i not in T]
        theta_new = deriv_coef.copy()
        theta_T_old = deriv_coef[T]
        dim_T = len(T)

        G_TT = G[np.ix_(T, T)]
        self.G_TT = G_TT

        # 1. Evaluate estimating equation residual Psi(theta^(t))
        Psi = G_TT @ theta_T_old - v_vec[T]
        if len(Tc) > 0:
            Psi = Psi + G[np.ix_(T, Tc)] @ deriv_coef[Tc]

        # 2. Continuous regularization of A (prevents binary fallback chattering)
        A_solve = A.copy()
        eig_real_min = np.linalg.eigvals(A_solve).real.min()
        if eig_real_min < self.tol:
            shift = (self.tol - eig_real_min) + lm_damping * np.linalg.norm(G_TT, ord=2)
            A_solve = A_solve + shift * np.eye(dim_T)

        # 3. Solve A_solve @ delta_theta = -Psi
        try:
            delta_theta = -np.linalg.solve(A_solve, Psi)
        except np.linalg.LinAlgError:
            delta_theta = -np.linalg.lstsq(A_solve, Psi, rcond=self.tol)[0]

        # 4. Trust-region step clipping (prevents jumps across SURE mode switches)
        theta_scale = max(np.linalg.norm(theta_T_old), 1.0)
        step_norm = np.linalg.norm(delta_theta)
        max_step = max_rel_step * theta_scale
        if step_norm > max_step:
            delta_theta = delta_theta * (max_step / step_norm)

        # 5. Damped update
        theta_new[T] = theta_T_old + relaxation * delta_theta
        return theta_new

    def compute_cov_beta(self, omega_matrix, weights, H_lam, H_r, V, sig2, n, gamma):
        E_D = np.diag(omega_matrix @ weights)  # (p, p)

        A_grid = H_r  # (p, n_lambda): d - alpha(lam)
        B_grid = H_lam * H_r  # (p, n_lambda): (1 - omega) * (d - alpha(lam))
        B_centered = B_grid - (B_grid @ weights)[:, np.newaxis]
        Cov_w = A_grid @ (B_centered * weights[np.newaxis, :]).T

        Delta = (2.0 * n / gamma) * Cov_w  # (p, p)
        Xi = E_D + Delta

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
        if sp.issparse(TV):
            TV = TV.toarray()
        d_vec = self.d_vec_ if has_bc else np.zeros(p)

        H_matrices = []
        force_proj_grad = np.zeros((p_red, K))
        for i, t_i in enumerate(self._trainable_idx):
            k_i = self._coef_keys[t_i]
            H_k = np.zeros((p, p))
            for j, k_j in enumerate(self._coef_keys):
                H_k += theta[j] * (omega_dict[(k_i, k_j)] + omega_dict[(k_j, k_i)])

            H_proj = TV.T @ H_k @ TV
            f_proj = TV.T @ (F_g_dict[t_i] - np.array(H_k @ d_vec).flatten())

            H_matrices.append(H_proj)
            force_proj_grad[:, i] = f_proj

        J_OLS = np.zeros((p_red, K))
        for k in range(K):
            J_OLS[:, k] = H_matrices[k] @ d - force_proj_grad[:, k]
        return J_OLS, H_matrices, force_proj_grad

    def compute_grad_beta_lambda(
        self,
        H_s: np.ndarray,
        H_r: np.ndarray,
        J_OLS: np.ndarray,
        H_matrices: list[np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        """Computes the gradient of candidate splines with respect to theta_k on lambda grid."""
        p, n_lam = H_r.shape
        K = J_OLS.shape[1]
        grad_alpha = np.zeros((p, n_lam, K))

        for k in range(K):
            term1 = -H_s * J_OLS[:, k, np.newaxis]
            term2 = H_s * np.array(H_matrices[k] @ H_r)
            grad_alpha[:, :, k] = term1 + term2
        return grad_alpha

    def compute_grad_sure(
        self,
        grad_alpha: np.ndarray,
        H_r: np.ndarray,
        omega_matrix: np.ndarray,
        H_s: np.ndarray,
        H_matrices: list[np.ndarray] | np.ndarray,
        n: int,
        sig2: float,
    ) -> np.ndarray:
        """Computes the SURE risk gradient matrix G_SURE across the candidate grid."""
        K = grad_alpha.shape[2]
        d_RSS = -2.0 * n * np.sum(H_r[:, :, np.newaxis] * grad_alpha, axis=0)

        W_df = omega_matrix * H_s
        D_H = np.column_stack([np.diag(H_matrices[k]) for k in range(K)])
        d_df = -2.0 * sig2 * (W_df.T @ D_H)

        G_SURE = d_RSS + d_df
        return G_SURE

    def compute_grad_beta_ewa(
        self,
        grad_alpha: np.ndarray,
        G_SURE: np.ndarray,
        weights: np.ndarray,
        H_r: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        g_bar = weights @ G_SURE
        J_w = -(1.0 / gamma) * weights[:, np.newaxis] * (G_SURE - g_bar[np.newaxis, :])
        J_direct = np.sum(grad_alpha * weights[np.newaxis, :, np.newaxis], axis=1)
        J_adapt = -H_r @ J_w
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
        n = len(y_fit)
        T = self._trainable_idx
        K = len(T)

        sig2 = fit_res['sigma_est'] ** 2
        gamma = fit_res['gamma']
        weights = fit_res["weights"]
        shrinkage_matrix = fit_res['shrinkage_matrix']
        alpha_ewa = fit_res['alpha_hat']
        p = len(alpha_ewa)

        d = U.T @ y_fit / n
        f_p = np.zeros(p) if force_proj is None else force_proj

        # Exact division-free shrinkage operators valid for all s >= 0 (including s == 0)
        omega_matrix = fit_res["omega_matrix"]
        H_lam = 1.0 - omega_matrix
        H_s = lam_vec[np.newaxis, :] * shrinkage_matrix
        H_r = H_s * ((s * d - f_p)[:, np.newaxis])

        Xi, cov_beta_sqrt = self.compute_cov_beta(omega_matrix, weights, H_lam, H_r, V, sig2, n, gamma)
        J_OLS, H_matrices, force_proj_grad = self.compute_pde_projected_operators(
            theta, V, d, self._omega_dict, self.F_g_dict
        )

        grad_alpha = self.compute_grad_beta_lambda(H_s, H_r, J_OLS, H_matrices)
        G_SURE = self.compute_grad_sure(grad_alpha, H_r, omega_matrix, H_s, H_matrices, n, sig2)
        J_alpha_EWA = self.compute_grad_beta_ewa(grad_alpha, G_SURE, weights, H_r, gamma)

        has_bc = len(self.boundary_conditions) > 0
        TV = (self.T_matrix_ @ V) if has_bc else V
        d_vec = self.d_vec_ if has_bc else np.zeros(self.total_n_splines)
        beta = TV @ alpha_ewa + d_vec

        gram_new = self.compute_theta_system_pde(
            self._omega_dict,
            beta,
            theta=theta,
            sigma=self.sigma_est_,
            bias_corrected=self.bias_corrected,
        )[0]
        J_TT = gram_new[np.ix_(T, T)]

        K_tilde = np.zeros((K, p))
        for k in range(K):
            K_tilde[k, :] = H_matrices[k] @ alpha_ewa - force_proj_grad[:, k]

        A = J_TT + K_tilde @ J_alpha_EWA
        K_Xi = K_tilde @ Xi
        B = (sig2 / n) * (K_Xi @ K_Xi.T)  # TODO: Unbiased meat matrix
        return A, B, cov_beta_sqrt, K_tilde, J_alpha_EWA
