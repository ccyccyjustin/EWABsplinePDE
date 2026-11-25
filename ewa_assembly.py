from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

import util
from Bsplines import compute_spline_integral_gl


@util.time_all_methods
class PDEAssemblyMixin:
    """Strong/weak PDE operator integrals, penalty matrices, and forcing assembly."""

    def compute_K_uv(self, d_u, d_v):
        K_uv = None
        for p in range(self.dim):
            d_up = d_u[p]
            d_vp = d_v[p]
            # Check the Local 1D Micro-Cache
            cache_key_1d = (p, d_up, d_vp)
            if cache_key_1d not in self._integral_1d_cache:
                self._integral_1d_cache[cache_key_1d] = compute_spline_integral_gl(
                    self.spline_objs[p], u=d_up, v=d_vp, sparse=self.sparse
                )

            K_1d = self._integral_1d_cache[cache_key_1d]

            # Assemble the Kronecker Product
            if K_uv is None:
                K_uv = K_1d
            else:
                if self.sparse:
                    K_uv = sp.kron(K_uv, K_1d, format='csr')
                else:
                    K_uv = np.kron(K_uv, K_1d)
        return K_uv

    def _weak_deriv_splits(self, deriv_order: Tuple[int, ...]) -> List[
        Tuple[Tuple[int, ...], Tuple[int, ...], float]
    ]:
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
        First compute u-th and v-th order partial derivatives,
        then compute the integral of their products.
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
        Computes weak-form penalty matrices projected onto the interior test space.
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
            np.abs(penalty_matrix - penalty_matrix.T).max(), coef_vals
        )
        penalty_matrix = (penalty_matrix + penalty_matrix.T) / 2.0

        # Hard Constrained Reduced Space Projection
        if len(self.boundary_conditions) > 0:
            T_mat = self.T_matrix_
            if not sp.issparse(penalty_matrix) and sp.issparse(T_mat):
                T_mat = T_mat.toarray()
            A_reduced = T_mat.T @ penalty_matrix @ T_mat
            v_reduced = T_mat.T @ (forcing_vector - np.array(penalty_matrix @ self.d_vec_).ravel())
            return A_reduced, v_reduced

        return penalty_matrix, forcing_vector
