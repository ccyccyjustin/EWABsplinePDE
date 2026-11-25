import logging
from itertools import product
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import util
from Bsplines import (
    build_custom_sparse_tensor_basis,
    build_custom_tensor_basis,
    tensor_prod,
)

logger = logging.getLogger(__name__)


@util.time_all_methods
class BoundaryConditionMixin:
    """Function projection and affine boundary condition elimination for EWABSplinePDE."""

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

    def project_bc(self, bc_func, d, x_star):
        dims_sub = [j for j in range(self.dim) if j != d]
        # --- 1. Project boundary conditions on the N-1 dimension sub-grid ---
        if self.dim > 1:
            grids_sub = [self.grids_[j] if j != d else np.array([x_star]) for j in range(self.dim)]
            eval_points = tensor_prod(*grids_sub)
            if self.sparse:
                B_sub = build_custom_sparse_tensor_basis(
                    eval_points, self.n_splines, self.all_knots, self.deg, self.dim, dims_sub
                )
            else:
                B_sub = build_custom_tensor_basis(eval_points, self.spline_objs, self.dim, dims_sub)
            c_sub = self._project_function(
                bc_func,
                B_sub,
                eval_points,
                func_name=f'Boundary Condition at dim {self.variables[d]}={x_star}',
            )
        else:
            eval_points = np.array([[x_star]])
            c_sub = bc_func(eval_points).ravel()
        return c_sub

    def locate_boundary_idx(self):
        """
        Locates the flat indices of boundary splines (idx_B) and interior splines (idx_I)
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
        T_matrix : np.ndarray or scipy.sparse.csr_matrix
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
            assert cond_num < 1e12, (
                f"A_B is too ill-conditioned for stable inversion! Condition number: {cond_num:.2e}"
            )

            invA_c = np.linalg.solve(A_B, c_full)
            invA_AI = np.linalg.solve(A_B, A_I)

            self.T_matrix_ = np.zeros((M_total, len(self.idx_I_BC)))
            self.T_matrix_[self.idx_I_BC, :] = np.eye(len(self.idx_I_BC))
            self.T_matrix_[self.idx_B_BC, :] = -invA_AI

        self.d_vec_ = np.zeros(M_total)
        self.d_vec_[self.idx_B_BC] = invA_c
        return self.T_matrix_, self.d_vec_
