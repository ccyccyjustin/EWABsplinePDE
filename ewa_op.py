import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ewa import EWABSplinePDE
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, List

@dataclass
class SubOperator:
    deriv_u: Tuple[int, ...]
    deriv_v: Tuple[int, ...]
    weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    coeff: float = 1.0

@dataclass
class Operator:
    name: str
    sub_operators: List[SubOperator]
    initial_val: float = 1.0
    trainable: bool = True

class FunctionalEWABSplinePDE(EWABSplinePDE):
    """
    Subclass of EWABSplinePDE that supports functional/spatially varying
    coefficients via pre-composed Operator and SubOperator lists.
    """
    def __init__(self, *args, operators: Optional[List[Operator]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.operators = operators

    def _parse_physics_configuration(self):
        if self.operators is not None:
            self.mode = 'PDE'
            self._coef_keys = list(range(len(self.operators)))
            self.coef_keys = [op.name for op in self.operators]
            self.deriv_map = dict(zip(self._coef_keys, self.coef_keys))
            self._initial_coef_vals = np.array([op.initial_val for op in self.operators], dtype=float)
            self._trainable_idx = [i for i, op in enumerate(self.operators) if op.trainable]
            self._zero_deriv = tuple([0] * self.dim)
        else:
            super()._parse_physics_configuration()

    def assemble_functional_pde_operators(self, spline_objs, operators: List[Operator], source_fn: Callable, deg: int,
                                          dim: int):
        """
        Evaluates sub-operators at ND Gauss-Legendre quadrature nodes across knot spans
        and constructs the precomputed dictionaries _omega_dict and F_g_dict.
        """
        n_gl = max(4, deg + 2)
        xi, wi = np.polynomial.legendre.leggauss(n_gl)
        pts_1d, wts_1d = [], []
        for p in range(dim):
            spl = spline_objs[p]
            int_knots = spl.t[spl.k:-spl.k]
            x_L, x_R = int_knots[:-1], int_knots[1:]
            valid = x_R > x_L
            x_L, x_R = x_L[valid], x_R[valid]
            mid = 0.5 * (x_L + x_R)
            hw = 0.5 * (x_R - x_L)
            pts_1d.append((mid[:, None] + hw[:, None] * xi[None, :]).ravel())
            wts_1d.append((hw[:, None] * wi[None, :]).ravel())

        grids = np.meshgrid(*pts_1d, indexing='ij')
        quad_pts = np.column_stack([g.ravel() for g in grids])
        quad_w = wts_1d[0]
        for p in range(1, dim):
            quad_w = np.kron(quad_w, wts_1d[p])

        # 1. Identify and cache unique derivative orders
        needed_derivs = {(0,) * dim}
        for op in operators:
            for sub in op.sub_operators:
                needed_derivs.add(sub.deriv_u)
                needed_derivs.add(sub.deriv_v)

        basis_cache = {}
        for deriv_order in needed_derivs:
            B_1d_list = [sp.csr_matrix(spline_objs[p](pts_1d[p], nu=deriv_order[p])) for p in range(dim)]
            B_nd = B_1d_list[0]
            for p in range(1, dim):
                B_nd = sp.kron(B_nd, B_1d_list[p], format='csr')
            basis_cache[deriv_order] = B_nd

        n_splines = np.array([s.c.shape[0] for s in spline_objs])
        P_total = np.prod(n_splines)

        # 2. Assemble elementary operator matrices K_k
        K_list = []
        for op in operators:
            K_op = sp.csr_matrix((P_total, P_total), dtype=float)
            for sub in op.sub_operators:
                B_u = basis_cache[sub.deriv_u]
                B_v = basis_cache[sub.deriv_v]
                w_total = quad_w if sub.weight_fn is None else quad_w * np.asarray(sub.weight_fn(quad_pts),
                                                                                   dtype=float).ravel()
                W_diag = sp.diags(w_total * sub.coeff, format='csr')
                K_sub = B_u.T @ (W_diag @ B_v)
                K_op = K_op + K_sub
            K_list.append(K_op.tocsr())

        # 3. Interior splines and unweighted Mass matrix M_0
        is_interior = np.zeros(n_splines, dtype=bool)
        is_interior[tuple(slice(1, -1) for _ in range(dim))] = True
        idx_I = np.flatnonzero(is_interior)

        B_0 = basis_cache[(0,) * dim]
        M_full = B_0.T @ (sp.diags(quad_w, format='csr') @ B_0)
        M_0 = M_full[idx_I, :][:, idx_I].tocsc()
        M_0_solver = spla.factorized(M_0)

        # 4. Source forcing vector f
        s_vals = np.asarray(source_fn(quad_pts), dtype=float).ravel()
        f_full = (B_0.T @ (quad_w * s_vals)).ravel()
        invM_f = M_0_solver(f_full[idx_I])

        # 5. Build _omega_dict and F_g_dict
        K_I_list = [K[idx_I, :].tocsc() for K in K_list]
        F_g_dict = {u: np.asarray(K_I_list[u].T @ invM_f).ravel() for u in range(len(operators))}

        omega_dict = {}
        for v in range(len(operators)):
            invM_Kv = M_0_solver(K_I_list[v].toarray())
            for u in range(len(operators)):
                omega_dict[(u, v)] = sp.csr_matrix(K_I_list[u].T @ invM_Kv)

        return omega_dict, F_g_dict, idx_I, M_full

    def precompute_forcing_components(self):
        if self.operators is not None:
            # High-performance quadrature assembly of sub-operators
            omega_dict, F_g_dict, idx_I, M_full = self.assemble_functional_pde_operators(
                self.spline_objs, self.operators, self.force, self.deg, self.dim
            )
            self.idx_I = idx_I
            self.gram = M_full
            self._omega_dict = omega_dict
            self.F_g_dict = F_g_dict
            self.force_norm = 0.0
        else:
            super().precompute_forcing_components()