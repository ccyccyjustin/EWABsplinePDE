import numpy as np
from pathlib import Path
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Tuple

import ewa
from ewa import EWABSplinePDE
from run_example import run_pde_experiment

# Import our functional operator extensions
from ewa_op import SubOperator, Operator

PDE_NAME = Path(__file__).stem
X_MAX_SIZE = 1_000_000
DIM = 2
OVERSAMPLE = 100
VARIABLES = ['x', 'y']
boundary_conditions = 'dirichlet'

# --- Benchmark Hyperparameters ---
PDE_PARAMS = {
    'p_rf': {'flag': 'reflector_D', 'type': float, 'default': 0.9},
    'p_cf': {'flag': 'core_fission', 'type': float, 'default': 0.060},
    'p_sm': {'flag': 'source_mult', 'type': float, 'default': 1.0},
    'p_L':  {'flag': 'bound_L', 'type': float, 'default': 100.0}
}

# -----------------------------------------------------------------------------
# 1. Physics: Source and Finite-Difference Ground Truth Solver
# -----------------------------------------------------------------------------
def force(x: np.ndarray, p_sm: float = 1.0, **kwargs) -> np.ndarray:
    """External neutron startup source distribution S(x, y)."""
    assert np.ndim(x) == 2 and x.shape[1] == 2
    X, Y = x[:, 0], x[:, 1]
    S = np.zeros(len(x))
    sources = [
        (25.0, 25.0, 40.0 * p_sm, 80.0),
        (75.0, 25.0, 40.0 * p_sm, 80.0),
        (50.0, 75.0, 60.0 * p_sm, 100.0)
    ]
    for sx, sy, strength, spread in sources:
        dist_sq = (X - sx)**2 + (Y - sy)**2
        S += strength * np.exp(-dist_sq / spread)
    return S


def solve_fd_reactor(reflector_D: float, core_fission: float, source_mult: float, Lx: float = 100.0, n_grid: int = 101):
    """Solves the 2D finite difference benchmark on a high-resolution grid."""
    nx = ny = n_grid
    dx = dy = Lx / (nx - 1)
    N = nx * ny

    x_1d = np.linspace(0.0, Lx, nx)
    y_1d = np.linspace(0.0, Lx, ny)
    X, Y = np.meshgrid(x_1d, y_1d, indexing='ij')

    D = np.full((nx, ny), reflector_D)
    Sigma_rem = np.full((nx, ny), 0.001)

    in_core = (X >= 25.0) & (X <= 75.0) & (Y >= 25.0) & (Y <= 75.0)
    D[in_core] = 1.0
    Sigma_rem[in_core] = 0.07 - core_fission

    S_2D = force(np.column_stack([X.ravel(), Y.ravel()]), p_sm=source_mult).reshape(nx, ny)
    b = S_2D.ravel()

    A = sp.lil_matrix((N, N))
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                A[k, k] = 1.0
                b[k] = 0.0
            else:
                D_left   = (D[i, j] + D[i - 1, j]) / 2.0
                D_right  = (D[i, j] + D[i + 1, j]) / 2.0
                D_bottom = (D[i, j] + D[i, j - 1]) / 2.0
                D_top    = (D[i, j] + D[i, j + 1]) / 2.0

                center = (D_right + D_left) / dx**2 + (D_top + D_bottom) / dy**2 + Sigma_rem[i, j]
                A[k, k] = center
                A[k, k - ny] = -D_left / dx**2
                A[k, k + ny] = -D_right / dx**2
                A[k, k - 1]  = -D_bottom / dy**2
                A[k, k + 1]  = -D_top / dy**2

    phi = spsolve(A.tocsr(), b).reshape((nx, ny))
    return x_1d, y_1d, phi


def true_sol_with_derivatives(x: np.ndarray, p_rf: float, p_cf: float, p_sm: float, p_L: float) -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Interpolates high-resolution finite difference solution and its spatial derivatives.
    Returns dictionary mapping (u, v) -> derivative arrays.
    """
    assert np.ndim(x) == 2 and x.shape[1] == 2
    x_1d, y_1d, phi = solve_fd_reactor(reflector_D=p_rf, core_fission=p_cf, source_mult=p_sm, Lx=p_L, n_grid=121)

    rgi = RegularGridInterpolator((x_1d, y_1d), phi, method='cubic', bounds_error=False, fill_value=0.0)
    u_vals = rgi(x)

    # Compute numerical derivatives for Sobolev error diagnostics
    dx = x_1d[1] - x_1d[0]
    dy = y_1d[1] - y_1d[0]
    dphi_dx, dphi_dy = np.gradient(phi, dx, dy)
    d2phi_dx2, d2phi_dxdy = np.gradient(dphi_dx, dx, dy)
    d2phi_dydx, d2phi_dy2 = np.gradient(dphi_dy, dx, dy)

    intp_kwargs = {'method': 'cubic', 'bounds_error': False, 'fill_value': 0.0}
    rgi_dx = RegularGridInterpolator((x_1d, y_1d), dphi_dx, **intp_kwargs)
    rgi_dy = RegularGridInterpolator((x_1d, y_1d), dphi_dy,  **intp_kwargs)
    rgi_dx2 = RegularGridInterpolator((x_1d, y_1d), d2phi_dx2,  **intp_kwargs)
    rgi_dxdy = RegularGridInterpolator((x_1d, y_1d), d2phi_dxdy,  **intp_kwargs)
    rgi_dydx = RegularGridInterpolator((x_1d, y_1d), d2phi_dydx,  **intp_kwargs)
    rgi_dy2 = RegularGridInterpolator((x_1d, y_1d), d2phi_dy2,  **intp_kwargs)

    return {
        (0, 0): u_vals,
        (1, 0): rgi_dx(x),
        (0, 1): rgi_dy(x),
        (2, 0): rgi_dx2(x),
        (1, 1): (rgi_dxdy(x) + rgi_dydx(x)) / 2,
        (0, 2): rgi_dy2(x),
    }

# -----------------------------------------------------------------------------
# 2. Setup Configuration: Assemble Sub-Operators and Operators
# -----------------------------------------------------------------------------
def setup_config_func(params):
    """
    Constructs the 6 sub-operators, bundles them into the 3 physical operators,
    and returns configuration dictionaries matching the EWA runner contract.
    """
    domain = [(0.0, params['p_L']), (0.0, params['p_L'])]

    # Spatial Material Indicator Functions
    chi_c = lambda x: ((x[:, 0] >= 25.0) & (x[:, 0] <= 75.0) &
                       (x[:, 1] >= 25.0) & (x[:, 1] <= 75.0)).astype(float)
    chi_r = lambda x: 1.0 - chi_c(x)
    w_sigma_a = lambda x: 0.07 * chi_c(x) + 0.001 * chi_r(x)

    # 1. Sub-Operators (Weighted Derivative Integrals)
    sub1 = SubOperator(deriv_u=(1, 0), deriv_v=(1, 0), weight_fn=chi_c, coeff=1.0)
    sub2 = SubOperator(deriv_u=(0, 1), deriv_v=(0, 1), weight_fn=chi_c, coeff=1.0)
    sub3 = SubOperator(deriv_u=(0, 0), deriv_v=(0, 0), weight_fn=w_sigma_a, coeff=1.0)

    sub4 = SubOperator(deriv_u=(1, 0), deriv_v=(1, 0), weight_fn=chi_r, coeff=1.0)
    sub5 = SubOperator(deriv_u=(0, 1), deriv_v=(0, 1), weight_fn=chi_r, coeff=1.0)

    sub6 = SubOperator(deriv_u=(0, 0), deriv_v=(0, 0), weight_fn=chi_c, coeff=-1.0)

    # 2. Group into Operators
    op1 = Operator(name='baseline', sub_operators=[sub1, sub2, sub3], initial_val=1.0, trainable=False)
    op2 = Operator(name='reflector_D', sub_operators=[sub4, sub5], initial_val=1.0, trainable=True)
    op3 = Operator(name='core_fission', sub_operators=[sub6], initial_val=0.1, trainable=True)

    operators = [op1, op2, op3]

    pde_kwargs = {
        'operators': operators,
        'force': lambda x: force(x, p_sm=params['p_sm'])
    }

    true_params = {
        'baseline': 1.0,
        'reflector_D': params['p_rf'],
        'core_fission': params['p_cf']
    }

    num_int_knots = 400
    return domain, pde_kwargs, true_params, num_int_knots


# -----------------------------------------------------------------------------
# 3. Main Runner Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    run_pde_experiment(
        pde_name=PDE_NAME,
        dim=DIM,
        variables=VARIABLES,
        pde_params=PDE_PARAMS,
        true_sol_with_derivatives=true_sol_with_derivatives,
        boundary_conditions=boundary_conditions,
        setup_config_func=setup_config_func,
        x_max_size=X_MAX_SIZE,
        oversample=OVERSAMPLE,
    )