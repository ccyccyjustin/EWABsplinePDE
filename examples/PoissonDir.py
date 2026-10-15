import numpy as np
from pathlib import Path
import inspect
from functools import partial
from run_example import run_pde_experiment

PDE_NAME = Path(__file__).stem
X_MAX_SIZE = 1_000_000
DIM = 2
OVERSAMPLE = 100
VARIABLES = ['x', 'y']
boundary_conditions = 'dirichlet'

# --- Experiment Parameters ---
PDE_PARAMS = {
    'p_a': {'flag': 'anisotropy', 'type': float, 'default': 3},
    'p_K': {'flag': 'amplitude', 'type': float, 'default': 100.},
    'p_m': {'flag': 'freq_x', 'type': float, 'default': 5.},
    'p_n': {'flag': 'freq_y', 'type': float, 'default': 2.},
    'p_L': {'flag': 'bound_x', 'type': float, 'default': 2 * np.pi},
    'p_H': {'flag': 'bound_y', 'type': float, 'default': np.pi}
}

def true_sol_with_derivatives(x, p_a, p_m, p_n, p_K, p_L, p_H):
    """
    Calculates the exact analytical partial derivatives up to order 2
    for the Generalized Poisson solution:
    u_xx + a^2 u_yy = K cos(m pi x / L) cos(n pi y / H)

    Returns a dictionary mapping derivative order tuples (u, v) to evaluated arrays.
    """
    assert np.ndim(x) == 2 and x.shape[1] == 2

    X1 = x[:, 0]
    X2 = x[:, 1]

    # Scaled frequencies
    m2 = p_m * np.pi / p_L
    n2 = p_n * np.pi / p_H

    # Precompute base trigonometric evaluations
    cos_m2x = np.cos(m2 * X1)
    sin_m2x = np.sin(m2 * X1)

    cos_n2y = np.cos(n2 * X2)
    sin_n2y = np.sin(n2 * X2)

    # Global amplitude constant
    c = - (p_K / (m2 ** 2 + (p_a * n2) ** 2))

    derivs = {}
    derivs[(0, 0)] = c * cos_m2x * cos_n2y
    derivs[(1, 0)] = -c * m2 * sin_m2x * cos_n2y
    derivs[(0, 1)] = -c * n2 * cos_m2x * sin_n2y
    derivs[(2, 0)] = -c * (m2 ** 2) * cos_m2x * cos_n2y
    derivs[(0, 2)] = -c * (n2 ** 2) * cos_m2x * cos_n2y
    derivs[(1, 1)] = c * m2 * n2 * sin_m2x * sin_n2y
    return derivs

def force(x, p_m, p_n, p_K, p_L, p_H):
    """External Forcing Function g(x)"""
    assert np.ndim(x) == 2 and x.shape[1] == 2
    m2, n2 = p_m * np.pi / p_L, p_n * np.pi / p_H
    return p_K * np.cos(m2 * x[:, 0]) * np.cos(n2 * x[:, 1])


def setup_config(params):
    """
    Dynamically maps the parsed terminal arguments into the PDE configuration.
    """
    domain = [(0, params['p_L']), (0, params['p_H'])]

    # Isolate parameters mapped to the force function and partially apply them
    p_force = inspect.signature(force).parameters.keys()
    force_known = partial(force, **{k: v for k, v in params.items() if k in p_force})

    # --- Exact Original Parameter Settings Retained ---
    PDE_KWARGS = {
        'deriv_coef': {'x,x': 1, 'y,y': 3},
        'trainable': ['y,y'],
        'force': force_known
    }

    true_params = {'p_a': params['p_a']}
    num_int_knots = 500
    return domain, PDE_KWARGS, true_params, num_int_knots

if __name__ == '__main__':
    # Dispatches the physics definitions to the general testing suite
    run_pde_experiment(
        pde_name=PDE_NAME,
        dim=DIM,
        variables=VARIABLES,
        pde_params=PDE_PARAMS,
        true_sol_with_derivatives=true_sol_with_derivatives,
        boundary_conditions=boundary_conditions,
        setup_config_func=setup_config,
        x_max_size=X_MAX_SIZE,
        oversample=OVERSAMPLE,
    )