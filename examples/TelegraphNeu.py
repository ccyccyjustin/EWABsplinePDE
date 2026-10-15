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
boundary_conditions = 'neumann'

# --- Experiment Parameters ---
PDE_PARAMS = {
    'p_c': {'flag': 'wave_speed', 'type': float, 'default': 2.0},
    'p_gamma': {'flag': 'damping', 'type': float, 'default': 0.5},
    'p_K': {'flag': 'amplitude', 'type': float, 'default': 100.},
    'p_m': {'flag': 'freq_x', 'type': float, 'default': 5.},
    'p_n': {'flag': 'freq_y', 'type': float, 'default': 2.},
    'p_L': {'flag': 'bound_x', 'type': float, 'default': 2 * np.pi},
    'p_H': {'flag': 'bound_y', 'type': float, 'default': np.pi}
}


def true_sol_with_derivatives(x, p_c, p_gamma, p_m, p_n, p_K, p_L, p_H):
    """
    Calculates the exact analytical partial derivatives up to order 2
    for the Telegraph Equation solution:
    u_yy + 2*gamma u_y - c^2 u_xx = g(x,y)

    Using an exponentially damped standing wave:
    u(x,y) = K * exp(-gamma * y) * sin(m pi x / L) * cos(n pi y / H)
    """
    assert np.ndim(x) == 2 and x.shape[1] == 2

    X1 = x[:, 0]
    X2 = x[:, 1]

    # Scaled frequencies
    m2 = p_m * np.pi / p_L
    n2 = p_n * np.pi / p_H
    gamma = p_gamma

    # Precompute base evaluations
    E = np.exp(-gamma * X2)

    cos_m2x = np.cos(m2 * X1)
    sin_m2x = np.sin(m2 * X1)

    cos_n2y = np.cos(n2 * X2)
    sin_n2y = np.sin(n2 * X2)

    c = p_K

    derivs = {}

    # 0th order (Exact Solution)
    u = c * E * sin_m2x * cos_n2y
    derivs[(0, 0)] = u

    # 1st order derivatives
    derivs[(1, 0)] = c * m2 * E * cos_m2x * cos_n2y
    derivs[(0, 1)] = -c * E * sin_m2x * (gamma * cos_n2y + n2 * sin_n2y)

    # 2nd order derivatives
    derivs[(2, 0)] = -(m2 ** 2) * u
    derivs[(0, 2)] = c * E * sin_m2x * ((gamma ** 2 - n2 ** 2) * cos_n2y + 2 * gamma * n2 * sin_n2y)
    derivs[(1, 1)] = -c * m2 * E * cos_m2x * (gamma * cos_n2y + n2 * sin_n2y)

    return derivs


def force(x, p_c, p_gamma, p_m, p_n, p_K, p_L, p_H):
    """External Forcing Function g(x,y) for the Telegraph Equation"""
    assert np.ndim(x) == 2 and x.shape[1] == 2
    m2 = p_m * np.pi / p_L
    n2 = p_n * np.pi / p_H

    # Substituting the analytical solution into L(u) causes the sin(n2 * y)
    # terms from u_yy and 2*gamma*u_y to cancel out:
    dispersion_factor = (p_c ** 2) * (m2 ** 2) - (n2 ** 2) - (p_gamma ** 2)
    amplitude = p_K * dispersion_factor

    return amplitude * np.exp(-p_gamma * x[:, 1]) * np.sin(m2 * x[:, 0]) * np.cos(n2 * x[:, 1])


def setup_config(params):
    """
    Dynamically maps the parsed terminal arguments into the PDE configuration.
    """
    domain = [(0, params['p_L']), (0, params['p_H'])]

    # Isolate parameters mapped to the force function and partially apply them
    p_force = inspect.signature(force).parameters.keys()
    force_known = partial(force, **{k: v for k, v in params.items() if k in p_force})

    wave_speed_sq = params['p_c'] ** 2
    gamma_coeff = 2 * params['p_gamma']

    # PDE: -c^2 u_xx + 1 u_yy + 2\gamma u_y = g(x,y)
    PDE_KWARGS = {
        'deriv_coef': {'x,x': -wave_speed_sq, 'y,y': 1, 'y': gamma_coeff},
        'trainable': ['x,x'],
        'force': force_known
    }

    true_params = {'p_c': params['p_c']}
    num_int_knots = 1000
    return domain, PDE_KWARGS, true_params, num_int_knots



if __name__ == '__main__':
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