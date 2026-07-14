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

# --- Experiment Parameters ---
PDE_PARAMS = {
    'p_a': {'flag': 'anisotropy', 'type': float, 'default': 2},
    'p_K': {'flag': 'amplitude', 'type': float, 'default': 1.},
    'p_m': {'flag': 'freq_x', 'type': float, 'default': 3.},
    'p_n': {'flag': 'freq_y', 'type': float, 'default': 4.},
    'p_L': {'flag': 'bound_x', 'type': float, 'default': 2 * np.pi},
    'p_H': {'flag': 'bound_y', 'type': float, 'default': np.pi}
}


def true_sol(x, p_a, p_m, p_n, p_K, p_L, p_H):
    """
    Generalized Poisson on [0, L] x [0, H]:
    u_xx + a^2 u_yy = 0
    g(x, y) = K sin(m pi x / L) sin(n pi y / H)
    Dirichlet BC: f=0 on dOmega
    """
    assert np.ndim(x) == 2 and x.shape[1] == 2
    m2 = p_m * np.pi / p_L
    n2 = m2 / p_a
    return np.cos(m2 * x[:, 0]) * np.cosh(n2 * x[:, 1])


def setup_config(params):
    """
    Dynamically maps the parsed terminal arguments into the PDE configuration.
    """
    domain = [(0, params['p_L']), (0, params['p_H'])]

    # Isolate parameters mapped to the force function and partially apply them
    # p_force = inspect.signature(force).parameters.keys()
    force_known = None #partial(force, **{k: v for k, v in params.items() if k in p_force})

    # --- Exact Original Parameter Settings Retained ---
    PDE_KWARGS = {
        'deriv_coef': {'x,x': 1, 'y,y': 3},
        'trainable': ['y,y'],
        'force': force_known
    }

    # TODO: ALWAYS PASS NEGATIVE!
    ACTION_KWARGS = {}
    action_coef = {
        ('x', 'x'): -1,  # u_x * u_x
        ('y', 'y'): -3,  # u_y * u_y
    }
    ACTION_KWARGS['action_coef'] = action_coef
    ACTION_KWARGS['trainable'] = [('y', 'y')] #, ('y', 'y')
    ACTION_KWARGS['force'] = force_known

    true_params = {'p_a': params['p_a']}

    k_sqr = (params['p_m'] ** 2 + params['p_n'] ** 2) * (np.pi ** 2)
    k = np.sqrt(k_sqr)

    required_knots_per_dim = int((k / np.pi) * 3)
    num_int_knots = required_knots_per_dim ** DIM
    return domain, PDE_KWARGS, ACTION_KWARGS, true_params, num_int_knots


if __name__ == '__main__':
    # Dispatches the physics definitions to the general testing suite
    run_pde_experiment(
        pde_name=PDE_NAME,
        dim=DIM,
        variables=VARIABLES,
        pde_params=PDE_PARAMS,
        true_sol_func=true_sol,
        setup_config_func=setup_config,
        x_max_size=X_MAX_SIZE,
        oversample=OVERSAMPLE,
    )