import numpy as np
from pathlib import Path
from run_example import run_pde_experiment

PDE_NAME = Path(__file__).stem
X_MAX_SIZE = 1_000_000
DIM = 2
OVERSAMPLE = 100

VARIABLES = ['x', 'y']
boundary_conditions= 'dirichlet'

def setup_config_func(params):
    DOMAIN = [(0, 1), (0, 1)]

    alpha = params['p_alpha']
    beta = (params['p_m'] ** 2 + alpha * params['p_n'] ** 2) * (np.pi ** 2)
    true_params = {'': beta, 'x,x': 1.0, 'y,y': alpha} #{'beta': beta, 'alpha': alpha}

    PDE_KWARGS = {}
    # Strong form: u_xx + alpha * u_yy + beta * u = 0
    PDE_KWARGS['deriv_coef'] = {'': 1.0, 'x,x': 1.0, 'y,y': 1.0}
    PDE_KWARGS['trainable'] = ['', 'y,y']

    return DOMAIN, PDE_KWARGS, true_params


PDE_PARAMS = {
    'p_m': {'flag': 'wave_m', 'type': float, 'default': 3.2},
    'p_n': {'flag': 'wave_n', 'type': float, 'default': 4.7},
    'p_alpha': {'flag': 'anisotropy', 'type': float, 'default': 0.6}, # Anisotropy ratio
    'p_a1': {'flag': 'amp_1', 'type': float, 'default': 1.1},
    'p_a2': {'flag': 'amp_2', 'type': float, 'default': 0.4},
}


def true_sol_with_derivatives(x, p_m, p_n, p_alpha, p_a1, p_a2):
    X1 = x[:, 0]
    X2 = x[:, 1]

    # Mode 1: Defined by the passed parameters (default m=2, n=2)
    m1_pi = p_m * np.pi
    n1_pi = p_n * np.pi
    assert np.abs(p_m ** 2 - p_alpha * p_n ** 2) > 1e-4

    # Mode 2: Hardcoded to guarantee the shared eigenvalue at alpha = 0.6
    m2_pi = (np.sqrt(p_alpha) * p_n) * np.pi
    n2_pi = (p_m / np.sqrt(p_alpha)) * np.pi

    cos_m1x, sin_m1x = np.cos(m1_pi * X1), np.sin(m1_pi * X1)
    cos_n1y, sin_n1y = np.cos(n1_pi * X2), np.sin(n1_pi * X2)

    cos_m2x, sin_m2x = np.cos(m2_pi * X1), np.sin(m2_pi * X1)
    cos_n2y, sin_n2y = np.cos(n2_pi * X2), np.sin(n2_pi * X2)

    derivs = {}

    # Order (0, 0): u
    u1 = p_a1 * (cos_m1x * cos_n1y)
    u2 = p_a2 * (sin_m2x * sin_n2y)
    derivs[(0, 0)] = u1 + u2

    # 1st-order spatial derivatives
    derivs[(1, 0)] = p_a1 * (-m1_pi * sin_m1x * cos_n1y) + p_a2 * (m2_pi * cos_m2x * sin_n2y)
    derivs[(0, 1)] = p_a1 * (-n1_pi * cos_m1x * sin_n1y) + p_a2 * (n2_pi * sin_m2x * cos_n2y)

    # 2nd-order spatial derivatives (Breaking the collinearity)
    derivs[(2, 0)] = -(m1_pi ** 2) * u1 - (m2_pi ** 2) * u2
    derivs[(0, 2)] = -(n1_pi ** 2) * u1 - (n2_pi ** 2) * u2

    # Mixed derivative
    derivs[(1, 1)] = p_a1 * (m1_pi * n1_pi * sin_m1x * sin_n1y) + p_a2 * (m2_pi * n2_pi * cos_m2x * cos_n2y)
    return derivs


if __name__ == '__main__':
    # Ensure it runs in strong form
    import sys

    if '-w' not in sys.argv and '--weak_form' not in sys.argv:
        sys.argv.extend(['-w', '0'])

    run_pde_experiment(
        pde_name=PDE_NAME,
        dim=DIM,
        variables=VARIABLES,
        pde_params=PDE_PARAMS,
        true_sol_with_derivatives=true_sol_with_derivatives,
        boundary_conditions=boundary_conditions,
        setup_config_func=setup_config_func,
        x_max_size=X_MAX_SIZE,
        oversample=OVERSAMPLE
    )