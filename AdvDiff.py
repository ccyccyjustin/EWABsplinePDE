import numpy as np
import sys
from pathlib import Path
from run_example import run_pde_experiment

PDE_NAME = Path(__file__).stem
X_MAX_SIZE = 1_000_000
DIM = 2
OVERSAMPLE = 100

VARIABLES = ['x', 'y']


def setup_config_func(params):
    DOMAIN = [(0, 1), (0, 1)]

    alpha = params['p_alpha']
    v = params['p_v']
    true_params = {'alpha': alpha, 'v': v}

    PDE_KWARGS = {}
    # True Strong form: alpha * u_xx + u_yy - v * u_x = 0
    # We initialize all parameters to a blind guess of 1.0
    PDE_KWARGS['deriv_coef'] = {'y,y': 1.0, 'x,x': 1.0, 'x': 1.0}

    # We lock 'y,y' as our reference scale, and train alpha ('x,x') and velocity ('x')
    PDE_KWARGS['trainable'] = ['x,x', 'x']
    ACTION_KWARGS = {}

    peclet = abs(v / alpha)
    required_knots_per_dim = int(min(60, max(15, peclet / 2)))
    num_int_knots = required_knots_per_dim ** 2

    return DOMAIN, PDE_KWARGS, ACTION_KWARGS, true_params, num_int_knots


PDE_PARAMS = {
    'p_alpha': {'flag': 'diffusion', 'type': float, 'default': 0.1},
    'p_v': {'flag': 'vel', 'type': float, 'default': 1.5},
    'p_a1': {'flag': 'amp_1', 'type': float, 'default': 1.0},
    'p_a2': {'flag': 'amp_2', 'type': float, 'default': 0.5},  # Added second mode amplitude
}


def true_sol_with_derivatives(x, p_alpha, p_v, p_a1, p_a2):
    """
    Calculates the exact analytical partial derivatives for a 2-mode
    Advection-Diffusion solution to guarantee linear independence.
    """
    X1 = x[:, 0]
    X2 = x[:, 1]

    # Calculate the spatial decay constants for the two different frequencies
    lam1 = (p_v - np.sqrt(p_v ** 2 + 4 * p_alpha * (np.pi ** 2))) / (2 * p_alpha)
    lam2 = (p_v - np.sqrt(p_v ** 2 + 16 * p_alpha * (np.pi ** 2))) / (2 * p_alpha)

    # Base evaluations
    exp1 = p_a1 * np.exp(lam1 * X1)
    exp2 = p_a2 * np.exp(lam2 * X1)

    sin1 = np.sin(np.pi * X2)
    sin2 = np.sin(2 * np.pi * X2)
    cos1 = np.cos(np.pi * X2)
    cos2 = np.cos(2 * np.pi * X2)

    derivs = {}
    derivs[(0, 0)] = exp1 * sin1 + exp2 * sin2
    derivs[(1, 0)] = lam1 * exp1 * sin1 + lam2 * exp2 * sin2
    derivs[(0, 1)] = np.pi * exp1 * cos1 + 2 * np.pi * exp2 * cos2
    derivs[(2, 0)] = (lam1 ** 2) * exp1 * sin1 + (lam2 ** 2) * exp2 * sin2
    derivs[(0, 2)] = -(np.pi ** 2) * exp1 * sin1 - (4 * np.pi ** 2) * exp2 * sin2
    derivs[(1, 1)] = lam1 * np.pi * exp1 * cos1 + lam2 * 2 * np.pi * exp2 * cos2

    return derivs


def get_dirichlet_bc_dict(p_alpha, p_v, p_a1, p_a2, domain):
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]

    def force_boundary_eval(x_input, dim, fixed_val):
        x_exact = x_input.copy()
        x_exact[:, dim] = fixed_val
        return true_sol_with_derivatives(x_exact, p_alpha, p_v, p_a1, p_a2)[(0, 0)]

    bc_dict = {
        ('0_min', 0): lambda x: force_boundary_eval(x, dim=0, fixed_val=x_min),
        ('0_max', 0): lambda x: force_boundary_eval(x, dim=0, fixed_val=x_max),
        ('1_min', 0): lambda x: force_boundary_eval(x, dim=1, fixed_val=y_min),
        ('1_max', 0): lambda x: force_boundary_eval(x, dim=1, fixed_val=y_max)
    }
    return bc_dict


if __name__ == '__main__':
    # Force weak_form flag to '0' since this is a strong-form setup
    import sys
    if '-w' not in sys.argv and '--weak_form' not in sys.argv:
        sys.argv.extend(['-w', '0'])

    run_pde_experiment(
        pde_name=PDE_NAME,
        dim=DIM,
        variables=VARIABLES,
        pde_params=PDE_PARAMS,
        true_sol_with_derivatives=true_sol_with_derivatives,
        boundary_conditions=get_dirichlet_bc_dict,
        setup_config_func=setup_config_func,
        x_max_size=X_MAX_SIZE,
        oversample=OVERSAMPLE
    )