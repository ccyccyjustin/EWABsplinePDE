import numpy as np
from pathlib import Path
from run_example import run_pde_experiment

PDE_NAME = Path(__file__).stem
X_MAX_SIZE = 1_000_000
DIM = 2
OVERSAMPLE = 100


VARIABLES = ['x', 'y']

def setup_config_func(params):
    DOMAIN = [(0, 1), (0, 1)]

    PDE_KWARGS = {}
    PDE_KWARGS['deriv_coef'] = {'x,x': 1., 'y,y': 1., '': 1.}
    PDE_KWARGS['trainable'] = ['']

    ACTION_KWARGS = {}
    action_coef = {
        ('x', 'x'): -1.,  # u_x * u_x
        ('y', 'y'): -1.,  # u_y * u_y
        ('', ''): 1.  # -(k ** 2)  # u * u
    }
    ACTION_KWARGS['action_coef'] = action_coef
    ACTION_KWARGS['trainable'] = [('', '')]

    k_sqr = (params['p_m'] ** 2 + params['p_n'] ** 2) * (np.pi ** 2)
    k = np.sqrt(k_sqr)
    true_params = {'k_sqr': k_sqr}

    required_knots_per_dim = int((k / np.pi) * 3)
    num_int_knots = required_knots_per_dim ** 2
    return DOMAIN, PDE_KWARGS, ACTION_KWARGS, true_params, num_int_knots

#TODO: Weak only works when true grad=0 on boundaries
PDE_PARAMS = {'p_m':
                  {'flag': 'wave_m',
                   'type': float,
                   'default': 5.5}, #5.5

              'p_n':
                  {'flag': 'wave_n',
                   'type': float,
                   'default': 12.5}, # 11.2

              'p_a1':
                  {'flag': 'amp_1',
                   'type': float,
                   'default': 1},

              'p_a2':
                  {'flag': 'amp_2',
                   'type': float,
                   'default': -1},
              }

def true_sol(x, p_m, p_n, p_a1, p_a2):
    # Helmholtz:
    # u_xx + u_yy + (m^2 + n^2) pi^2 u = 0
    # Zero BC: u = 0
    u_1 = np.cos(p_m * np.pi * x[:, 0]) * np.cos(p_n * np.pi * x[:, 1])
    u_2 = np.cos(p_n * np.pi * x[:, 0]) * np.cos(p_m * np.pi * x[:, 1])
    return p_a1 * u_1 + p_a2 * u_2

def true_sol_with_derivatives(x, p_m, p_n, p_a1, p_a2):
    """
    Calculates the exact analytical partial derivatives up to order 2
    for the specified Helmholtz solution.

    Returns a dictionary mapping derivative order tuples (u, v) to evaluated arrays.
    """
    X1 = x[:, 0]
    X2 = x[:, 1]

    m_pi = p_m * np.pi
    n_pi = p_n * np.pi

    # 1. Precompute base trigonometric evaluations to avoid redundant grid calls
    cos_mx = np.cos(m_pi * X1)
    sin_mx = np.sin(m_pi * X1)

    cos_nx = np.cos(n_pi * X1)
    sin_nx = np.sin(n_pi * X1)

    cos_my = np.cos(m_pi * X2)
    sin_my = np.sin(m_pi * X2)

    cos_ny = np.cos(n_pi * X2)
    sin_ny = np.sin(n_pi * X2)

    derivs = {}

    # Order (0, 0): u
    derivs[(0, 0)] = p_a1 * (cos_mx * cos_ny) + p_a2 * (cos_nx * cos_my)
    derivs[(1, 0)] = -p_a1 * m_pi * (sin_mx * cos_ny) - p_a2 * n_pi * (sin_nx * cos_my)
    derivs[(0, 1)] = -p_a1 * n_pi * (cos_mx * sin_ny) - p_a2 * m_pi * (cos_nx * sin_my)
    derivs[(2, 0)] = -p_a1 * (m_pi ** 2) * (cos_mx * cos_ny) - p_a2 * (n_pi ** 2) * (cos_nx * cos_my)
    derivs[(0, 2)] = -p_a1 * (n_pi ** 2) * (cos_mx * cos_ny) - p_a2 * (m_pi ** 2) * (cos_nx * cos_my)
    derivs[(1, 1)] = p_a1 * m_pi * n_pi * (sin_mx * sin_ny) + p_a2 * m_pi * n_pi * (sin_nx * sin_my)
    return derivs


def get_neumann_bc_dict(p_m, p_n, p_a1, p_a2, domain):
    """
    Constructs the 1st-order derivative (Neumann) boundary conditions.
    Explicitly forces the spatial coordinates to align with domain bounds.
    """
    # Extract the exact boundary constants from the domain definition
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]

    def force_boundary_eval(x_input, dim, fixed_val, deriv_order):
        # Create a copy to prevent mutating the original evaluation array
        x_exact = x_input.copy()
        x_exact[:, dim] = fixed_val
        return true_sol_with_derivatives(x_exact, p_m, p_n, p_a1, p_a2)[deriv_order]

    bc_dict = {
        # --- Dimension 0 (X-axis) Boundaries ---
        ('0_min', 1): lambda x: force_boundary_eval(x, dim=0, fixed_val=x_min, deriv_order=(1, 0)),
        ('0_max', 1): lambda x: force_boundary_eval(x, dim=0, fixed_val=x_max, deriv_order=(1, 0)),

        # --- Dimension 1 (Y-axis) Boundaries ---
        ('1_min', 1): lambda x: force_boundary_eval(x, dim=1, fixed_val=y_min, deriv_order=(0, 1)),
        ('1_max', 1): lambda x: force_boundary_eval(x, dim=1, fixed_val=y_max, deriv_order=(0, 1))
    }

    return bc_dict

if __name__ == '__main__':
    # Dispatches the physics definitions to the general testing suite
    run_pde_experiment(
        pde_name=PDE_NAME,
        dim=DIM,
        variables=VARIABLES,
        pde_params=PDE_PARAMS,
        true_sol_with_derivatives=true_sol_with_derivatives,
        boundary_conditions=get_neumann_bc_dict,
        setup_config_func=setup_config_func,
        x_max_size=X_MAX_SIZE,
        oversample=OVERSAMPLE
    )