"""Pure First-Order Transport Benchmark Problem.

Governing Equation (Strong Form):
    alpha * u_xx + c_x * u_x + c_y * u_y = g(x, y)   on Omega = [0, 1] x [0, 1]

True Parameters:
    alpha = 0.0 (strictly zero second-order diffusion)
    c_x   = 1.0 (reference transport velocity locked to 1.0)
    c_y   = 2.0 (cross-transport velocity)

Failure Mode:
    Least-squares solvers on spline bases tend to regularize convective residuals by
    introducing artificial numerical diffusion (alpha > 0).
    Tests if the solver correctly drives the estimated alpha to 0.0.
"""

from functools import partial
import inspect
from pathlib import Path
import sys
import numpy as np
from run_example import run_pde_experiment

PDE_NAME = Path(__file__).stem
X_MAX_SIZE = 1_000_000
DIM = 2
OVERSAMPLE = 100

VARIABLES = ['x', 'y']
boundary_conditions = 'dirichlet'

PDE_PARAMS = {
    'p_cx': {'flag': 'vel_x', 'type': float, 'default': 1.0},
    'p_cy': {'flag': 'vel_y', 'type': float, 'default': 2.0},
    'p_m1': {'flag': 'freq_1', 'type': float, 'default': 1.0},
    'p_m2': {'flag': 'freq_2', 'type': float, 'default': 2.0},
    'p_a1': {'flag': 'amp_1', 'type': float, 'default': 1.0},
    'p_a2': {'flag': 'amp_2', 'type': float, 'default': 0.5},
}


def true_sol(x, p_m1, p_m2, p_a1, p_a2, **kwargs):
  X1 = x[:, 0]
  X2 = x[:, 1]
  u1 = p_a1 * np.sin(p_m1 * np.pi * X1) * np.sin(p_m1 * np.pi * X2)
  u2 = p_a2 * np.sin(p_m2 * np.pi * X1) * np.sin(p_m2 * np.pi * X2)
  return u1 + u2


def true_sol_with_derivatives(x, p_m1, p_m2, p_a1, p_a2, **kwargs):
  assert np.ndim(x) == 2 and x.shape[1] == 2
  X1 = x[:, 0]
  X2 = x[:, 1]

  k1 = p_m1 * np.pi
  k2 = p_m2 * np.pi

  s1_x, c1_x = np.sin(k1 * X1), np.cos(k1 * X1)
  s1_y, c1_y = np.sin(k1 * X2), np.cos(k1 * X2)

  s2_x, c2_x = np.sin(k2 * X1), np.cos(k2 * X1)
  s2_y, c2_y = np.sin(k2 * X2), np.cos(k2 * X2)

  u1 = p_a1 * s1_x * s1_y
  u2 = p_a2 * s2_x * s2_y
  u = u1 + u2

  u_x = p_a1 * k1 * c1_x * s1_y + p_a2 * k2 * c2_x * s2_y
  u_y = p_a1 * k1 * s1_x * c1_y + p_a2 * k2 * s2_x * c2_y

  u_xx = -(k1**2) * u1 - (k2**2) * u2
  u_yy = -(k1**2) * u1 - (k2**2) * u2
  u_xy = p_a1 * (k1**2) * c1_x * c1_y + p_a2 * (k2**2) * c2_x * c2_y

  derivs = {
      (0, 0): u,
      (1, 0): u_x,
      (0, 1): u_y,
      (2, 0): u_xx,
      (0, 2): u_yy,
      (1, 1): u_xy,
  }
  return derivs


def force(x, p_cx, p_cy, p_m1, p_m2, p_a1, p_a2, **kwargs):
  """g(x, y) = c_x * u_x + c_y * u_y (pure convection without viscosity)."""
  assert np.ndim(x) == 2 and x.shape[1] == 2
  derivs = true_sol_with_derivatives(
      x, p_m1=p_m1, p_m2=p_m2, p_a1=p_a1, p_a2=p_a2
  )
  return p_cx * derivs[(1, 0)] + p_cy * derivs[(0, 1)]


def setup_config_func(params):
  DOMAIN = [(0.0, 1.0), (0.0, 1.0)]

  p_force = inspect.signature(force).parameters.keys()
  force_known = partial(
      force, **{k: v for k, v in params.items() if k in p_force}
  )

  # Lock 'x' to 1.0 (reference c_x)
  # Trainable: 'y' (target: c_y = 2.0) and 'x,x' (target: alpha = 0.0, initialized to 1.0)
  PDE_KWARGS = {
      'deriv_coef': {'x': 1.0, 'y': 1.0, 'x,x': 1.0},
      'trainable': ['y', 'x,x'],
      'force': force_known,
  }

  true_params = {'vel_y': params['p_cy'], 'diff_xx': 0.0}

  required_knots_per_dim = int(max(30, params['p_m2'] * 15))
  num_int_knots = required_knots_per_dim**DIM

  return DOMAIN, PDE_KWARGS, true_params, num_int_knots


if __name__ == '__main__':
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
      oversample=OVERSAMPLE,
  )