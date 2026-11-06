"""Collinear Harmonic Trap Benchmark Problem.

Governing Equation (Strong Form):
    u_xx + c * u_yy + d * u_x = 0   on Omega = [0, 1] x [0, 1]

True Parameters:
    c = 1.0 (diffusion in y)
    d = 0.0 (zero advection)
    Reference locked to 'x,x' = 1.0.

Failure Mode:
    Because u(x, y) = exp(k*x) * sin(k*y), we have:
        u_xx = k^2 * u
        u_yy = -k^2 * u
        u_x  = k * u
    Hence, u_xx + u_yy = 0 and u_xx - k * u_x = 0 simultaneously.
    Any parameter pair satisfying c - (d / k) = 1 is a valid solution.
    Tests if the solver detects rank deficiency in the Gram matrix or outputs arbitrary parameters.
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
boundary_conditions = 'neumann'

PDE_PARAMS = {
    'p_k': {'flag': 'freq_k', 'type': float, 'default': 1.0},
    'p_amp': {'flag': 'amplitude', 'type': float, 'default': 1.0},
}


def true_sol(x, p_k, p_amp, **kwargs):
  X1 = x[:, 0]
  X2 = x[:, 1]
  return p_amp * np.exp(p_k * X1) * np.sin(p_k * X2)


def true_sol_with_derivatives(x, p_k, p_amp, **kwargs):
  assert np.ndim(x) == 2 and x.shape[1] == 2
  X1 = x[:, 0]
  X2 = x[:, 1]

  E = p_amp * np.exp(p_k * X1)
  sin_ky = np.sin(p_k * X2)
  cos_ky = np.cos(p_k * X2)

  u = E * sin_ky
  u_x = p_k * u
  u_y = p_k * E * cos_ky
  u_xx = (p_k**2) * u
  u_yy = -(p_k**2) * u
  u_xy = (p_k**2) * E * cos_ky

  derivs = {
      (0, 0): u,
      (1, 0): u_x,
      (0, 1): u_y,
      (2, 0): u_xx,
      (0, 2): u_yy,
      (1, 1): u_xy,
  }
  return derivs


def force(x, **kwargs):
  assert np.ndim(x) == 2 and x.shape[1] == 2
  return np.zeros(x.shape[0], dtype=float)


def setup_config_func(params):
  DOMAIN = [(0.0, 1.0), (0.0, 1.0)]

  p_force = inspect.signature(force).parameters.keys()
  force_known = partial(
      force, **{k: v for k, v in params.items() if k in p_force}
  )

  # Model is configured to learn c ('y,y') and d ('x')
  # Locking 'x,x' to 1.0; true values are c = 1.0, d = 0.0
  PDE_KWARGS = {
      'deriv_coef': {'x,x': 1.0, 'y,y': 1.5, 'x': 0.5},
      'trainable': ['y,y', 'x'],
      'force': force_known,
  }

  true_params = {'x,x': 1.0, 'y,y': 1.0, 'x': 0.0}

  num_int_knots = 1000 #required_knots_per_dim**DIM

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