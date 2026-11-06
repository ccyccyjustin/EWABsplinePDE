"""High-Péclet Advection-Dominated Benchmark Problem.

Governing Equation (Strong Form):
    alpha * u_xx + u_yy - v * u_x = 0   on Omega = [0, 1] x [0, 1]
where Pe = v / alpha >> 1 (default Pe = 100 with alpha = 0.01, v = 1.0).

Numerical Stability Design:
The stiff boundary layer mode at x = 1 is formulated as exp(r_2 * (x - 1)).
Since (x - 1) <= 0 on [0, 1], the exponential argument is strictly non-positive,
preventing float64 overflow even as Pe -> 10^4.
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

# --- Experiment Parameters ---
PDE_PARAMS = {
    'p_alpha': {'flag': 'diffusion', 'type': float, 'default': 0.01},
    'p_v': {'flag': 'vel', 'type': float, 'default': 1.0},
    'p_a1': {'flag': 'amp_decay_1', 'type': float, 'default': 1.0},
    'p_a2': {'flag': 'amp_decay_2', 'type': float, 'default': 0.5},
    'p_bl1': {'flag': 'amp_bl_1', 'type': float, 'default': 1.0},
    'p_bl2': {'flag': 'amp_bl_2', 'type': float, 'default': 0.5},
}


def true_sol_with_derivatives(x, p_alpha, p_v, p_a1, p_a2, p_bl1, p_bl2):
  """Calculates the exact analytical partial derivatives up to order 2

  for a 2-mode Advection-Diffusion solution exhibiting boundary layers at x = 1.
  """
  assert np.ndim(x) == 2 and x.shape[1] == 2

  X1 = x[:, 0]
  X2 = x[:, 1]

  # Mode 1: k1 = 1
  k1 = 1
  d1 = np.sqrt(p_v**2 + 4 * p_alpha * (k1 * np.pi) ** 2)
  r1_decay = (p_v - d1) / (2 * p_alpha)  # Smooth decay root (< 0)
  r1_bl = (p_v + d1) / (2 * p_alpha)  # Stiff boundary layer root (> 0)

  # Mode 2: k2 = 2
  k2 = 2
  d2 = np.sqrt(p_v**2 + 4 * p_alpha * (k2 * np.pi) ** 2)
  r2_decay = (p_v - d2) / (2 * p_alpha)
  r2_bl = (p_v + d2) / (2 * p_alpha)

  # Stable evaluations in X1:
  # r_decay * X1 <= 0, and r_bl * (X1 - 1.0) <= 0 on [0, 1]
  exp1_decay = p_a1 * np.exp(r1_decay * X1)
  exp1_bl = p_bl1 * np.exp(r1_bl * (X1 - 1.0))
  X_mode1 = exp1_decay + exp1_bl
  dX_mode1 = (r1_decay * exp1_decay) + (r1_bl * exp1_bl)
  ddX_mode1 = (r1_decay**2 * exp1_decay) + (r1_bl**2 * exp1_bl)

  exp2_decay = p_a2 * np.exp(r2_decay * X1)
  exp2_bl = p_bl2 * np.exp(r2_bl * (X1 - 1.0))
  X_mode2 = exp2_decay + exp2_bl
  dX_mode2 = (r2_decay * exp2_decay) + (r2_bl * exp2_bl)
  ddX_mode2 = (r2_decay**2 * exp2_decay) + (r2_bl**2 * exp2_bl)

  # Y2 trigonometric evaluations
  sin1 = np.sin(np.pi * X2)
  cos1 = np.cos(np.pi * X2)
  sin2 = np.sin(2 * np.pi * X2)
  cos2 = np.cos(2 * np.pi * X2)

  # Assemble derivatives
  u = X_mode1 * sin1 + X_mode2 * sin2
  u_x = dX_mode1 * sin1 + dX_mode2 * sin2
  u_y = np.pi * X_mode1 * cos1 + 2 * np.pi * X_mode2 * cos2
  u_xx = ddX_mode1 * sin1 + ddX_mode2 * sin2
  u_yy = -(np.pi**2) * X_mode1 * sin1 - (4 * np.pi**2) * X_mode2 * sin2
  u_xy = np.pi * dX_mode1 * cos1 + 2 * np.pi * dX_mode2 * cos2

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
  """External forcing function g(x) = 0 for the homogeneous equation."""
  assert np.ndim(x) == 2 and x.shape[1] == 2
  return np.zeros(x.shape[0], dtype=float)


def setup_config_func(params):
  """Dynamically maps parsed terminal arguments into the PDE configuration."""
  domain = [(0.0, 1.0), (0.0, 1.0)]

  p_force = inspect.signature(force).parameters.keys()
  force_known = partial(
      force, **{k: v for k, v in params.items() if k in p_force}
  )

  alpha = params['p_alpha']
  v = params['p_v']
  true_params = {'x,x': alpha, 'y,y': 1.0, 'x': -v}

  # Strong Form: alpha * u_xx + u_yy - v * u_x = 0
  # Reference scale locked to u_yy = 1.0
  PDE_KWARGS = {
      'deriv_coef': {'x,x': 1.0, 'y,y': 1.0, 'x': -1.0},
      'trainable': ['x,x', 'x'],
      'force': force_known,
  }

  num_int_knots = 1000
  return domain, PDE_KWARGS, true_params, num_int_knots


if __name__ == '__main__':
  # Force weak_form flag to '0' since this is a strong-form setup
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