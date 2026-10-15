"""Strongly Anisotropic Tilted Diffusion Benchmark Problem.

Governing Equation (Strong Form):
    u_xx + b * u_xy + c * u_yy = g(x, y)   on Omega = [0, 1] x [0, 1]

Properties:
    - Discriminant: (b / 2)^2 - c = (1.9 / 2)^2 - 1.0 = -0.0975 < 0 (nearly degenerate elliptic).
    - Strong coordinate cross-coupling requiring simultaneous recovery of:
        - b (cross-derivative term 'x,y', default 1.9)
        - c (anisotropic diffusion term 'y,y', default 1.0)
    - Reference scale locked to 'x,x' = 1.0.
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

# --- Experiment Parameters ---
PDE_PARAMS = {
    'p_b': {
        'flag': 'cross_diff',
        'type': float,
        'default': 1.95,  # Strongly tilted cross-coupling (b^2 / 4 - c ≈ -0.0975)
    },
    'p_c': {
        'flag': 'diff_y',
        'type': float,
        'default': 1.0,  # Anisotropic y-direction diffusion
    },
    'p_m': {
        'flag': 'wave_m',
        'type': float,
        'default': 2.0,  # Integer modes ensure homogeneous Neumann BCs
    },
    'p_n': {
        'flag': 'wave_n',
        'type': float,
        'default': 3.0,
    },
    'p_a1': {'flag': 'amp_1', 'type': float, 'default': 1.0},
    'p_a2': {'flag': 'amp_2', 'type': float, 'default': -0.5},
}


def true_sol(x, p_m, p_n, p_a1, p_a2, **kwargs):
  """Analytical 2-mode solution ensuring linear independence across u_xx, u_yy, and u_xy."""
  X1 = x[:, 0]
  X2 = x[:, 1]
  u_1 = np.cos(p_m * np.pi * X1) * np.cos(p_n * np.pi * X2)
  u_2 = np.cos(p_n * np.pi * X1) * np.cos(p_m * np.pi * X2)
  return p_a1 * u_1 + p_a2 * u_2


def true_sol_with_derivatives(x, p_m, p_n, p_a1, p_a2, **kwargs):
  """Calculates exact analytical partial derivatives up to order 2

  for the anisotropic tilted diffusion solution.
  """
  assert np.ndim(x) == 2 and x.shape[1] == 2

  X1 = x[:, 0]
  X2 = x[:, 1]

  m_pi = p_m * np.pi
  n_pi = p_n * np.pi

  # Precompute base trigonometric evaluations
  cos_mx = np.cos(m_pi * X1)
  sin_mx = np.sin(m_pi * X1)

  cos_nx = np.cos(n_pi * X1)
  sin_nx = np.sin(n_pi * X1)

  cos_my = np.cos(m_pi * X2)
  sin_my = np.sin(m_pi * X2)

  cos_ny = np.cos(n_pi * X2)
  sin_ny = np.sin(n_pi * X2)

  derivs = {}

  # 0th-order: u
  derivs[(0, 0)] = p_a1 * (cos_mx * cos_ny) + p_a2 * (cos_nx * cos_my)

  # 1st-order: u_x, u_y
  derivs[(1, 0)] = -p_a1 * m_pi * (sin_mx * cos_ny) - p_a2 * n_pi * (
      sin_nx * cos_my
  )
  derivs[(0, 1)] = -p_a1 * n_pi * (cos_mx * sin_ny) - p_a2 * m_pi * (
      cos_nx * sin_my
  )

  # 2nd-order pure: u_xx, u_yy
  derivs[(2, 0)] = -p_a1 * (m_pi**2) * (cos_mx * cos_ny) - p_a2 * (
      n_pi**2
  ) * (cos_nx * cos_my)
  derivs[(0, 2)] = -p_a1 * (n_pi**2) * (cos_mx * cos_ny) - p_a2 * (
      m_pi**2
  ) * (cos_nx * cos_my)

  # 2nd-order cross: u_xy
  derivs[(1, 1)] = p_a1 * (m_pi * n_pi) * (sin_mx * sin_ny) + p_a2 * (
      n_pi * m_pi
  ) * (sin_nx * sin_my)

  return derivs


def force(x, p_m, p_n, p_a1, p_a2, p_b, p_c, **kwargs):
  """External forcing function g(x, y) = u_xx + b * u_xy + c * u_yy."""
  assert np.ndim(x) == 2 and x.shape[1] == 2
  derivs = true_sol_with_derivatives(
      x, p_m=p_m, p_n=p_n, p_a1=p_a1, p_a2=p_a2, p_b=p_b, p_c=p_c
  )
  return derivs[(2, 0)] + p_b * derivs[(1, 1)] + p_c * derivs[(0, 2)]


def setup_config_func(params):
  """Dynamically maps parsed terminal arguments into the PDE configuration."""
  DOMAIN = [(0.0, 1.0), (0.0, 1.0)]

  p_force = inspect.signature(force).parameters.keys()
  force_known = partial(
      force, **{k: v for k, v in params.items() if k in p_force}
  )

  # Reference scale locked to 'x,x' = 1.0
  # Blind initial guesses set to 1.0 for trainable parameters
  PDE_KWARGS = {
      'deriv_coef': {'x,x': 1.0, 'x,y': 1.0, 'y,y': 1.0},
      'trainable': ['x,y', 'y,y'],
      'force': force_known,
  }

  true_params = {'b_cross': params['p_b'], 'c_yy': params['p_c']}

  # Knot allocation resolving highest spatial frequency
  k_sqr = (params['p_m'] ** 2 + params['p_n'] ** 2) * (np.pi**2)
  k = np.sqrt(k_sqr)
  required_knots_per_dim = int(max(30, (k / np.pi) * 6))
  num_int_knots = required_knots_per_dim**DIM

  return DOMAIN, PDE_KWARGS, true_params, num_int_knots


if __name__ == '__main__':
  # Force strong-form execution by default
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