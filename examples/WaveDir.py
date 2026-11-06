"""Strictly Hyperbolic / Space-Time Wave Benchmark Problem.

Governing Equation (Strong Form):
    u_yy - c^2 * u_xx = 0   on Omega = [0, L] x [0, H]
where:
    x: spatial coordinate
    y: temporal coordinate (t)
    c: characteristic wave speed

Solution Design:
    Two-mode standing wave solution to ensure linear independence (persistent excitation):
        u(x, y) = a1 * sin(k1 * x) * cos(w1 * y) + a2 * sin(k2 * x) * cos(w2 * y)
    where:
        k_i = m_i * pi / L
        w_i = c * k_i
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
boundary_conditions='dirichlet'

# --- Experiment Parameters ---
PDE_PARAMS = {
    'p_c': {'flag': 'wavespeed', 'type': float, 'default': 1.5},
    'p_m1': {'flag': 'freq_x1', 'type': float, 'default': 1.0},
    'p_m2': {'flag': 'freq_x2', 'type': float, 'default': 2.0},
    'p_a1': {'flag': 'amp_1', 'type': float, 'default': 1.0},
    'p_a2': {'flag': 'amp_2', 'type': float, 'default': 0.5},
    'p_L': {'flag': 'bound_x', 'type': float, 'default': 1.0},
    'p_H': {'flag': 'bound_y', 'type': float, 'default': 1.0},
}


def true_sol_with_derivatives(x, p_c, p_m1, p_m2, p_a1, p_a2, p_L, p_H):
  """Calculates the exact analytical partial derivatives up to order 2

  for the 2-mode space-time wave equation.
  """
  assert np.ndim(x) == 2 and x.shape[1] == 2

  X1 = x[:, 0]
  X2 = x[:, 1]

  # Spatial and temporal angular frequencies
  k1 = p_m1 * np.pi / p_L
  k2 = p_m2 * np.pi / p_L
  w1 = p_c * k1
  w2 = p_c * k2

  # Trigonometric terms
  sin_k1x = np.sin(k1 * X1)
  cos_k1x = np.cos(k1 * X1)
  cos_w1y = np.cos(w1 * X2)
  sin_w1y = np.sin(w1 * X2)

  sin_k2x = np.sin(k2 * X1)
  cos_k2x = np.cos(k2 * X1)
  cos_w2y = np.cos(w2 * X2)
  sin_w2y = np.sin(w2 * X2)

  # Analytical solution: u = u1 + u2
  u1 = p_a1 * sin_k1x * cos_w1y
  u2 = p_a2 * sin_k2x * cos_w2y
  u = u1 + u2

  # 1st-order derivatives
  u_x = p_a1 * k1 * cos_k1x * cos_w1y + p_a2 * k2 * cos_k2x * cos_w2y
  u_y = -p_a1 * w1 * sin_k1x * sin_w1y - p_a2 * w2 * sin_k2x * sin_w2y

  # 2nd-order derivatives
  u_xx = -(k1**2) * u1 - (k2**2) * u2
  u_yy = -(w1**2) * u1 - (w2**2) * u2  # u_yy = c^2 * u_xx
  u_xy = -p_a1 * k1 * w1 * cos_k1x * sin_w1y - p_a2 * k2 * w2 * cos_k2x * sin_w2y

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
  """External forcing function g(x) = 0 for the homogeneous wave equation."""
  assert np.ndim(x) == 2 and x.shape[1] == 2
  return np.zeros(x.shape[0], dtype=float)


def setup_config_func(params):
  """Dynamically maps parsed terminal arguments into the PDE configuration."""
  domain = [(0.0, params['p_L']), (0.0, params['p_H'])]

  p_force = inspect.signature(force).parameters.keys()
  force_known = partial(
      force, **{k: v for k, v in params.items() if k in p_force}
  )

  c = params['p_c']
  true_params = {'x,x': -c**2, 'y,y': 1.0}

  # Strong Form: u_yy - c^2 * u_xx = 0
  # Locking 'y,y' to 1.0; 'x,x' target is -c^2 (initialized to blind guess -1.0)
  PDE_KWARGS = {
      'deriv_coef': {'x,x': -1.0, 'y,y': 1.0},
      'trainable': ['x,x'],
      'force': force_known,
  }

  # Ensure knot density adequately resolves the highest spatio-temporal frequency
  max_freq = max(
      params['p_m2'] * np.pi / params['p_L'],
      params['p_c'] * params['p_m2'] * np.pi / params['p_H'],
  )
  required_knots_per_dim = int(max(30, (max_freq / np.pi) * 8))
  num_int_knots = required_knots_per_dim**DIM

  return domain, PDE_KWARGS, true_params, num_int_knots


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