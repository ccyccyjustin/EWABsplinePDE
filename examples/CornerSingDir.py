"""Corner Singularity / Re-entrant Corner Benchmark Problem.

Governing Equation (Strong Form):
    u_xx + c * u_yy = 0   on Omega = [0, 1] x [0, 1]

True Parameter:
    c = 1.0 (Laplace operator with reference locked to 'x,x' = 1.0)

Analytical Field:
    u(x, y) = Im( (x + i*y)^(2/3) ) = r^(2/3) * sin(2*theta / 3)
    This function is strictly harmonic (u_xx + u_yy = 0) everywhere on (0, 1]^2,
    but possesses an unbounded gradient and Hessian at the corner (0, 0):
        |grad u| ~ r^(-1/3) -> inf as r -> 0
        |D^2 u|  ~ r^(-4/3) -> inf as r -> 0

Failure Mode:
    Tests whether singular boundary derivatives destabilize global B-spline quadratures
    or corrupt recovery of c = 1.0.
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
    'p_alpha': {
        'flag': 'singular_power',
        'type': float,
        'default': 2.0 / 3.0,
    },  # 2/3 corner singularity
    'p_scale': {'flag': 'scale', 'type': float, 'default': 1.5},
}


def true_sol(x, p_alpha, p_scale, **kwargs):
  X1 = x[:, 0]
  X2 = x[:, 1]
  r = np.maximum(np.hypot(X1, X2), 1e-12)
  theta = np.arctan2(X2, X1)
  return p_scale * (r**p_alpha) * np.sin(p_alpha * theta)


def true_sol_with_derivatives(x, p_alpha, p_scale, **kwargs):
  assert np.ndim(x) == 2 and x.shape[1] == 2
  X1 = x[:, 0]
  X2 = x[:, 1]

  # Floor r to 1e-12 to prevent numerical Inf/NaN at the singular corner (0, 0)
  r = np.maximum(np.hypot(X1, X2), 1e-12)
  theta = np.arctan2(X2, X1)
  a = p_alpha

  # Analytical derivatives via complex holomorphic function f(z) = z^a = v + i*u
  # u    = Im(z^a)   = r^a * sin(a * theta)
  # u_x  = Im(f'(z)) = -a * r^(a - 1) * sin((1 - a) * theta)
  # u_y  = Re(f'(z)) =  a * r^(a - 1) * cos((1 - a) * theta)
  # u_xx = Im(f''(z)) = a * (1 - a) * r^(a - 2) * sin((2 - a) * theta)
  # u_yy = -u_xx (harmonic identity)
  u = p_scale * (r**a) * np.sin(a * theta)
  u_x = -p_scale * a * (r ** (a - 1.0)) * np.sin((1.0 - a) * theta)
  u_y = p_scale * a * (r ** (a - 1.0)) * np.cos((1.0 - a) * theta)

  u_xx = p_scale * a * (1.0 - a) * (r ** (a - 2.0)) * np.sin((2.0 - a) * theta)
  u_yy = -u_xx
  u_xy = -p_scale * a * (1.0 - a) * (r ** (a - 2.0)) * np.cos((2.0 - a) * theta)

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

  # Strong Form: u_xx + c * u_yy = 0
  # Locking 'x,x' to 1.0; target for 'y,y' is c = 1.0
  PDE_KWARGS = {
      'deriv_coef': {'': 0.0, 'x,x': 1.0, 'y,y': 1.0},
      'trainable': ['', 'y,y'],
      'force': force_known,
  }

  true_params = {'diff_y': 1.0}

  # Allocate higher resolution near domain bounds
  num_int_knots = 1000 #45**DIM

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