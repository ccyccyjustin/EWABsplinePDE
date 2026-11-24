"""Zero-Signal / Noise-Only Nullity Benchmark Problem.

Governing Physics:
    True field is identically null: u(x, y) = 0 everywhere on Omega = [0, 1] x [0, 1].
    Boundary conditions: homogeneous Dirichlet u|_{dOmega} = 0.
    Forcing: g(x, y) = 0.

Observed Data:
    y_i = 0 + eta_i, where eta_i ~ N(0, sigma^2).

Failure Mode:
    Tests whether the EWA estimator identifies the absence of a coherent physical signal
    (assigning lambda -> 0 or flagging infinite parameter variance) rather than overfitting
    the noise profile to a spurious differential equation.
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
    'p_dummy': {'flag': 'dummy_param', 'type': float, 'default': 1.0},
}


class ZeroSignalArray(np.ndarray):
  """Custom ndarray returning nominal standard deviation of 1.0 for noise scaling."""

  def std(self, *args, **kwargs):
    return 1.0


def true_sol(x, **kwargs):
  N = x.shape[0]
  return np.zeros(N, dtype=float).view(ZeroSignalArray)


def true_sol_with_derivatives(x, **kwargs):
  assert np.ndim(x) == 2 and x.shape[1] == 2
  N = x.shape[0]

  zero_y = np.zeros(N, dtype=float).view(ZeroSignalArray)
  zero_deriv = np.zeros(N, dtype=float)

  derivs = {
      (0, 0): zero_y,
      (1, 0): zero_deriv,
      (0, 1): zero_deriv,
      (2, 0): zero_deriv,
      (0, 2): zero_deriv,
      (1, 1): zero_deriv,
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

  # Blindly attempts to identify a Laplacian balance u_xx + c * u_yy = 0 from pure noise
  PDE_KWARGS = {
      'deriv_coef': {'': 1.0, 'x,x': 1.0, 'y,y': 1.0},
      'trainable': ['x,x', 'y,y'],
      'force': force_known,
  }

  true_params = {'': 0.0, 'x,x': 0.0, 'y,y': 0.0}

  return DOMAIN, PDE_KWARGS, true_params


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