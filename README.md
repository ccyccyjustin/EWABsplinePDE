# EWABSplinePDE

**Exponentially Weighted Aggregation (EWA) B-Spline Estimator for Physics-Informed Surface Reconstruction, PDE Parameter Discovery, and Calibrated Uncertainty Quantification**

`EWABSplinePDE` is a Python package for jointly reconstructing noisy multidimensional physical fields $u(\mathbf{x})$ and inferring unknown governing Partial Differential Equation (PDE) coefficients $\theta$ with finite-sample-calibrated confidence intervals. Rather than selecting a single smoothing parameter by cross-validation, it aggregates an entire continuum of physics-regularized B-spline estimators via Stein's Unbiased Risk Estimate (SURE) in an orthogonal eigenbasis.

---

## Key Features

* **Data-Adaptive Spectral EWA Smoothing:** Solves a Generalized Eigenvalue Problem (GEVP) to evaluate candidate PDE regularization weights $\lambda \in (0, 1)$ in closed form and combines them via SURE-weighted Gibbs aggregation (`util.py`)
* **Strong-Form & Variational Weak-Form Penalties:** Supports strong-form PDE residuals alongside integration-by-parts variational weak forms (`weak_form=True`) (`ewa_assembly.py`, `Bsplines.py`)
* **Boundary Enforcement:** Enforces boundary conditions in coefficient space (`ewa_bc.py`)
* **Uncertainty Quantification for Both Fields and PDE Parameters:** Computes exact pointwise predictive standard errors using the SURE-adaptive sensitivity matrix and asymptotic parameter covariance (`ewa_inference.py`)
* **Constant- and Variable-Coefficient PDE Support:** Provides `EWABSplinePDE` for constant-coefficient linear PDEs and `FunctionalEWABSplinePDE` for spatially varying operator benchmarks (`ewa.py`, `ewa_op.py`)
* **Diagnostics Plots & Benchmark Examples:** Reproducible PDE benchmarks (`examples/`) and multi-panel Sobolev, calibration, spectral-filter, and parameter-recovery dashboards (`ewa_plot.py`, `plot.py`, `run_example.py`)

---
## Mathematical Overview

Consider noisy observations $`y_i = u^{\ast}(\mathbf{x}_i) + \varepsilon_i`$ ($`i = 1, \dots, n`$) on a bounded domain $`\Omega \subset \mathbb{R}^d`$, where $`\mathbb{E}[\varepsilon_i] = 0`$, $`\mathrm{Var}(\varepsilon_i) = \sigma^2`$, and $`u^{\ast}`$ satisfies a linear PDE

$$\mathcal{L}(\theta) u := \sum_{a=1}^p \theta_a D_a u = g(\mathbf{x}), \qquad \mathbf{x} \in \Omega,$$

with trainable parameters $`\theta_T`$ and fixed normalization parameters $`\theta_{T^c}`$ (e.g., $`\theta_{(2,0)} = 1`$).
### 1. Spectral Orthogonalization & SURE-Weighted EWA

Given the current PDE penalty $`P(\theta) = \sum_{a,b} \theta_a \theta_b \Omega_{ab}`$, reduced interior matrix $`A_{\text{red}}(\theta) = T^T P(\theta) T`$, and forcing $`v_{\text{red}}(\theta) = T^T(F_g(\theta) - P(\theta)d_0)`$, the solver diagonalizes the pair $`(A_{\text{red}}(\theta), G_n)`$:

$$A_{\text{red}}(\theta) V = G_n V S, \qquad V^T G_n V = I_{n_I}, \qquad S = \mathrm{diag}(s_1, \dots, s_{n_I}).$$

In the orthonormal basis $`U = B_{\text{red}} V`$ ($`\frac{1}{n}U^T U = I_{n_I}`$) with empirical projection $`d = \frac{1}{n} U^T y_{\text{adj}}`$ and projected forcing $`f = V^T v_{\text{red}}`$, every penalized candidate along the $`\lambda_m \in (0, 1)`$ grid has a diagonal closed form:

$$\hat{\alpha}_j(\lambda_m) = \omega_{jm} d_j + h_{jm} f_j, \qquad \omega_{jm} = \frac{(1-\lambda_m)n}{(1-\lambda_m)n + \lambda_m s_j}, \qquad h_{jm} = \frac{\lambda_m}{(1-\lambda_m)n + \lambda_m s_j}.$$

Using the exact spectral SURE risk $`\mathrm{SURE}(\lambda_m) = \lVert y_\perp \rVert_2^2 + n \sum_j [h_{jm}(s_j d_j - f_j)]^2 + 2\hat{\sigma}^2 \sum_j \omega_{jm}`$ and temperature $`\gamma = \gamma_{\text{mult}} \hat{\sigma}^2`$, EWA aggregates all candidates via Gibbs weights:

$$w_m = \frac{\pi_m \exp(-\mathrm{SURE}(\lambda_m)/\gamma)}{\sum_\ell \pi_\ell \exp(-\mathrm{SURE}(\lambda_\ell)/\gamma)}, \qquad \hat{\alpha}_{\text{EWA}} = \sum_{m=1}^L w_m \hat{\alpha}(\lambda_m), \qquad \hat{\beta}_{\text{full}} = T V \hat{\alpha}_{\text{EWA}} + d_0.$$

### 2. Parameter Estimation & Coupled Sandwich Covariance

Alternating with the EWA surface update, $`\theta_T`$ solves the (optionally EIV-debiased) linear PDE normal equations $`\Psi(\theta, \hat{\beta}) = G_{TT}\theta_T + G_{TT^c}\theta_{T^c} - v_T = 0`$. Differentiating both the linear filter and the data-dependent SURE weights yields:
- **Adaptive modal sensitivity:** $`\Xi = \frac{\partial \hat{\alpha}_{\text{EWA}}}{\partial d} = \mathrm{diag}(\bar{\omega}) + \frac{2n}{\gamma} H_r \mathrm{diag}(w) B_{\text{centered}}^T`$, giving predictive field covariance $`\widehat{\mathrm{Cov}}(\hat{\beta}_{\text{full}}) = \frac{\hat{\sigma}^2}{n}(TV\Xi)(TV\Xi)^T`$.
- **Coupled parameter Jacobian (Bread) and Score Covariance (Meat):** Propagating $`\theta`$-perturbations through the candidate path, SURE gradient $`G_{\text{SURE}}`$, and EWA weights yields $`J_{\alpha,\text{EWA}} = V^{-1}\partial_{\theta_T} \hat{\beta}_{I,\text{EWA}}`$, giving

$$A = J_{TT} + K J_{\alpha,\text{EWA}}, \qquad B = \frac{\hat{\sigma}^2}{n}(K \Xi)(K \Xi)^T, \qquad \widehat{\mathrm{Cov}}(\hat{\theta}_T) = A^{-1} B A^{-T},$$

where $`K_k = H_k \hat{\alpha}_{\text{EWA}} - g_k`$ is the modal gradient of the PDE score $`\Psi_k`$.
---

## Repository Structure

```text
EWABsplinePDE/
├── ewa.py               # Core EWABSplinePDE scikit-learn estimator & BCD/Newton fit loop
├── ewa_assembly.py      # Strong, projected, and weak PDE Gram & forcing matrix assembly
├── ewa_bc.py            # Generalized Dirichlet/Neumann boundary condition elimination (T, d_vec)
├── ewa_inference.py     # SURE differentiation, adaptive sensitivity Xi, EIV debiasing & sandwich SE
├── ewa_op.py            # FunctionalEWABSplinePDE for spatially varying / operator PDEs
├── ewa_plot.py          # Estimator diagnostic plots (derivatives, basis functions, lambda PDF, spectrum)
├── Bsplines.py          # Sparse CSR/dense tensor B-splines & vectorized Gauss-Legendre quadrature
├── util.py              # Fast LAPACK/sparse GEVP, spectral SURE, EWA weights, GSJ variance, parallelization
├── run_example.py       # Unified Monte Carlo experiment runner, CLI parser & Sobolev/UQ evaluation
├── plot.py              # Publication-ready comparative benchmark visualization scripts
└── examples/            # 12 reproducible PDE discovery benchmarks
    ├── AnisoHelmholtzDir.py
    ├── CollinHarmNeu.py
    ├── ConstNeu.py
    ├── CornerSingDir.py
    ├── DegenEllipNeu.py
    ├── HighHelmholtzNeu.py
    ├── HighPecletNeu.py
    ├── NuclearDir.py
    ├── PoissonDir.py
    ├── TelegraphNeu.py
    ├── TransportDir.py
    └── WaveDir.py
```

---

## Installation

Clone the repository and install the scientific Python dependencies:

```bash
git clone [https://github.com/ccyccyjustin/ewabsplinepde.git](https://github.com/ccyccyjustin/ewabsplinepde.git)
cd ewabsplinepde
pip install numpy scipy pandas scikit-learn matplotlib seaborn joblib tqdm
```

---

## Quick Start

Fit `EWABSplinePDE` on noisy 2D observations from the Anisotropic Helmholtz equation $u_{xx} + \alpha u_{yy} + \beta u = 0$ with Dirichlet boundary conditions:

```python
import numpy as np
from ewa import EWABSplinePDE
from run_example import get_bc_dict, align_pde_params
from examples.AnisoHelmholtzDir import (
    PDE_PARAMS, setup_config_func, true_sol_with_derivatives
)
import util

# 1. Generate synthetic noisy training data on [0, 1]^2
params = {k: v['default'] for k, v in PDE_PARAMS.items()}
domain, pde_kwargs, true_params = setup_config_func(params)

x_grid = util.tensor_prod(np.linspace(0, 1, 500), np.linspace(0, 1, 500))
u_grid = true_sol_with_derivatives(x_grid, **params)[(0, 0)]
sigma = 0.1 * np.std(u_grid)

rng = np.random.default_rng(42)
idx = np.sort(rng.choice(len(x_grid), size=3000))
X_train = x_grid[idx]
y_train = u_grid[idx] + rng.normal(0.0, sigma, size=3000)

# 2. Construct exact Dirichlet boundary conditions
bc_dict = get_bc_dict(
    true_sol_with_derivatives, domain, boundary_conditions='dirichlet', **params
)

# 3. Fit EWABSplinePDE (jointly reconstructs u and discovers beta, alpha)
model = EWABSplinePDE(
    dim=2,
    variables=['x', 'y'],
    domain=domain,
    deg=3,
    boundary_conditions=bc_dict,
    weak_form=False,
    bias_corrected=True,
    **pde_kwargs,
)
model.fit(X_train, y_train)

# 4. Inspect recovered PDE coefficients, sandwich SEs, and 95% CIs
param_table = align_pde_params(true_params, model=model)
print(param_table[['Operator', 'Role', 'Estimate', 'True_Value', 'Std_Error', 'Z_Score', 'Covered_95']])

# 5. Predict field values with calibrated pointwise CIs and partial derivatives
preds = model.predict(X_train[:10])            # columns: fhat, se, ci_lower, ci_upper
u_yy_hat = model.predict_derivative(X_train[:10], deriv_orders=(0, 2))
```

---

## Running Benchmark Experiments

Every script in `examples/` uses the shared CLI runner in `run_example.py` for single-run visual debugging or parallel Monte Carlo evaluation:

```bash
# Single debug run with full diagnostic dashboards on Anisotropic Helmholtz
python examples/AnisoHelmholtzDir.py -n 3000 -sf 0.1 -deg 3 -w 0 -b 1 -d 1

# 100-replicate Monte Carlo study (saved to CSV)
python examples/AnisoHelmholtzDir.py -n 3000 -sf 0.1 -deg 3 -w 0 -b 1 -i 100 -s 1

# Run in variational weak form
python examples/HighPecletNeu.py -n 3000 -sf 0.1 -w 1 -b 1 -i 50
```

### Command-Line Arguments

| Flag | Long Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `-n` | `--sample_size` | `1000` | Number of scattered training observations $n$ |
| `-sf` | `--sigma_frac` | `0.1` | Noise-to-signal ratio ($\sigma / \operatorname{std}(u^*)$) |
| `-deg` | `--degree` | `3` | B-spline polynomial degree $p$ |
| `-a` | `--alpha` | `0.05` | Significance level for $(1 - \alpha)$ confidence intervals |
| `-q` | `--quantile` | `'0'` | Use quantile-spaced (`1`) vs. uniform (`0`) interior knots |
| `-w` | `--weak_form` | `'0'` | Use variational weak form (`1`) vs. strong form (`0`) |
| `-b` | `--bias_corrected` | `'1'` | Enable (`1`) or disable (`0`) EIV bias correction |
| `-i` | `--iteration` | `100` | Number of Monte Carlo replications (executed in parallel) |
| `-d` | `--debug` | `'0'` | Run a single iteration with full diagnostic plots (`1`) |
| `-s` | `--save` | `'1'` | Save Monte Carlo summary metrics to CSV (`1`) |

### Included PDE Benchmarks (`examples/`)

* **Dirichlet Boundary Benchmarks:** `AnisoHelmholtzDir.py`, `CornerSingDir.py`, `NuclearDir.py`, `PoissonDir.py`, `TransportDir.py`, `WaveDir.py`
* **Neumann Boundary Benchmarks:** `CollinHarmNeu.py`, `ConstNeu.py`, `DegenEllipNeu.py`, `HighHelmholtzNeu.py`, `HighPecletNeu.py`, `TelegraphNeu.py`

---

## Diagnostics & Evaluation Metrics

Running with `-d 1` or invoking the plotting helpers produces comprehensive field and parameter diagnostics:
- **Sobolev & Spatial Accuracy:** Relative $`L^2`$, $`H^1`$, and $`H^2`$ semi-norm errors, interior vs. boundary RMSE, $`L^\infty`$ error, and derivative error heatmaps (`evaluate_metrics`, `model.plot_derivatives`)
- **Uncertainty Calibration:** Empirical 90%/95%/99% pointwise coverage, Mean Prediction Interval Width (MPIW), Winkler score, Gaussian CRPS, reliability curves, and conditional error funnels (`plot_diagnostics`)
- **Spectral & Regularization Inspection:** EWA posterior weight distribution $`w(\lambda)`$ overlaid with $`\mathrm{SURE}(\lambda)`$, GEVP spectrum $`s_j`$ vs. filter decay $`\bar{\omega}_j`$, and dominant eigenfunction decompositions (`plot_lambda_weight`, `plot_eigenvalues`, `plot_basis_functions`)
- **Parameter Inference:** Analytical multivariate normal density pairplots (`plot_coef_dist`) and multi-configuration studentized $`Z`$-score / 95% CI coverage boxplots (`plot_res_summary`)
---

## Citation

If you use `EWABSplinePDE` in your research, please cite the repository:

```bibtex
@software{ewabsplinepde,
  author = {Cheung, Chun Yin},
  title  = {EWABSplinePDE: Exponentially Weighted Aggregation B-Spline Estimator for PDE Discovery and Uncertainty Quantification},
  url    = {[https://github.com/ccyccyjustin/ewabsplinepde](https://github.com/ccyccyjustin/ewabsplinepde)},
  year   = {2026}
}
```
