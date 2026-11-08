import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Callable

import ewa
import util
from ewa import EWABSplinePDE
from ewa_op import FunctionalEWABSplinePDE
from util import canberra_similarity, parallel, tensor_prod, CMAP
from scipy.stats import norm


def to_bool(arg):
    assert arg in ['0', '1']
    return True if arg == '1' else False

def get_bc_dict(true_sol_with_derivatives, domain, boundary_conditions='neumann', **kwargs):
    """
    Constructs the 1st-order derivative (Neumann) boundary conditions.
    Explicitly forces the spatial coordinates to align with domain bounds.
    """
    dim = len(domain)
    bc_dict = {}
    if boundary_conditions is None:
        return bc_dict

    assert isinstance(boundary_conditions, str)
    if boundary_conditions.lower() == 'neumann':
        k = 1
    elif boundary_conditions.lower() == 'dirichlet':
        k = 0
    else:
        raise ValueError

    def force_boundary_eval(x_input, d_idx, fixed_val, deriv_tuple):
        x_exact = x_input.copy()
        x_exact[:, d_idx] = fixed_val
        return true_sol_with_derivatives(x_exact, **kwargs)[deriv_tuple]

    for d in range(dim):
        d_min, d_max = domain[d]
        deriv_order = [0] * dim
        deriv_order[d] = k
        deriv_order = tuple(deriv_order)
        # Bind loop variables by value using default arguments to prevent closure corruption
        bc_dict[(f'{d}_min', k)] = (
            lambda x, d_idx=d, val=d_min, order=deriv_order: force_boundary_eval(
                x, d_idx=d_idx, fixed_val=val, deriv_tuple=order
            )
        )
        bc_dict[(f'{d}_max', k)] = (
            lambda x, d_idx=d, val=d_max, order=deriv_order: force_boundary_eval(
                x, d_idx=d_idx, fixed_val=val, deriv_tuple=order
            )
        )

    return bc_dict

def align_pde_params(
    true_params: dict,
    model: Optional[EWABSplinePDE] = None,
    res: Optional[Union[pd.Series, pd.DataFrame]] = None,
    pde_kwargs: Optional[dict] = None,
    variables: Optional[List[str]] = None,
    dim: Optional[int] = None,
    model_class=None,
) -> pd.DataFrame:
    """
    Aligns true_params with the model coefficients using exact derivative keys ('', 'x,x', 'y,y').
    Works BOTH inside run_one (using model) AND outside run_one at line 774 (using res_df + pde_kwargs).
    """
    # If model is not provided (e.g., at line 774 outside run_one), build a lightweight metadata instance
    if model is None:
        if pde_kwargs is None or variables is None or dim is None or model_class is None:
            raise ValueError("Must provide either 'model' or ('pde_kwargs', 'variables', 'dim').")
        model = model_class(dim=dim, variables=variables, **pde_kwargs)
        model._parse_physics_configuration()

    K = len(model._coef_keys)
    T = list(model._trainable_idx)

    # 1. Extract Coefficient Estimates and Variances
    if res is not None:
        if isinstance(res, pd.DataFrame):
            res = res.mean(axis=0)
        coef_mean = np.array([float(res.get(f'coef_mean_{i}', np.nan)) for i in range(K)])

        var_keys = [k for k in res.index if k.startswith('coef_var_')]
        dim_v = int(np.sqrt(len(var_keys))) if var_keys else 0
        coef_var = np.zeros((dim_v, dim_v))
        for r in range(dim_v):
            for c in range(dim_v):
                coef_var[r, c] = float(res.get(f'coef_var_{r}_{c}', 0.0))
    else:
        coef_mean = model.coef_mean
        coef_var = model.coef_var
        dim_v = coef_var.shape[0]

    is_active_only = (dim_v == len(T) and len(T) < K)

    # 3. Assemble aligned summary table
    rows = []
    for i in range(K):
        d_ord = model._coef_keys[i]
        op_label = model.coef_keys[i]
        raw_key = model.deriv_map[d_ord]
        is_trainable = (i in T)
        role = "Trainable" if is_trainable else "Locked"

        if is_trainable:
            t_pos = T.index(i)
            var_i = float(coef_var[t_pos, t_pos]) if is_active_only else float(coef_var[i, i])
        else:
            var_i = 0.0  # Locked reference parameters have zero estimation variance

        se_i = float(np.sqrt(max(var_i, 0.0)))
        est_val = float(coef_mean[i])
        true_val = true_params.get(raw_key, np.nan)

        if not is_trainable and np.isnan(true_val):
            true_val = est_val

        if not np.isnan(true_val):
            abs_err = abs(est_val - true_val)
            if abs(true_val) < 1e-6:
                rel_err = np.nan
            else:
                rel_err = abs_err / abs(true_val)
            z_score = (est_val - true_val) / max(se_i, 1e-12) if is_trainable else 0.0
            ci_95_l = est_val - 1.95996 * se_i
            ci_95_u = est_val + 1.95996 * se_i
            covered = bool(ci_95_l <= true_val <= ci_95_u) if is_trainable else True
        else:
            abs_err = rel_err = z_score = np.nan
            ci_95_l = ci_95_u = est_val
            covered = np.nan

        rows.append({
            'Index': f'coef_mean_{i}',
            'Operator': op_label,
            'Raw_Key': raw_key,
            'Role': role,
            'Estimate': est_val,
            'True_Value': true_val,
            'Std_Error': se_i,
            'Variance': var_i,
            'CI_95_Lower': ci_95_l,
            'CI_95_Upper': ci_95_u,
            'Abs_Error': abs_err,
            'Rel_Error': rel_err,
            'Z_Score': z_score,
            'Covered_95': covered
        })

    return pd.DataFrame(rows)


def evaluate_metrics(
        model: EWABSplinePDE,
        domain: List[Tuple[float, float]],
        true_sol_func: Callable,
        true_params: dict,
        prefix: str = "grid"
) -> Dict[str, float]:
    """
    Evaluates field approximation, continuous Sobolev norms, and uncertainty
    calibration on a fixed, equally-spaced Cartesian grid using the true solution.
    """
    dim = len(domain)
    x_grid = model.gridpoints_
    n_eval = len(x_grid)

    # 2. Evaluate ground truth field and derivatives
    true_dict = true_sol_func(x_grid, **true_params)
    u_true = true_dict[(0, 0)] if isinstance(true_dict, dict) else true_dict

    # 3. Model prediction (field and function standard error se_f)
    # Note: noiseless=True returns se_f of the mean surface, not observation error
    pred_res = model.predict(x_grid)
    u_pred, se_f = pred_res['fhat'], pred_res['se']

    err = u_pred - u_true
    l2_denom = np.linalg.norm(u_true)

    # --- Sobolev Norms ---
    if l2_denom < 1e-6:
        rel_l2 = np.nan
    else:
        rel_l2 = np.linalg.norm(err) / l2_denom

    rmse = np.sqrt(np.mean(err ** 2))
    nrmse = rmse / (np.max(u_true) - np.min(u_true) + 1e-12)
    linf = np.max(np.abs(err))

    # 3. H^1 and H^2 Sobolev Semi-Norm Relative Errors
    h1_err_sq, h1_true_sq = 0.0, 0.0
    h2_err_sq, h2_true_sq = 0.0, 0.0

    for ord_tup, d_true in true_dict.items():
        d_pred = model.predict_derivative(x_grid, ord_tup)
        nom = float(np.nansum((d_pred - d_true) ** 2))
        denom = float(np.nansum(d_true ** 2))
        if denom < 1e-6:
            continue

        if sum(ord_tup) == 1:
            h1_err_sq += nom
            h1_true_sq += denom
        elif sum(ord_tup) == 2:
            h2_err_sq += nom
            h2_true_sq += denom

    rel_h1 = float(np.sqrt(h1_err_sq) / max(np.sqrt(h1_true_sq), 1e-12)) if h1_true_sq > 0 else np.nan
    rel_h2 = float(np.sqrt(h2_err_sq) / max(np.sqrt(h2_true_sq), 1e-12)) if h2_true_sq > 0 else np.nan

    # --- Interior vs. Boundary Shell Partitioning ---
    bnd_mask = np.zeros(n_eval, dtype=bool)
    for p, (a, b) in enumerate(domain):
        margin = 0.10 * (b - a)
        bnd_mask |= (x_grid[:, p] <= a + margin) | (x_grid[:, p] >= b - margin)
    int_mask = ~bnd_mask

    rmse_int = np.sqrt(np.mean(err[int_mask] ** 2)) if np.any(int_mask) else np.nan
    rmse_bnd = np.sqrt(np.mean(err[bnd_mask] ** 2)) if np.any(bnd_mask) else np.nan

    # --- Uncertainty Quantification Calibration (vs. True Mean Surface) ---
    z90, z95, z99 = 1.64485, 1.95996, 2.57583
    se_safe = np.maximum(se_f, 1e-12)

    cov90 = float(np.mean(np.abs(err) <= z90 * se_safe))
    cov95 = float(np.mean(np.abs(err) <= z95 * se_safe))
    cov99 = float(np.mean(np.abs(err) <= z99 * se_safe))
    mpiw95 = float(np.mean(2 * z95 * se_safe))

    # 95% Winkler Interval Score
    alpha_uq = 0.05
    lower_95 = u_pred - z95 * se_safe
    upper_95 = u_pred + z95 * se_safe
    winkler = (upper_95 - lower_95) + \
              (2.0 / alpha_uq) * np.maximum(lower_95 - u_true, 0.0) + \
              (2.0 / alpha_uq) * np.maximum(u_true - upper_95, 0.0)
    winkler95 = float(np.mean(winkler))

    # Gaussian CRPS against ground truth
    z_std = err / se_safe
    crps = se_safe * (z_std * (2 * norm.cdf(z_std) - 1) + 2 * norm.pdf(z_std) - 1.0 / np.sqrt(np.pi))
    crps_score = float(np.mean(crps))

    # Canberra R^2 on the grid
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum(u_true ** 2) + 1e-12
    r2_grid = 1.0 - (ss_res / ss_tot)

    # Condition number of trainable PDE Gram matrix (if available)
    cond_G_TT = float(np.linalg.cond(model.G_TT))

    return {
        f"R2_{prefix}": r2_grid,
        f"rel_l2_{prefix}": rel_l2,
        f"rel_h1_{prefix}": rel_h1,
        f"rel_h2_{prefix}": rel_h2,
        f"rmse_{prefix}": rmse,
        f"nrmse_{prefix}": nrmse,
        f"linf_{prefix}": linf,
        f"rmse_int_{prefix}": rmse_int,
        f"rmse_bnd_{prefix}": rmse_bnd,
        f"cov90_{prefix}": cov90,
        f"cov95_{prefix}": cov95,
        f"cov99_{prefix}": cov99,
        f"mpiw95_{prefix}": mpiw95,
        f"winkler95_{prefix}": winkler95,
        f"crps_{prefix}": crps_score,
        f"cond_G_TT_{prefix}": cond_G_TT,
    }


def plot_diagnostics(model, x_test, y_test, e_test, true_params, pde_name):
    """
    6-Panel Academic & Industrial Diagnostic Dashboard:
      1. Parity Plot (u* vs \hat{u})
      2. Pointwise Spatial Error Map |u* - \hat{u}|
      3. Standardized Residual Q-Q Plot
      4. Calibration Reliability Curve (Empirical vs. Nominal Coverage)
      5. Spectral Filter Factor Decay (\bar{\omega}_m vs Eigenvalues s_m)
      6. Parameter Forest Plot (\hat{\theta}_T +/- 95% Sandwich CI vs \theta^*)
    """
    u_true = y_test - e_test
    preds = model.predict(x_test)
    u_pred = preds['fhat'].values
    se_mean = preds['se'].values
    err = u_pred - u_true

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), layout='constrained')

    # Panel 1: Parity Plot
    ax1 = axes[0, 0]
    ax1.scatter(u_true, u_pred, alpha=0.4, s=14, color='#1f77b4')
    lims = [min(u_true.min(), u_pred.min()), max(u_true.max(), u_pred.max())]
    ax1.plot(lims, lims, 'r--', lw=1.8, label='Ideal ($y = x$)')
    ax1.set(title="1. Field Parity ($u^*$ vs $\\hat{u}$)", xlabel="True $u^*(\\mathbf{x})$", ylabel="Predicted $\\hat{u}(\\mathbf{x})$")
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend(loc='best')

    # Panel 2: Spatial Absolute Error Map
    ax2 = axes[0, 1]
    if model.dim == 2:
        sc = ax2.scatter(x_test[:, 0], x_test[:, 1], c=np.abs(err), cmap='magma', s=16)
        fig.colorbar(sc, ax=ax2, shrink=0.85, label='$|\\hat{u} - u^*|$')
        ax2.set(xlabel=model.variables[0], ylabel=model.variables[1])
    else:
        idx_s = np.argsort(x_test[:, 0])
        ax2.plot(x_test[idx_s, 0], np.abs(err[idx_s]), color='crimson', lw=1.5, label='$|\\hat{u} - u^*|$')
        ax2.plot(x_test[idx_s, 0], 1.96 * se_mean[idx_s], 'b--', lw=1.2, label='95% CI Half-Width')
        ax2.legend(loc='best')
    ax2.set_title("2. Pointwise Spatial Abs Error Map")
    ax2.grid(True, ls='--', alpha=0.5)

    # Panel 3: Conditional Error Funnel
    ax3 = axes[0, 2]

    # 1. Plot raw errors against their specific predictive standard error
    ax3.scatter(se_mean, err, s=8, color='#1f77b4', alpha=0.4, edgecolors='none', label='Observed Error')

    # 2. Draw theoretical distribution bounds (No division required)
    se_line = np.linspace(0, se_mean.max(), 100)
    ax3.plot(se_line, 1.96 * se_line, color='crimson', ls='--', lw=1.5, label='95% Theoretical Bound')
    ax3.plot(se_line, -1.96 * se_line, color='crimson', ls='--', lw=1.5)

    # 3. Fill the expected nominal region
    ax3.fill_between(se_line, -1.96 * se_line, 1.96 * se_line, color='crimson', alpha=0.08)

    ax3.set_xlabel(r"Predictive Standard Error ($\mathrm{se\_mean}$)", fontweight='bold')
    ax3.set_ylabel(r"Raw Empirical Error ($u - \hat{u}$)", fontweight='bold')
    ax3.set_title("3. Conditional Error Funnel", fontweight='bold')
    ax3.set_xlim(0, se_mean.max() * 1.05)
    ax3.grid(True, ls='--', alpha=0.5)
    ax3.legend(loc="upper left", frameon=True)

    # Panel 4: Calibration Reliability Curve
    ax4 = axes[1, 0]
    nom_grid = np.linspace(0.05, 0.99, 20)
    emp_cov = [np.mean(np.abs(err) <= stats.norm.ppf(1.0 - (1.0 - a) / 2.0) * se_mean) for a in nom_grid]
    ax4.plot(nom_grid, emp_cov, 'o-', color='#2ca02c', lw=2, ms=4, label='Empirical Coverage')
    ax4.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Ideal Calibration')
    ax4.set(title="4. Uncertainty Calibration Curve", xlabel="Nominal Confidence", ylabel="Empirical Coverage", xlim=(0, 1), ylim=(0, 1.02))
    ax4.grid(True, ls='--', alpha=0.5)
    ax4.legend(loc='lower right')

    # Panel 5: Spectral Filter Decay
    ax5 = axes[1, 1]
    s_vals = np.maximum(model.s_, 1e-10)
    w_vals = model.ewa_res_['shrink_weight']
    modes = np.arange(1, len(s_vals) + 1)
    ax5.plot(modes, w_vals, color='darkorange', lw=2, label=r'Filter $\bar{\omega}_m$')
    ax5_t = ax5.twinx()
    ax5_t.plot(modes, s_vals, color='steelblue', ls=':', lw=1.5, label=r'Eigenvalue $s_m$')
    ax5_t.set_yscale('log')
    ax5.set(title="5. Spectral Filter Decay", xlabel="Mode Index $m$")
    ax5.set_ylabel(r"Filter Weight $\bar{\omega}_m$", color='darkorange')
    ax5_t.set_ylabel(r"Penalty Eigenvalue $s_m$", color='steelblue')
    ax5.grid(True, ls='--', alpha=0.5)

    # Panel 6: Parameter Recovery Forest Plot
    ax6 = axes[1, 2]
    T = model._trainable_idx
    if len(T) > 0:
        y_pos = np.arange(len(T))
        ests = model.coef_mean[T]
        se_t = np.sqrt(np.maximum(np.diag(model.coef_var), 1e-16))
        names = [model.coef_keys[idx] for idx in T]
        ax6.errorbar(ests, y_pos, xerr=1.96 * se_t, fmt='o', color='midnightblue', ecolor='royalblue', elinewidth=2.5, capsize=5, label='Estimate ± 95% CI')
        if true_params is not None:
            t_list = [v for v in true_params.values() if isinstance(v, (int, float))]
            if len(t_list) == len(T):
                ax6.scatter(t_list, y_pos, color='crimson', marker='D', s=55, zorder=5, label='True $\\theta^*$')
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(names)
        ax6.set(title="6. PDE Parameter Forest Plot", xlabel="Parameter Value")
        ax6.grid(True, ls='--', alpha=0.5)
        ax6.legend(loc='best')

    fig.suptitle(f"EWABSplinePDE 6-Panel Diagnostic Dashboard: {pde_name}", fontsize=14, fontweight='bold')
    plt.show()

def format_res_summary(
    res: pd.Series,
    model=None,
    true_params: dict = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Formats a flat `res_series` (or mean of `res_df`) into two structured DataFrames:
      1. perf_df  : Side-by-side Train vs. Test performance & calibration table
      2. param_df : Aligned PDE parameter discovery table with SE and 95% CIs
    """

    # 1. Build Train vs. Test Performance Table
    metric_groups = [
        ("Data Fit", "mse", "Observation MSE"),
        ("Data Fit", "R2", "Canberra Residual R2"),
        ("Data Fit", "se_max", "Max Rel. Std Error (SE/sigma)"),
        ("Sobolev Norms", "rel_l2", "Relative L2 Error"),
        ("Sobolev Norms", "rel_h1", "Relative H1 Semi-Norm Error"),
        ("Sobolev Norms", "rel_h2", "Relative H2 Semi-Norm Error"),
        ("Spatial Error", "rmse", "Overall Surface RMSE"),
        ("Spatial Error", "nrmse", "Normalized RMSE (NRMSE)"),
        ("Spatial Error", "linf", "Max Pointwise Error (L_inf)"),
        ("Spatial Error", "rmse_int", "Interior Core RMSE (Inner 80%)"),
        ("Spatial Error", "rmse_bnd", "Boundary Shell RMSE (Outer 10%)"),
        ("UQ Calibration", "cov90", "Empirical 90% CI Coverage"),
        ("UQ Calibration", "cov95", "Empirical 95% CI Coverage"),
        ("UQ Calibration", "cov99", "Empirical 99% CI Coverage"),
        ("UQ Calibration", "mpiw95", "95% Mean Interval Width (MPIW)"),
        ("UQ Calibration", "winkler95", "95% Winkler Interval Score"),
        ("UQ Calibration", "crps", "Gaussian CRPS"),
        ("Conditioning", "cond_G_TT", "Gram Condition Number Cond(G_TT)"),
    ]

    perf_rows = []
    for group, key, label in metric_groups:
        if f"{key}_train" in res or f"{key}_test" in res:
            tr_v = float(res.get(f"{key}_train", np.nan))
            te_v = float(res.get(f"{key}_test", np.nan))
            diff = te_v - tr_v
            rel_pct = (diff / tr_v * 100.0) if (not np.isnan(tr_v) and tr_v != 0) else np.nan
            perf_rows.append({
                "Category": group,
                "Metric": label,
                "Key": key,
                "Train": tr_v,
                "Test": te_v,
                "Gap (Test-Train)": diff,
                "Rel. Diff (%)": rel_pct,
            })

    perf_df = pd.DataFrame(perf_rows).set_index(["Category", "Metric"])

    # 2. Build PDE Parameter Summary Table
    coef_keys_in_res = sorted(
        [k for k in res.index if k.startswith("coef_mean_")],
        key=lambda x: int(x.split("_")[-1])
    )
    K = len(coef_keys_in_res)
    T = model._trainable_idx if (model is not None and hasattr(model, "_trainable_idx")) else None
    op_names = model.coef_keys if (model is not None and hasattr(model, "coef_keys")) else [f"theta_{i}" for i in range(K)]

    # Reconstruct covariance matrix from coef_var_r_c entries
    var_keys = [k for k in res.index if k.startswith("coef_var_")]
    dim_v = int(np.sqrt(len(var_keys))) if var_keys else 0

    if T is None and dim_v < K:
        # Infer fixed index as the entry exactly equal to 1.0
        T = [i for i in range(K) if not np.isclose(res[f"coef_mean_{i}"], 1.0, atol=1e-12)]

    param_rows = []
    for i in range(K):
        est = float(res[f"coef_mean_{i}"])
        if dim_v == K:
            var_i = float(res.get(f"coef_var_{i}_{i}", 0.0))
            role = "Trainable" if (T is None or i in T) else "Locked"
        elif T is not None and i in T:
            t_pos = T.index(i)
            var_i = float(res.get(f"coef_var_{t_pos}_{t_pos}", 0.0))
            role = "Trainable"
        else:
            var_i = 0.0
            role = "Locked"

        se_i = float(np.sqrt(max(var_i, 0.0)))

        true_v = np.nan
        if true_params is not None:
            raw_key = model._raw_str_names[i]
            if raw_key in true_params:
                true_v = float(true_params[raw_key])
            elif op_names[i] in true_params:
                true_v = float(true_params[op_names[i]])

        param_rows.append({
            "Index": f"coef_mean_{i}",
            "Operator": op_names[i],
            "Role": role,
            "Estimate": est,
            "True_Value": true_v,
            "Variance": var_i,
            "Std_Error": se_i,
            "CI_95_Lower": est - 1.96 * se_i,
            "CI_95_Upper": est + 1.96 * se_i,
        })

    param_df = pd.DataFrame(param_rows)
    return perf_df, param_df


def plot_res_summary(
        res: pd.DataFrame | pd.Series,
        param_df: pd.DataFrame | dict = None,
        pde_name: str = "PDE Benchmark",
        save_path: str = None,
        metric_suffix: str = "test",
        annot=False
):
    """
    4-panel publication dashboard with native MultiIndex / comparative group support:
      - Accepts single DataFrame/Series (single-run) OR MultiIndex DataFrame (e.g. pd.concat(res, names=['weak'])).
      - Accepts param_df as a single DataFrame OR dict of DataFrames (e.g. params={'0': p0, '1': p1}).
      - Produces side-by-side dodged boxplots comparing formulations across all 4 panels.
    """
    df = res.to_frame().T if isinstance(res, pd.Series) else res
    df = df.astype(float)

    # =========================================================================
    # 1. MultiIndex Detection & Group Extraction
    # =========================================================================
    is_multi = False
    groups = [None]
    group_labels = ["All"]
    badge_prefixes = [""]
    grp_level_names = []
    grp_levels = []

    if isinstance(df.index, pd.MultiIndex):
        is_multi = True
        # Exclude internal iteration/sample level if present at innermost level
        if df.index.nlevels > 1 and df.index.names[-1] in ('iter', 'iteration', 'rep', None):
            grp_levels = list(range(df.index.nlevels - 1))
        else:
            grp_levels = list(range(df.index.nlevels))

        grp_level_names = [df.index.names[i] or f"Level_{i}" for i in grp_levels]

        # Extract unique group tuples in order of appearance
        if len(grp_levels) == df.index.nlevels:
            groups = list(dict.fromkeys(df.index.tolist()))
        else:
            drop_lvls = [i for i in range(df.index.nlevels) if i not in grp_levels]
            grp_index = df.index.droplevel(drop_lvls)
            groups = list(dict.fromkeys(grp_index.tolist()))

    elif "weak" in df.columns:
        is_multi = True
        groups = list(dict.fromkeys(df["weak"]))
        grp_level_names = ["weak"]

    # =========================================================================
    # 2. Adaptive Group Label Formatting
    # =========================================================================
    palette = ["#1f77b4", "#e6550d", "#2ca02c", "#7b3294", "#d62728", "#8c564b", "#e377c2", "#17becf"]
    n_groups = len(groups)
    group_colors = [palette[i % len(palette)] for i in range(n_groups)]

    def _format_token(name: str, val, short: bool = False):
        """Translates known experimental parameter names into clean typography."""
        name_lower = str(name).lower()
        v_int = int(val) if isinstance(val, (int, float, np.integer, np.floating)) and np.isclose(val,
                                                                                                  int(val)) else val

        if name_lower == "weak":
            if short:
                return "Weak" if v_int == 1 else "Strong"
            return "Weak Form" if v_int == 1 else "Strong Form"

        if name_lower in ("bias_correct", "bias_corrected", "bc"):
            if short:
                return "+BC" if v_int == 1 else ""
            return "Bias-Corrected" if v_int == 1 else "Uncorrected"

        if name_lower in ("quantile", "quantile_based"):
            if short:
                return "+Qtl" if v_int == 1 else "+Uni"
            return "Quantile Knots" if v_int == 1 else "Uniform Knots"

        # Generic parameter fallback
        return f"{name}={v_int}"

    if is_multi:
        palette = ["#1f77b4", "#e6550d", "#2ca02c", "#7b3294"]
        group_colors = [palette[i % len(palette)] for i in range(len(groups))]

        def _format_token(name: str, val, short: bool = False):
            v_int = int(val) if isinstance(val, (int, float, np.integer, np.floating)) and np.isclose(val,
                                                                                                      int(val)) else val
            return f"{name}={v_int}"

        group_labels, badge_prefixes = [], []
        for g in groups:
            vals = g if isinstance(g, (tuple, list)) else [g]
            group_labels.append(" | ".join(_format_token(n, v, False) for n, v in zip(grp_level_names, vals)))
            b_tokens = [_format_token(n, v, True) for n, v in zip(grp_level_names, vals)]
            badge_prefixes.append(f"{''.join(b_tokens).strip('+')}: ")

    width = 0.35 if n_groups == 2 else (0.75 / n_groups if n_groups > 1 else 0.45)
    offsets = np.linspace(-width * (n_groups - 1) / 2, width * (n_groups - 1) / 2, n_groups) if n_groups > 1 else [0.0]

    def get_df_group(g):
        if isinstance(df.index, pd.MultiIndex):
            lvl_arg = grp_levels[0] if len(grp_levels) == 1 else grp_levels
            return df.xs(g, level=lvl_arg)
        else:
            return df

    # Helper to extract metrics safely
    def get_val(sub_df: pd.DataFrame, key: str) -> np.ndarray:
        for k in [f"{key}_{metric_suffix}", f"{key}_grid", f"{key}_test", key]:
            if k in sub_df.columns:
                vals = sub_df[k].dropna().values
                if len(vals) > 0:
                    return vals
        return np.array([0.0])

    # Helper to resolve param_df for each group
    def get_param_df_group(g):
        if isinstance(param_df, dict):
            if g in param_df:
                return param_df[g]
            elif str(g) in param_df:
                return param_df[str(g)]
            return next(iter(param_df.values()), None)
        return param_df

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")

    # =========================================================================
    # Panel A: Sobolev Relative Error Hierarchy (%)
    # =========================================================================
    ax1 = axes[0, 0]
    sob_items = [("rel_l2", "Rel. $L^2$ Error"), ("rel_h1", "Rel. $H^1$ Error"), ("rel_h2", "Rel. $H^2$ Error")]
    x = np.arange(len(sob_items))
    all_max_sob = 1.0

    for g_idx, g in enumerate(groups):
        df_g = get_df_group(g)
        sob_vals = [get_val(df_g, k) * 100 for k, _ in sob_items]
        pos = x + offsets[g_idx]

        bp1 = ax1.boxplot(
            sob_vals,
            positions=pos,
            widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=group_colors[g_idx], alpha=0.75, edgecolor="black", linewidth=0.9),
            medianprops=dict(color="darkred" if g_idx == 0 else "black", linewidth=1.6),
            flierprops=dict(marker="o", markersize=3, alpha=0.4)
        )

        for i, data in enumerate(sob_vals):
            med = np.median(data)
            all_max_sob = max(all_max_sob, np.max(data) if len(data) > 0 else 1.0)
            ha = "right" if (is_multi and g_idx == 0) else ("left" if is_multi else "center")
            x_shift = -3 if (is_multi and g_idx == 0) else (3 if is_multi else 0)
            if annot:
                ax1.annotate(
                    f"{med:.3f}%",
                    xy=(pos[i], med),
                    xytext=(x_shift, 4),
                    textcoords="offset points",
                    ha=ha,
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=group_colors[g_idx] if is_multi else "#12466b"
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in sob_items], fontsize=10)
    ax1.set_ylabel("Relative Error (%)", fontweight="bold")
    ax1.set_ylim(0, max(all_max_sob * 1.3, 0.5))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_title("A. Sobolev Relative Error Hierarchy ($L^2, H^1, H^2$)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    if is_multi:
        proxies = [Patch(facecolor=group_colors[i], alpha=0.75, label=group_labels[i]) for i in range(n_groups)]
        ax1.legend(handles=proxies, loc="upper left", frameon=True, facecolor="white")

    # =========================================================================
    # Panel B: Spatial Error Decomposition & Interval Sharpness
    # =========================================================================
    ax2 = axes[0, 1]
    sp_items = [
        ("crps", "CRPS Score"),
        ("rmse_bnd", "Boundary RMSE (Outer 10%)"),
        ("rmse", "Overall RMSE"),
        ("rmse_int", "Interior RMSE (Inner 80%)"),
        ("mpiw95", "95% MPIW (Interval Width)"),
        ("linf", "Max Error ($L^\\infty$)"),
        ("winkler95", "Winkler Score (95%)"),
    ]
    y_pos = np.arange(len(sp_items))
    all_max_sp = 0.01

    for g_idx, g in enumerate(groups):
        df_g = get_df_group(g)
        sp_vals = [get_val(df_g, k) for k, _ in sp_items]
        pos = y_pos + offsets[g_idx]

        bp2 = ax2.boxplot(
            sp_vals,
            positions=pos,
            vert=False,
            widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=group_colors[g_idx], alpha=0.75, edgecolor="black", linewidth=0.9),
            medianprops=dict(color="darkred" if g_idx == 0 else "black", linewidth=1.6),
            flierprops=dict(marker="o", markersize=3, alpha=0.4)
        )

        for i, data in enumerate(sp_vals):
            med = np.median(data)
            all_max_sp = max(all_max_sp, np.max(data) if len(data) > 0 else 0.01)
            y_shift = -3 if (is_multi and g_idx == 0) else (3 if is_multi else 0)
            va = "top" if (is_multi and g_idx == 0) else ("bottom" if is_multi else "center")
            if annot:
                ax2.annotate(
                    f"{med:.4f}",
                    xy=(med, pos[i]),
                    xytext=(4, y_shift),
                    textcoords="offset points",
                    ha="left",
                    va=va,
                    fontsize=7.5,
                    fontweight="bold",
                    color=group_colors[g_idx] if is_multi else "black"
                )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl in sp_items], fontsize=9)
    ax2.set_xlim(0, all_max_sp * 1.3)
    ax2.set_xlabel("Magnitude (Field Units)", fontweight="bold")
    ax2.set_title("B. Spatial Error Decomposition & Interval Sharpness", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    if is_multi:
        proxies = [Patch(facecolor=group_colors[i], alpha=0.75, label=group_labels[i]) for i in range(n_groups)]
        ax2.legend(handles=proxies, loc="lower right", frameon=True, facecolor="white")

    # =========================================================================
    # Panel C: Empirical vs. Nominal Coverage Calibration
    # =========================================================================
    ax3 = axes[1, 0]
    nom = np.array([90.0, 95.0, 99.0])
    xc = np.arange(3)
    min_emp = 80.0

    # Draw shared target lines
    ax3.scatter(xc, nom, color="#2ca02c", s=70, zorder=5, marker="D", label="Nominal Target")
    for i in range(len(nom)):
        span = width * n_groups / 2 + 0.1
        ax3.hlines(nom[i], xc[i] - span, xc[i] + span, colors="#2ca02c", linestyles="--", linewidth=1.5, zorder=4)

    for g_idx, g in enumerate(groups):
        df_g = get_df_group(g)
        emp = [get_val(df_g, k) * 100 for k in ["cov90", "cov95", "cov99"]]
        pos = xc + offsets[g_idx]

        bp3 = ax3.boxplot(
            emp,
            positions=pos,
            widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=group_colors[g_idx], alpha=0.75, edgecolor="black", linewidth=0.9),
            medianprops=dict(color="black", linewidth=1.6),
            flierprops=dict(marker="o", markersize=3, alpha=0.4)
        )

        for i, data in enumerate(emp):
            med = np.median(data)
            min_emp = min(min_emp, np.min(data) if len(data) > 0 else 80.0)
            ha = "right" if (is_multi and g_idx == 0) else ("left" if is_multi else "center")
            x_shift = -3 if (is_multi and g_idx == 0) else (3 if is_multi else 0)
            if annot:
                ax3.annotate(
                    f"{med:.1f}%",
                    xy=(pos[i], med),
                    xytext=(x_shift, 4),
                    textcoords="offset points",
                    ha=ha,
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=group_colors[g_idx] if is_multi else "black"
                )

    ax3.set_xticks(xc)
    ax3.set_xticklabels(["90% CI", "95% CI", "99% CI"], fontsize=10)
    ax3.set_ylim(max(0.0, min(50.0, min_emp - 5)), 105.0)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.set_ylabel("Coverage Probability (%)", fontweight="bold")
    ax3.set_title("C. Uncertainty Interval Calibration ($90\\%, 95\\%, 99\\%$)", fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    c_proxies = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c", markersize=7, label="Nominal Target")]
    if is_multi:
        c_proxies.extend([Patch(facecolor=group_colors[i], alpha=0.75, label=group_labels[i]) for i in range(n_groups)])
    ax3.legend(handles=c_proxies, loc="lower left", frameon=True, facecolor="white")

    # =========================================================================
    # Panel D: Parameter Z-Score Distribution & Empirical 95% Coverage
    # =========================================================================
    ax4 = axes[1, 1]
    ref_param_df = get_param_df_group(groups[0])

    # 1. Identify trainable parameters from ref_param_df (or infer from DataFrame columns)
    trainable_descriptors = []
    if ref_param_df is not None and not ref_param_df.empty:
        trainable_rows = ref_param_df[
            ref_param_df["Role"].astype(str).str.lower() == "trainable"
            ]
        for t_pos, (_, row) in enumerate(trainable_rows.iterrows()):
            idx_str = str(row["Index"])
            param_idx = (
                int(idx_str.split("_")[-1])
                if "_" in idx_str
                else int(row.name if isinstance(row.name, int) else t_pos)
            )
            op_label = row["Operator"]
            true_v = (
                float(row["True_Value"])
                if ("True_Value" in row and pd.notna(row["True_Value"]))
                else np.nan
            )

            lbl = (
                f"{op_label} ($\\theta^*={true_v:.3g}$)"
                if not np.isnan(true_v)
                else f"{op_label}"
            )
            trainable_descriptors.append((param_idx, t_pos, lbl, true_v))
    else:
        # Fallback: Infer trainable indices from coef_mean_ and coef_var_ columns
        var_keys = [k for k in df.columns if k.startswith("coef_var_")]
        dim_v = int(np.sqrt(len(var_keys))) if var_keys else 0
        coef_cols = sorted(
            [c for c in df.columns if c.startswith("coef_mean_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        for t_pos in range(min(dim_v, len(coef_cols))):
            param_idx = int(coef_cols[t_pos].split("_")[-1])
            trainable_descriptors.append(
                (param_idx, t_pos, f"$\\theta_{{{param_idx}}}$", np.nan)
            )

    n_params = len(trainable_descriptors)
    y_p = np.arange(n_params)

    # 2. Reference targets: ideal zero line, 95% nominal bounds, and shaded coverage band
    ax4.axvline(
        0.0, color="crimson", ls="-", lw=1.5, zorder=3, label="Target ($z=0$)"
    )
    ax4.axvline(
        -1.96,
        color="black",
        ls="--",
        lw=1.2,
        alpha=0.7,
        zorder=2,
        label=r"Nominal 95% ($\pm 1.96$)",
    )
    ax4.axvline(1.96, color="black", ls="--", lw=1.2, alpha=0.7, zorder=2)
    ax4.axvspan(-1.96, 1.96, color="#2ca02c", alpha=0.08, zorder=1)

    # Collect z-score arrays across groups and parameters: shape (n_groups, n_params)
    all_z_groups = []
    all_se_groups = []
    global_min_z, global_max_z = -2.5, 2.5

    for g_idx, g in enumerate(groups):
        df_g = get_df_group(g)
        p_df_g = get_param_df_group(g)

        # Determine covariance layout: active-only (|T| x |T|) vs full (K x K)
        var_keys = [k for k in df_g.columns if k.startswith("coef_var_")]
        dim_v = int(np.sqrt(len(var_keys))) if var_keys else 0

        group_z_list = []
        group_se_list = []
        for param_idx, t_pos, _, true_v in trainable_descriptors:
            # (a) Extract estimates
            est_key = f"coef_mean_{param_idx}"
            if est_key in df_g.columns:
                ests = df_g[est_key].dropna().values
            elif p_df_g is not None and not p_df_g.empty:
                ests = np.array([float(p_df_g.iloc[param_idx]["Estimate"])])
            else:
                ests = np.array([0.0])

            # (b) Extract sandwich standard errors
            v_key_active = f"coef_var_{t_pos}_{t_pos}"
            v_key_full = f"coef_var_{param_idx}_{param_idx}"

            if v_key_active in df_g.columns:
                vars_ = df_g[v_key_active].dropna().values
            elif v_key_full in df_g.columns:
                vars_ = df_g[v_key_full].dropna().values
            elif p_df_g is not None and not p_df_g.empty:
                v_val = float(p_df_g.iloc[param_idx].get("Variance", 0.0))
                if v_val == 0.0 and "Std_Error" in p_df_g.iloc[param_idx]:
                    v_val = float(p_df_g.iloc[param_idx]["Std_Error"]) ** 2
                vars_ = np.array([v_val])
            else:
                vars_ = np.ones_like(ests)

            min_len = min(len(ests), len(vars_))
            ests = ests[:min_len]
            ses = np.sqrt(np.maximum(vars_[:min_len], 1e-16))

            # (c) Calculate studentized z-scores
            if not np.isnan(true_v):
                z = (ests - true_v) / np.maximum(ses, 1e-12)
            else:
                z = ests / np.maximum(ses, 1e-12)

            group_z_list.append(z)
            group_se_list.append(ses)
            if len(z) > 0:
                global_min_z = min(global_min_z, float(np.min(z)))
                global_max_z = max(global_max_z, float(np.max(z)))

        all_z_groups.append(group_z_list)
        all_se_groups.append(group_se_list)

        # 3. Draw horizontal boxplots for group g
        pos = y_p + offsets[g_idx]
        bp4 = ax4.boxplot(
            group_z_list,
            positions=pos,
            vert=False,
            widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(
                facecolor=group_colors[g_idx],
                alpha=0.75,
                edgecolor="black",
                linewidth=0.9,
            ),
            medianprops=dict(color="black", linewidth=1.6),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )

    # 4. Annotate coverage probability and median z-score per parameter
    for i in range(n_params):
        # Determine flush x-anchor beyond upper whiskers across all groups
        row_whiskers = []
        for g_idx in range(n_groups):
            zd = all_z_groups[g_idx][i]
            if len(zd) > 0:
                q75 = np.percentile(zd, 75)
                q25 = np.percentile(zd, 25)
                filtered = zd[zd <= q75 + 1.5 * (q75 - q25)]
                row_whiskers.append(np.max(filtered) if len(filtered) > 0 else 1.96)
            else:
                row_whiskers.append(1.96)
        x_anchor = max(max(row_whiskers), 2.15)

        for g_idx in range(n_groups):
            zd = all_z_groups[g_idx][i]
            pos_y = y_p[i] + offsets[g_idx]
            df_g_curr = get_df_group(groups[g_idx])
            n_obs = int(df_g_curr['n_obs'].iloc[0])

            if len(zd) > 0:
                med = np.median(zd)
                cov = np.mean(np.abs(zd) <= 1.95996)
                med_se_norm = np.median(all_se_groups[g_idx][i]) * np.sqrt(n_obs)
            else:
                med, cov, med_se_norm = 0.0, 0.0, 0.0

            grp_prefix = badge_prefixes[g_idx] if is_multi else ""
            badge_text = rf"Cov={cov:.0%}, aSE={med_se_norm:.3g}"

            ax4.annotate(
                badge_text,
                xy=(x_anchor, pos_y),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color=group_colors[g_idx],
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor=group_colors[g_idx],
                    alpha=0.85,
                ),
            )

    # 5. Styling, limits, and unified legend
    ax4.set_yticks(y_p)
    ax4.set_yticklabels([desc[2] for desc in trainable_descriptors], fontsize=10)
    ax4.set_xlabel(
        r"Parameter Z-Score $(\hat{\theta} - \theta^*) / \mathrm{SE}(\hat{\theta})$",
        fontweight="bold",
    )
    ax4.set_title(
        "D. Trainable Parameter Z-Score & 95% CI Coverage", fontweight="bold", pad=10
    )
    ax4.grid(axis="x", ls="--", alpha=0.5)

    # Set x-limits with room for the annotation badges
    x_lim_left = min(global_min_z * 1.15, -3.2)
    x_lim_right = max(global_max_z * 1.25, 5.5)
    ax4.set_xlim(x_lim_left, x_lim_right)

    d_proxies = [
        plt.Line2D(
            [0], [0], color="crimson", ls="-", lw=1.5, label="Target ($z=0$)"
        ),
        plt.Line2D(
            [0], [0], color="black", ls="--", lw=1.2, label=r"95% ($\pm 1.96$)"
        ),
    ]
    if is_multi:
        d_proxies.extend([
            Patch(facecolor=group_colors[i], alpha=0.75, label=group_labels[i])
            for i in range(n_groups)
        ])
    ax4.legend(
        handles=d_proxies,
        loc="lower left",
        fontsize=8,
        frameon=True,
        facecolor="white",
    )

    title_suffix = f" (Comparing {', '.join(grp_level_names)})" if is_multi else ""
    fig.suptitle(f"EWABSplinePDE Summary Statistics: {pde_name}{title_suffix}", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def run_pde_experiment(
        pde_name,
        dim,
        variables,
        pde_params,
        true_sol_with_derivatives,
        boundary_conditions,
        setup_config_func,
        x_max_size=1_000_000,
        oversample=100
):
    """
    Generalized runner for PDE parameter discovery experiments.
    Handles arg parsing, data generation, training loops, and metrics.
    """
    parser = argparse.ArgumentParser(description=f"Run {dim}D {pde_name} PDE")
    parser.add_argument("-n", "--sample_size", type=int, default=1000)
    parser.add_argument("-sf", "--sigma_frac", type=float, default=0.1, help="Noise to Signal Ratio")
    parser.add_argument("-a", "--alpha", type=float, default=.05)
    parser.add_argument("-q", "--quantile", type=str, default='0')
    parser.add_argument("-i", "--iteration", type=int, default=100)
    parser.add_argument("-s", "--save", type=str, default='1')
    parser.add_argument("-d", "--debug", type=str, default='0')
    parser.add_argument("-w", "--weak_form", type=str, default='0')
    parser.add_argument("-b", "--bias_corrected", type=str, default='1')

    for p_name, p_dict in pde_params.items():
        p_flag = p_dict['flag']
        arg_kwargs = {k: v for k, v in p_dict.items() if k != 'flag'}
        parser.add_argument(f"-{p_name}", f"--{p_flag}", **arg_kwargs)

    args = parser.parse_args()

    n = args.sample_size
    sigma_frac = args.sigma_frac
    alpha = args.alpha
    n_iter = args.iteration

    save = to_bool(args.save)
    debug = to_bool(args.debug)
    quantile_based = to_bool(args.quantile)
    weak_form = to_bool(args.weak_form)
    bias_corrected = to_bool(args.bias_corrected)

    if debug:
        n_iter = 1

    # Extract parsed PDE Params
    params = {p_name: getattr(args, p_dict['flag']) for p_name, p_dict in pde_params.items()}

    # Calculate required knots and metadata specific to this PDE
    domain, pde_kwargs, true_params, num_int_knots = setup_config_func(params)
    if any(['operator' in k.lower() for k in pde_kwargs.keys()]):
        MOD = FunctionalEWABSplinePDE
        weak_form = True  # TODO?
    else:
        MOD = EWABSplinePDE

    # print(f'Sample Size: {n}')

    # Grid Generation
    grid_size_1d = int(min(x_max_size ** (1 / dim), oversample * n))
    grids = [np.linspace(*d, grid_size_1d) for d in domain]
    x = tensor_prod(*grids)

    y_derivs = true_sol_with_derivatives(x, **params)
    y = next(iter(y_derivs.values()))
    sigma_scaled = np.std(y) * sigma_frac

    # boundary_conditions(**params, domain=domain)
    bc_dict = get_bc_dict(true_sol_with_derivatives, domain, boundary_conditions=boundary_conditions, **params)
    fname = f'Bayes{pde_name}' + str(args).replace('Namespace', ' ') + '.csv'

    def run_one(i):
        rng = np.random.default_rng(i)
        noise = rng.normal(size=len(x)) * sigma_scaled
        y_noise = y + noise

        # Train Split
        index_train = rng.choice(range(len(x)), n)
        index_train = np.sort(index_train)
        x_train, y_train, e_train = x[index_train], y_noise[index_train], noise[index_train]

        # Test Split
        index_test = rng.choice(range(len(x)), n)
        index_test = np.sort(index_test)
        x_test, y_test, e_test = x[index_test], y_noise[index_test], noise[index_test]

        if debug:
            dataset = pd.DataFrame({'x': x_train[:, 0], 'y': x_train[:, 1], 'z': y_train})
            dataset.to_csv(f'/Users/cheungjustin/PDE/EWA/dataset/{pde_name}.csv')

        # print(f"True Sigma: {sigma_scaled}")
        model = MOD(dim=dim, variables=variables, domain=domain, boundary_conditions=bc_dict,
                              alpha=alpha, quantile_based=quantile_based, num_int_knots=num_int_knots,
                              total_grid_size=10000, weak_form=weak_form, bias_corrected=bias_corrected,
                              **pde_kwargs)
        model.fit(x_train, y_train)


        y_pred_train = model.predict(x_train)['fhat']
        y_pred_test = model.predict(x_test)['fhat']
        res_series = pd.Series({
            'n_obs': n,
            'mse_train': float(np.mean((y_train - y_pred_train) ** 2)),
            'mse_test': float(np.mean((y_test - y_pred_test) ** 2)),
        })

        grid_metrics = evaluate_metrics(
            model=model,
            domain=domain,
            true_sol_func=true_sol_with_derivatives,
            true_params=params,
            prefix="test"  # Uses prefix='test' to maintain downstream plot compatibility
        )

        for k, v in grid_metrics.items():
            res_series[k] = v

        coef_mean, coef_var = model.coef_mean, model.coef_var
        for idx_c, m_i in enumerate(coef_mean):
            res_series[f'coef_mean_{idx_c}'] = m_i
        for (idx_r, idx_c) in np.ndindex(coef_var.shape):
            res_series[f'coef_var_{idx_r}_{idx_c}'] = coef_var[idx_r, idx_c]

        if debug:
            model.plot_derivatives(deriv_orders_list=[(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)],
                                   true_deriv_func=true_sol_with_derivatives,
                                   true_deriv_kwargs=params)

            # 1. Build the aligned parameter summary table
            param_df = align_pde_params(true_params, model=model)
            print("\n=== Aligned PDE Parameter Discovery Summary ===")
            print(param_df.to_string(index=False, float_format=lambda x: f"{x:.4e}"))

            plot_diagnostics(model, x_test, y_test, e_test, true_params, pde_name)
            # model.plot_fit_stats()
            # model.galerkin_force_diagnostics()
            # model.galerkin_penalty_diagnostics()

            trainable_true_vals = param_df.iloc[model._trainable_idx]['True_Value'].values
            model.plot_coef_dist(true_vals=trainable_true_vals)

            # Plot True Solution (If 2D)
            # if debug and (dim == 2):
            #     X_grid = x[:, 0].reshape(grid_size_1d, grid_size_1d)
            #     Y_grid = x[:, 1].reshape(grid_size_1d, grid_size_1d)
            #     Z_grid = y.reshape(grid_size_1d, grid_size_1d)
            #     vmax = np.abs(Z_grid).max()
            #
            #     fig, ax = util.contour_plot(X_grid, Y_grid, Z_grid, vmin=-vmax, vmax=vmax)
            #
            #     title_str = f"True Solution for {pde_name} + {true_params}"
            #     ax.set(title=title_str, xlabel=variables[0], ylabel=variables[1])
            #     plt.show()

            # Plot Fitted
            #model.plot_fit(x_test=x_test, y_test=y_test)
            model.plot_lambda_weight()
            model.plot_eigenvalues()
            model.plot_basis_functions()

        return res_series

    t0 = time.time()
    res_list = parallel(run_one, range(n_iter), disable_par=debug)
    res_df = pd.concat(res_list, names=['iter'], axis=1).T

    # --- Summarize & Plot Monte Carlo Aggregate Results (when n_iter > 1) ---
    final_param_df = align_pde_params(
        true_params,
        res=res_df,
        pde_kwargs=pde_kwargs,
        variables=variables,
        dim=dim,
        model_class=MOD
    )
    print(f"\n================ {pde_name} Parameter Recovery Table ================")
    print(final_param_df.to_string(index=False, float_format=lambda x: f"{x:.4e}"))

    if save:
        # Falls back to local output directory if absolute path is missing
        out_dir = Path('/Users/cheungjustin/PDE/EWA/output/')
        if not out_dir.exists():
            out_dir = Path('./output')
            out_dir.mkdir(exist_ok=True)

        res_df.to_csv(out_dir / fname)
        print(f'{fname} written to {out_dir}')

    t1 = time.time()
    print(f"Elapsed Time: {t1 - t0:.2f}s")
    mode_str = "VARIATIONAL" if weak_form else "PDE"
    print(f'PDE Mode: {mode_str} | Meta Info: {true_params}')

    return res_df, final_param_df