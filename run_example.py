import argparse
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from Bsplines import tensor_prod
from ewa import EWABSplinePDE
from ewa_op import FunctionalEWABSplinePDE
from plot import plot_diagnostics, plot_res_summary  # noqa: F401
from util import parallel


def to_bool(arg):
    assert arg in ['0', '1']
    return True if arg == '1' else False


def get_bc_dict(true_sol_with_derivatives, domain, boundary_conditions='neumann', **kwargs):
    """
    Constructs the 1st-order derivative (Neumann) or 0th-order (Dirichlet) boundary conditions.
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
    Works BOTH inside run_one (using model) AND outside run_one (using res_df + pde_kwargs).
    """
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

    # 2. Assemble aligned summary table
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
            var_i = 0.0

        se_i = float(np.sqrt(max(var_i, 0.0)))
        est_val = float(coef_mean[i])
        true_val = true_params.get(raw_key, np.nan)

        if not is_trainable and np.isnan(true_val):
            true_val = est_val

        if not np.isnan(true_val):
            abs_err = abs(est_val - true_val)
            rel_err = np.nan if abs(true_val) < 1e-6 else abs_err / abs(true_val)
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
            'Covered_95': covered,
        })

    return pd.DataFrame(rows)


def evaluate_metrics(
    model: EWABSplinePDE,
    domain: List[Tuple[float, float]],
    true_sol_func: Callable,
    true_params: dict,
    prefix: str = "grid",
) -> Dict[str, float]:
    """
    Evaluates field approximation, continuous Sobolev norms, and uncertainty
    calibration on a fixed, equally-spaced Cartesian grid using the true solution.
    """
    x_grid = model.gridpoints_
    n_eval = len(x_grid)

    true_dict = true_sol_func(x_grid, **true_params)
    u_true = true_dict[(0, 0)] if isinstance(true_dict, dict) else true_dict

    pred_res = model.predict(x_grid)
    u_pred, se_f = pred_res['fhat'], pred_res['se']

    err = u_pred - u_true
    l2_denom = np.linalg.norm(u_true)

    rel_l2 = np.nan if l2_denom < 1e-6 else np.linalg.norm(err) / l2_denom
    rmse = np.sqrt(np.mean(err ** 2))
    nrmse = rmse / (np.max(u_true) - np.min(u_true) + 1e-12)
    linf = np.max(np.abs(err))

    # H^1 and H^2 Sobolev Semi-Norm Relative Errors
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

    # Interior vs. Boundary Shell Partitioning
    bnd_mask = np.zeros(n_eval, dtype=bool)
    for p, (a, b) in enumerate(domain):
        margin = 0.10 * (b - a)
        bnd_mask |= (x_grid[:, p] <= a + margin) | (x_grid[:, p] >= b - margin)
    int_mask = ~bnd_mask

    rmse_int = np.sqrt(np.mean(err[int_mask] ** 2)) if np.any(int_mask) else np.nan
    rmse_bnd = np.sqrt(np.mean(err[bnd_mask] ** 2)) if np.any(bnd_mask) else np.nan

    # Uncertainty Quantification Calibration (vs. True Mean Surface)
    z90, z95, z99 = 1.64485, 1.95996, 2.57583
    se_safe = np.maximum(se_f, 1e-12)

    cov90 = float(np.mean(np.abs(err) <= z90 * se_safe))
    cov95 = float(np.mean(np.abs(err) <= z95 * se_safe))
    cov99 = float(np.mean(np.abs(err) <= z99 * se_safe))
    mpiw95 = float(np.mean(2 * z95 * se_safe))

    alpha_uq = 0.05
    lower_95 = u_pred - z95 * se_safe
    upper_95 = u_pred + z95 * se_safe
    winkler = (
        (upper_95 - lower_95)
        + (2.0 / alpha_uq) * np.maximum(lower_95 - u_true, 0.0)
        + (2.0 / alpha_uq) * np.maximum(u_true - upper_95, 0.0)
    )
    winkler95 = float(np.mean(winkler))

    z_std = err / se_safe
    crps = se_safe * (z_std * (2 * norm.cdf(z_std) - 1) + 2 * norm.pdf(z_std) - 1.0 / np.sqrt(np.pi))
    crps_score = float(np.mean(crps))

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum(u_true ** 2) + 1e-12
    r2_grid = 1.0 - (ss_res / ss_tot)

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


def format_res_summary(
    res: pd.Series,
    model=None,
    true_params: dict = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Formats a flat `res_series` (or mean of `res_df`) into two structured DataFrames:
      1. perf_df  : Side-by-side Train vs. Test performance & calibration table
      2. param_df : Aligned PDE parameter discovery table with SE and 95% CIs
    """
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

    coef_keys_in_res = sorted(
        [k for k in res.index if k.startswith("coef_mean_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    K = len(coef_keys_in_res)
    T = model._trainable_idx if (model is not None and hasattr(model, "_trainable_idx")) else None
    op_names = (
        model.coef_keys
        if (model is not None and hasattr(model, "coef_keys"))
        else [f"theta_{i}" for i in range(K)]
    )

    var_keys = [k for k in res.index if k.startswith("coef_var_")]
    dim_v = int(np.sqrt(len(var_keys))) if var_keys else 0

    if T is None and dim_v < K:
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


def run_pde_experiment(
    pde_name,
    dim,
    variables,
    pde_params,
    true_sol_with_derivatives,
    boundary_conditions,
    setup_config_func,
    x_max_size=1_000_000,
    oversample=100,
):
    """
    Generalized runner for PDE parameter discovery experiments.
    Handles arg parsing, data generation, training loops, and metrics.
    """
    parser = argparse.ArgumentParser(description=f"Run {dim}D {pde_name} PDE")
    parser.add_argument("-n", "--sample_size", type=int, default=1000)
    parser.add_argument("-sf", "--sigma_frac", type=float, default=0.1, help="Noise to Signal Ratio")
    parser.add_argument("-deg", "--degree", type=int, default=3)
    parser.add_argument("-a", "--alpha", type=float, default=.05)
    parser.add_argument("-q", "--quantile", type=str, default='0')
    parser.add_argument("-i", "--iteration", type=int, default=100)
    parser.add_argument("-s", "--save", type=str, default='1')
    parser.add_argument("-o", "--output_dir", type=str, default='output', help="Directory for saving experiment CSVs")
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
    deg = args.degree
    alpha = args.alpha
    n_iter = args.iteration

    save = to_bool(args.save)
    debug = to_bool(args.debug)
    quantile_based = to_bool(args.quantile)
    weak_form = to_bool(args.weak_form)
    bias_corrected = to_bool(args.bias_corrected)

    if debug:
        n_iter = 1

    params = {p_name: getattr(args, p_dict['flag']) for p_name, p_dict in pde_params.items()}

    domain, pde_kwargs, true_params = setup_config_func(params)
    if any(['operator' in k.lower() for k in pde_kwargs.keys()]):
        MOD = FunctionalEWABSplinePDE
        weak_form = True
    else:
        MOD = EWABSplinePDE

    grid_size_1d = int(min(x_max_size ** (1 / dim), oversample * n))
    grids = [np.linspace(*d, grid_size_1d) for d in domain]
    x = tensor_prod(*grids)

    y_derivs = true_sol_with_derivatives(x, **params)
    y = next(iter(y_derivs.values()))
    sigma_scaled = np.std(y) * sigma_frac

    bc_dict = get_bc_dict(true_sol_with_derivatives, domain, boundary_conditions=boundary_conditions, **params)
    args_for_name = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != 'output_dir'})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_one(i):
        rng = np.random.default_rng(i)
        noise = rng.normal(size=len(x)) * sigma_scaled
        y_noise = y + noise

        index_train = np.sort(rng.choice(range(len(x)), n))
        x_train, y_train, e_train = x[index_train], y_noise[index_train], noise[index_train]

        index_test = np.sort(rng.choice(range(len(x)), n))
        x_test, y_test, e_test = x[index_test], y_noise[index_test], noise[index_test]

        if debug:
            dataset_dir = Path("dataset")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset = pd.DataFrame(x_train, columns=variables)
            dataset['z'] = y_train
            dataset.to_csv(dataset_dir / f"{pde_name}.csv", index=False)

        model = MOD(
            dim=dim, variables=variables, domain=domain, boundary_conditions=bc_dict,
            alpha=alpha, quantile_based=quantile_based, deg=deg,
            total_grid_size=10000, weak_form=weak_form, bias_corrected=bias_corrected,
            **pde_kwargs,
        )
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
            prefix="test",
        )
        for k, v in grid_metrics.items():
            res_series[k] = v

        coef_mean, coef_var = model.coef_mean, model.coef_var
        for idx_c, m_i in enumerate(coef_mean):
            res_series[f'coef_mean_{idx_c}'] = m_i
        for (idx_r, idx_c) in np.ndindex(coef_var.shape):
            res_series[f'coef_var_{idx_r}_{idx_c}'] = coef_var[idx_r, idx_c]

        if debug:
            model.plot_derivatives(
                deriv_orders_list=[(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)],
                true_deriv_func=true_sol_with_derivatives,
                true_deriv_kwargs=params,
            )
            model.plot_basis_functions()
            model.plot_lambda_weight()
            plot_diagnostics(model, x_test, y_test, e_test, true_params, pde_name)

        return res_series

    t0 = time.time()
    res_list = parallel(run_one, range(n_iter), disable_par=debug)
    res_df = pd.concat(res_list, names=['iter'], axis=1).T

    final_param_df = align_pde_params(
        true_params,
        res=res_df,
        pde_kwargs=pde_kwargs,
        variables=variables,
        dim=dim,
        model_class=MOD,
    )
    print(f"\n================ {pde_name} Parameter Recovery Table ================")
    print(final_param_df.to_string(index=False, float_format=lambda x: f"{x:.4e}"))

    if save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = f'{pde_name}_{timestamp}'
        csv_path = out_dir / f'{stem}.csv'
        suffix = 1
        while csv_path.exists():
            stem = f'{pde_name}_{timestamp}_{suffix}'
            csv_path = out_dir / f'{stem}.csv'
            suffix += 1

        fname = csv_path.name
        meta_fname = f'{stem}_meta.json'
        meta_path = out_dir / meta_fname

        res_df.to_csv(csv_path)
        metadata = {
            'pde_name': pde_name,
            'timestamp': timestamp,
            'output_file': fname,
            'args': vars(args_for_name),
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f'{fname} written to {out_dir}')
        print(f'{meta_fname} written to {out_dir}')

    t1 = time.time()
    print(f"Elapsed Time: {t1 - t0:.2f}s")
    mode_str = "VARIATIONAL" if weak_form else "PDE"
    print(f'PDE Mode: {mode_str} | Meta Info: {true_params}')

    return res_df, final_param_df
