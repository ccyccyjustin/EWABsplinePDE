import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
from pathlib import Path

import ewa
import util
from ewa import EWABSplinePDE
from util import canberra_similarity, parallel, tensor_prod, CMAP


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
    action = to_bool(args.weak_form)

    if debug:
        n_iter = 1

    # Extract parsed PDE Params
    params = {p_name: getattr(args, p_dict['flag']) for p_name, p_dict in pde_params.items()}

    # Calculate required knots and metadata specific to this PDE
    domain, pde_kwargs, action_kwargs, true_params, num_int_knots = setup_config_func(params)

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
    kwargs = action_kwargs if action else pde_kwargs
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
        model = EWABSplinePDE(dim=dim, variables=variables, domain=domain, boundary_conditions=bc_dict,
                              alpha=alpha, quantile_based=quantile_based, num_int_knots=num_int_knots,
                              total_grid_size=10000, norm_omega=False,
                              sigma=None, #sigma_scaled, #TODO!!!!!
                              **kwargs)
        model.fit(x_train, y_train)

        def get_metrics(x_ev, y_ev, e_ev):
            preds = model.predict(x_ev)
            resid = y_ev - preds['fhat']
            mse = np.mean(resid ** 2)
            R2 = canberra_similarity(resid, e_ev)
            prob = ((preds['ci_lower'] <= y_ev) & (y_ev <= preds['ci_upper'])).mean()
            se_max = preds['se'].max() / sigma_scaled
            return mse, R2, prob, se_max

        mse_train, R2_train, prob_train, sem_train = get_metrics(x_train, y_train, e_train)
        mse_test, R2_test, prob_test, sem_test = get_metrics(x_test, y_test, e_test)

        res_arr = np.array([mse_train, R2_train, prob_train, sem_train,
                            mse_test, R2_test, prob_test, sem_test])

        res_series = pd.Series(res_arr, index=['mse_train', 'R2_train', 'prob_train', 'se_max_train',
                                               'mse_test', 'R2_test', 'prob_test', 'se_max_test'])

        coef_mean, coef_var = model.coef_mean, model.coef_var

        for idx_c, m_i in enumerate(coef_mean):
            res_series[f'coef_mean_{idx_c}'] = m_i

        for (idx_r, idx_c) in np.ndindex(coef_var.shape):
            res_series[f'coef_var_{idx_r}_{idx_c}'] = coef_var[idx_r, idx_c]

        if debug:
            # model.plot_fit_stats()
            # model.galerkin_force_diagnostics()
            # model.galerkin_penalty_diagnostics()

            # Plot True Solution (If 2D)
            if dim == 2:
                X_grid = x[:, 0].reshape(grid_size_1d, grid_size_1d)
                Y_grid = x[:, 1].reshape(grid_size_1d, grid_size_1d)
                Z_grid = y.reshape(grid_size_1d, grid_size_1d)
                vmax = np.abs(Z_grid).max()

                fig, ax = util.contour_plot(X_grid, Y_grid, Z_grid, vmin=-vmax, vmax=vmax)

                title_str = f"True Solution for {pde_name} + {true_params}"
                ax.set(title=title_str, xlabel=variables[0], ylabel=variables[1])
                plt.show()

            # Plot Fitted
            model.plot_fit(x_test=x_test, y_test=y_test)
            plt.show()
            # print(res_series)

            model.plot_lambda_weight()
            model.plot_eigenvalues()
            model.plot_basis_functions()
            model.plot_derivatives(deriv_orders_list=[(0, 0), (1, 0), (0, 1), (2, 0), (1,1), (0, 2)],
                                   true_deriv_func=true_sol_with_derivatives,
                                   true_deriv_kwargs=params)
        return res_series

    t0 = time.time()
    res_list = parallel(run_one, range(n_iter), disable_par=True)
    res_df = pd.concat(res_list, names=['iter'], axis=1).T

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
    mode_str = "DUAL" if action else "PDE"
    print(f'PDE Mode: {mode_str} | Meta Info: {true_params}')

    return res_df