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
    parser.add_argument("-nf", "--sample_size_frac", type=float, default=1., help='Data to Spline Ratio')
    parser.add_argument("-sf", "--sigma_frac", type=float, default=0.1, help="Noise to Signal Ratio")
    parser.add_argument("-a", "--alpha", type=float, default=.05)
    parser.add_argument("-q", "--quantile", type=str, default='0')
    parser.add_argument("-i", "--iteration", type=int, default=100)
    parser.add_argument("-s", "--save", type=str, default='1')
    parser.add_argument("-d", "--debug", type=str, default='0')
    parser.add_argument("-w", "--weak_form", type=str, default='0')

    # Dynamically inject PDE-specific arguments from the dictionary
    for p_name, p_dict in pde_params.items():
        p_flag = p_dict.pop('flag')
        parser.add_argument(f"-{p_name}", f"--{p_flag}", **p_dict)
        p_dict['flag'] = p_flag

    args = parser.parse_args()

    nf = args.sample_size_frac
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

    if num_int_knots is not None:
        n = int(num_int_knots * nf)
    else:
        n = int(3000 ** (dim / 2))  # Fallback if knots are strictly dynamically determined

    print(f'Sample Size: {n}')

    # Grid Generation
    grid_size_1d = int(min(x_max_size ** (1 / dim), oversample * n))
    grids = [np.linspace(*d, grid_size_1d) for d in domain]
    x = tensor_prod(*grids)

    y_derivs = true_sol_with_derivatives(x, **params)
    y = next(iter(y_derivs.values()))
    sigma_scaled = np.std(y) * sigma_frac

    bc_dict = boundary_conditions(**params, domain=domain)
    kwargs = action_kwargs if action else pde_kwargs

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
        dataset = pd.DataFrame({'x': x_train[:,0], 'y': x_train[:, 1], 'z': y_train})

        print(f"True Sigma: {sigma_scaled}")
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
            model.plot_basis_functions()

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
            print(res_series)

            model.plot_derivatives(deriv_orders_list=[(0, 0), (1, 0), (0, 1), (2, 0), (1,1), (0, 2)],
                                   true_deriv_func=true_sol_with_derivatives,
                                   true_deriv_kwargs=params)
        return res_series

    t0 = time.time()
    res_list = parallel(run_one, range(n_iter), disable_par=True)
    res_df = pd.concat(res_list, names=['iter'], axis=1).T

    fname = f'Bayes{pde_name}' + str(args).replace('Namespace', ' ') + '.csv'

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