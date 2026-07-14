import numpy as np
from pathlib import Path
from run_example import run_pde_experiment

PDE_NAME = Path(__file__).stem
X_MAX_SIZE = 1_000_000
DIM = 2
OVERSAMPLE = 100


VARIABLES = ['x', 'y']

def setup_config_func(params):
    DOMAIN = [(0, 1), (0, 1)]

    PDE_KWARGS = {}
    PDE_KWARGS['deriv_coef'] = {'x,x': 1., 'y,y': 1., '': 1.}
    PDE_KWARGS['trainable'] = ['']

    ACTION_KWARGS = {}
    action_coef = {
        ('x', 'x'): -1.,  # u_x * u_x
        ('y', 'y'): -1.,  # u_y * u_y
        ('', ''): 1.  # -(k ** 2)  # u * u
    }
    ACTION_KWARGS['action_coef'] = action_coef
    ACTION_KWARGS['trainable'] = [('', '')]

    k_sqr = (params['p_m'] ** 2 + params['p_n'] ** 2) * (np.pi ** 2)
    k = np.sqrt(k_sqr)
    true_params = {'k_sqr': k_sqr}

    required_knots_per_dim = int((k / np.pi) * 3)
    num_int_knots = required_knots_per_dim ** 2
    return DOMAIN, PDE_KWARGS, ACTION_KWARGS, true_params, num_int_knots

PDE_PARAMS = {'p_m':
                  {'flag': 'wave_m',
                   'type': float,
                   'default': 5.},

              'p_n':
                  {'flag': 'wave_n',
                   'type': float,
                   'default': 12.},

              'p_a1':
                  {'flag': 'amp_1',
                   'type': float,
                   'default': 1},

              'p_a2':
                  {'flag': 'amp_2',
                   'type': float,
                   'default': -1},
              }

def true_sol(x, p_m, p_n, p_a1, p_a2):
    # Helmholtz:
    # u_xx + u_yy + (m^2 + n^2) pi^2 u = 0
    # Zero BC: u = 0
    u_1 = np.cos(p_m * np.pi * x[:, 0]) * np.cos(p_n * np.pi * x[:, 1])
    u_2 = np.cos(p_n * np.pi * x[:, 0]) * np.cos(p_m * np.pi * x[:, 1])
    return p_a1 * u_1 + p_a2 * u_2



if __name__ == '__main__':
    # Dispatches the physics definitions to the general testing suite
    run_pde_experiment(
        pde_name=PDE_NAME,
        dim=DIM,
        variables=VARIABLES,
        pde_params=PDE_PARAMS,
        true_sol_func=true_sol,
        setup_config_func=setup_config_func,
        x_max_size=X_MAX_SIZE,
        oversample=OVERSAMPLE
    )

# if __name__ == '__main__':
#     import time
#     t0 = time.time()
#     def to_bool(arg):
#         assert arg in ['0', '1']
#         return True if arg == '1' else False
#
#     import argparse
#     parser = argparse.ArgumentParser(description="Run 2D Helmholtz PDE")
#     parser.add_argument("-nf", "--sample_size_frac", type=float, default=1., help='Data to Spline Ratio')
#     parser.add_argument("-sf", "--sigma_frac", type=float, default=0.1, help="Noise to Signal Ratio")
#
#     parser.add_argument("-a", "--alpha", type=float, default=.05)
#     parser.add_argument("-q", "--quantile", type=str, default='0')
#     parser.add_argument("-i", "--iteration", type=int, default=100)
#     parser.add_argument("-s", "--save", type=str, default='1')
#     parser.add_argument("-d", "--debug", type=str, default='0')
#     parser.add_argument("-w", "--weak_form", type=str, default='0')
#
#
#     for p_name, p_dict in PDE_PARAMS.items():
#         p_flag = p_dict.pop('flag')
#         parser.add_argument(f"-{p_name}", f"--{p_flag}", **p_dict)
#         p_dict['flag'] = p_flag
#     args = parser.parse_args()
#
#     nf = args.sample_size_frac #1000 # TODO: Slow for large sample size...
#     sigma_frac = args.sigma_frac
#     alpha = args.alpha
#     n_iter = args.iteration
#
#     save = to_bool(args.save)
#     debug = to_bool(args.debug)
#     quantile_based = to_bool(args.quantile)
#     action = to_bool(args.weak_form)
#
#     if debug:
#         n_iter = 1
#
#     # PDE Params
#     params = {p_name: args.__getattribute__(p_dict['flag']) for p_name, p_dict in PDE_PARAMS.items()}
#
#     k_sqr = (params['p_m'] ** 2 + params['p_n'] ** 2) * (np.pi ** 2)
#     k = np.sqrt(k_sqr)
#     required_knots_per_dim = int((k / np.pi) * 3)
#     num_int_knots = required_knots_per_dim ** 2
#
#     n = int(num_int_knots * nf)
#     grid_size_1d = int(min(X_MAX_SIZE ** (1 / DIM), OVERSAMPLE * n))
#     x1 = np.linspace(*DOMAIN[0], grid_size_1d)
#     x2 = np.linspace(*DOMAIN[1], grid_size_1d)
#     x = tensor_prod(x1, x2)
#
#     y = true_sol(x, **params)
#     sigma_scaled = np.std(y) * sigma_frac
#
#     if action:
#         kwargs = ACTION_KWARGS
#     else:
#         kwargs = PDE_KWARGS
#
#     def run_one(i):
#         rng = np.random.default_rng(i)
#         noise = rng.normal(size=len(x)) * sigma_scaled
#         y_noise = y + noise
#
#         index_train = rng.choice(range(len(x)), n)
#         index_train = np.sort(index_train)
#         x_train, y_train, e_train = x[index_train], y_noise[index_train], noise[index_train]
#
#         # TODO: Change CI after adding PDE loss
#         index_test = rng.choice(range(len(x)), n)
#         index_test = np.sort(index_test)
#         x_test, y_test, e_test = x[index_test], y_noise[index_test], noise[index_test]  # TODO: sort
#         # dataset = pd.DataFrame({'x_train': x_obs, 'y_train': y_obs, 'x_test': x_test, 'y_test': y_test})
#
#         model = EWABSplinePDE(dim=DIM, variables=VARIABLES, domain=DOMAIN,
#                               alpha=alpha, quantile_based=quantile_based, num_int_knots=num_int_knots,
#                               total_grid_size=10000, norm_omega=False, **kwargs)
#         model.fit(x_train, y_train)
#
#         def get_metrics(x, y, e):
#             preds = model.predict(x)
#             resid = y - preds['fhat']
#             mse = np.mean(resid ** 2)
#             R2 = canberra_similarity(resid, e) #(resid.T @ e) / (e.T @ e)
#             prob = ((preds['ci_lower'] <= y) & (y <= preds['ci_upper'])).mean()
#             se_max = preds['se'].max() / sigma_scaled
#             return mse, R2, prob, se_max
#
#         mse_train, R2_train, prob_train, sem_train = get_metrics(x_train, y_train, e_train)
#         mse_test, R2_test, prob_test, sem_test = get_metrics(x_test, y_test, e_test)
#
#         res = np.array([mse_train, R2_train, prob_train, sem_train,
#                         mse_test, R2_test, prob_test, sem_test])
#
#         res = pd.Series(res, index=['mse_train', 'R2_train', 'prob_train', 'se_max_train',
#                                     'mse_test', 'R2_test', 'prob_test', 'se_max_test'])
#
#         coef_mean, coef_var = model.coef_mean, model.coef_var
#
#         for i, m_i in enumerate(coef_mean):
#             res[f'coef_mean_{i}'] = m_i
#
#         for (i, j) in np.ndindex(coef_var.shape):
#             res[f'coef_var_{i}_{j}'] = coef_var[i, j]
#
#         if debug:
#             model.plot_basis_functions()
#
#             # Plot True Solution
#             X = x[:, 0].reshape(grid_size_1d, grid_size_1d)
#             Y = x[:, 1].reshape(grid_size_1d, grid_size_1d)
#             Z = y.reshape(grid_size_1d, grid_size_1d)
#
#             fig, ax = plt.subplots(figsize=(6, 5))
#             c = ax.contourf(X, Y, Z, levels=50, cmap=ewa.CMAP)
#             fig.colorbar(c, ax=ax)
#             title = f"True Solution for {PDE_NAME} with u_xx + u_yy + {k_sqr:.4g}u = 0"
#             ax.set(title=title, xlabel="X1", ylabel="X2")
#             plt.show()
#
#             # Plot Fitted
#             model.plot_fit(show_ci=True, x_test=x_test, y_test=y_test)
#             plt.show()
#             print(res)
#         return res
#
#     res = parallel(run_one, range(n_iter), disable_par=True)
#     res = pd.concat(res, names=['iter'], axis=1).T
#
#     fname='BayesHelmholtz' + str(args).replace('Namespace', ' ') + '.csv'
#
#     if save:
#         res.to_csv(f'/Users/cheungjustin/PDE/EWA/output/{fname}')
#         print(f'{fname} written')
#
#     t1 = time.time()
#     print(t1-t0)
#     print(f'PDE Mode: {"DUAL" if action else "PDE"}, k_true^2: {k_sqr}')