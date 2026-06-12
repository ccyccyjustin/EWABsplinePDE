import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import util
from ewa import EWABSplinePDE
from util import canberra_similarity, parallel

X_MAX_SIZE = 1_000_000
DIM = 2
OVERSAMPLE = 100

PDE_PARAMS = {'p_a':
                  {'flag': 'anisotropy',
                   'type': float,
                   'default': 1.},

              'p_K':
                  {'flag': 'amplitude',
                   'type': float,
                   'default': 1.},

              'p_m':
                  {'flag': 'freq_x',
                   'type': float,
                   'default': 1.},

              'p_n':
                  {'flag': 'freq_y',
                   'type': float,
                   'default': 1.},

              'p_L':
                  {'flag': 'bound_x',
                   'type': float,
                   'default': 2 * np.pi},

              'p_H':
                  {'flag': 'bound_y',
                   'type': float,
                   'default': np.pi},
              }


if __name__ == '__main__':
    def to_bool(arg):
        assert arg in ['0', '1']
        return True if arg == '1' else False

    import argparse
    parser = argparse.ArgumentParser(description="Run Helmholtz PDE")
    parser.add_argument("-n", "--sample_size", type=int, default=30)
    parser.add_argument("-sf", "--sigma_frac", type=float, default=0.1, help="Noise to Signal Ratio")

    parser.add_argument("-a", "--alpha", type=float, default=.05)
    parser.add_argument("-q", "--quantile", type=str, default='0')
    parser.add_argument("-i", "--iteration", type=int, default=300)
    parser.add_argument("-s", "--save", type=str, default='1')
    parser.add_argument("-d", "--debug", type=str, default='0')

    for p_name, p_dict in PDE_PARAMS.items():
        p_flag = p_dict.pop('flag')
        parser.add_argument(f"-{p_name}", f"--{p_flag}", **p_dict)
        p_dict['flag'] = p_flag
    args = parser.parse_args()

    n = args.sample_size #1000 # TODO: Slow for large sample size...
    sigma_frac = args.sigma_frac
    alpha = args.alpha
    n_iter = args.iteration

    save = to_bool(args.save)
    debug = to_bool(args.debug)
    quantile_based = to_bool(args.quantile)

    # PDE Params
    params = {p_name: args.__getattribute__(p_dict['flag']) for p_name, p_dict in PDE_PARAMS.items()}

    if debug:
        n_iter = 1

    def true_sol(x, p_a, p_m, p_n, p_K, p_L, p_H):
        # Generalized Poisson on [0, L] x [0, H]:
        # u_xx + a^2 u_yy = g
        # g(x, y) = K sin(m pi x / L) sin(n pi y / H)
        # Dirichlet BC: f=0 on dOmega
        # -> u(x, y) = - (K / m^2 + a^2n^2) [sin (mx) sin (ny)]
        assert np.ndim(x) == 2 and x.shape[1] == 2
        m2, n2 = p_m * np.pi / p_L, p_n * np.pi / p_H
        return - (p_K / (m2 ** 2 + (p_a * n2) ** 2)) * np.sin(m2 * x[:, 0]) * np.sin(n2 * x[:, 1])

    # take 1d_grid of size (n_sample * oversample)
    grid_size_1d = int(min(X_MAX_SIZE ** (1 / DIM), OVERSAMPLE * n))

    x1 = np.linspace(0, params['p_L'], grid_size_1d)
    x2 = np.linspace(0, params['p_H'], grid_size_1d)
    x = util.tensor_prod(x1, x2)
    y = true_sol(x, **params)
    sigma_scaled = np.std(y) * sigma_frac
    # print(k_true**2, sigma_scaled)

    def run_one(i):
        rng = np.random.default_rng(i)
        noise = rng.normal(size=len(x)) * sigma_scaled
        y_noise = y + noise

        index_train = rng.choice(range(len(x)), n)
        index_train = np.sort(index_train)
        x_train, y_train, e_train = x[index_train], y_noise[index_train], noise[index_train]

        # TODO: Change CI after adding PDE loss
        index_test = rng.choice(range(len(x)), n)
        index_test = np.sort(index_test)
        x_test, y_test, e_test = x[index_test], y_noise[index_test], noise[index_test]  # TODO: sort
        # dataset = pd.DataFrame({'x_train': x_obs, 'y_train': y_obs, 'x_test': x_test, 'y_test': y_test})

        deriv_coef = [1., 0., 1.]
        trainable = [0]

        model = EWABSplinePDE(deriv_coef=deriv_coef, domain=[(0, params['p_L']), (0, params['p_H'])],
                              alpha=alpha, quantile_based=quantile_based,
                              grid_size=10000, trainable=trainable, norm_omega=False)
        model.fit(x_train, y_train)

        def get_metrics(x, y, e):
            preds = model.predict(x)
            resid = y - preds['fhat']
            mse = np.mean(resid ** 2)
            R2 = canberra_similarity(resid, e) #(resid.T @ e) / (e.T @ e)
            prob = ((preds['ci_lower'] <= y) & (y <= preds['ci_upper'])).mean()
            se_max = preds['se'].max() / sigma_scaled
            return mse, R2, prob, se_max

        mse_train, R2_train, prob_train, sem_train = get_metrics(x_train, y_train, e_train)
        mse_test, R2_test, prob_test, sem_test = get_metrics(x_test, y_test, e_test)

        res = np.array([mse_train, R2_train, prob_train, sem_train,
                        mse_test, R2_test, prob_test, sem_test])

        res = pd.Series(res, index=['mse_train', 'R2_train', 'prob_train', 'se_max_train',
                                    'mse_test', 'R2_test', 'prob_test', 'se_max_test'])

        coef_mean, coef_var = model.coef_mean, model.coef_var

        for i, m_i in enumerate(coef_mean):
            res[f'coef_mean_{i}'] = m_i

        for (i, j) in np.ndindex(coef_var.shape):
            res[f'coef_var_{i}_{j}'] = coef_var[i, j]

        if debug:
            basis_df = model.basis_functions
            basis_df.plot()
            plt.show()

            model.plot_fit(show_ci=True, x_test=x_test, y_test=y_test)
            plt.show()
            print(res)
        return res

    res = parallel(run_one, range(n_iter), disable_par=True)
    res = pd.concat(res, names=['iter'], axis=1).T

    fname='BayesHelmholtz' + str(args).replace('Namespace', ' ') + '.csv'

    if save:
        res.to_csv(f'/Users/cheungjustin/PDE/EWA/output/{fname}')
        print(f'{fname} written')