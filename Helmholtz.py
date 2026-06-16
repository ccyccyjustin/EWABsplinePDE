import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ewa import EWABSplinePDE
from util import canberra_similarity, parallel, tensor_prod

X_MAX_SIZE = 1_000_000
DIM = 1
OVERSAMPLE = 100

PDE_PARAMS = {'p_k':
                  {'flag': 'wavenumber',
                   'type': float,
                   'default': 2 * np.pi},

              'p_l':
                  {'flag': 'boundary',
                   'type': float,
                   'default': 1.},

              'p_ul':
                  {'flag': 'boundary_value',
                   'type': float,
                   'default': 1.},
              }

def true_sol(x, p_k, p_l, p_ul):
    # Helmholtz:
    # k^2 u(x) + u''(x) = 0
    # BC: u(0) = 0, u(l) = ul
    return p_ul * np.sin(p_k * x) / np.sin(p_k * p_l)

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

    if debug:
        n_iter = 1

    # PDE Params
    params = {p_name: args.__getattribute__(p_dict['flag']) for p_name, p_dict in PDE_PARAMS.items()}

    grid_size_1d = int(min(X_MAX_SIZE ** (1 / DIM), OVERSAMPLE * n))
    x = np.linspace(0, params['p_l'], grid_size_1d)
    y = true_sol(x, **params)
    x = x.reshape(-1, 1)
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

        deriv_coef = {'': 1, 'x': 0, 'x,x': 1}
        variables = ['x']
        trainable = None #['', 'x']

        model = EWABSplinePDE(dim=DIM, variables=variables,
                              deriv_coef=deriv_coef, domain=[(0, params['p_l'])],
                              alpha=alpha, quantile_based=quantile_based,
                              total_grid_size=10000, trainable=trainable, norm_omega=False)
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