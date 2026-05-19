import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, t
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import util
from util import orthogonalize_basis, gsj_est, fit_ewa, compute_spline_integral_gl, orthogonalize_basis_robust
import pandas as pd
from scipy.interpolate import BSpline

def create_knots(x, df=None, knots=None, degree=3, intercept=False, boundary_knots=None):
    """
    Python implementation of R's splines::bs()
    """
    assert np.ndim(x) == 1, "support 1d now..."
    x = np.asarray(x)

    # 1. Determine Boundary Knots
    if boundary_knots is None:
        boundary_knots = [np.min(x), np.max(x)]
    else:
        boundary_knots = np.sort(boundary_knots)

    # 2. Determine Internal Knots
    if knots is None:
        if df is None:
            # Default behavior if neither df nor knots provided
            int_knots = []
        else:
            # Calculate internal knots based on degrees of freedom (quantiles)
            # R logic: n_internal_knots = df - degree - (1 if intercept else 0)
            n_int_knots = df - degree - (1 if intercept else 0)
            if n_int_knots < 0:
                raise ValueError(f"df={df} is too small for degree={degree}")

            # Place knots at quantiles within the boundary
            qs = np.linspace(0, 100, n_int_knots + 2)[1:-1]
            int_knots = np.percentile(x, qs)
    else:
        int_knots = np.sort(knots)

    # 3. Construct Augmented Knot Vector
    # We repeat the boundary knots (degree + 1) times at both ends
    all_knots = np.r_[np.repeat(boundary_knots[0], degree + 1), int_knots, np.repeat(boundary_knots[1], degree + 1)]
    return all_knots

def bs(x, all_knots, degree, intercept=False):
    # 4. Generate Basis Matrix
    basis_matrix = BSpline.design_matrix(x, all_knots, degree).toarray()

    # 5. Handle Intercept
    # R's bs() by default (intercept=FALSE) drops the first basis function
    if not intercept:
        basis_matrix = basis_matrix[:, 1:]
    return basis_matrix

# class EWABSpline(BaseEstimator, RegressorMixin):
#     def __init__(self, deg=3, deriv_order=2, num_int_knots=None,
#                  sigma=None, gamma_mult=8, prior_type="light",
#                  alpha=0.05, log_lamb=(-30, 20), log_lamb_step=0.01, prior_min=1e-10):
#         self.deg = deg
#         self.deriv_order = deriv_order
#         self.num_int_knots = num_int_knots
#         self.sigma = sigma
#         self.gamma_mult = gamma_mult
#         self.prior_type = prior_type
#         self.alpha = alpha
#         self.log_lamb = log_lamb
#         self.log_lamb_step = log_lamb_step
#         self.prior_min = prior_min
#
#     def fit(self, X, y):
#         """
#         1. Knot
#         2. Create Basis and Calculate D_q
#         3. Write SVD -> (U, V, s) (U refers to the A matrix in that paper)
#         4. Fit EWA on (U, y, s)
#         :param X:
#         :param y:
#         :return:
#         """
#         X, y = check_X_y(X, y, accept_sparse=False)
#         n_samples = X.shape[0]
#
#         self.a_, self.b_ = X.min(), X.max()
#         self.domain = (self.a_, self.b_)
#
#         # 1. Knot Setup
#         if self.num_int_knots is None:
#             # Rule of thumb for knot density
#             self.n_knots = int(15 * n_samples ** (1 / (2 * self.deg + 3)))
#         else:
#             self.n_knots = self.num_int_knots
#
#         self.int_knots_ = np.linspace(*self.domain, self.n_knots + 2)[1:-1]
#
#         # 2. Basis & Orthogonalization
#         omega = form_omega(*self.domain, self.int_knots_, self.deriv_order, self.deg)
#
#         B_s = [bs(X[:, i], knots=self.int_knots_, degree=self.deg, intercept=True, boundary_knots=self.domain)
#                for i in range(X.shape[1])][0] #TODO: Hack for now...
#
#         orth = orthogonalize_basis(B_s, omega) #TODO: Check if need sort eigen
#
#         self.V_s_, self.U_s_ = orth['V'], orth['U']
#         self.sigma_est = gsj_est(y) if self.sigma is None else self.sigma
#
#         # 3. Lambda & Prior Vectorization
#         log_lams = np.arange(self.log_lamb[0], self.log_lamb[1], self.log_lamb_step)
#         lam_grid = np.exp(np.sort(log_lams)[::-1])
#
#         if self.prior_type == "light":
#             p_dist = gamma(a=0.3, scale=1 / 0.3)
#             prior_vals = p_dist.cdf(lam_grid)
#         else:
#             prior_vals = 2 * t.cdf(25 * lam_grid, df=3) - 1
#
#         prior = np.diff(np.r_[0, prior_vals[::-1]])[::-1]
#         prior, lam_grid = prior[prior > self.prior_min], lam_grid[prior > self.prior_min]
#
#         # 4. EWA Logic
#         U, V, s = orth['U'], orth['V'], orth['s']
#         res = fit_ewa(
#             y=y, ftrue=y, U=U, V=V, s=s,
#             lam=lam_grid, prior=prior,
#             sig_eps_est=self.sigma_est, gamma_mult=self.gamma_mult,
#         )
#
#         # 5. Extract Coefficients
#         # We only need the mean weights across the aggregation
#         w_vec = res['weights'].reshape(-1, len(lam_grid))[0, :] # approx correct
#         fhat_obs = res['fhatps'] @ w_vec # approx correct
#
#         self.theta_hat_ = self.U_s_.T @ fhat_obs
#         self.lambda_best = float(np.atleast_1d(res['lam_best'])[0])
#         self.X_fit_, self.y_fit_ = X, y
#         self.ewa_res = res
#         self.fit_n = len(X)
#         return self
#
#     def predict(self, X):
#         check_is_fitted(self)
#         X = check_array(X, ensure_2d=False)
#         assert np.ndim(X) == 1
#
#         # Project new X into the orthogonalized basis
#         B_new = bs(X, knots=self.int_knots_, degree=self.deg, intercept=True,
#                    boundary_knots=self.domain) #TODO: 2d
#         U_new = B_new @ self.V_s_
#         y_hat = U_new @ self.theta_hat_
#
#         # Optimized SE calculation: avoid full matrix products
#         # sqrt(diag(U @ U.T)) is just the row-wise norm
#         variances = np.sum(np.square(U_new), axis=1)
#         se = self.sigma_est * np.sqrt(variances)
#         interval = norm.ppf(1 - self.alpha / 2) * se
#         preds = pd.DataFrame({'fhat': y_hat, 'se': se,
#                              'ci_lower': y_hat - interval, 'ci_upper': y_hat + interval},
#                              index=X)
#         return preds
#
#     def plot_fit(self, n_grid=500):
#         check_is_fitted(self)
#         x_grid = np.linspace(*self.domain, n_grid)
#         preds = self.predict(x_grid)
#
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.fill_between(x_grid, preds['ci_lower'], preds['ci_upper'],
#                         color='C0', alpha=0.15, label=f'{1 - self.alpha:.0%} CI')
#         ax.plot(x_grid, preds['fhat'], color='C0', lw=2, label='EWA Prediction')
#         ax.scatter(self.X_fit_, self.y_fit_, s=10, color='black', alpha=0.5, label='Data')
#
#         title = (
#             f"EWA fit [n={len(self.X_fit_)}, deg={self.deg}, "
#             f"nk={self.n_knots}, sigma={self.sigma_est:.3f}, "
#             f"{100 * (1 - self.alpha):.0f}% CI]"
#         )
#         ax.set(title=title, xlabel="X", ylabel="y")
#         ax.legend()
#         plt.show()
#
#     def __repr__(self):
#         status = "Fitted" if hasattr(self, 'theta_hat_') else "Unfitted"
#         fit_info = {k: self.__getattribute__(k)
#                     for k in ['fit_n', 'domain', 'deg', 'n_knots',
#                               'sigma_est', 'gamma_mult', 'prior_type', 'alpha',
#                               'lambda_best']}
#         return f"EWABSpline({status}, {fit_info})"
#

def knots_data_count(X: np.ndarray, all_knots: list, deg: int):
    knot_counts = []
    for dim, knots_1d in enumerate(all_knots):
        knot_counts_1d = np.zeros(len(knots_1d) - 1 - 2 * deg)
        for i, a_i in enumerate(knots_1d[deg:-(1 + deg)]):
            b_i = knots_1d[deg + i + 1]
            knot_counts_1d[i] = ((X[:, dim] >= a_i) & (X[:, dim] < b_i)).sum()
        knot_counts.append(knot_counts_1d)
    return knot_counts


class EWABSplinePDE(BaseEstimator, RegressorMixin):
    def __init__(self, deg=3, deriv_coef=[0., 0., 1.],
                 domain=None, num_int_knots=None, quantile_based=False,
                 sigma=None, gamma_mult=8, prior_type="light", norm_omega=True,
                 alpha=0.05, log_lamb=(-30, 20), log_lamb_step=0.01, tol=1e-10):
        self.deg = deg
        self.deriv_coef = deriv_coef
        self.domain = domain
        self.num_int_knots = num_int_knots
        self.quantile_based = quantile_based
        self.sigma = sigma
        self.gamma_mult = gamma_mult
        self.prior_type = prior_type
        self.norm_omega = norm_omega
        self.alpha = alpha
        self.log_lamb = log_lamb
        self.log_lamb_step = log_lamb_step
        self.tol = tol

    def fit(self, X, y):
        """
        1. Knot
        2. Create Basis and Calculate D_q
        3. Write SVD -> (U, V, s) (U refers to the A matrix in that paper)
        4. Fit EWA on (U, y, s)
        :param X:
        :param y:
        :return:
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = X.shape[0]

        self.a_, self.b_ = X.min(), X.max()
        if self.domain is None:
            self.domain = (self.a_, self.b_)

        # 1. Knot Setup
        if self.num_int_knots is None:
            # Rule of thumb for knot density
            self.n_knots = int(15 * n_samples ** (1 / (2 * self.deg + 3)))
        else:
            self.n_knots = self.num_int_knots

        # TODO...
        self.int_grid = np.linspace(*self.domain, self.n_knots + 2)[1:-1]

        if self.quantile_based:
            self.int_knots_ = np.percentile(X, np.linspace(0, 100, self.n_knots + 2)[1:-1])
        else:
            self.int_knots_ = self.int_grid

        self.step_size = np.diff(self.int_knots_).mean()

        self.all_knots = [create_knots(X[:, i], knots=self.int_knots_, degree=self.deg,
                                       intercept=True, boundary_knots=self.domain)
                          for i in range(X.shape[1])] # Could have different #knots

        self.all_grids = [create_knots(X[:, i], knots=self.int_grid, degree=self.deg,
                                       intercept=True, boundary_knots=self.domain)
                          for i in range(X.shape[1])] # Could have different #knots

        self.knot_counts = knots_data_count(X, self.all_knots, self.deg)

        # 2. Basis & Orthogonalization
        B_s = [bs(X[:, i], all_knots=self.all_knots[i], degree=self.deg, intercept=True)
               for i in range(X.shape[1])][0] #TODO: Hack for now...

        omega = []
        for u, c_u in enumerate(self.deriv_coef):
            if abs(c_u) < 1e-6:
                continue
            for v, c_v in enumerate(self.deriv_coef[:u + 1]):
                if abs(c_v) < 1e-6:
                    continue
                c = c_u * c_v
                omega_uv = compute_spline_integral_gl(self.all_knots[0], u=u, v=v, deg=self.deg)#, x_eval=self.all_grids[0])
                # omega_uv = compute_spline_integral_simp(self.all_knots[0], u=u, v=v, deg=self.deg)
                # diff = np.abs(omega_uv - omega_uv_old).sum()
                # print(f"Check omega methods, diff, u, v :{diff, u, v}")
                # if diff  > 1e-8:
                #     raise ValueError#(f"Check omega methods, diff, u, v :{diff, u, v}")
                omega.append(omega_uv * c)

                if u != v:
                    omega.append(omega_uv.T * c)

        omega = np.add.reduce(omega) #Shape: (kn, kn)
        # TODO: looks deriv will blow up when more knots
        if self.norm_omega:
            highest_deriv = len(self.deriv_coef) - 1
            omega *= (self.step_size ** (highest_deriv * 2))
            # print(omega.max(axis=1))

        orth = orthogonalize_basis(B_s, omega)
        self.V_, self.U_, self.s_ = orth['V'], orth['U'], orth['s']
        self.sigma_est = gsj_est(y) if self.sigma is None else self.sigma
        # print(self.s_)

        # 3. Lambda & Prior Vectorization
        log_lams = np.arange(self.log_lamb[0], self.log_lamb[1], self.log_lamb_step)
        lam_grid = np.exp(np.sort(log_lams)[::-1])

        if self.prior_type == "light":
            p_dist = gamma(a=0.3, scale=1 / 0.3)
            prior_vals = p_dist.cdf(lam_grid)
        else:
            prior_vals = 2 * t.cdf(25 * lam_grid, df=3) - 1

        prior = np.diff(np.r_[0, prior_vals[::-1]])[::-1]
        prior, lam_grid = prior[prior > self.tol], lam_grid[prior > self.tol]

        if self.norm_omega:
            highest_deriv = len(self.deriv_coef) - 1
            lam_grid /= self.step_size ** (highest_deriv * 2)

        # 4. EWA Logic
        res = fit_ewa(
            y=y, ftrue=y, U=self.U_, V=self.V_, s=self.s_,
            lam=lam_grid, prior=prior,
            sig_eps_est=self.sigma_est, gamma_mult=self.gamma_mult,
        )

        # 5. Extract Coefficients
        # We only need the mean weights across the aggregation
        w_vec = res['weights'].reshape(-1, len(lam_grid))[0, :]
        fhat_ewa = res['fhat_ewa'].flatten()

        self.ewa_res = res
        self.D_ewa = res['shrink_weight']
        self.hat = res['hat_matrix']

        self.weights = pd.Series(w_vec, index=lam_grid).sort_index()
        self.sigma_regr_adj = np.sqrt(np.sum((fhat_ewa - y)**2) / (len(y) - np.trace(self.hat)))
        self.sigma_regr = np.sqrt(np.mean((fhat_ewa - y)**2))
        self.theta_hat_ = self.U_.T @ fhat_ewa
        self.lambda_best = float(np.atleast_1d(res['lam_best'])[0])
        self.X_fit_, self.y_fit_ = X, y
        self.fit_n = len(X)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False)
        assert np.ndim(X) == 1

        # Project new X into the orthogonalized basis
        B_new = bs(X, all_knots=self.all_knots[0], degree=self.deg, intercept=True)#TODO: 2d
        U_new = B_new @ self.V_
        y_hat = U_new @ self.theta_hat_

        # np.diag(U_new @ U_new.T) + 1
        variances = np.sum((U_new ** 2) * (self.D_ewa ** 2)[None, :], axis=1) + 1 #TODO: **2?
        se = self.sigma_regr * np.sqrt(variances) # TODO: wide CI because of large (self.V_ @ self.V_.T)
        interval = norm.ppf(1 - self.alpha / 2) * se
        preds = pd.DataFrame({'fhat': y_hat, 'se': se,
                             'ci_lower': y_hat - interval, 'ci_upper': y_hat + interval},
                             index=X)
        return preds

    def plot_fit(self, x_test=None, y_test=None, n_grid=500, show_ci=True):
        check_is_fitted(self)

        if x_test is None:
            x_grid = np.linspace(*self.domain, n_grid)
            fit_preds = self.predict(x_grid)
            x_scatter, y_scatter = self.X_fit_, self.y_fit_
        else:
            x_grid = x_test #np.linspace(*self.domain, n_grid)
            fit_preds = self.predict(x_grid)
            x_scatter, y_scatter = x_test, y_test
            #preds = self.predict(x)

        fig, ax = plt.subplots(figsize=(10, 5))
        if show_ci:
            ax.fill_between(x_grid, fit_preds['ci_lower'], fit_preds['ci_upper'],
                            color='C0', alpha=0.15, label=f'{1 - self.alpha:.0%} CI (during Fit)')
        ax.plot(x_grid, fit_preds['fhat'], color='C0', lw=2, label='EWA Prediction')
        ax.scatter(x_scatter, y_scatter, s=10, color='black', alpha=0.5, label='Data')

        title = (
            f"EWA fit [n={len(self.X_fit_)}, deg={self.deg}, deriv_coef={self.deriv_coef}, "
            f"nk={self.n_knots}, sigma={self.sigma_est:.3f}, "
            f"{100 * (1 - self.alpha):.0f}% CI]"
        )
        ax.set(title=title, xlabel="X", ylabel="y")
        ax.legend()
        plt.show()

    def __repr__(self):
        status = "Fitted" if hasattr(self, 'theta_hat_') else "Unfitted"
        fit_info = {k: self.__getattribute__(k)
                    for k in ['fit_n', 'domain', 'deg', 'deriv_coef', 'n_knots',
                              'sigma_est', 'gamma_mult', 'prior_type', 'alpha',
                              'lambda_best']}
        return f"EWABSpline({status}, {fit_info})"


if __name__ == '__main__':
    def to_bool(arg):
        assert arg in ['0', '1']
        return True if arg == '1' else False

    import argparse
    parser = argparse.ArgumentParser(description="Run Helmholtz PDE")
    parser.add_argument("-n", "--sample_size", type=int, default=30)
    parser.add_argument("-sf", "--sigma_frac", type=float, default=0.1, help="Noise to Signal Ratio")
    parser.add_argument("-k", "--wavenumber", type=float, default=2*np.pi)
    parser.add_argument("-l", "--boundary", type=float, default=10.)
    parser.add_argument("-ul", "--boundary_value", type=float, default=1.)
    parser.add_argument("-a", "--alpha", type=float, default=.05)
    parser.add_argument("-q", "--quantile", type=str, default='0')
    parser.add_argument("-i", "--iteration", type=int, default=100)
    parser.add_argument("-s", "--save", type=str, default='1')
    parser.add_argument("-d", "--debug", type=str, default='0')
    args = parser.parse_args()

    n = args.sample_size #1000 # TODO: Slow for large sample size...
    sigma_frac = args.sigma_frac
    k_true = args.wavenumber
    l = args.boundary
    ul = args.boundary_value
    alpha = args.alpha
    n_iter = args.iteration

    save = to_bool(args.save)
    debug = to_bool(args.debug)
    quantile_based = to_bool(args.quantile)

    if debug:
        n_iter = 1

    def true_sol(x, k, l, ul):
        # Helmholtz:
        # k^2 u(x) + u''(x) = 0
        # BC: u(0) = 0, u(l) = ul
        return ul * np.sin(k * x) / np.sin(k * l)

    x = np.arange(0, l, 0.01/n) #np.linspace(0, l, l * n * 100)
    y = true_sol(x, k=k_true, l=l, ul=ul)
    sigma_scaled = np.std(y) * sigma_frac
    print(sigma_scaled)

    def run_one(i):
        res = {}
        rng = np.random.default_rng(i)
        y_noise = y + rng.normal(size=len(x)) * sigma_scaled

        index_obs = rng.choice(range(len(x)), n)
        index_obs = np.sort(index_obs)
        x_obs, y_obs = x[index_obs], y_noise[index_obs]

        # TODO: Change CI after adding PDE loss
        index_test = rng.choice(range(len(x)), n)
        index_test = np.sort(index_test)
        x_test, y_test = x[index_test], y_noise[index_test]  # TODO: sort
        # dataset = pd.DataFrame({'x_train': x_obs, 'y_train': y_obs, 'x_test': x_test, 'y_test': y_test})
        best_loss, best_model = np.inf, None

        for k_guess in np.linspace(k_true // 3, k_true * 3, 100):
            deriv_coef = [k_guess**2, 0, 1]

            model = EWABSplinePDE(deriv_coef=deriv_coef, domain=(0, l), alpha=alpha, quantile_based=quantile_based)
            model.fit(x_obs.reshape(-1, 1), y_obs)

            fit_preds = model.predict(x_obs)
            mse_train = np.mean((y_obs - fit_preds['fhat']) ** 2)
            prob_train = ((fit_preds['ci_lower'] <= y_obs) & (y_obs <= fit_preds['ci_upper'])).mean()

            test_preds = model.predict(x_test)
            mse_test = np.mean((y_test - test_preds['fhat']) ** 2)
            prob_test = ((test_preds['ci_lower'] <= y_test) & (y_test <= test_preds['ci_upper'])).mean()
            se_max = test_preds['se'].max()
            print(k_guess, model.sigma_regr, model.sigma_regr_adj, model.sigma_est)

            if mse_test < best_loss:
                best_loss = mse_test
                best_model = model

            res[k_guess] = np.array([mse_train, prob_train, mse_test, prob_test, se_max])

        res = pd.DataFrame(res, index=['mse_train', 'prob_train', 'mse_test', 'prob_test', 'se_max']).T
        if debug:
            best_model.plot_fit(show_ci=True, x_test=x_test, y_test=y_test)
            plt.show()
            print(res)
        return res

    res = util.parallel(run_one, range(n_iter), disable_par=debug)
    res = pd.concat(res, names=['iter', 'k_guess'])
    res_mean = res.groupby('k_guess').mean()

    fname='NewHelmholtz' + str(args).replace('Namespace', ' ') + '.csv'
    if save:
        res_mean.to_csv(f'/Users/cheungjustin/PDE/EWA/output/{fname}')
        print(f'{fname} written')

    # res_mean[['mse_train', 'mse_test']].plot()
    # plt.axvline(x=k_true, color='k', linestyle='--')
    # plt.show()
    #
    # res_mean[['prob_train', 'prob_test']].plot()
    # plt.axvline(x=k_true, color='k', linestyle='--')
    # plt.axhline(1-alpha, color='r', linestyle='--')
    # plt.show()
    #
    # res_mean[['se_max']].plot()
    # plt.axvline(x=k_true, color='k', linestyle='--')
    # plt.show()
    from statsmodels.api import OLS