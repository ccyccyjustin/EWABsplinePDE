import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, t
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from util import form_omega, orthogonalize_basis, gsj_est, fit_ewa
import pandas as pd
from scipy.interpolate import BSpline


def bs(x, df=None, knots=None, degree=3, intercept=False, boundary_knots=None):
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

    # 4. Generate Basis Matrix
    basis_matrix = BSpline.design_matrix(x, all_knots, degree).toarray()

    # 5. Handle Intercept
    # R's bs() by default (intercept=FALSE) drops the first basis function
    if not intercept:
        basis_matrix = basis_matrix[:, 1:]
    return basis_matrix

class EWABSpline(BaseEstimator, RegressorMixin):
    def __init__(self, deg=3, deriv_order=2, num_int_knots=None,
                 sigma=None, gamma_mult=8, prior_type="light",
                 alpha=0.05, log_lamb=(-30, 20), log_lamb_step=0.01, prior_min=1e-10):
        self.deg = deg
        self.deriv_order = deriv_order
        self.num_int_knots = num_int_knots
        self.sigma = sigma
        self.gamma_mult = gamma_mult
        self.prior_type = prior_type
        self.alpha = alpha
        self.log_lamb = log_lamb
        self.log_lamb_step = log_lamb_step
        self.prior_min = prior_min

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
        self.domain = (self.a_, self.b_)

        # 1. Knot Setup
        if self.num_int_knots is None:
            # Rule of thumb for knot density
            self.n_knots = int(15 * n_samples ** (1 / (2 * self.deg + 3)))
        else:
            self.n_knots = self.num_int_knots

        self.int_knots_ = np.linspace(*self.domain, self.n_knots + 2)[1:-1]

        # 2. Basis & Orthogonalization
        omega = form_omega(*self.domain, self.int_knots_, self.deriv_order, self.deg)

        B_s = [bs(X[:, i], knots=self.int_knots_, degree=self.deg, intercept=True, boundary_knots=self.domain)
               for i in range(X.shape[1])][0] #TODO: Hack for now...

        orth = orthogonalize_basis(B_s, omega) #TODO: Check if need sort eigen

        self.V_s_, self.U_s_ = orth['V'], orth['U']
        self.sigma_est = gsj_est(y) if self.sigma is None else self.sigma

        # 3. Lambda & Prior Vectorization
        log_lams = np.arange(self.log_lamb[0], self.log_lamb[1], self.log_lamb_step)
        lam_grid = np.exp(np.sort(log_lams)[::-1])

        if self.prior_type == "light":
            p_dist = gamma(a=0.3, scale=1 / 0.3)
            prior_vals = p_dist.cdf(lam_grid)
        else:
            prior_vals = 2 * t.cdf(25 * lam_grid, df=3) - 1

        prior = np.diff(np.r_[0, prior_vals[::-1]])[::-1]
        prior, lam_grid = prior[prior > self.prior_min], lam_grid[prior > self.prior_min]

        # 4. EWA Logic
        U, V, s = orth['U'], orth['V'], orth['s']
        res = fit_ewa(
            y=y, ftrue=y, U=U, V=V, s=s,
            lam=lam_grid, prior=prior,
            sig_eps_est=self.sigma_est, gamma_mult=self.gamma_mult,
        )

        # 5. Extract Coefficients
        # We only need the mean weights across the aggregation
        w_vec = res['weights'].reshape(-1, len(lam_grid))[0, :] # approx correct
        fhat_obs = res['fhatps'] @ w_vec # approx correct

        self.theta_hat_ = self.U_s_.T @ fhat_obs
        self.lambda_best = float(np.atleast_1d(res['lam_best'])[0])
        self.X_fit_, self.y_fit_ = X, y
        self.ewa_res = res
        self.fit_n = len(X)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False)
        assert np.ndim(X) == 1

        # Project new X into the orthogonalized basis
        B_new = bs(X, knots=self.int_knots_, degree=self.deg, intercept=True,
                   boundary_knots=self.domain) #TODO: 2d
        U_new = B_new @ self.V_s_
        y_hat = U_new @ self.theta_hat_

        # Optimized SE calculation: avoid full matrix products
        # sqrt(diag(U @ U.T)) is just the row-wise norm
        variances = np.sum(np.square(U_new), axis=1)
        se = self.sigma_est * np.sqrt(variances)
        interval = norm.ppf(1 - self.alpha / 2) * se
        preds = pd.DataFrame({'fhat': y_hat, 'se': se,
                             'ci_lower': y_hat - interval, 'ci_upper': y_hat + interval},
                             index=X)
        return preds

    def plot_fit(self, n_grid=500):
        check_is_fitted(self)
        x_grid = np.linspace(*self.domain, n_grid)
        preds = self.predict(x_grid)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(x_grid, preds['ci_lower'], preds['ci_upper'],
                        color='C0', alpha=0.15, label=f'{1 - self.alpha:.0%} CI')
        ax.plot(x_grid, preds['fhat'], color='C0', lw=2, label='EWA Prediction')
        ax.scatter(self.X_fit_, self.y_fit_, s=10, color='black', alpha=0.5, label='Data')

        title = (
            f"EWA fit [n={len(self.X_fit_)}, deg={self.deg}, "
            f"nk={self.n_knots}, sigma={self.sigma_est:.3f}, "
            f"{100 * (1 - self.alpha):.0f}% CI]"
        )
        ax.set(title=title, xlabel="X", ylabel="y")
        ax.legend()
        plt.show()

    def __repr__(self):
        status = "Fitted" if hasattr(self, 'theta_hat_') else "Unfitted"
        fit_info = {k: self.__getattribute__(k)
                    for k in ['fit_n', 'domain', 'deg', 'n_knots',
                              'sigma_est', 'gamma_mult', 'prior_type', 'alpha',
                              'lambda_best']}
        return f"EWABSpline({status}, {fit_info})"



class EWABSplinePDE(BaseEstimator, RegressorMixin):
    def __init__(self, deg=3, deriv_coef=[0., 0., 1.],
                 domain=None, num_int_knots=None,
                 sigma=None, gamma_mult=8, prior_type="light",
                 alpha=0.05, log_lamb=(-30, 20), log_lamb_step=0.01, prior_min=1e-10):
        self.deg = deg
        self.deriv_coef = deriv_coef
        self.domain = domain
        self.num_int_knots = num_int_knots
        self.sigma = sigma
        self.gamma_mult = gamma_mult
        self.prior_type = prior_type
        self.alpha = alpha
        self.log_lamb = log_lamb
        self.log_lamb_step = log_lamb_step
        self.prior_min = prior_min

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

        self.int_knots_ = np.linspace(*self.domain, self.n_knots + 2)[1:-1]

        # 2. Basis & Orthogonalization
        omega = []
        for q, c in enumerate(self.deriv_coef):
            omega.append(form_omega(*self.domain, self.int_knots_, deriv_order=q, deg=self.deg) * c)
        omega = np.add.reduce(omega)

        B_s = [bs(X[:, i], knots=self.int_knots_, degree=self.deg, intercept=True, boundary_knots=self.domain)
               for i in range(X.shape[1])][0] #TODO: Hack for now...

        orth = orthogonalize_basis(B_s, omega) #TODO: Check if need sort eigen

        self.V_s_, self.U_s_ = orth['V'], orth['U']
        self.sigma_est = gsj_est(y) if self.sigma is None else self.sigma

        # 3. Lambda & Prior Vectorization
        log_lams = np.arange(self.log_lamb[0], self.log_lamb[1], self.log_lamb_step)
        lam_grid = np.exp(np.sort(log_lams)[::-1])

        if self.prior_type == "light":
            p_dist = gamma(a=0.3, scale=1 / 0.3)
            prior_vals = p_dist.cdf(lam_grid)
        else:
            prior_vals = 2 * t.cdf(25 * lam_grid, df=3) - 1

        prior = np.diff(np.r_[0, prior_vals[::-1]])[::-1]
        prior, lam_grid = prior[prior > self.prior_min], lam_grid[prior > self.prior_min]

        # 4. EWA Logic
        U, V, s = orth['U'], orth['V'], orth['s']
        res = fit_ewa(
            y=y, ftrue=y, U=U, V=V, s=s,
            lam=lam_grid, prior=prior,
            sig_eps_est=self.sigma_est, gamma_mult=self.gamma_mult,
        )

        # 5. Extract Coefficients
        # We only need the mean weights across the aggregation
        w_vec = res['weights'].reshape(-1, len(lam_grid))[0, :] # approx correct
        fhat_obs = res['fhatps'] @ w_vec # approx correct

        self.theta_hat_ = self.U_s_.T @ fhat_obs
        self.lambda_best = float(np.atleast_1d(res['lam_best'])[0])
        self.X_fit_, self.y_fit_ = X, y
        self.ewa_res = res
        self.fit_n = len(X)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=False)
        assert np.ndim(X) == 1

        # Project new X into the orthogonalized basis
        B_new = bs(X, knots=self.int_knots_, degree=self.deg, intercept=True,
                   boundary_knots=self.domain) #TODO: 2d
        U_new = B_new @ self.V_s_
        y_hat = U_new @ self.theta_hat_

        # Optimized SE calculation: avoid full matrix products
        # sqrt(diag(U @ U.T)) is just the row-wise norm
        variances = np.sqrt(np.sum(np.square(U_new), axis=1)) #TODO: CHECK!
        se = self.sigma_est # * np.sqrt(variances)
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
    def true_sol(x, k=5.0, L=10, ul=1):
        return ul * np.sin(k * x) / np.sin(k * L)


    n = 50
    sigma_frac = 0.25
    L = 10
    k_true = 5.0
    ul = 1

    x = np.linspace(0, L, L * n * 100)
    y = true_sol(x, k=k_true, L=L, ul=ul)
    sigma_scaled = np.std(y) * sigma_frac
    print(sigma_scaled ** 2)

    n_iter = 30
    res = {}
    for i in range(n_iter):
        rng = np.random.default_rng(i)
        y_noise = y + rng.normal(size=len(x)) * sigma_scaled

        index_obs = rng.choice(range(len(x)), n)
        index_obs = np.sort(index_obs)
        x_obs, y_obs = x[index_obs], y_noise[index_obs]

        # TODO: Change CI after adding PDE loss
        index_test = rng.choice(range(len(x)), n)
        index_test = np.sort(index_test)
        x_test, y_test = x[index_test], y_noise[index_test]  # TODO: sort

        for k_guess in np.linspace(1, 100, 100):
            deriv_coef = [k_guess**2, 0, 1]

            model = EWABSplinePDE(deriv_coef=deriv_coef, domain=(0, L))
            fit = model.fit(x_obs.reshape(-1, 1), y_obs)
            # fit_preds = model.predict(x_obs)
            # # fit_preds.plot()
            # mse_train = np.mean((y_obs - fit_preds['fhat']) ** 2)
            # hit_prob = ((fit_preds['ci_lower'] <= y_obs) & (y_obs <= fit_preds['ci_upper'])).mean()
            # print(mse_train, hit_prob)


            preds = model.predict(x_test)
            mse_test = np.mean((y_test - preds['fhat']) ** 2)
            hit_prob = ((preds['ci_lower'] <= y_test) & (y_test <= preds['ci_upper'])).mean()
            # model.plot_fit(show_ci=True, x_test=x_test, y_test=y_test)

            res[k_guess, i] = np.array([mse_test, hit_prob])

    res = pd.DataFrame(res, index=['mse_test', 'hit_prob']).T
    res = res.rename_axis(['k_guess', 'iter'])
    res_mean = res.groupby('k_guess').mean()

    res_mean['mse_test'].plot()
    plt.axvline(x=k_true, color='k', linestyle='--')
    plt.show()

    res_mean['hit_prob'].plot()
    plt.axhline(0.95, color='r', linestyle='--')
    plt.show()
    # print(preds)

    # df = pd.read_csv('~/PDE/EWA/ewa_demo/data.csv', index_col=0)
    # x, y = df[['x']].values, df.y.values
    #
    # # -- 2. Fit --------------------------------------------------------------------
    # # This calls the translated Python ewa_fit which uses the NumPy-based util.py
    # model = EWABSpline(log_lamb_step=0.1)
    #
    # fit = model.fit(x, y)
    #
    # # Print summary (calls the __repr__ we defined in the EWAFit class)
    # print(fit)
    #
    # # -- 3. Plot -------------------------------------------------------------------
    # # This replicates the ggplot2 behavior using Matplotlib
    # # model.plot_fit()
    # preds = model.predict([0.25, 0.5, 0.75])
    # print(preds)