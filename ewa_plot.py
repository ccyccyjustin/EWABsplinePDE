import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.utils.validation import check_is_fitted

import plot

logger = logging.getLogger(__name__)


class EWAPlotMixin:
    """Diagnostic and visualization methods for EWABSplinePDE."""

    def plot_fit(self, x_test=None, y_test=None, true_coef_vals=None):
        """
        Plots diagnostic subplots for the fitted EWA surface:
        1. Scatter plot of Data Residuals (Y - X * beta_ewa) with CI Error bounds
        2. Heatmap of Learnt Pointwise PDE Loss
        3. Heatmap of True Pointwise PDE Loss (Optional)
        """
        check_is_fitted(self)
        x_grid = self.gridpoints_

        # Get continuous predictions on the grid for the CI Error Bounds
        fit_preds_grid = self.predict(x_grid)

        # --- 1. Compute Data Residuals ---
        if x_test is None:
            x_scatter, y_scatter = self.X_fit_, self.y_fit_
        else:
            x_scatter, y_scatter = x_test, y_test

        fit_preds_scatter = self.predict(x_scatter)
        data_residuals = y_scatter - fit_preds_scatter['fhat'].values

        # Define Error CI Bounds relative to the fitted surface
        err_ci_lower = fit_preds_grid['ci_lower'] - fit_preds_grid['fhat']
        err_ci_upper = fit_preds_grid['ci_upper'] - fit_preds_grid['fhat']

        # --- 2. Compute Pointwise PDE Loss (Learnt Parameters) ---
        pde_loss_learnt = -self.force(x_grid).ravel()
        for i, (k, v) in enumerate(self.deriv_coef.items()):
            order = self._coef_keys[i]
            pde_loss_learnt += v * self.predict_derivative(x_grid, order)

        # --- 3. Compute Pointwise PDE Loss (True Parameters) ---
        pde_loss_true = None
        if true_coef_vals is not None:
            pde_loss_true = -self.force(x_grid).ravel()

            if isinstance(true_coef_vals, dict):
                parsed_true_coefs = {self._get_deriv_order(k, self.variables): v for k, v in true_coef_vals.items()}
                for i, order in enumerate(self._coef_keys):
                    theta_true = parsed_true_coefs.get(order, 0.0)
                    pde_loss_true += theta_true * self.predict_derivative(x_grid, order)
            else:
                for i, order in enumerate(self._coef_keys):
                    pde_loss_true += true_coef_vals[i] * self.predict_derivative(x_grid, order)

        # --- 4. Plotting Setup ---
        n_subplots = 3 if pde_loss_true is not None else 2
        fig, axes = plt.subplots(1, n_subplots, figsize=(5.5 * n_subplots, 5), layout='constrained')

        if n_subplots == 2:
            axes = [axes[0], axes[1]]

        # --- 1D Case ---
        if self.dim == 1:
            axes[0].fill_between(
                x_grid[:, 0], err_ci_lower, err_ci_upper, color='C0', alpha=0.15,
                label=f'{1 - self.alpha:.0%} CI',
            )
            axes[0].scatter(x_scatter[:, 0], data_residuals, color='black', alpha=0.6, s=15, label='Residuals')
            axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
            axes[0].set(title=r"Data Residuals $(Y - \hat{Y})$", xlabel="X", ylabel="Residual")
            axes[0].legend(loc='best')

            axes[1].plot(x_grid[:, 0], pde_loss_learnt, color='tab:orange', lw=2)
            axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
            axes[1].set(title="Learnt PDE Pointwise Loss", xlabel="X", ylabel="PDE Residual")

            if pde_loss_true is not None:
                axes[2].plot(x_grid[:, 0], pde_loss_true, color='tab:green', lw=2)
                axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
                axes[2].set(title="True PDE Pointwise Loss", xlabel="X", ylabel="PDE Residual")

        # --- 2D Case ---
        elif self.dim == 2:
            xlabel, ylabel = self.variables
            X_mesh = x_grid[:, 0].reshape(*self.grid_size_)
            Y_mesh = x_grid[:, 1].reshape(*self.grid_size_)

            vmax_res = np.max(np.abs(data_residuals))
            if vmax_res == 0:
                vmax_res = 1e-8

            err_ci_upper_mesh = err_ci_upper.values.reshape(*self.grid_size_)
            cs = axes[0].contour(
                X_mesh, Y_mesh, err_ci_upper_mesh, levels=5, colors='black', alpha=0.4, linestyles='dashed'
            )
            axes[0].clabel(cs, inline=True, fontsize=8, fmt='CI: ±%.2f')

            norm_res = plot.norm_color(vmin=-vmax_res, vmax=vmax_res)
            sc = axes[0].scatter(
                x_scatter[:, 0], x_scatter[:, 1], c=data_residuals,
                norm=norm_res, cmap=plot.CMAP, edgecolors='black', s=20,
            )
            fig.colorbar(sc, ax=axes[0], shrink=0.8, pad=0.04)
            axes[0].set(title=r"Data Residuals $(Y - \hat{Y})$", xlabel=xlabel, ylabel=ylabel)

            Z_learnt = pde_loss_learnt.reshape(*self.grid_size_)
            vmax_pde = np.max(np.abs(Z_learnt))
            if vmax_pde == 0:
                vmax_pde = 1e-8

            _, ax1 = plot.contour_plot(
                X_mesh, Y_mesh, Z_learnt, ax=axes[1], vmin=-vmax_pde, vmax=vmax_pde, levels=30, cbar=False
            )
            sm_learnt = plot.map_color(vmin=-vmax_pde, vmax=vmax_pde)
            fig.colorbar(sm_learnt, ax=axes[1], shrink=0.8, pad=0.04)
            axes[1].set(title="Learnt PDE Pointwise Loss", xlabel=xlabel, ylabel=ylabel)

            if pde_loss_true is not None:
                Z_true = pde_loss_true.reshape(*self.grid_size_)
                vmax_true = np.max(np.abs(Z_true))
                if vmax_true == 0:
                    vmax_true = 1e-8

                _, ax2 = plot.contour_plot(
                    X_mesh, Y_mesh, Z_true, ax=axes[2], vmin=-vmax_true, vmax=vmax_true, levels=30, cbar=False
                )
                sm_true = plot.map_color(vmin=-vmax_true, vmax=vmax_true)
                fig.colorbar(sm_true, ax=axes[2], shrink=0.8, pad=0.04)
                axes[2].set(title="True PDE Pointwise Loss", xlabel=xlabel, ylabel=ylabel)

        else:
            logger.warning(f"Plotting not supported for dimension {self.dim} > 2.")

        # --- 5. Common Formatting ---
        coef_vals = getattr(self, 'coef_mean', list(self.deriv_coef.values()))
        deriv_coef = {str(k): float(f"{v:.4g}") for k, v in zip(self.coef_keys, coef_vals)}
        title = (
            f"EWA Fit Diagnostics | n={len(x_scatter)} | deg={self.deg}\n"
            f"Learnt Coefs: {deriv_coef}"
        )
        fig.suptitle(title, fontsize=14, y=1.05)
        plt.show()

    def plot_basis_functions(self, thres=0.999, max_subplots=12, **kwargs):
        """
        Plots the top basis functions that explain the given variance threshold.
        Handles 1D (line plots), 2D (heatmaps), and 3D (2D cross-section heatmaps).

        :param thres: Cumulative variance threshold to determine how many bases to plot.
        :param max_subplots: Cap the number of subplots to prevent memory/rendering blowouts.
        """
        check_is_fitted(self)
        df2 = self.basis_functions.copy()

        exp_var = df2.columns.values.cumsum()
        n_col = np.argmax(exp_var > thres) + 1
        n_col = min(n_col, max_subplots)
        df2.columns = df2.columns.map('{:.0%}'.format)

        top_bases = df2.iloc[:, :n_col]
        global_vmax = np.abs(top_bases.values).max()

        # --- 1D Case: All bases on a single line plot ---
        if self.dim == 1:
            ax = top_bases.plot(title=f'Top {n_col} Basis Functions ({exp_var[n_col - 1] * 100:.1f}% Var)', **kwargs)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[0:n_col], labels[0:n_col])
            plt.show()

        # --- 2D Case: Grid of Heatmaps ---
        elif self.dim == 2:
            xlabel, ylabel = self.variables
            grid_shape = (len(self.grids_[0]), len(self.grids_[1]))
            X1, X2 = np.meshgrid(self.grids_[0], self.grids_[1], indexing='ij')

            cols = min(n_col, 3)
            rows = int(np.ceil(n_col / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows),
                                     squeeze=False, layout='constrained')

            for idx in range(n_col):
                r, c = divmod(idx, cols)
                ax = axes[r, c]
                Z = top_bases.iloc[:, idx].values.reshape(grid_shape)
                _, ax = plot.contour_plot(
                    X1, X2, Z, ax=ax, vmin=-global_vmax, vmax=global_vmax, levels=30, cbar=False
                )
                ax.set(title=f"Basis {idx + 1} [Var: {df2.columns[idx]}]", xlabel=xlabel, ylabel=ylabel)

            for idx in range(n_col, rows * cols):
                r, c = divmod(idx, cols)
                axes[r, c].axis('off')

            sm = plot.map_color(vmin=-global_vmax, vmax=global_vmax)
            fig.colorbar(sm, ax=axes.ravel().tolist(), aspect=40, pad=0.03, label='Basis Weight Magnitude')
            fig.suptitle(f'Top {n_col} 2D Basis Functions ({exp_var[n_col - 1] * 100:.1f}% Variance)', fontsize=14)
            plt.show()

        # --- 3D Case: Grid of Heatmaps (Sliced at middle of Z-axis) ---
        elif self.dim == 3:
            grid_shape = (len(self.grids_[0]), len(self.grids_[1]), len(self.grids_[2]))
            z_mid = len(self.grids_[2]) // 2
            z_val = self.grids_[2][z_mid]

            cols = min(n_col, 3)
            rows = int(np.ceil(n_col / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

            for idx in range(n_col):
                r, c = divmod(idx, cols)
                ax = axes[r, c]
                Z_full = top_bases.iloc[:, idx].values.reshape(grid_shape)
                Z_slice = Z_full[:, :, z_mid]
                fig, ax = plot.contour_plot(self.grids_[0], self.grids_[1], Z_slice, ax=ax, levels=30)
                ax.set(title=f"Basis {idx + 1} [Var: {df2.columns[idx]}]", xlabel="X1", ylabel="X2")

            for idx in range(n_col, rows * cols):
                r, c = divmod(idx, cols)
                axes[r, c].axis('off')

            fig.suptitle(f'Top {n_col} 3D Bases (Central Slice X3 ≈ {z_val:.2g})', fontsize=14)
            plt.tight_layout()
            plt.show()

        else:
            logger.warning(f"Plotting basis functions not supported for dimension {self.dim} > 3.")

    def plot_coef_dist(self, true_vals=None, **kwargs):
        labels = [rf'$\theta_{{{i}}}$' for i in self._trainable_idx]
        plot.mvn_pairplot(
            self.coef_mean[self._trainable_idx],
            self.coef_var,
            labels=labels,
            title=r'Density of Physical $\theta$ given $\hat \beta$',
            true_vals=true_vals,
            **kwargs,
        )

    def plot_lambda_weight(self, show_sure: bool = True, figsize: tuple = (13, 5), save_path: str = None):
        """
        Visualizes the EWA posterior probability weights w(lambda) alongside
        the continuous SURE risk curve and theoretical asymptotic target.
        """
        check_is_fitted(self, 'ewa_res_')

        lams = self.lam_grid_
        weights = self.ewa_res_['weights']
        sure = self.ewa_res_.get('sure_est', None)

        lam_mle = float(lams[np.argmax(weights)])
        lam_mean = float(np.sum(lams * weights))

        cum_w = np.cumsum(weights)
        idx_lq = np.argmax(cum_w >= self.alpha / 2.0)
        idx_uq = np.argmax(cum_w >= 1.0 - self.alpha / 2.0)
        lam_theo = getattr(self, 'lam_asymp_', None)

        n_plots = 2 if (show_sure and sure is not None) else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, layout='constrained')
        ax_w = axes[0] if n_plots == 2 else axes

        # PANEL 1: Posterior Weights w(lambda)
        ax_w.plot(lams, weights, color='#1f77b4', lw=2.2, label=r'Posterior $w(\lambda)$')
        ax_w.fill_between(lams, 0, weights, color='#1f77b4', alpha=0.25)
        ax_w.fill_between(
            lams[idx_lq:idx_uq + 1],
            0,
            weights[idx_lq:idx_uq + 1],
            color='#aec7e8',
            alpha=0.5,
            label=f'{1 - self.alpha:.0%} Credible Interval',
        )

        ax_w.axvline(lam_mle, color='crimson', ls='--', lw=1.8, label=rf'$\lambda_{{\mathrm{{MLE}}}} = {lam_mle:.4f}$')
        ax_w.axvline(lam_mean, color='darkorange', ls=':', lw=2.0, label=rf'$\mathbb{{E}}[\lambda] = {lam_mean:.4f}$')

        if lam_theo is not None:
            ax_w.axvline(lam_theo, color='forestgreen', ls='-.', lw=1.8,
                         label=rf'$\lambda_{{\mathrm{{theo}}}} = {lam_theo:.4f}$')

        ax_w.set_title(r"EWA Posterior Weights Distribution $w(\lambda)$", fontsize=12, fontweight='bold')
        ax_w.set_xlabel(r"Regularization Parameter $\lambda$", fontsize=11)
        ax_w.set_ylabel("Posterior Probability Weight", fontsize=11)
        ax_w.set_xlim(0.0, 1.0)
        ax_w.set_ylim(bottom=0.0)
        ax_w.grid(True, ls='--', alpha=0.5)
        ax_w.legend(loc='best', frameon=True, fontsize=9)

        # PANEL 2: Objective SURE(lambda) Landscape
        if n_plots == 2:
            ax_s = axes[1]
            ax_s.plot(lams, sure, color='#2ca02c', lw=2.0, label=r'$\mathrm{SURE}(\lambda) - \min$')
            ax_s.axvline(lam_mle, color='crimson', ls='--', lw=1.8, label=r'Argmin $\lambda^*$')

            if lam_theo is not None:
                ax_s.axvline(lam_theo, color='forestgreen', ls='-.', lw=1.8, label=r'Theoretical $\lambda^*$')

            ax_s.set_title(r"Objective Landscape $\mathrm{SURE}(\lambda)$", fontsize=12, fontweight='bold')
            ax_s.set_xlabel(r"Regularization Parameter $\lambda$", fontsize=11)
            ax_s.set_ylabel(r"Relative Risk $(\mathrm{SURE} - \min)$", fontsize=11)
            ax_s.set_xlim(0.0, 1.0)
            ax_s.grid(True, ls='--', alpha=0.5)
            ax_s.legend(loc='best', frameon=True, fontsize=9)

        fig.suptitle(
            rf"Regularization Diagnostics | $n={self.n_train}$ | $\sigma_{{\mathrm{{est}}}}={self.sigma_est_:.3e}$",
            fontsize=13,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_eigenvalues(self, figsize=(12, 5)):
        """
        Plots the GEVP spectrum s_i and the corresponding eigenfunction coefficients.
        """
        s = np.asarray(self.s_)
        k = len(s)
        mode_indices = np.arange(1, k + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Subplot 1: GEVP Eigenvalue Spectrum Growth
        ax1.plot(mode_indices, s, 'o-', color='#1f77b4', markersize=3.5, linewidth=1.2, label=r'$s_i$ (Penalty Energy)')
        ax1.set_yscale('symlog', linthresh=1e-5)
        ax1.set_xlabel('Eigenmode Index ($i$)', fontsize=11)
        ax1.set_ylabel(r'Eigenvalue $s_i$ (Symlog Scale)', fontsize=11)
        ax1.set_title(f'GEVP Spectrum ($k={k}$ Computed Modes)', fontsize=12, fontweight='bold')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        near_zero_mask = np.abs(s) < 1e-4
        n_zeros = np.sum(near_zero_mask)
        ax1.axvline(n_zeros, color='crimson', linestyle=':', alpha=0.8)

        # Subplot 2: Eigenfunction Coefficients (alpha_hat)
        alpha_hat_abs = np.abs(self.ewa_res_['alpha_hat'])
        ax2.plot(
            mode_indices[:len(alpha_hat_abs)], alpha_hat_abs, color='#2ca02c',
            linewidth=1.2, label=r'$\hat{\alpha}_{\mathrm{ewa}}$',
        )
        ax2.set_xlabel('Eigenmode Index ($i$)', fontsize=11)
        ax2.set_ylabel(r'$|\hat{\alpha}_i|$', fontsize=11)
        ax2.set_title(r'Abs. Eigenfunction Coefficients $\hat{\alpha}_{\mathrm{ewa}}$', fontsize=12, fontweight='bold')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_derivatives(self, deriv_orders_list: List[Tuple[int, ...]], max_cols: int = 3,
                         true_deriv_func: callable = None, true_deriv_kwargs: dict = None, **kwargs):
        """
        Plots the requested partial derivatives over the evaluation grid.
        Handles 1D (line plots), 2D (heatmaps), and 3D (2D cross-section heatmaps).
        If `true_deriv_func` is provided, plots the Error (Predicted - True) directly below each prediction.
        """
        check_is_fitted(self, 'ewa_res_')

        if true_deriv_func is not None:
            t_kwargs = true_deriv_kwargs or {}
            true_derivs = true_deriv_func(self.gridpoints_, **t_kwargs)
            plot_error = True
        else:
            true_derivs = None
            plot_error = False

        n_plots = len(deriv_orders_list)
        cols = min(n_plots, max_cols)
        primary_rows = int(np.ceil(n_plots / cols))
        rows = primary_rows * 2 if plot_error else primary_rows

        # --- 1D Case ---
        if self.dim == 1:
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                                     squeeze=False, layout='constrained')

            for idx, order in enumerate(deriv_orders_list):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                ax_pred = axes[base_row, c]

                y_deriv = self.predict_derivative(self.gridpoints_, order)
                ax_pred.plot(self.grids_[0], y_deriv, lw=2, color='tab:orange')
                ax_pred.set(title=f"Derivative {order}", xlabel="X", ylabel="Value")
                ax_pred.grid(True, linestyle='--', alpha=0.6)

                if plot_error:
                    ax_err = axes[base_row + 1, c]
                    error_vals = y_deriv - true_derivs[order]
                    ax_err.plot(self.grids_[0], error_vals, lw=2, color='tab:red')
                    ax_err.set(title=f"Error {order}", xlabel="X", ylabel="Error (Pred - True)")
                    ax_err.grid(True, linestyle='--', alpha=0.6)

            for idx in range(n_plots, primary_rows * cols):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                axes[base_row, c].axis('off')
                if plot_error:
                    axes[base_row + 1, c].axis('off')

            fig.suptitle("Fitted Surface Derivatives" + (" & Error Profiles" if plot_error else ""), fontsize=14)
            plt.show()

        # --- 2D Case ---
        elif self.dim == 2:
            xlabel, ylabel = self.variables
            grid_shape = (len(self.grids_[0]), len(self.grids_[1]))
            X1, X2 = np.meshgrid(self.grids_[0], self.grids_[1], indexing='ij')

            fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                                     squeeze=False, layout='constrained')

            for idx, order in enumerate(deriv_orders_list):
                order_str = self.format_deriv_orders(order)
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                ax_pred = axes[base_row, c]

                Z_flat = self.predict_derivative(self.gridpoints_, order)
                Z = Z_flat.reshape(grid_shape)

                vmax = np.max(np.abs(Z))
                if vmax == 0:
                    vmax = 1e-8

                _, ax_pred = plot.contour_plot(
                    X1, X2, Z, ax=ax_pred, vmin=-vmax, vmax=vmax, levels=30, cbar=False
                )
                sm = plot.map_color(vmin=-vmax, vmax=vmax)
                fig.colorbar(sm, ax=ax_pred, shrink=0.8, pad=0.04)
                ax_pred.set(title=f"Predicted {order_str}", xlabel=xlabel, ylabel=ylabel)

                if plot_error:
                    ax_err = axes[base_row + 1, c]
                    Z_true = true_derivs[order]
                    Z_err = Z_flat - Z_true

                    Z_true_norm = np.sqrt(np.nansum(Z_true ** 2))
                    Z_rel_err = (Z_err / Z_true_norm).reshape(grid_shape)
                    Z_rel_err_norm = np.sqrt(np.nansum(Z_rel_err ** 2))

                    vmax_err = np.nanmax(np.abs(Z_rel_err))
                    if vmax_err == 0:
                        vmax_err = 1e-8

                    _, ax_err = plot.contour_plot(
                        X1, X2, Z_rel_err, ax=ax_err, vmin=-vmax_err, vmax=vmax_err, levels=30, cbar=False
                    )
                    sm_err = plot.map_color(vmin=-vmax_err, vmax=vmax_err)
                    fig.colorbar(sm_err, ax=ax_err, shrink=0.8, pad=0.04, format=mtick.StrMethodFormatter('{x:.0%}'))
                    ax_err.set(title=f"{order_str} Relative Error: {Z_rel_err_norm:.4%}", xlabel=xlabel, ylabel=ylabel)

            for idx in range(n_plots, primary_rows * cols):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                axes[base_row, c].axis('off')
                if plot_error:
                    axes[base_row + 1, c].axis('off')

            fig.suptitle("2D Fitted Surface Derivatives" + (" & Error Heatmaps" if plot_error else ""), fontsize=14)
            plt.show()

        # --- 3D Case ---
        elif self.dim == 3:
            grid_shape = (len(self.grids_[0]), len(self.grids_[1]), len(self.grids_[2]))
            z_mid = len(self.grids_[2]) // 2
            z_val = self.grids_[2][z_mid]

            fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                                     squeeze=False, layout='constrained')

            for idx, order in enumerate(deriv_orders_list):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                ax_pred = axes[base_row, c]

                Z_flat = self.predict_derivative(self.gridpoints_, order)
                Z_full = Z_flat.reshape(grid_shape)
                Z_slice = Z_full[:, :, z_mid]

                vmax = np.max(np.abs(Z_slice))
                if vmax == 0:
                    vmax = 1e-8

                _, ax_pred = plot.contour_plot(
                    self.grids_[0], self.grids_[1], Z_slice,
                    ax=ax_pred, vmin=-vmax, vmax=vmax, levels=30, cbar=False,
                )
                sm = plot.map_color(vmin=-vmax, vmax=vmax)
                fig.colorbar(sm, ax=ax_pred, shrink=0.8, pad=0.04)
                ax_pred.set(title=f"Predicted {order}", xlabel="X1", ylabel="X2")

                if plot_error:
                    ax_err = axes[base_row + 1, c]
                    Z_err_full = (Z_flat - true_derivs[order]).reshape(grid_shape)
                    Z_err_slice = Z_err_full[:, :, z_mid]

                    vmax_err = np.max(np.abs(Z_err_slice))
                    if vmax_err == 0:
                        vmax_err = 1e-8

                    _, ax_err = plot.contour_plot(
                        self.grids_[0], self.grids_[1], Z_err_slice,
                        ax=ax_err, vmin=-vmax_err, vmax=vmax_err, levels=30, cbar=False,
                    )
                    sm_err = plot.map_color(vmin=-vmax_err, vmax=vmax_err)
                    fig.colorbar(sm_err, ax=ax_err, shrink=0.8, pad=0.04)
                    ax_err.set(title=f"Error {order}", xlabel="X1", ylabel="X2")

            for idx in range(n_plots, primary_rows * cols):
                base_row = (idx // cols) * (2 if plot_error else 1)
                c = idx % cols
                axes[base_row, c].axis('off')
                if plot_error:
                    axes[base_row + 1, c].axis('off')

            fig.suptitle(f"3D Derivatives (Central Slice X3 ≈ {z_val:.2g})" + (" & Errors" if plot_error else ""),
                         fontsize=14)
            plt.show()

        else:
            logger.warning(f"Plotting derivatives not supported for dimension {self.dim} > 3.")
