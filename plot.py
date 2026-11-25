import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, multivariate_normal

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import seaborn as sns

CMAP = sns.diverging_palette(20, 220, as_cmap=True)


def norm_color(vmin, vmax):
    return Normalize(vmin=vmin, vmax=vmax)


def map_color(vmin, vmax):
    norm_obj = norm_color(vmin=vmin, vmax=vmax)
    return ScalarMappable(norm_obj, cmap=CMAP)


def contour_plot(X, Y, Z, fig=None, ax=None, vmin=None, vmax=None, cbar=True, **kwargs):
    Z_plot = np.ma.masked_invalid(Z)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    norm_obj = Normalize(vmin=vmin, vmax=vmax)
    ax.contourf(X, Y, Z_plot, cmap=CMAP, norm=norm_obj)
    if cbar:
        sm = map_color(vmin, vmax)
        fig.colorbar(sm, ax=ax)
    return fig, ax


def mvn_pairplot(mu, sigma, true_vals=None, title='MVN Density Pairplot',
                 labels=None, num_grid_points=100, one_figsize=2.2, **kwargs):
    """
    Generates a lower-triangular corner pair plot displaying the EXACT
    analytical probability density functions of a Multivariate Normal
    distribution of any dimension (no simulations or random samples used).
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    dim = len(mu)
    if true_vals is not None:
        assert len(true_vals) == dim
        assert np.isfinite(true_vals).all()

    if sigma.shape != (dim, dim):
        raise ValueError("Covariance matrix shape must match mean vector length.")

    if labels is None:
        labels = [f'$X_{i}$' for i in range(dim)]

    stdevs = np.sqrt(np.diagonal(sigma))
    mins = mu - 3.5 * stdevs
    maxs = mu + 3.5 * stdevs
    if true_vals is not None:
        true_vals = np.asarray(true_vals, dtype=float).ravel()
        mins = np.minimum(mins, true_vals - 1.5 * stdevs)
        maxs = np.maximum(maxs, true_vals + 1.5 * stdevs)

    fig, axes = plt.subplots(dim, dim, figsize=(one_figsize * dim, one_figsize * dim), squeeze=False)

    for i in range(dim):
        for j in range(dim):
            ax = axes[i, j]

            if j > i:
                ax.axis('off')
                continue

            if i == j:
                x_eval = np.linspace(mins[i], maxs[i], num_grid_points * 2)
                pdf_eval = norm.pdf(x_eval, loc=mu[i], scale=stdevs[i])
                ax.plot(x_eval, pdf_eval, color='midnightblue', linewidth=2)
                ax.fill_between(x_eval, pdf_eval, color='midnightblue', alpha=0.3)
                ax.axvline(mu[i], color='darkorange', ls='--', lw=1.5, label='Estimate')
                if true_vals is not None:
                    ax.axvline(true_vals[i], color='crimson', ls='-', lw=1.8, label='True $\\theta^*$')

                ax.set_xlim(mins[i], maxs[i])
                ax.get_yaxis().set_visible(False)
                ax.grid(True, linestyle=':', alpha=0.5)
            else:
                x_eval = np.linspace(mins[j], maxs[j], num_grid_points)
                y_eval = np.linspace(mins[i], maxs[i], num_grid_points)
                X, Y = np.meshgrid(x_eval, y_eval)
                pos = np.dstack((X, Y))

                mu_sub = np.array([mu[j], mu[i]])
                sigma_sub = np.array([
                    [sigma[j, j], sigma[j, i]],
                    [sigma[i, j], sigma[i, i]]
                ])

                mvn_sub = multivariate_normal(mean=mu_sub, cov=sigma_sub, allow_singular=True)
                Z = mvn_sub.pdf(pos)

                ax.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.85)
                ax.contour(X, Y, Z, levels=5, colors='midnightblue', linewidths=0.5, alpha=0.5)
                ax.scatter(mu[j], mu[i], color='crimson', marker='x', s=40, zorder=5)
                if true_vals is not None:
                    ax.scatter(true_vals[j], true_vals[i], color='crimson', marker='x', s=50, zorder=6)

                ax.set_xlim(mins[j], maxs[j])
                ax.set_ylim(mins[i], maxs[i])
                ax.grid(True, linestyle=':', alpha=0.5)

            if i == dim - 1:
                ax.set_xlabel(labels[j], fontsize=11, labelpad=10)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=11, labelpad=10)
            if i != dim - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)

    plt.suptitle(title, y=0.99, fontsize=13, fontweight='bold', color='midnightblue')
    plt.tight_layout()
    return fig, axes


def plot_diagnostics(model, x_test, y_test, e_test, true_params, pde_name):
    """
    6-Panel Academic & Industrial Diagnostic Dashboard:
      1. Parity Plot (u* vs \\hat{u})
      2. Pointwise Spatial Error Map |u* - \\hat{u}|
      3. Conditional Error Funnel
      4. Calibration Reliability Curve (Empirical vs. Nominal Coverage)
      5. Spectral Filter Factor Decay (\\bar{\\omega}_m vs Eigenvalues s_m)
      6. Parameter Forest Plot (\\hat{\\theta}_T +/- 95% Sandwich CI vs \\theta^*)
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
    ax1.set(
        title="1. Field Parity ($u^*$ vs $\\hat{u}$)",
        xlabel="True $u^*(\\mathbf{x})$",
        ylabel="Predicted $\\hat{u}(\\mathbf{x})$",
    )
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
    ax3.scatter(se_mean, err, s=8, color='#1f77b4', alpha=0.4, edgecolors='none', label='Observed Error')
    se_line = np.linspace(0, se_mean.max(), 100)
    ax3.plot(se_line, 1.96 * se_line, color='crimson', ls='--', lw=1.5, label='95% Theoretical Bound')
    ax3.plot(se_line, -1.96 * se_line, color='crimson', ls='--', lw=1.5)
    ax3.fill_between(se_line, -1.96 * se_line, 1.96 * se_line, color='crimson', alpha=0.08)
    ax3.set_xlabel(r"Predictive Standard Error ($\mathrm{se\_mean}$)", fontweight='bold')
    ax3.set_ylabel(r"Raw Empirical Error ($u - \hat{u}$)", fontweight='bold')
    ax3.set_title("3. Conditional Error Funnel", fontweight='bold')
    ax3.grid(True, ls='--', alpha=0.5)
    ax3.legend(loc="upper left", frameon=True)

    # Panel 4: Calibration Reliability Curve
    ax4 = axes[1, 0]
    nom_grid = np.linspace(0.05, 0.99, 20)
    emp_cov = [np.mean(np.abs(err) <= stats.norm.ppf(1.0 - (1.0 - a) / 2.0) * se_mean) for a in nom_grid]
    ax4.plot(nom_grid, emp_cov, 'o-', color='#2ca02c', lw=2, ms=4, label='Empirical Coverage')
    ax4.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Ideal Calibration')
    ax4.set(
        title="4. Uncertainty Calibration Curve",
        xlabel="Nominal Confidence",
        ylabel="Empirical Coverage",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
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
        ax6.errorbar(
            ests, y_pos, xerr=1.96 * se_t, fmt='o', color='midnightblue',
            ecolor='royalblue', elinewidth=2.5, capsize=5, label='Estimate ± 95% CI',
        )
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


def plot_res_summary(
    res: pd.DataFrame | pd.Series,
    param_df: pd.DataFrame | dict = None,
    pde_name: str = "PDE Benchmark",
    save_path: str = None,
    metric_suffix: str = "test",
    annot=False,
):
    """
    4-panel publication dashboard with native MultiIndex / comparative group support:
      - Accepts single DataFrame/Series (single-run) OR MultiIndex DataFrame (e.g. pd.concat(res, names=['weak'])).
      - Accepts param_df as a single DataFrame OR dict of DataFrames (e.g. params={'0': p0, '1': p1}).
      - Produces side-by-side dodged boxplots comparing formulations across all 4 panels.
    """
    df = res.to_frame().T if isinstance(res, pd.Series) else res
    df = df.astype(float)

    # 1. MultiIndex Detection & Group Extraction
    is_multi = False
    groups = [None]
    group_labels = ["All"]
    badge_prefixes = [""]
    grp_level_names = []
    grp_levels = []

    if isinstance(df.index, pd.MultiIndex):
        is_multi = True
        if df.index.nlevels > 1 and df.index.names[-1] in ('iter', 'iteration', 'rep', None):
            grp_levels = list(range(df.index.nlevels - 1))
        else:
            grp_levels = list(range(df.index.nlevels))

        grp_level_names = [df.index.names[i] or f"Level_{i}" for i in grp_levels]

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

    # 2. Adaptive Group Label Formatting
    palette = ["#1f77b4", "#e6550d", "#2ca02c", "#7b3294", "#d62728", "#8c564b", "#e377c2", "#17becf"]
    n_groups = len(groups)
    group_colors = [palette[i % len(palette)] for i in range(n_groups)]

    if is_multi:
        multi_palette = ["#1f77b4", "#e6550d", "#2ca02c", "#7b3294"]
        group_colors = [multi_palette[i % len(multi_palette)] for i in range(len(groups))]

        def _format_token(name: str, val):
            v_int = (
                int(val)
                if isinstance(val, (int, float, np.integer, np.floating)) and np.isclose(val, int(val))
                else val
            )
            return f"{name}={v_int}"

        group_labels, badge_prefixes = [], []
        for g in groups:
            vals = g if isinstance(g, (tuple, list)) else [g]
            tokens = [_format_token(n, v) for n, v in zip(grp_level_names, vals)]
            group_labels.append(" | ".join(tokens))
            badge_prefixes.append(f"{''.join(tokens).strip('+')}: ")

    width = 0.35 if n_groups == 2 else (0.75 / n_groups if n_groups > 1 else 0.45)
    offsets = (
        np.linspace(-width * (n_groups - 1) / 2, width * (n_groups - 1) / 2, n_groups)
        if n_groups > 1
        else [0.0]
    )

    def get_df_group(g):
        if isinstance(df.index, pd.MultiIndex):
            lvl_arg = grp_levels[0] if len(grp_levels) == 1 else grp_levels
            return df.xs(g, level=lvl_arg)
        return df

    def get_val(sub_df: pd.DataFrame, key: str) -> np.ndarray:
        for k in [f"{key}_{metric_suffix}", f"{key}_grid", f"{key}_test", key]:
            if k in sub_df.columns:
                vals = sub_df[k].dropna().values
                if len(vals) > 0:
                    return vals
        return np.array([0.0])

    def get_param_df_group(g):
        if isinstance(param_df, dict):
            if g in param_df:
                return param_df[g]
            if str(g) in param_df:
                return param_df[str(g)]
            return next(iter(param_df.values()), None)
        return param_df

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")

    # Panel A: Sobolev Relative Error Hierarchy (%)
    ax1 = axes[0, 0]
    sob_items = [("rel_l2", "Rel. $L^2$ Error"), ("rel_h1", "Rel. $H^1$ Error"), ("rel_h2", "Rel. $H^2$ Error")]
    x = np.arange(len(sob_items))

    for g_idx, g in enumerate(groups):
        df_g = get_df_group(g)
        sob_vals = [get_val(df_g, k) * 100 for k, _ in sob_items]
        pos = x + offsets[g_idx]

        ax1.boxplot(
            sob_vals,
            positions=pos,
            widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=group_colors[g_idx], alpha=0.75, edgecolor="black", linewidth=0.9),
            medianprops=dict(color="darkred" if g_idx == 0 else "black", linewidth=1.6),
            showfliers=False,
        )

        if annot:
            for i, data in enumerate(sob_vals):
                med = np.median(data)
                ha = "right" if (is_multi and g_idx == 0) else ("left" if is_multi else "center")
                x_shift = -3 if (is_multi and g_idx == 0) else (3 if is_multi else 0)
                ax1.annotate(
                    f"{med:.3f}%",
                    xy=(pos[i], med),
                    xytext=(x_shift, 4),
                    textcoords="offset points",
                    ha=ha,
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=group_colors[g_idx] if is_multi else "#12466b",
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in sob_items], fontsize=10)
    ax1.set_ylabel("Relative Error (%)", fontweight="bold")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_title("A. Sobolev Relative Error Hierarchy ($L^2, H^1, H^2$)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    if is_multi:
        proxies = [Patch(facecolor=group_colors[i], alpha=0.75, label=group_labels[i]) for i in range(n_groups)]
        ax1.legend(handles=proxies, loc="upper left", frameon=True, facecolor="white")

    # Panel B: Spatial Error Decomposition & Interval Sharpness
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

    for g_idx, g in enumerate(groups):
        df_g = get_df_group(g)
        sp_vals = [get_val(df_g, k) for k, _ in sp_items]
        pos = y_pos + offsets[g_idx]

        ax2.boxplot(
            sp_vals,
            positions=pos,
            vert=False,
            widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=group_colors[g_idx], alpha=0.75, edgecolor="black", linewidth=0.9),
            medianprops=dict(color="darkred" if g_idx == 0 else "black", linewidth=1.6),
            showfliers=False,
        )

        if annot:
            for i, data in enumerate(sp_vals):
                med = np.median(data)
                y_shift = -3 if (is_multi and g_idx == 0) else (3 if is_multi else 0)
                va = "top" if (is_multi and g_idx == 0) else ("bottom" if is_multi else "center")
                ax2.annotate(
                    f"{med:.4f}",
                    xy=(med, pos[i]),
                    xytext=(4, y_shift),
                    textcoords="offset points",
                    ha="left",
                    va=va,
                    fontsize=7.5,
                    fontweight="bold",
                    color=group_colors[g_idx] if is_multi else "black",
                )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl in sp_items], fontsize=9)
    ax2.set_xlabel("Magnitude (Field Units)", fontweight="bold")
    ax2.set_title("B. Spatial Error Decomposition & Interval Sharpness", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    if is_multi:
        proxies = [Patch(facecolor=group_colors[i], alpha=0.75, label=group_labels[i]) for i in range(n_groups)]
        ax2.legend(handles=proxies, loc="lower right", frameon=True, facecolor="white")

    # Panel C: Empirical vs. Nominal Coverage Calibration
    ax3 = axes[1, 0]
    nom = np.array([90.0, 95.0, 99.0])
    xc = np.arange(3)

    ax3.scatter(xc, nom, color="#2ca02c", s=70, zorder=5, marker="D", label="Nominal Target")
    for i in range(len(nom)):
        span = width * n_groups / 2 + 0.1
        ax3.hlines(nom[i], xc[i] - span, xc[i] + span, colors="#2ca02c", linestyles="--", linewidth=1.5, zorder=4)

    for g_idx, g in enumerate(groups):
        df_g = get_df_group(g)
        emp = [get_val(df_g, k) * 100 for k in ["cov90", "cov95", "cov99"]]
        pos = xc + offsets[g_idx]

        ax3.boxplot(
            emp,
            positions=pos,
            widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=group_colors[g_idx], alpha=0.75, edgecolor="black", linewidth=0.9),
            medianprops=dict(color="black", linewidth=1.6),
            showfliers=False,
        )

        if annot:
            for i, data in enumerate(emp):
                med = np.median(data)
                ha = "right" if (is_multi and g_idx == 0) else ("left" if is_multi else "center")
                x_shift = -3 if (is_multi and g_idx == 0) else (3 if is_multi else 0)
                ax3.annotate(
                    f"{med:.1f}%",
                    xy=(pos[i], med),
                    xytext=(x_shift, 4),
                    textcoords="offset points",
                    ha=ha,
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=group_colors[g_idx] if is_multi else "black",
                )

    ax3.set_xticks(xc)
    ax3.set_xticklabels(["90% CI", "95% CI", "99% CI"], fontsize=10)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.set_ylabel("Coverage Probability (%)", fontweight="bold")
    ax3.set_title(r"C. Uncertainty Interval Calibration ($90\%, 95\%, 99\%$)", fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    c_proxies = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c", markersize=7, label="Nominal Target")
    ]
    if is_multi:
        c_proxies.extend([Patch(facecolor=group_colors[i], alpha=0.75, label=group_labels[i]) for i in range(n_groups)])
    ax3.legend(handles=c_proxies, loc="lower left", frameon=True, facecolor="white")

    # Panel D: Parameter Z-Score Distribution & Empirical 95% Coverage
    ax4 = axes[1, 1]
    ref_param_df = get_param_df_group(groups[0])

    trainable_descriptors = []
    if ref_param_df is not None and not ref_param_df.empty:
        trainable_rows = ref_param_df[ref_param_df["Role"].astype(str).str.lower() == "trainable"]
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
            lbl = f"{op_label} ($\\theta^*={true_v:.3g}$)" if not np.isnan(true_v) else f"{op_label}"
            trainable_descriptors.append((param_idx, t_pos, lbl, true_v))
    else:
        var_keys = [k for k in df.columns if k.startswith("coef_var_")]
        dim_v = int(np.sqrt(len(var_keys))) if var_keys else 0
        coef_cols = sorted(
            [c for c in df.columns if c.startswith("coef_mean_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        for t_pos in range(min(dim_v, len(coef_cols))):
            param_idx = int(coef_cols[t_pos].split("_")[-1])
            trainable_descriptors.append((param_idx, t_pos, f"$\\theta_{{{param_idx}}}$", np.nan))

    n_params = len(trainable_descriptors)
    y_p = np.arange(n_params)

    ax4.axvline(0.0, color="crimson", ls="-", lw=1.5, zorder=3, label="Target ($z=0$)")
    ax4.axvline(-1.96, color="black", ls="--", lw=1.2, alpha=0.7, zorder=2, label=r"Nominal 95% ($\pm 1.96$)")
    ax4.axvline(1.96, color="black", ls="--", lw=1.2, alpha=0.7, zorder=2)
    ax4.axvspan(-1.96, 1.96, color="#2ca02c", alpha=0.08, zorder=1)

    all_z_groups = []
    all_se_groups = []

    for g_idx, g in enumerate(groups):
        df_g = get_df_group(g)
        p_df_g = get_param_df_group(g)

        group_z_list = []
        group_se_list = []
        for param_idx, t_pos, _, true_v in trainable_descriptors:
            est_key = f"coef_mean_{param_idx}"
            if est_key in df_g.columns:
                ests = df_g[est_key].dropna().values
            elif p_df_g is not None and not p_df_g.empty:
                ests = np.array([float(p_df_g.iloc[param_idx]["Estimate"])])
            else:
                ests = np.array([0.0])

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

            if not np.isnan(true_v):
                z = (ests - true_v) / np.maximum(ses, 1e-12)
            else:
                z = ests / np.maximum(ses, 1e-12)

            group_z_list.append(z)
            group_se_list.append(ses)

        all_z_groups.append(group_z_list)
        all_se_groups.append(group_se_list)

        pos = y_p + offsets[g_idx]
        ax4.boxplot(
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
            showfliers=False,
        )

    for i in range(n_params):
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
                cov = np.mean(np.abs(zd) <= 1.95996)
                med_se_norm = np.median(all_se_groups[g_idx][i]) * np.sqrt(n_obs)
            else:
                cov, med_se_norm = 0.0, 0.0

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

    ax4.set_yticks(y_p)
    ax4.set_yticklabels([desc[2] for desc in trainable_descriptors], fontsize=10)
    ax4.set_xlabel(
        r"Parameter Z-Score $(\hat{\theta} - \theta^*) / \mathrm{SE}(\hat{\theta})$",
        fontweight="bold",
    )
    ax4.set_title("D. Trainable Parameter Z-Score & 95% CI Coverage", fontweight="bold", pad=10)
    ax4.grid(axis="x", ls="--", alpha=0.5)

    d_proxies = [
        plt.Line2D([0], [0], color="crimson", ls="-", lw=1.5, label="Target ($z=0$)"),
        plt.Line2D([0], [0], color="black", ls="--", lw=1.2, label=r"95% ($\pm 1.96$)"),
    ]
    if is_multi:
        d_proxies.extend([
            Patch(facecolor=group_colors[i], alpha=0.75, label=group_labels[i])
            for i in range(n_groups)
        ])
    ax4.legend(handles=d_proxies, loc="lower left", fontsize=8, frameon=True, facecolor="white")

    title_suffix = f" (Comparing {', '.join(grp_level_names)})" if is_multi else ""
    fig.suptitle(f"EWABSplinePDE Summary Statistics: {pde_name}{title_suffix}", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
