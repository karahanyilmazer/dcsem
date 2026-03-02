# %% Imports and config
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from dcsem import SpectralDCM, get_colormap, set_style, to_latex_label
from dcsem.diagnostics import (
    compute_2d_loss_landscape,
    compute_hessian_diagnostics,
    parametric_bootstrap_uncertainty,
    profile_likelihood_1d,
)
from dcsem.numerics import (
    compute_confidence_intervals,
    compute_correlation_matrix,
    compute_standard_errors,
    safe_hessian_inversion,
)
from utils import get_out_dir, get_width_height_latex, log_run

set_style()
width, height = get_width_height_latex()
cmap = get_colormap("YlGnBu_r")
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

SEED = 42
N_RUNS = 50
SNR_GRID = np.array([1, 2, 5, 10, 20, 50, 100])

N_BOOTSTRAP = 0  # set to 50 for bootstrap uncertainty (slow)
DIAG_SNR_IDX = 2  # SNR_GRID[2] = 5 — representative problem SNR
DIAG_RUN_IDX = 0  # run index for landscape / profile figures
DIAG_N_PROFILE_RUNS = 5  # number of MC runs for profile likelihood figure
HESS_NEAR_SINGULAR_THRESH = 1e8
HESS_STEP = 1e-3  # step size for Hessian in scaled space

# %% SpectralDCM instance + model spec

_spdcm = SpectralDCM(
    n_rois=2, TR=1.0, self_connection=-1.0, freq_lo=0.01, freq_hi=0.1, n_freqs=32
)
theta_true = np.array([0.4, 0.6, np.log(0.05)])
theta_zero = np.array([0.2, 0.2, np.log(0.1)])
param_names = ["a01", "a10", "log_sigma_e"]
param_bounds = [(0.0, 1.0), (0.0, 1.0), (-10.0, 2.0)]


def model(theta, _x):
    return _spdcm.predict_csd(theta)


n_snrs = len(SNR_GRID)
n_params = len(theta_true)

# Parameter scaling helpers (map param_bounds to [0, 1]^n)
_lowers = np.array([b[0] for b in param_bounds])  # [0, 0, -10]
_scales = np.array([b[1] - b[0] for b in param_bounds])  # [1, 1, 12]


def _to_scaled(theta):
    return (theta - _lowers) / _scales


def _from_scaled(s):
    return s * _scales + _lowers


# %% make_objective helper (copied verbatim from spdcm_generic.py)


def make_objective(model_func, y_obs, x_data, loss_fn, normalize=True):
    """Define objective once; reuse for optimization, Hessian, and landscapes.

    When predict_csd returns np.inf (unstable A), the objective returns 1e10
    to guide the optimizer away from unstable regions.
    """
    if normalize:
        y_mean = y_obs.mean(axis=0, keepdims=True)
        y_std = y_obs.std(axis=0, keepdims=True) + 1e-12
        y_obs_norm = (y_obs - y_mean) / y_std

        def objective(theta):
            y_pred = model_func(theta, x_data)
            if not np.all(np.isfinite(y_pred)):
                return 1e10
            y_pred_norm = (y_pred - y_mean) / y_std
            return loss_fn(y_obs_norm, y_pred_norm)

        return objective, y_mean, y_std, y_obs_norm
    else:

        def objective(theta):
            y_pred = model_func(theta, x_data)
            if not np.all(np.isfinite(y_pred)):
                return 1e10
            return loss_fn(y_obs, y_pred)

        return objective, None, None, None


def run_snr_sweep():
    """Run full SNR sweep and produce all diagnostic figures.

    Produces Figures 1-9. Set N_BOOTSTRAP>0 for bootstrap uncertainty.
    Set N_BOOTSTRAP_CALIB>0 for the calibration experiment.
    """
    # %% Result arrays

    theta_est_all = np.full((n_snrs, N_RUNS, n_params), np.nan)
    se_all = np.full((n_snrs, N_RUNS, n_params), np.nan)
    ci_all = np.full((n_snrs, N_RUNS, n_params, 2), np.nan)
    mse_all = np.full((n_snrs, N_RUNS), np.nan)
    converged = np.zeros((n_snrs, N_RUNS), dtype=bool)

    # New diagnostic arrays
    hess_cond_all = np.full((n_snrs, N_RUNS), np.nan)
    hess_eigmin_all = np.full((n_snrs, N_RUNS), np.nan)
    n_neg_eigvals_all = np.zeros((n_snrs, N_RUNS), dtype=int)
    near_singular_all = np.zeros((n_snrs, N_RUNS), dtype=bool)
    corr_max_all = np.full((n_snrs, N_RUNS), np.nan)
    bs_se_all = np.full((n_snrs, N_RUNS, n_params), np.nan)
    bs_ci95_all = np.full((n_snrs, N_RUNS, n_params, 2), np.nan)

    # Store objectives for diagnostic figures (keyed by (i, j))
    _stored_objs = {}
    _stored_ymean = {}
    _stored_ystd = {}

    # %% Monte Carlo sweep

    for i, snr in enumerate(SNR_GRID):
        for j in range(N_RUNS):
            rng_ij = np.random.default_rng(SEED * 1000 + i * N_RUNS + j)
            y_obs = _spdcm.generate_noisy_csd(theta_true, snr=snr, rng=rng_ij)
            obj, y_mean, y_std, y_obs_norm = make_objective(
                model, y_obs, None, mean_squared_error
            )
            res = minimize(obj, theta_zero, method="L-BFGS-B", bounds=param_bounds)
            failed = not res.success or res.fun >= 1e9
            if failed:
                continue
            converged[i, j] = True
            theta_est_all[i, j] = res.x
            mse_all[i, j] = res.fun

            # Store obj for diagnostic figures at DIAG_SNR_IDX
            if i == DIAG_SNR_IDX and j < DIAG_N_PROFILE_RUNS:
                _stored_objs[(i, j)] = obj
                _stored_ymean[(i, j)] = y_mean
                _stored_ystd[(i, j)] = y_std

            # --- Hessian in scaled space (all params on [0,1] for numerical stability) ---
            def _scaled_obj(s, _obj=obj):
                return _obj(_from_scaled(s))

            theta_s = _to_scaled(res.x)
            try:
                H_s = nd.Hessian(_scaled_obj, step=HESS_STEP)(theta_s)
                H_s = 0.5 * (H_s + H_s.T)
                # Back-transform: H_theta[i,j] = H_s[i,j] / (scales[i] * scales[j])
                H = H_s / np.outer(_scales, _scales)
            except Exception:
                H = nd.Hessian(obj)(res.x)
                H = 0.5 * (H + H.T)

            # --- Hessian diagnostics (positive-spectrum condition number) ---
            hd = compute_hessian_diagnostics(H, HESS_NEAR_SINGULAR_THRESH)
            hess_cond_all[i, j] = hd["condition_number"]
            hess_eigmin_all[i, j] = hd["eigvals_min"]
            n_neg_eigvals_all[i, j] = hd["n_negative_eigvals"]
            near_singular_all[i, j] = hd["is_near_singular"]

            # --- Canonical NLL Hessian for SE (avoids sigma_sq ambiguity) ---
            # NLL = 0.5 * SSE / sigma2 in normalized space; Cov = H_NLL^{-1} exactly.
            r_norm = y_obs_norm - (model(res.x, None) - y_mean) / y_std
            sigma2_norm = float(
                np.sum(r_norm**2) / max(len(r_norm.ravel()) - n_params, 1)
            )

            def _nll_norm(
                theta, _ym=y_mean, _ys=y_std, _yo=y_obs_norm, _s2=sigma2_norm
            ):
                y_pred = model(theta, None)
                if not np.all(np.isfinite(y_pred)):
                    return 1e10
                r = (_yo - (y_pred - _ym) / _ys).ravel()
                return 0.5 * float(np.dot(r, r)) / _s2

            def _nll_scaled(s, _nll=_nll_norm):
                return _nll(_from_scaled(s))

            try:
                H_nll_s = nd.Hessian(_nll_scaled, step=HESS_STEP)(theta_s)
                H_nll_s = 0.5 * (H_nll_s + H_nll_s.T)
                H_nll = H_nll_s / np.outer(_scales, _scales)
                # Cov = H_NLL^{-1} via pinvh; sigma_sq=1.0 (absorbed into NLL)
                cov, _ = safe_hessian_inversion(
                    H_nll, 1.0, regularization=1e-6, method="pinvh"
                )
                se_all[i, j] = compute_standard_errors(cov, warn_negative=False)
                ci_all[i, j] = compute_confidence_intervals(res.x, se_all[i, j])
                corr = compute_correlation_matrix(cov)
                corr_max_all[i, j] = float(np.max(np.abs(corr - np.eye(n_params))))
            except (np.linalg.LinAlgError, Exception):
                pass  # leave as NaN

            # --- Bootstrap (only if N_BOOTSTRAP > 0) ---
            if N_BOOTSTRAP > 0:

                def _make_obj_fn(y, _model=model):
                    return make_objective(_model, y, None, mean_squared_error)

                bs_se, bs_ci = parametric_bootstrap_uncertainty(
                    _spdcm, res.x, snr, _make_obj_fn, param_bounds, N_BOOTSTRAP, rng_ij
                )
                bs_se_all[i, j] = bs_se
                if 95 in bs_ci:
                    bs_ci95_all[i, j] = bs_ci[95]

        n_conv = int(converged[i].sum())
        print(f"SNR={snr}: {n_conv}/{N_RUNS} converged")

    # %% Summary statistics

    bias = np.nanmean(theta_est_all - theta_true, axis=1)  # (n_snrs, n_params)
    bias_sd = np.nanstd(theta_est_all - theta_true, axis=1)
    rmse = np.sqrt(np.nanmean((theta_est_all - theta_true) ** 2, axis=1))
    rmse_sd = np.nanstd((theta_est_all - theta_true) ** 2, axis=1) / (
        2 * rmse + 1e-30
    )  # propagated SD
    emp_sd = np.nanstd(theta_est_all, axis=1)
    mean_se = np.nanmean(se_all, axis=1)

    # CI coverage at 95% (Hessian)
    inside_hess = (ci_all[..., 0] <= theta_true) & (theta_true <= ci_all[..., 1])
    coverage_hess95 = np.nanmean(
        inside_hess.astype(float), axis=1
    )  # (n_snrs, n_params)

    # Bootstrap CI coverage (if collected)
    if N_BOOTSTRAP > 0:
        inside_bs95 = (bs_ci95_all[..., 0] <= theta_true) & (
            theta_true <= bs_ci95_all[..., 1]
        )
        coverage_bs95 = np.nanmean(inside_bs95.astype(float), axis=1)
    else:
        coverage_bs95 = None

    failure_rate = 1.0 - np.mean(converged, axis=1)
    mean_mse = np.nanmean(mse_all, axis=1)

    IMG_DIR = get_out_dir(
        type="img",
        subfolder="inversion",
        extra_subfolders=["L-BFGS-B", "spdcm_2roi", "noise_sweep"],
    )
    LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {IMG_DIR}")

    latex_labels = [to_latex_label(name) for name in param_names]

    # %% Figure 1 — Parameter recovery (Bias and RMSE) with ±1 SD bands

    fig, axes = plt.subplots(2, n_params, figsize=(width, height), sharex=True)

    for p in range(n_params):
        # Top row: Bias ± 1 SD
        ax = axes[0, p]
        ax.semilogx(SNR_GRID, bias[:, p], color=default_colors[0], marker="o", lw=1.5)
        ax.fill_between(
            SNR_GRID,
            bias[:, p] - bias_sd[:, p],
            bias[:, p] + bias_sd[:, p],
            color=default_colors[0],
            alpha=0.2,
        )
        ax.axhline(0, color="gray", linestyle="--", lw=1.0)
        ax.set_title(f"Bias — {latex_labels[p]}")
        ax.set_ylabel("Bias")
        ax.grid(True, alpha=0.3)

        # Bottom row: RMSE ± propagated SD
        ax = axes[1, p]
        ax.semilogx(SNR_GRID, rmse[:, p], color=default_colors[1], marker="s", lw=1.5)
        ax.fill_between(
            SNR_GRID,
            np.maximum(rmse[:, p] - rmse_sd[:, p], 0),
            rmse[:, p] + rmse_sd[:, p],
            color=default_colors[1],
            alpha=0.2,
        )
        ax.set_title(f"RMSE — {latex_labels[p]}")
        ax.set_xlabel("SNR")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"\textbf{Parameter Recovery vs SNR}", y=1.01)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "recovery_bias_rmse.png", bbox_inches="tight")
    plt.savefig(LATEX_DIR / "spdcm_sweep_recovery.pdf", bbox_inches="tight")
    plt.show()

    # %% Figure 2 — Uncertainty calibration (log-log SD vs SE, CI coverage)

    fig, axes = plt.subplots(2, n_params, figsize=(width, height), sharex=True)

    for p in range(n_params):
        # Top row: log-log Empirical SD vs Hessian SE (and bootstrap SE if available)
        ax = axes[0, p]
        ax.loglog(
            SNR_GRID,
            emp_sd[:, p],
            color=default_colors[0],
            marker="o",
            lw=1.5,
            label="Empirical SD",
        )
        ax.loglog(
            SNR_GRID,
            mean_se[:, p],
            color=default_colors[2],
            marker="^",
            lw=1.5,
            linestyle="--",
            label="Hessian SE (local quadratic approx)",
        )
        if N_BOOTSTRAP > 0:
            mean_bs_se = np.nanmean(bs_se_all, axis=1)
            ax.loglog(
                SNR_GRID,
                mean_bs_se[:, p],
                color=default_colors[3],
                marker="v",
                lw=1.5,
                linestyle=":",
                label=f"Bootstrap SE (N={N_BOOTSTRAP})",
            )
        ax.set_title(f"SD vs SE — {latex_labels[p]}")
        ax.set_ylabel("Std / SE")
        ax.grid(True, alpha=0.3, which="both")
        if p == 0:
            ax.legend(fontsize="small")

        # Bottom row: CI coverage (Hessian + bootstrap)
        ax = axes[1, p]
        ax.semilogx(
            SNR_GRID,
            coverage_hess95[:, p],
            color=default_colors[3],
            marker="D",
            lw=1.5,
            label="Hessian CI 95%",
        )
        if coverage_bs95 is not None:
            ax.semilogx(
                SNR_GRID,
                coverage_bs95[:, p],
                color=default_colors[4],
                marker="P",
                lw=1.5,
                linestyle="--",
                label=f"Bootstrap CI 95% (N={N_BOOTSTRAP})",
            )
        ax.axhline(0.95, color="gray", linestyle="--", lw=1.0)
        ax.set_title(f"CI Coverage — {latex_labels[p]}")
        ax.set_xlabel("SNR")
        ax.set_ylabel("Coverage")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if p == 0:
            ax.legend(fontsize="small")

    subtitle = (
        "Note: log\\_sigma\\_e coverage may be anomalous due to identifiability ridge."
    )
    fig.suptitle(r"\textbf{Uncertainty Calibration vs SNR}" + f"\n{subtitle}", y=1.03)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "calibration_sd_coverage.png", bbox_inches="tight")
    plt.savefig(LATEX_DIR / "spdcm_sweep_calibration.pdf", bbox_inches="tight")
    plt.show()

    # %% Figure 3 — Summary (Mean CSD MSE and Failure rate)

    fig, axes = plt.subplots(1, 2, figsize=(width, height / 1.5))

    ax = axes[0]
    ax.loglog(SNR_GRID, mean_mse, color=default_colors[0], marker="o", lw=1.5)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Mean CSD MSE")
    ax.set_title(r"Mean CSD MSE vs SNR")
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    ax.semilogx(SNR_GRID, failure_rate, color=default_colors[1], marker="s", lw=1.5)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Failure Rate")
    ax.set_title("Failure Rate vs SNR")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle(r"\textbf{Sweep Summary}", y=1.01)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "summary_mse_failure.png", bbox_inches="tight")
    plt.savefig(LATEX_DIR / "spdcm_sweep_summary.pdf", bbox_inches="tight")
    plt.show()

    # %% Figure 4 — Hessian diagnostics vs SNR

    # Replace inf condition numbers with a large sentinel for display
    cond_display = np.where(
        np.isinf(hess_cond_all) | np.isnan(hess_cond_all),
        HESS_NEAR_SINGULAR_THRESH * 10,
        hess_cond_all,
    )
    log10_cond = np.log10(cond_display + 1e-30)
    mean_log10_cond = np.nanmean(log10_cond, axis=1)
    sd_log10_cond = np.nanstd(log10_cond, axis=1)
    frac_near_singular = np.mean(near_singular_all, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(width, height / 1.5))

    ax = axes[0]
    ax.semilogx(SNR_GRID, mean_log10_cond, color=default_colors[0], marker="o", lw=1.5)
    ax.fill_between(
        SNR_GRID,
        mean_log10_cond - sd_log10_cond,
        mean_log10_cond + sd_log10_cond,
        color=default_colors[0],
        alpha=0.2,
    )
    ax.axhline(
        np.log10(HESS_NEAR_SINGULAR_THRESH),
        color="red",
        linestyle="--",
        lw=1.0,
        label=f"threshold = {HESS_NEAR_SINGULAR_THRESH:.0e}",
    )
    ax.set_xlabel("SNR")
    ax.set_ylabel(r"$\log_{10}$(condition number)")
    ax.set_title("Hessian Condition Number")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx(
        SNR_GRID, frac_near_singular, color=default_colors[1], marker="s", lw=1.5
    )
    ax.set_xlabel("SNR")
    ax.set_ylabel("Fraction near-singular")
    ax.set_title("Near-Singular Fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle(r"\textbf{Hessian Diagnostics vs SNR}", y=1.01)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "hessian_diagnostics.png", bbox_inches="tight")
    plt.show()

    # %% Figure 5 — Eigenvalue magnitude + correlation vs SNR

    eigmin_display = np.where(
        np.isnan(hess_eigmin_all), 1e-30, np.abs(hess_eigmin_all) + 1e-30
    )
    log10_eigmin = np.log10(eigmin_display)
    mean_log10_eigmin = np.nanmean(log10_eigmin, axis=1)
    sd_log10_eigmin = np.nanstd(log10_eigmin, axis=1)
    mean_corr_max = np.nanmean(corr_max_all, axis=1)
    sd_corr_max = np.nanstd(corr_max_all, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(width, height / 1.5))

    ax = axes[0]
    ax.semilogx(
        SNR_GRID, mean_log10_eigmin, color=default_colors[0], marker="o", lw=1.5
    )
    ax.fill_between(
        SNR_GRID,
        mean_log10_eigmin - sd_log10_eigmin,
        mean_log10_eigmin + sd_log10_eigmin,
        color=default_colors[0],
        alpha=0.2,
    )
    ax.set_xlabel("SNR")
    ax.set_ylabel(r"$\log_{10}|\lambda_{\min}|$")
    ax.set_title("Min Hessian Eigenvalue Magnitude")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx(SNR_GRID, mean_corr_max, color=default_colors[1], marker="s", lw=1.5)
    ax.fill_between(
        SNR_GRID,
        np.maximum(mean_corr_max - sd_corr_max, 0),
        np.minimum(mean_corr_max + sd_corr_max, 1),
        color=default_colors[1],
        alpha=0.2,
    )
    ax.axhline(0.9, color="red", linestyle="--", lw=1.0, label="0.9 threshold")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Max off-diagonal |correlation|")
    ax.set_title("Parameter Correlation")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    fig.suptitle(r"\textbf{Eigenvalue and Correlation vs SNR}", y=1.01)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "eigval_corr.png", bbox_inches="tight")
    plt.show()

    # %% Figure 6 — SE comparison (3 columns, log y-axis)

    fig, axes = plt.subplots(1, n_params, figsize=(width, height / 1.5))

    for p in range(n_params):
        ax = axes[p]
        ax.semilogy(
            SNR_GRID,
            emp_sd[:, p],
            color=default_colors[0],
            marker="o",
            lw=1.5,
            label="Empirical SD",
        )
        ax.semilogy(
            SNR_GRID,
            mean_se[:, p],
            color=default_colors[2],
            marker="^",
            lw=1.5,
            linestyle="--",
            label="Hessian SE (pinvh+NLL)",
        )
        if N_BOOTSTRAP > 0:
            mean_bs_se = np.nanmean(bs_se_all, axis=1)
            ax.semilogy(
                SNR_GRID,
                mean_bs_se[:, p],
                color=default_colors[3],
                marker="v",
                lw=1.5,
                linestyle=":",
                label=f"Bootstrap SE (N={N_BOOTSTRAP})",
            )
        ax.set_xlabel("SNR")
        ax.set_ylabel("SE")
        ax.set_title(latex_labels[p])
        ax.grid(True, alpha=0.3, which="both")
        if p == 0:
            ax.legend(fontsize="small")

    fig.suptitle(r"\textbf{SE Comparison vs SNR (log y-scale)}", y=1.01)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "se_comparison.png", bbox_inches="tight")
    plt.show()

    # %% Figure 7 — 2D loss landscapes at DIAG_SNR

    diag_key = (DIAG_SNR_IDX, DIAG_RUN_IDX)
    if diag_key in _stored_objs and not np.all(
        np.isnan(theta_est_all[DIAG_SNR_IDX, DIAG_RUN_IDX])
    ):
        obj_diag = _stored_objs[diag_key]
        theta_est_diag = theta_est_all[DIAG_SNR_IDX, DIAG_RUN_IDX]
        diag_snr = SNR_GRID[DIAG_SNR_IDX]

        p1_vals_1, p2_vals_1, Z1 = compute_2d_loss_landscape(
            obj_diag, theta_est_diag, p1_idx=0, p2_idx=2, n_grid=40
        )
        p1_vals_2, p2_vals_2, Z2 = compute_2d_loss_landscape(
            obj_diag, theta_est_diag, p1_idx=1, p2_idx=2, n_grid=40
        )

        fig, axes = plt.subplots(1, 2, figsize=(width, height / 1.5))

        for ax, p1v, p2v, Z, pidx1, pidx2 in [
            (axes[0], p1_vals_1, p2_vals_1, Z1, 0, 2),
            (axes[1], p1_vals_2, p2_vals_2, Z2, 1, 2),
        ]:
            Z_finite = Z[np.isfinite(Z)]
            if len(Z_finite) == 0:
                ax.set_title("No finite values")
                continue
            z_min, z_max = Z_finite.min(), Z_finite.max()
            levels = np.logspace(np.log10(max(z_min, 1e-10)), np.log10(z_max), 20)
            Z_plot = np.clip(Z, z_min, z_max)
            ct = ax.contourf(p2v, p1v, Z_plot, levels=levels, cmap="viridis")
            plt.colorbar(ct, ax=ax)
            ax.plot(
                theta_est_diag[pidx2], theta_est_diag[pidx1], "w*", ms=10, label="MLE"
            )
            ax.plot(
                theta_true[pidx2], theta_true[pidx1], "rx", ms=10, mew=2, label="True"
            )
            ax.set_xlabel(latex_labels[pidx2])
            ax.set_ylabel(latex_labels[pidx1])
            ax.set_title(f"{latex_labels[pidx1]} vs {latex_labels[pidx2]}")
            ax.legend(fontsize="small")

        fig.suptitle(
            rf"\textbf{{2D Loss Landscape}} (SNR={diag_snr}, run {DIAG_RUN_IDX})",
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(IMG_DIR / "loss_landscape.png", bbox_inches="tight")
        plt.show()
    else:
        print(
            f"Skipping Figure 7: run {DIAG_RUN_IDX} at SNR index {DIAG_SNR_IDX} did not converge."
        )

    # %% Figure 8 — Profile likelihood for log_sigma_e

    profile_snr = SNR_GRID[DIAG_SNR_IDX]
    profile_grid = np.linspace(-8.0, 0.0, 50)
    free_bounds_profile = [(0.0, 1.0), (0.0, 1.0)]  # bounds for a01, a10

    profile_runs = []
    profile_mles = []

    for j in range(DIAG_N_PROFILE_RUNS):
        key = (DIAG_SNR_IDX, j)
        if key not in _stored_objs:
            continue
        theta_est_j = theta_est_all[DIAG_SNR_IDX, j]
        if np.any(np.isnan(theta_est_j)):
            continue
        _, pvals = profile_likelihood_1d(
            _stored_objs[key],
            theta_est_j,
            fixed_idx=2,
            param_grid=profile_grid,
            free_bounds=free_bounds_profile,
        )
        profile_runs.append(pvals)
        profile_mles.append(theta_est_j[2])

    if len(profile_runs) >= 1:
        fig, ax = plt.subplots(figsize=(width / 2, height / 1.5))
        profile_arr = np.array(profile_runs)
        for k, pv in enumerate(profile_arr):
            ax.plot(profile_grid, pv, color="gray", alpha=0.5, lw=1.0)
            ax.axvline(profile_mles[k], color="gray", alpha=0.3, lw=0.8)
        ax.plot(
            profile_grid,
            np.mean(profile_arr, axis=0),
            color=default_colors[0],
            lw=2.0,
            label="Mean profile",
        )
        ax.axvline(
            theta_true[2], color="red", linestyle="--", lw=1.5, label="True value"
        )
        ax.set_xlabel(latex_labels[2])
        ax.set_ylabel("Profile objective")
        ax.set_title(
            rf"\textbf{{Profile Likelihood}}: {latex_labels[2]}"
            f" (SNR={profile_snr})"
        )
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "profile_likelihood.png", bbox_inches="tight")
        plt.show()
    else:
        print("Skipping Figure 8: insufficient converged runs for profile likelihood.")

    # %% Figure 9 — Scatter: connectivity vs log_sigma_e tradeoff

    scatter_snr_indices = [1, 3, 5]  # SNR=2, 10, 50
    # Clamp to available indices
    scatter_snr_indices = [idx for idx in scatter_snr_indices if idx < n_snrs]

    fig, axes = plt.subplots(
        2, len(scatter_snr_indices), figsize=(width, height), sharex=False
    )
    if len(scatter_snr_indices) == 1:
        axes = axes[:, np.newaxis]  # ensure 2D indexing

    for col, si in enumerate(scatter_snr_indices):
        snr_val = SNR_GRID[si]
        for row, pidx in enumerate([0, 1]):
            ax = axes[row, col]
            x_vals = theta_est_all[si, :, 2]  # log_sigma_e
            y_vals = theta_est_all[si, :, pidx]
            valid = np.isfinite(x_vals) & np.isfinite(y_vals)
            if valid.sum() >= 2:
                r = np.corrcoef(x_vals[valid], y_vals[valid])[0, 1]
                ax.scatter(
                    x_vals[valid],
                    y_vals[valid],
                    s=10,
                    alpha=0.6,
                    color=default_colors[col],
                )
                ax.set_title(f"SNR={snr_val}, r={r:.2f}")
            else:
                ax.set_title(f"SNR={snr_val}")
            ax.plot(theta_true[2], theta_true[pidx], "rx", ms=10, mew=2, label="True")
            ax.set_xlabel(latex_labels[2])
            ax.set_ylabel(latex_labels[pidx])
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize="small")

    fig.suptitle(r"\textbf{Connectivity vs $\log\sigma_e$ Tradeoff}", y=1.01)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "connectivity_sigma_scatter.png", bbox_inches="tight")
    plt.show()

    # %% Bootstrap calibration experiment (slow — set N_BOOTSTRAP_CALIB > 0 to run)

    CALIB_SNR_INDICES = [1, 3]  # SNR=2 and SNR=10
    N_BOOTSTRAP_CALIB = 0  # set to 50 to run

    if N_BOOTSTRAP_CALIB > 0:
        bs_se_calib = np.full((len(CALIB_SNR_INDICES), N_RUNS, n_params), np.nan)
        bs_ci_calib = np.full((len(CALIB_SNR_INDICES), N_RUNS, n_params, 2), np.nan)

        for ci_row, si in enumerate(CALIB_SNR_INDICES):
            snr_c = SNR_GRID[si]
            for j in range(N_RUNS):
                if not converged[si, j]:
                    continue
                rng_c = np.random.default_rng(SEED * 9999 + si * N_RUNS + j)
                theta_c = theta_est_all[si, j]

                def _make_obj_fn_c(y, _model=model):
                    return make_objective(_model, y, None, mean_squared_error)

                bs_se_c, bs_ci_c = parametric_bootstrap_uncertainty(
                    _spdcm,
                    theta_c,
                    snr_c,
                    _make_obj_fn_c,
                    param_bounds,
                    N_BOOTSTRAP_CALIB,
                    rng_c,
                )
                bs_se_calib[ci_row, j] = bs_se_c
                if 95 in bs_ci_c:
                    bs_ci_calib[ci_row, j] = bs_ci_c[95]

            print(f"Calibration: SNR={snr_c} done")

        # Calibration comparison figure
        fig, axes = plt.subplots(1, n_params, figsize=(width, height / 1.5))
        for p in range(n_params):
            ax = axes[p]
            for ci_row, si in enumerate(CALIB_SNR_INDICES):
                snr_val = SNR_GRID[si]
                ax.scatter(
                    [snr_val],
                    [np.nanmean(se_all[si, :, p])],
                    marker="^",
                    color=default_colors[ci_row],
                    label=f"Hessian SE (SNR={snr_val})",
                    zorder=3,
                    s=60,
                )
                ax.scatter(
                    [snr_val],
                    [np.nanmean(bs_se_calib[ci_row, :, p])],
                    marker="o",
                    color=default_colors[ci_row],
                    label=f"Bootstrap SE (SNR={snr_val})",
                    zorder=3,
                    s=60,
                    facecolors="none",
                    linewidths=1.5,
                )
                ax.scatter(
                    [snr_val],
                    [emp_sd[si, p]],
                    marker="s",
                    color=default_colors[ci_row],
                    label=f"Empirical SD (SNR={snr_val})",
                    zorder=3,
                    s=60,
                    alpha=0.5,
                )
            ax.set_title(latex_labels[p])
            ax.set_ylabel("SE / SD")
            ax.set_xlabel("SNR")
            ax.grid(True, alpha=0.3)
            if p == 0:
                ax.legend(fontsize="x-small")
        fig.suptitle(r"\textbf{Bootstrap Calibration Comparison}", y=1.01)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "bootstrap_calibration.png", bbox_inches="tight")
        plt.show()

    # %% Save results

    np.savez(
        IMG_DIR / "sweep_results.npz",
        SNR_GRID=SNR_GRID,
        theta_true=theta_true,
        theta_est_all=theta_est_all,
        se_all=se_all,
        ci_all=ci_all,
        mse_all=mse_all,
        converged=converged,
        hess_cond_all=hess_cond_all,
        hess_eigmin_all=hess_eigmin_all,
        n_neg_eigvals_all=n_neg_eigvals_all,
        near_singular_all=near_singular_all,
        corr_max_all=corr_max_all,
        bs_se_all=bs_se_all,
        bs_ci95_all=bs_ci95_all,
    )
    print(f"\nResults saved to {IMG_DIR / 'sweep_results.npz'}")

    # %% Auto-generate diagnostics_summary.md

    lines = ["# Spectral DCM Diagnostics Summary\n"]

    lines.append("## 1. Root Cause Confirmation\n")
    lines.append(
        "| SNR | Median log10(cond) | Frac near-singular | Median log10|eigmin| |\n"
    )
    lines.append(
        "|-----|-------------------|--------------------|---------------------|\n"
    )
    for i, snr in enumerate(SNR_GRID):
        med_cond = np.nanmedian(log10_cond[i])
        frac_ns = frac_near_singular[i]
        med_eigmin = np.nanmedian(log10_eigmin[i])
        lines.append(f"| {snr} | {med_cond:.2f} | {frac_ns:.2f} | {med_eigmin:.2f} |\n")
    lines.append(
        "\nConclusion: If median condition number > 8 (log10) and near-singular fraction > 0.5, "
        "Hessian blowup is genuine degeneracy (not numerical artifact).\n"
    )

    lines.append("\n## 2. SE Calibration Table\n")
    lines.append("| SNR | Param | Empirical SD | Hessian SE (pinvh+NLL) | Ratio |\n")
    lines.append("|-----|-------|-------------|----------------------|-------|\n")
    for i, snr in enumerate(SNR_GRID):
        for p, pname in enumerate(param_names):
            esd = emp_sd[i, p]
            hse = mean_se[i, p]
            ratio = hse / (esd + 1e-30)
            lines.append(f"| {snr} | {pname} | {esd:.4f} | {hse:.4f} | {ratio:.2f} |\n")
    lines.append(
        "\nRatio >> 1: residual inflation. Ratio ≈ 1: calibration fix worked.\n"
    )

    lines.append("\n## 3. log_sigma_e Identifiability\n")
    lines.append("| SNR | r(a01, log_sigma_e) | r(a10, log_sigma_e) |\n")
    lines.append("|-----|---------------------|---------------------|\n")
    for i, snr in enumerate(SNR_GRID):
        data = theta_est_all[i]
        valid = np.all(np.isfinite(data), axis=1)
        if valid.sum() >= 2:
            r01 = np.corrcoef(data[valid, 0], data[valid, 2])[0, 1]
            r10 = np.corrcoef(data[valid, 1], data[valid, 2])[0, 1]
        else:
            r01 = r10 = float("nan")
        lines.append(f"| {snr} | {r01:.3f} | {r10:.3f} |\n")
    lines.append(
        "\nLarge |r| confirms identifiability ridge between connectivity and log_sigma_e.\n"
    )

    lines.append("\n## 4. Recommended Uncertainty Method\n")
    if N_BOOTSTRAP > 0 and coverage_bs95 is not None:
        ok_a01 = abs(np.nanmean(coverage_bs95[:, 0]) - 0.95) < 0.1
        ok_a10 = abs(np.nanmean(coverage_bs95[:, 1]) - 0.95) < 0.1
        if ok_a01 and ok_a10:
            lines.append(
                "Bootstrap SE/CI is recommended for a01 and a10: "
                "bootstrap 95% CI coverage ≈ 0.95 for both.\n"
            )
        else:
            lines.append(
                "Bootstrap CI coverage is not consistently near 0.95; "
                "further investigation needed.\n"
            )
    else:
        lines.append(
            "Bootstrap was not run (N_BOOTSTRAP=0). "
            "Hessian SE (pinvh+NLL) is a fast alternative for identifiable parameters "
            "but should be validated against bootstrap before use as primary uncertainty report.\n"
        )
    lines.append(
        "\nFor log_sigma_e: identifiability is weak (see ridge above). "
        "Recommend fixing sigma_e or using a prior. "
        "Flag runs where is_near_singular=True.\n"
    )

    summary_path = IMG_DIR / "diagnostics_summary.md"
    with open(summary_path, "w") as f:
        f.writelines(lines)
    print(f"Diagnostics summary written to {summary_path}")

    log_run(
        model_name="spdcm_2roi",
        method="noise_sweep_L-BFGS-B",
        seed=SEED,
        settings={
            "n_runs": N_RUNS,
            "snr_grid": SNR_GRID.tolist(),
            "n_freqs": len(_spdcm.freqs),
            "csd_dim": len(_spdcm.predict_csd(theta_true)),
            "n_bootstrap": N_BOOTSTRAP,
        },
        params={
            "names": param_names,
            "true": theta_true.tolist(),
            "init": theta_zero.tolist(),
        },
        hessian=None,
        performance={
            "failure_rates": failure_rate.tolist(),
            "mean_mse": mean_mse.tolist(),
        },
        overwrite=False,
    )


# %% Run
run_snr_sweep()
