# %% Imports and config
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from dcsem import SpectralDCM, get_colormap, set_style, to_latex_label
from dcsem.numerics import (
    compute_confidence_intervals,
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

# %% SpectralDCM instance + model spec

_spdcm = SpectralDCM(
    n_rois=2, TR=1.0, self_connection=-1.0, freq_lo=0.01, freq_hi=0.1, n_freqs=32
)
theta_true = np.array([0.4, 0.6, np.log(0.05)])
theta_zero = np.array([0.2, 0.2, np.log(0.1)])
param_names = ["a01", "a10", "log_sigma_e"]
param_bounds = [(0.0, 1.0), (0.0, 1.0), (-10.0, 2.0)]
model = lambda theta, _x: _spdcm.predict_csd(theta)

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


# %% Monte Carlo sweep

n_snrs = len(SNR_GRID)
n_params = len(theta_true)

theta_est_all = np.full((n_snrs, N_RUNS, n_params), np.nan)
se_all = np.full((n_snrs, N_RUNS, n_params), np.nan)
ci_all = np.full((n_snrs, N_RUNS, n_params, 2), np.nan)
mse_all = np.full((n_snrs, N_RUNS), np.nan)
converged = np.zeros((n_snrs, N_RUNS), dtype=bool)

for i, snr in enumerate(SNR_GRID):
    for j in range(N_RUNS):
        rng_ij = np.random.default_rng(SEED * 1000 + i * N_RUNS + j)
        y_obs = _spdcm.generate_noisy_csd(theta_true, snr=snr, rng=rng_ij)
        obj, y_mean, y_std, _ = make_objective(model, y_obs, None, mean_squared_error)
        res = minimize(obj, theta_zero, method="L-BFGS-B", bounds=param_bounds)
        failed = not res.success or res.fun >= 1e9
        if failed:
            continue
        converged[i, j] = True
        theta_est_all[i, j] = res.x
        mse_all[i, j] = res.fun
        # Hessian → SE and CI
        H = nd.Hessian(obj)(res.x)
        resid_var = np.var(
            (y_obs - model(res.x, None)) / (y_std + 1e-12), ddof=n_params
        )
        try:
            cov, _ = safe_hessian_inversion(H, resid_var, regularization=1e-6)
            se_all[i, j] = compute_standard_errors(cov, warn_negative=False)
            ci_all[i, j] = compute_confidence_intervals(res.x, se_all[i, j])
        except np.linalg.LinAlgError:
            pass  # leave as NaN

    n_conv = int(converged[i].sum())
    print(f"SNR={snr}: {n_conv}/{N_RUNS} converged")

# %% Summary statistics

# Per SNR, per parameter (nanmean/nanstd ignores failed runs)
bias = np.nanmean(theta_est_all - theta_true, axis=1)  # (n_snrs, n_params)
rmse = np.sqrt(np.nanmean((theta_est_all - theta_true) ** 2, axis=1))
emp_sd = np.nanstd(theta_est_all, axis=1)
mean_se = np.nanmean(se_all, axis=1)

# CI coverage: fraction of runs where theta_true is inside [ci_lo, ci_hi]
inside = (ci_all[..., 0] <= theta_true) & (theta_true <= ci_all[..., 1])
coverage = np.nanmean(inside.astype(float), axis=1)  # (n_snrs, n_params)

# Scalars per SNR
failure_rate = 1.0 - np.mean(converged, axis=1)  # (n_snrs,)
mean_mse = np.nanmean(mse_all, axis=1)  # (n_snrs,)

IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=["L-BFGS-B", "spdcm_2roi", "noise_sweep"],
)
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
IMG_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nResults will be saved to: {IMG_DIR}")

# %% Figure 1 — Parameter recovery (Bias and RMSE)

latex_labels = [to_latex_label(name) for name in param_names]

fig, axes = plt.subplots(2, n_params, figsize=(width, height), sharex=True)

for p in range(n_params):
    # Top row: Bias
    ax = axes[0, p]
    ax.semilogx(SNR_GRID, bias[:, p], color=default_colors[0], marker="o", lw=1.5)
    ax.axhline(0, color="gray", linestyle="--", lw=1.0)
    ax.set_title(f"Bias — {latex_labels[p]}")
    ax.set_ylabel("Bias")
    ax.grid(True, alpha=0.3)

    # Bottom row: RMSE
    ax = axes[1, p]
    ax.semilogx(SNR_GRID, rmse[:, p], color=default_colors[1], marker="s", lw=1.5)
    ax.set_title(f"RMSE — {latex_labels[p]}")
    ax.set_xlabel("SNR")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)

fig.suptitle(r"\textbf{Parameter Recovery vs SNR}", y=1.01)
plt.tight_layout()
plt.savefig(IMG_DIR / "recovery_bias_rmse.png", bbox_inches="tight")
plt.savefig(LATEX_DIR / "spdcm_sweep_recovery.pdf", bbox_inches="tight")
plt.show()

# %% Figure 2 — Uncertainty calibration (Empirical SD vs SE, CI coverage)

fig, axes = plt.subplots(2, n_params, figsize=(width, height), sharex=True)

for p in range(n_params):
    # Top row: Empirical SD vs mean Hessian SE
    ax = axes[0, p]
    ax.semilogx(
        SNR_GRID,
        emp_sd[:, p],
        color=default_colors[0],
        marker="o",
        lw=1.5,
        label="Empirical SD",
    )
    ax.semilogx(
        SNR_GRID,
        mean_se[:, p],
        color=default_colors[2],
        marker="^",
        lw=1.5,
        linestyle="--",
        label="Mean Hessian SE",
    )
    ax.set_title(f"SD vs SE — {latex_labels[p]}")
    ax.set_ylabel("Std / SE")
    ax.grid(True, alpha=0.3)
    if p == 0:
        ax.legend(fontsize="small")

    # Bottom row: CI coverage
    ax = axes[1, p]
    ax.semilogx(SNR_GRID, coverage[:, p], color=default_colors[3], marker="D", lw=1.5)
    ax.axhline(0.95, color="gray", linestyle="--", lw=1.0)
    ax.set_title(f"CI Coverage — {latex_labels[p]}")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle(r"\textbf{Uncertainty Calibration vs SNR}", y=1.01)
plt.tight_layout()
plt.savefig(IMG_DIR / "calibration_sd_coverage.png", bbox_inches="tight")
plt.savefig(LATEX_DIR / "spdcm_sweep_calibration.pdf", bbox_inches="tight")
plt.show()

# %% Figure 3 — Summary (Mean CSD MSE and Failure rate)

fig, axes = plt.subplots(1, 2, figsize=(width, height / 1.5))

# Left: Mean CSD MSE (log-log)
ax = axes[0]
ax.loglog(SNR_GRID, mean_mse, color=default_colors[0], marker="o", lw=1.5)
ax.set_xlabel("SNR")
ax.set_ylabel("Mean CSD MSE")
ax.set_title(r"Mean CSD MSE vs SNR")
ax.grid(True, alpha=0.3, which="both")

# Right: Failure rate
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
)

print(f"\nResults saved to {IMG_DIR / 'sweep_results.npz'}")

log_run(
    model_name="spdcm_2roi",
    method="noise_sweep_L-BFGS-B",
    seed=SEED,
    settings={
        "n_runs": N_RUNS,
        "snr_grid": SNR_GRID.tolist(),
        "n_freqs": len(_spdcm.freqs),
        "csd_dim": len(_spdcm.predict_csd(theta_true)),
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

# %%
