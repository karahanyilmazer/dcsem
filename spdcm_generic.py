# %% Imports and config
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from dcsem import SpectralDCM, get_colormap, set_style, to_latex_label
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
conf_cmap = load_cmap("Revolucion", cmap_type="continuous")
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# =============================================================================
# SPECTRAL DCM INSTANCE  (module-level, shared by all model funcs)
# =============================================================================

_spdcm = SpectralDCM(
    n_rois=2, TR=1.0, self_connection=-1.0, freq_lo=0.01, freq_hi=0.1, n_freqs=32
)


# =============================================================================
# MODEL REGISTRY
# =============================================================================


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    func: Callable
    param_names: list[str]
    theta_true: np.ndarray
    theta_zero: np.ndarray
    TR: float = 1.0
    param_bounds: Optional[list[tuple[float, float]]] = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "spdcm_2roi": ModelSpec(
        name="spdcm_2roi",
        display_name="2-ROI Spectral DCM (LS-spDCM)",
        func=lambda theta, _x: _spdcm.predict_csd(theta),
        param_names=["a01", "a10", "log_sigma_e"],
        theta_true=np.array([0.4, 0.6, np.log(0.05)]),
        theta_zero=np.array([0.2, 0.2, np.log(0.1)]),
        param_bounds=[(0.0, 1.0), (0.0, 1.0), (-10.0, 2.0)],
        TR=1.0,
    ),
}

# =============================================================================
# UNPACK SELECTED MODEL
# =============================================================================

ACTIVE_MODEL = "spdcm_2roi"

spec = MODEL_REGISTRY[ACTIVE_MODEL]
model = spec.func
model_name = spec.name
model_display_name = spec.display_name
param_names = spec.param_names
theta_true = spec.theta_true
theta_zero = spec.theta_zero
param_bounds = spec.param_bounds

# =============================================================================
# SETTINGS
# =============================================================================

loss_function = mean_squared_error
n_params = len(theta_zero)
opt_method = "L-BFGS-B"
title_suffix = "Least Squares"

IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=[opt_method, model_name],
)
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
IMG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using model: {model_display_name}")
print(f"Plots will be saved to: {IMG_DIR}")

# Plot toggles
PLOT_2D_LANDSCAPE = False  # slow for spectral model
PLOT_SPECTRAL = True
PLOT_BOLD = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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


# =============================================================================
# DATA GENERATION
# =============================================================================

x_data = None  # Not used for spectral DCM

# Analytical true CSD + structured noisy CSD
y_true = model(theta_true, x_data)
y_obs = _spdcm.generate_noisy_csd(theta_true, snr=10.0, rng=rng)

# Noise std for reporting
noise_std_actual = float(np.std(y_obs - y_true))

print(f"CSD vector length: {len(y_true)}")
print(f"Noise std (obs - true): {noise_std_actual:.4e}")

# =============================================================================
# FIT
# =============================================================================

loss_history = []

obj, y_mean, y_std, y_obs_norm = make_objective(
    model, y_obs, x_data, loss_function, normalize=True
)


def callback(theta):
    loss = obj(theta)
    loss_history.append(loss)
    print(f"Iteration {len(loss_history)}: loss = {loss:.6e}")


res = minimize(
    obj,
    theta_zero,
    method=opt_method,
    callback=callback,
    bounds=param_bounds,
)

if not res.success:
    print(f"Optimization did not converge: {res.message}")

theta_est = res.x
mse_est = obj(theta_est)

print("\nFit results:")
print(f"  True params: {np.round(theta_true, 4)}")
print(f"  Estimated  : {np.round(theta_est, 4)}")
print(f"  Loss: {mse_est:.4f}")

# =============================================================================
# STABILITY DIAGNOSTICS
# =============================================================================

a01_est, a10_est, log_sigma_e_est = theta_est
A_est = np.array([[-1.0, a10_est], [a01_est, -1.0]])
eigvals_A = np.linalg.eigvals(A_est)
print("\nStability diagnostics (estimated A):")
print(f"  Eigenvalues of A: {np.round(eigvals_A, 4)}")
if np.any(eigvals_A.real >= 0):
    print("  WARNING: A has non-negative eigenvalue — system may be unstable!")
else:
    print("  A is stable (all eigenvalues have negative real part).")

R = _spdcm.n_rois
max_cond = 0.0
for f in _spdcm.freqs:
    omega = 2 * np.pi * f
    cond = np.linalg.cond(1j * omega * np.eye(R) - A_est)
    if cond > max_cond:
        max_cond = cond
print(f"  Max condition number of (jωI − A) across freqs: {max_cond:.2e}")
if max_cond > 1e6:
    print("  WARNING: High condition number — numerical instability possible!")


# %% ==========================================================================
# SPECTRAL PLOTS
# =============================================================================

if PLOT_SPECTRAL:
    freqs = _spdcm.freqs

    # Reconstruct CSD matrices
    S_true = _spdcm._unvectorize_csd(y_true)  # (n_freqs, R, R)
    S_obs = _spdcm._unvectorize_csd(y_obs)  # (n_freqs, R, R)
    S_pred = _spdcm._unvectorize_csd(model(theta_est, None))  # (n_freqs, R, R)

    # --- Auto-spectra (log-log) ---
    fig, axes = plt.subplots(1, R, figsize=(width, height / 1.5), sharey=False)
    if R == 1:
        axes = [axes]

    for r in range(R):
        ax = axes[r]
        ax.semilogy(
            freqs,
            S_obs[:, r, r].real,
            color=default_colors[0],
            alpha=0.8,
            label="observed",
            lw=1.5,
        )
        ax.semilogy(
            freqs, S_pred[:, r, r].real, color=default_colors[2], label="fitted", lw=2
        )
        ax.semilogy(
            freqs,
            S_true[:, r, r].real,
            color=default_colors[1],
            linestyle="--",
            label="true",
            lw=1.5,
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (a.u.)")
        ax.set_title(f"ROI {r + 1} Auto-spectrum $S_{{{r + 1}{r + 1}}}(\\omega)$")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
    )
    fig.suptitle(rf"\textbf{{{model_display_name} — Auto-Spectra}}", y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "auto_spectra.png", bbox_inches="tight")
    plt.savefig(LATEX_DIR / f"{model_name}_autospectra.pdf", bbox_inches="tight")
    plt.show()

    # --- Cross-spectrum magnitude and coherence ---
    if R >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(width, height / 1.5))

        # |S12(ω)|
        ax = axes[0]
        ax.semilogy(
            freqs,
            np.abs(S_obs[:, 0, 1]),
            color=default_colors[0],
            alpha=0.8,
            label="observed",
            lw=1.5,
        )
        ax.semilogy(
            freqs,
            np.abs(S_pred[:, 0, 1]),
            color=default_colors[2],
            label="fitted",
            lw=2,
        )
        ax.semilogy(
            freqs,
            np.abs(S_true[:, 0, 1]),
            color=default_colors[1],
            linestyle="--",
            label="true",
            lw=1.5,
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"$|S_{12}(\omega)|$")
        ax.set_title(r"Cross-spectrum $|S_{12}(\omega)|$")
        ax.grid(True, alpha=0.3)

        # Coherence
        ax = axes[1]
        for S, lbl, col, ls in [
            (S_obs, "observed", default_colors[0], "-"),
            (S_pred, "fitted", default_colors[2], "-"),
            (S_true, "true", default_colors[1], "--"),
        ]:
            denom = S[:, 0, 0].real * S[:, 1, 1].real
            coh = np.abs(S[:, 0, 1]) ** 2 / (denom + 1e-30)
            ax.plot(freqs, coh, color=col, linestyle=ls, label=lbl, lw=1.5)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Coherence")
        ax.set_title(r"Coherence $|S_{12}|^2 / (S_{11} S_{22})$")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.08),
            frameon=True,
        )
        fig.suptitle(
            rf"\textbf{{{model_display_name} — Cross-Spectrum \& Coherence}}", y=1.02
        )
        plt.tight_layout()
        plt.savefig(IMG_DIR / "cross_spectrum_coherence.png", bbox_inches="tight")
        plt.savefig(
            LATEX_DIR / f"{model_name}_cross_coherence.pdf", bbox_inches="tight"
        )
        plt.show()


# %% ==========================================================================
# BOLD TIME SERIES
# =============================================================================

BOLD_SEED = SEED + 1

if PLOT_BOLD:
    # Canonical ground truth: pin to BOLD_SEED once.
    bold_true, tvec = _spdcm.simulate_bold(
        theta_true, T=200, rng=np.random.default_rng(BOLD_SEED)
    )
    # Same seed → same noise realization; differences reflect only parameter mismatch.
    bold_est, _ = _spdcm.simulate_bold(
        theta_est, T=200, rng=np.random.default_rng(BOLD_SEED)
    )

    fig, axes = plt.subplots(
        1, R, figsize=(width, height / 1.5), sharey=False, sharex=True
    )
    if R == 1:
        axes = [axes]

    for r in range(R):
        ax = axes[r]
        ax.plot(
            tvec,
            bold_true[:, r],
            color=default_colors[1],
            linestyle="--",
            label="true",
            lw=1.5,
        )
        ax.plot(tvec, bold_est[:, r], color=default_colors[2], label="fitted", lw=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BOLD (a.u.)")
        ax.set_title(f"ROI {r + 1}")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
    )
    fig.suptitle(rf"\textbf{{{model_display_name} — BOLD Time Series}}", y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "bold_timeseries.png", bbox_inches="tight")
    plt.savefig(LATEX_DIR / f"{model_name}_bold.pdf", bbox_inches="tight")
    plt.show()


# %% ==========================================================================
# HESSIAN-BASED DIAGNOSTICS
# =============================================================================

hess_func = nd.Hessian(obj)
H = hess_func(theta_est)

# Residual variance in the normalized space
if y_mean is not None:
    y_pred_norm = (model(theta_est, x_data) - y_mean) / y_std
    residuals = y_obs_norm - y_pred_norm
else:
    residuals = y_obs - model(theta_est, x_data)
sigma_sq_est = np.var(residuals, ddof=n_params)

try:
    cov, hess_diagnostics = safe_hessian_inversion(
        H, sigma_sq_est, regularization=1e-6, method="tikhonov"
    )

    eigvals_H = hess_diagnostics["eigenvalues"]
    cond = hess_diagnostics["condition_number"]
    rank_deficient = hess_diagnostics["rank_deficient"]

    se = compute_standard_errors(cov, warn_negative=True)
    ci = compute_confidence_intervals(theta_est, se, alpha=0.05)
    corr = compute_correlation_matrix(cov, handle_degenerate=True)
    max_offdiag_corr = np.nanmax(np.abs(corr - np.eye(n_params)))

    # Correlation heatmap
    latex_labels = [to_latex_label(name) for name in param_names]
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=conf_cmap,
        vmin=-1,
        vmax=1,
        xticklabels=latex_labels,
        yticklabels=latex_labels,
        ax=ax,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    ax.tick_params(which="both", left=False, bottom=False)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(which="both", size=0)
    ax.set_title(rf"\textbf{{{model_display_name} — Parameter Correlation Matrix}}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "correlation_matrix.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{title_suffix}_correlation_matrix.pdf")
    plt.show()

except np.linalg.LinAlgError:
    print("Failed to invert Hessian — matrix is singular!")
    eigvals_H = np.linalg.eigvalsh(H)
    cond = np.max(np.abs(eigvals_H)) / (np.min(np.abs(eigvals_H)) + 1e-12)
    se = np.full(n_params, np.nan)
    max_offdiag_corr = np.nan
    corr = None
    rank_deficient = True
    ci = np.full((n_params, 2), np.nan)

print("\nHessian diagnostics:")
print(f"  Eigenvalues: {np.round(eigvals_H, 4)}")
print(f"  Condition number: {cond:.2e}")

if cond > 1e6:
    print("  WARNING: High condition number — numerical instability likely!")
if np.min(eigvals_H) < 1e-6:
    print(
        f"  WARNING: Near-zero eigenvalue ({np.min(eigvals_H):.2e})"
        " — model may be degenerate!"
    )

print(f"  Estimated noise variance: {sigma_sq_est:.4e}")
print(f"  Standard errors: {np.round(se, 4)}")

if np.isfinite(max_offdiag_corr):
    print(f"  Max. off-diagonal correlation: {max_offdiag_corr:.3f}")
    if max_offdiag_corr > 0.95:
        print("  WARNING: High parameter correlation — identifiability issues!")

if not rank_deficient:
    print("  95% Confidence intervals:")
    for i, name in enumerate(param_names):
        print(
            f"    {name}: [{ci[i, 0]:.4f}, {ci[i, 1]:.4f}]  (True: {theta_true[i]:.4f})"
        )


# =============================================================================
# LOGGING
# =============================================================================

log_run(
    model_name=model_name,
    method=opt_method,
    seed=SEED,
    settings={
        "n_freqs": len(_spdcm.freqs),
        "csd_dim": len(y_obs),
        "snr": 10.0,
        "noise_std_actual": noise_std_actual,
    },
    params={
        "names": param_names,
        "true": theta_true.tolist(),
        "init": theta_zero.tolist(),
        "est": theta_est.tolist(),
        "se": se.tolist() if not rank_deficient else [float("nan")] * n_params,
        "corr_max": float(max_offdiag_corr) if np.isfinite(max_offdiag_corr) else None,
    },
    hessian={"cond": float(cond), "eigvals": eigvals_H.tolist()},
    performance={"mse": float(mse_est)},
    correlation=corr.tolist() if corr is not None else None,
    overwrite=False,
)

# %%
