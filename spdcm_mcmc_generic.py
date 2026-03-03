# %% Imports and config
from dataclasses import dataclass
from typing import Callable, Optional

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy import optimize
from sklearn.metrics import mean_squared_error

from dcsem import SpectralDCM, get_colormap, set_style, to_latex_label
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
# SPECTRAL DCM INSTANCE
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
    priors: list[tuple[float, float]]  # [(mu, sigma), ...] Normal priors
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
        priors=[
            (0.5, 0.3),  # a01 ~ N(0.5, 0.3) — centered in [0, 1]
            (0.5, 0.3),  # a10 ~ N(0.5, 0.3)
            (-3.0, 2.0),  # log_sigma_e ~ N(-3.0, 2.0), i.e. σ_e ≈ 0.05
        ],
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
priors = spec.priors
param_bounds = spec.param_bounds

# =============================================================================
# SETTINGS
# =============================================================================

n_params = len(theta_true)
n_walkers = max(24, 2 * n_params)
n_burn = 250
n_samples_mcmc = 3000
opt_method = "MCMC"

IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=[opt_method, model_name],
)
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
IMG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using model: {model_display_name}")
print(f"Plots will be saved to: {IMG_DIR}")

PLOT_CORNER = True
PLOT_POSTERIOR_BANDS = True
PLOT_BOLD_BANDS = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def log_prior(theta):
    """Independent Normal priors specified in priors list."""
    logp = 0.0
    for i, (mu, sigma) in enumerate(priors):
        z = (theta[i] - mu) / sigma
        logp += -0.5 * (z * z + np.log(2.0 * np.pi * sigma * sigma))
    return logp


def log_likelihood(theta, x, y_obs, sigma):
    """Gaussian likelihood on flattened CSD vector.

    NOTE: Uses a Gaussian likelihood on the real-valued CSD vector.
    TODO: Can be upgraded to a Wishart spectral likelihood for statistically
          principled inference.

    Returns -inf if the A matrix is unstable (predict_csd returns np.inf).
    """
    if sigma <= 0 or not np.isfinite(sigma):
        return -np.inf
    try:
        y_pred = model(theta, x)
        if not np.all(np.isfinite(y_pred)):  # unstable A
            return -np.inf
        r = (y_obs - y_pred) / sigma
        return -0.5 * (np.sum(r * r) + r.size * np.log(2.0 * np.pi * sigma**2))
    except Exception:
        return -np.inf


def log_posterior(theta, x, y, sigma):
    """Unnormalized log posterior."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, x, y, sigma)
    return lp + ll


def map_estimate(theta0, x, y, sigma):
    """Find MAP estimate via L-BFGS-B optimization."""

    def neg_logpost(th):
        return -log_posterior(th, x, y, sigma)

    res = optimize.minimize(neg_logpost, theta0, method="L-BFGS-B", bounds=param_bounds)
    return res.x


# =============================================================================
# DATA GENERATION
# =============================================================================

x_data = None  # Not used for spectral DCM

y_true = model(theta_true, x_data)
y_obs = _spdcm.generate_noisy_csd(theta_true, snr=10.0, rng=rng)
noise_std_actual = float(np.std(y_obs - y_true))

# Use noise std as the likelihood sigma
noise_sigma = noise_std_actual if noise_std_actual > 1e-12 else 1e-3

print(f"CSD vector length: {len(y_obs)}")
print(f"Noise std (obs - true): {noise_std_actual:.4e}")
print(f"Likelihood sigma: {noise_sigma:.4e}")


# =============================================================================
# MCMC SAMPLING
# =============================================================================

print("\nFinding MAP estimate for walker initialization...")
theta_est = map_estimate(theta_zero, x_data, y_obs, noise_sigma)
print(f"  MAP estimate: {np.round(theta_est, 4)}")

scale = np.maximum(0.05 * np.ones(n_params), 0.05 * np.abs(theta_est))
p0 = theta_est + rng.normal(0.0, scale, size=(n_walkers, n_params))

sampler = emcee.EnsembleSampler(
    n_walkers, n_params, log_posterior, args=(x_data, y_obs, noise_sigma)
)

print(f"\nRunning MCMC: {n_burn} burn-in + {n_samples_mcmc} production samples...")
state = sampler.run_mcmc(p0, n_burn, progress=True)
sampler.reset()
sampler.run_mcmc(state, n_samples_mcmc, progress=True)

# Extract chain and filter invalid samples
chain = sampler.get_chain(flat=True)
logp = sampler.get_log_prob(flat=True)
mask = np.isfinite(logp)
chain = chain[mask]
logp = logp[mask]

# Posterior summaries
theta_mean = np.mean(chain, axis=0)
theta_median = np.median(chain, axis=0)
map_idx = int(np.argmax(logp))
theta_est_post = chain[map_idx]
q025, q975 = np.percentile(chain, [2.5, 97.5], axis=0)

print("\nPosterior summary:")
print(f"  True params: {np.round(theta_true, 4)}")
print(f"  Mean       : {np.round(theta_mean, 4)}")
print(f"  Median     : {np.round(theta_median, 4)}")
print(f"  MAP        : {np.round(theta_est_post, 4)}")
print("  95% Credible intervals:")
for i, name in enumerate(param_names):
    print(f"    {name}: [{q025[i]:.4f}, {q975[i]:.4f}]  (True: {theta_true[i]:.4f})")


# =============================================================================
# DIAGNOSTICS
# =============================================================================

try:
    tau = sampler.get_autocorr_time(quiet=True)
    eff_per_walker = n_samples_mcmc / tau
    eff_total = np.sum(eff_per_walker)
    tau_str = np.round(tau, 1)
except Exception:
    tau_str = "n/a"
    eff_total = np.nan

acc_frac = np.mean(sampler.acceptance_fraction)

print("\nDiagnostics:")
print(f"  Acceptance fraction (mean): {acc_frac:.3f}")
print(f"  Autocorr time (per param) : {tau_str}")
if np.isfinite(eff_total):
    print(f"  Approx. effective samples : {int(eff_total)}")

if acc_frac < 0.05:
    print("  WARNING: Low acceptance (<0.05) — walkers may be stuck!")
elif acc_frac > 0.8:
    print("  WARNING: High acceptance (>0.8) — proposal may be too narrow!")


# %% ==========================================================================
# PLOT: POSTERIOR PREDICTIVE SPECTRAL BANDS
# =============================================================================

if PLOT_POSTERIOR_BANDS:
    nsamp = min(400, chain.shape[0])
    idx = rng.choice(chain.shape[0], size=nsamp, replace=False)
    thetas_sample = chain[idx]

    freqs = _spdcm.freqs
    R = _spdcm.n_rois

    # Sample CSD stacks for each posterior draw
    S_samples = []
    for th in thetas_sample:
        y_pred = model(th, None)
        if np.all(np.isfinite(y_pred)):
            S_samples.append(_spdcm._unvectorize_csd(y_pred))
    S_samples = np.array(S_samples)  # (nsamp_valid, n_freqs, R, R)

    # Reference spectra
    S_true = _spdcm._unvectorize_csd(y_true)
    S_obs = _spdcm._unvectorize_csd(y_obs)
    S_mean = _spdcm._unvectorize_csd(model(theta_mean, None))

    # --- Auto-spectra with 95% posterior bands ---
    fig, axes = plt.subplots(1, R, figsize=(width, height / 1.5))
    if R == 1:
        axes = [axes]

    for r in range(R):
        ax = axes[r]
        psd_samples = S_samples[:, :, r, r].real  # (nsamp, n_freqs)
        lo = np.percentile(psd_samples, 2.5, axis=0)
        hi = np.percentile(psd_samples, 97.5, axis=0)

        ax.semilogy(
            freqs,
            S_obs[:, r, r].real,
            color=default_colors[0],
            alpha=0.8,
            label="observed",
            lw=1.5,
        )
        ax.semilogy(
            freqs,
            S_mean[:, r, r].real,
            color=default_colors[2],
            label="posterior mean",
            lw=2,
        )
        ax.fill_between(
            freqs,
            np.maximum(lo, 1e-20),
            hi,
            color=default_colors[2],
            alpha=0.2,
            label="95% posterior band",
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
        ncol=4,
        bbox_to_anchor=(0.5, -0.10),
        frameon=True,
    )
    fig.suptitle(
        rf"\textbf{{{model_display_name} — Posterior Predictive Auto-Spectra ({opt_method})}}",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(IMG_DIR / "posterior_auto_spectra.png", bbox_inches="tight")
    plt.savefig(
        LATEX_DIR / f"{model_name}_{opt_method}_posterior_auto_spectra.pdf",
        bbox_inches="tight",
    )
    plt.show()

    # --- Cross-spectrum magnitude with 95% bands ---
    if R >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(width, height / 1.5))

        # |S12|
        ax = axes[0]
        cross_samples = np.abs(S_samples[:, :, 0, 1])  # (nsamp, n_freqs)
        lo12 = np.percentile(cross_samples, 2.5, axis=0)
        hi12 = np.percentile(cross_samples, 97.5, axis=0)

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
            np.abs(S_mean[:, 0, 1]),
            color=default_colors[2],
            label="posterior mean",
            lw=2,
        )
        ax.fill_between(
            freqs,
            np.maximum(lo12, 1e-20),
            hi12,
            color=default_colors[2],
            alpha=0.2,
            label="95% posterior band",
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
            (S_mean, "posterior mean", default_colors[2], "-"),
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
            ncol=4,
            bbox_to_anchor=(0.5, -0.10),
            frameon=True,
        )
        fig.suptitle(
            rf"\textbf{{{model_display_name} — Cross-Spectrum \& Coherence ({opt_method})}}",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(IMG_DIR / "posterior_cross_coherence.png", bbox_inches="tight")
        plt.savefig(
            LATEX_DIR / f"{model_name}_{opt_method}_cross_coherence.pdf",
            bbox_inches="tight",
        )
        plt.show()


# %% ==========================================================================
# BOLD POSTERIOR PREDICTIVE BANDS
# =============================================================================

# Ground truth: one canonical noise realization, pinned to BOLD_SEED.
# Resimulating with the same seed + theta_true reproduces it exactly,
# so theta_mean = theta_true → bold_mean overlaps bold_gt perfectly.
BOLD_SEED = SEED + 1
bold_gt, tvec = _spdcm.simulate_bold(
    theta_true, T=200, rng=np.random.default_rng(BOLD_SEED)
)

if PLOT_BOLD_BANDS:
    nsamp = min(50, chain.shape[0])
    idx = rng.choice(chain.shape[0], size=nsamp, replace=False)
    # Use a unique seed per sample so the band reflects genuine noise variation.
    bolds = []
    for i, th in enumerate(chain[idx]):
        b, _ = _spdcm.simulate_bold(
            th, T=200, rng=np.random.default_rng(BOLD_SEED + 10 + i)
        )
        bolds.append(b)
    bolds = np.array(bolds)  # (nsamp, n_steps, R)

    # Same seed as bold_gt → identical path when theta == theta_true.
    bold_mean, _ = _spdcm.simulate_bold(
        theta_mean, T=200, rng=np.random.default_rng(BOLD_SEED)
    )
    bold_true = bold_gt  # alias for clarity in the plot

    R = _spdcm.n_rois
    fig, axes = plt.subplots(
        1, R, figsize=(width, height / 1.5), sharey=False, sharex=True
    )
    if R == 1:
        axes = [axes]

    for r in range(R):
        ax = axes[r]
        lo = np.percentile(bolds[:, :, r], 2.5, axis=0)
        hi = np.percentile(bolds[:, :, r], 97.5, axis=0)
        ax.fill_between(
            tvec, lo, hi, color=default_colors[2], alpha=0.2, label="95% posterior band"
        )
        ax.plot(
            tvec,
            bold_mean[:, r],
            color=default_colors[2],
            label="posterior mean",
            lw=1.5,
        )
        ax.plot(
            tvec,
            bold_true[:, r],
            color=default_colors[1],
            linestyle="--",
            label="true",
            lw=1.5,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BOLD (a.u.)")
        ax.set_title(f"ROI {r + 1}")
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
        rf"\textbf{{{model_display_name} — Posterior Predictive BOLD ({opt_method})}}",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(IMG_DIR / "posterior_bold.png", bbox_inches="tight")
    plt.savefig(
        LATEX_DIR / f"{model_name}_{opt_method}_posterior_bold.pdf", bbox_inches="tight"
    )
    plt.show()


# %% ==========================================================================
# CORNER PLOT
# =============================================================================

if PLOT_CORNER:
    latex_labels = [to_latex_label(name) for name in param_names]
    fig = plt.figure(figsize=(width, width))
    fig = corner.corner(
        chain,
        labels=latex_labels,
        truths=theta_true,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        bins=50,
        smooth=0.8,
        truth_color=default_colors[1],
        fig=fig,
    )
    fig.suptitle(
        rf"\textbf{{{model_display_name} — Posterior Distributions ({opt_method})}}"
    )
    plt.tight_layout()
    plt.savefig(IMG_DIR / "corner_plot.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_corner_plot.pdf")
    plt.show()


# %% ==========================================================================
# CORRELATION PLOT
# =============================================================================

cov_post = np.cov(chain, rowvar=False)
diag_cov = np.diag(cov_post)
if np.any(diag_cov < 0):
    print("WARNING: Negative variance in posterior covariance!")
    diag_cov = np.clip(diag_cov, 0, None)

se_post = np.sqrt(diag_cov)
denom = np.outer(se_post, se_post)
with np.errstate(invalid="ignore", divide="ignore"):
    corr = np.where(denom > 0, cov_post / denom, 0)
max_offdiag_corr = np.nanmax(np.abs(corr - np.eye(n_params)))

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
    square=True,
    ax=ax,
)
ax.tick_params(which="both", left=False, bottom=False)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(which="both", size=0)
ax.set_title(
    rf"\textbf{{{model_display_name} — Parameter Correlation Matrix ({opt_method})}}"
)
plt.tight_layout()
plt.savefig(IMG_DIR / "correlation_matrix.png")
plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_correlation_matrix.pdf")
plt.show()


# %% ==========================================================================
# TRACE PLOTS
# =============================================================================

samples = sampler.get_chain()  # (n_steps, n_walkers, n_params)

fig, axes = plt.subplots(n_params, figsize=(width, 2 * n_params), sharex=True)
if n_params == 1:
    axes = [axes]

for i in range(n_params):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3, linewidth=0.5)
    ax.axhline(theta_true[i], color=default_colors[1], linestyle="--", label="true")
    ax.axhline(theta_mean[i], color=default_colors[2], linestyle="-", label="mean")
    ax.set_ylabel(to_latex_label(param_names[i]))
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right")

axes[-1].set_xlabel("Step number")
fig.suptitle(rf"\textbf{{MCMC Trace Plots — {model_display_name}}}")
plt.tight_layout()
plt.savefig(IMG_DIR / "trace_plots.png")
plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_trace_plots.pdf")
plt.show()


# =============================================================================
# LOGGING
# =============================================================================

theta_std = np.std(chain, axis=0)

log_run(
    model_name=model_name,
    method=opt_method,
    seed=SEED,
    settings={
        "n_freqs": len(_spdcm.freqs),
        "csd_dim": len(y_obs),
        "snr": 10.0,
        "noise_sigma": noise_sigma,
        "n_walkers": n_walkers,
        "n_burn": n_burn,
        "n_samples_mcmc": n_samples_mcmc,
    },
    params={
        "names": param_names,
        "true": theta_true.tolist(),
        "init": theta_zero.tolist(),
        "mean": theta_mean.tolist(),
        "median": theta_median.tolist(),
        "map": theta_est_post.tolist(),
        "std": theta_std.tolist(),
        "ci_lower": q025.tolist(),
        "ci_upper": q975.tolist(),
        "corr_max": float(max_offdiag_corr) if np.isfinite(max_offdiag_corr) else None,
    },
    diagnostics={
        "acceptance_fraction": float(acc_frac),
        "autocorr_time": tau_str if isinstance(tau_str, str) else tau_str.tolist(),
        "eff_samples": float(eff_total) if np.isfinite(eff_total) else None,
    },
    performance={
        "mse_mean": float(mean_squared_error(y_obs, model(theta_mean, x_data))),
        "mse_map": float(mean_squared_error(y_obs, model(theta_est_post, x_data))),
    },
    hessian=None,
    correlation=corr.tolist() if corr is not None else None,
    overwrite=False,
)

# %%
