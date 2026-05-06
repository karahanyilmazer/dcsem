"""Playground for ``spdcm_mcmc_generic.py`` — step the emcee MCMC
spectral-DCM pipeline cell by cell in VS Code's Interactive Window or
Jupyter.

Mirrors the body of ``spdcm_mcmc_generic.run_single_mcmc(cfg)`` at
module level with ``# %%`` cell markers. Helpers
(``_resolve_effective_tr``, ``_resolve_param_spec``,
``estimate_log_sigma_e``, ``is_chain_converged``) are imported from
their respective production modules so the source of truth stays
single-place.

Default sample counts here are reduced (``n_walkers=24``,
``n_burn=200``, ``n_samples_mcmc=2000``) so cells finish in seconds
rather than minutes.

NOTE: artifacts land in the same ``results/images/...`` path as a
production run.
"""

# %% Imports + style
import os

os.environ.setdefault("DCSEM_LATEX_DIR", "/tmp/dcsem_latex_playground")
os.makedirs("/tmp/dcsem_latex_playground", exist_ok=True)

import emcee  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402

from dcsem import SpectralDCM, to_latex_label  # noqa: E402
from dcsem.numerics import (  # noqa: E402
    compute_correlation_matrix,
    compute_standard_errors,
)
from dcsem.utils import is_chain_converged  # noqa: E402
from pipelines.spdcm_generic import _resolve_effective_tr, estimate_log_sigma_e  # noqa: E402
from pipelines.spdcm_mcmc_generic import RunConfig, _resolve_param_spec  # noqa: E402
from utils import get_out_dir  # noqa: E402

# %% Configure the run (small sample counts for fast iteration)
cfg = RunConfig(
    n_rois=2,
    TR=1.0,
    data_mode="synthetic_csd",  # "synthetic_csd" | "synthetic_bold" | "empirical"
    snr=10.0,
    seed=42,
    n_walkers=24,
    n_burn=200,
    n_samples_mcmc=2000,
)


# %% Resolve TR + build SpectralDCM
tr_effective, _bold_pre_loaded = _resolve_effective_tr(cfg)
spdcm = SpectralDCM(
    n_rois=cfg.n_rois,
    TR=tr_effective,
    self_connection=cfg.self_connection,
    freq_lo=cfg.freq_lo,
    freq_hi=cfg.freq_hi,
    n_freqs=cfg.n_freqs,
)
print(f"SpectralDCM(n_rois={spdcm.n_rois}, TR={spdcm.TR}, n_freqs={cfg.n_freqs})")


# %% Resolve param spec (returns priors too)
param_names, theta_true, theta_zero, param_bounds, priors = _resolve_param_spec(
    spdcm,
    cfg.theta_true,
    cfg.theta_zero,
    cfg.param_names,
    cfg.param_bounds,
    cfg.priors,
)
n_params = len(theta_true)
n_walkers = cfg.n_walkers if cfg.n_walkers is not None else max(24, 2 * n_params)
n_burn = cfg.n_burn
n_samples_mcmc = cfg.n_samples_mcmc


def model(theta, _x=None):
    return spdcm.predict_csd(theta)


print(f"param_names: {param_names}")
print(f"theta_true : {np.round(theta_true, 4)}")
print(f"theta_zero : {np.round(theta_zero, 4)}")
print(f"priors     : {priors}")
print(f"n_walkers={n_walkers}, n_burn={n_burn}, n_samples={n_samples_mcmc}")


# %% Output paths
opt_method = "MCMC"
model_name = f"spdcm_{cfg.n_rois}roi"
model_display_name = f"{cfg.n_rois}-ROI Spectral DCM (LS-spDCM)"
IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=[opt_method, model_name],
)
IMG_DIR.mkdir(parents=True, exist_ok=True)
print(f"NPZ + plots → {IMG_DIR}")


# %% Generate / load CSD data
rng = np.random.default_rng(cfg.seed)

if cfg.data_mode == "synthetic_csd":
    y_obs = spdcm.generate_noisy_csd(theta_true, snr=cfg.snr, rng=rng)
    has_ground_truth = True
elif cfg.data_mode == "synthetic_bold":
    bold_ref, _tvec = spdcm.simulate_bold(theta_true, T=cfg.T_sim, rng=rng)
    y_obs = spdcm.observed_csd(bold_ref, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    has_ground_truth = True
elif cfg.data_mode == "empirical":
    bold_ref = _bold_pre_loaded
    y_obs = spdcm.observed_csd(bold_ref, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    has_ground_truth = False
else:
    raise ValueError(f"Unknown data_mode: {cfg.data_mode!r}")

print(f"CSD vector length: {len(y_obs)}")


# %% Data-driven log_sigma_e prior (matches production behaviour)
if "log_sigma_e" in param_names:
    log_sigma_e_data = estimate_log_sigma_e(spdcm, y_obs)
    sigma_idx = param_names.index("log_sigma_e")
    priors[sigma_idx] = (float(log_sigma_e_data), 2.0)
    print(f"log_sigma_e prior centred at {log_sigma_e_data:.3f} (data-driven)")

# Likelihood sigma: estimated noise std from the residual when ground truth is
# known, otherwise 1e-3 fallback.
if has_ground_truth:
    y_true = model(theta_true)
    noise_std_actual = float(np.std(y_obs - y_true))
else:
    y_true = None
    noise_std_actual = np.nan

noise_sigma = (
    noise_std_actual
    if np.isfinite(noise_std_actual) and noise_std_actual > 1e-12
    else 1e-3
)
print(f"likelihood sigma: {noise_sigma:.4e}")


# %% Define log_prior, log_likelihood, log_posterior
def _in_bounds(theta):
    for val, (low, high) in zip(theta, param_bounds):
        if val < low or val > high:
            return False
    return True


def log_prior(theta):
    if not _in_bounds(theta):
        return -np.inf
    logp = 0.0
    for i, (mu, sigma) in enumerate(priors):
        z = (theta[i] - mu) / sigma
        logp += -0.5 * (z * z + np.log(2.0 * np.pi * sigma * sigma))
    return logp


def log_likelihood(theta, x, y, sigma):
    if sigma <= 0 or not np.isfinite(sigma):
        return -np.inf
    try:
        y_pred = model(theta, x)
        if not np.all(np.isfinite(y_pred)):  # unstable A
            return -np.inf
        r = (y - y_pred) / sigma
        return -0.5 * (np.sum(r * r) + r.size * np.log(2.0 * np.pi * sigma**2))
    except Exception:
        return -np.inf


def log_posterior(theta, x, y, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, sigma)


print(f"log_prior(theta_zero) = {log_prior(theta_zero):+.3f}")
print(f"log_prior(theta_true) = {log_prior(theta_true):+.3f}")


# %% MAP estimate (seeds the walkers)
def _neg_logpost(th):
    return -log_posterior(th, None, y_obs, noise_sigma)


theta_zero_clipped = np.clip(
    theta_zero, [b[0] for b in param_bounds], [b[1] for b in param_bounds]
)
_map_res = optimize.minimize(
    _neg_logpost, theta_zero_clipped, method="L-BFGS-B", bounds=param_bounds
)
theta_est = _map_res.x

print(f"MAP success: {_map_res.success}")
print(f"theta_zero: {np.round(theta_zero, 4)}")
print(f"theta_MAP : {np.round(theta_est, 4)}")
if has_ground_truth:
    print(f"theta_true: {np.round(theta_true, 4)}")


# %% Initialise walkers (Gaussian perturbation, clipped to bounds)
scale = np.maximum(0.05 * np.ones(n_params), 0.05 * np.abs(theta_est))
p0 = theta_est + rng.normal(0.0, scale, size=(n_walkers, n_params))
p0 = np.clip(p0, [b[0] for b in param_bounds], [b[1] for b in param_bounds])
print(f"p0 shape: {p0.shape}")


# %% Run sampler (burn-in + production)
sampler = emcee.EnsembleSampler(
    n_walkers, n_params, log_posterior, args=(None, y_obs, noise_sigma)
)
print(f"Burn-in ({n_burn} steps)…")
state = sampler.run_mcmc(p0, n_burn, progress=True)
sampler.reset()
print(f"Production ({n_samples_mcmc} steps)…")
sampler.run_mcmc(state, n_samples_mcmc, progress=True)


# %% Extract chain + posterior summaries
chain = sampler.get_chain(flat=True)
logp_flat = sampler.get_log_prob(flat=True)
mask = np.isfinite(logp_flat)
chain = chain[mask]
logp_flat = logp_flat[mask]

theta_mean = np.mean(chain, axis=0)
theta_median = np.median(chain, axis=0)
theta_map_post = chain[int(np.argmax(logp_flat))]
q025, q975 = np.percentile(chain, [2.5, 97.5], axis=0)

print(f"valid samples: {chain.shape[0]}")
print("Posterior summary:")
if has_ground_truth:
    print(f"  true   = {np.round(theta_true, 4)}")
print(f"  mean   = {np.round(theta_mean, 4)}")
print(f"  median = {np.round(theta_median, 4)}")
print(f"  MAP    = {np.round(theta_map_post, 4)}")
print("95% credible intervals:")
for i, name in enumerate(param_names):
    print(f"  {name}: [{q025[i]:+.4f}, {q975[i]:+.4f}]")


# %% Convergence diagnostics
acc_frac = float(np.mean(sampler.acceptance_fraction))
try:
    tau = sampler.get_autocorr_time(quiet=True)
    eff_total = float(np.sum(n_samples_mcmc / tau))
except Exception:
    tau = None
    eff_total = np.nan

converged = is_chain_converged(acc_frac, eff_total, n_params)
cov_is_calibrated = converged

print(f"acceptance_fraction (mean): {acc_frac:.3f}  (healthy [0.15, 0.80])")
print(f"autocorr time            : {tau}")
print(
    f"effective sample size    : {'n/a' if not np.isfinite(eff_total) else int(eff_total)}"
)
print(f"converged                : {converged}  (need ESS > {50 * n_params})")
print(f"cov_is_calibrated        : {cov_is_calibrated}")


# %% Trace plots
samples_full = sampler.get_chain()  # (n_steps, n_walkers, n_params)
fig, axes = plt.subplots(n_params, figsize=(8, 2 * n_params), sharex=True)
if n_params == 1:
    axes = [axes]
for i in range(n_params):
    axes[i].plot(samples_full[:, :, i], "k", alpha=0.3, lw=0.5)
    if has_ground_truth:
        axes[i].axhline(theta_true[i], color="C1", linestyle="--", label="true")
    axes[i].axhline(theta_mean[i], color="C2", lw=1, label="mean")
    axes[i].set_ylabel(to_latex_label(param_names[i]))
axes[-1].set_xlabel("step")
axes[0].legend()
fig.suptitle(f"{model_display_name} — MCMC traces")
plt.tight_layout()
plt.show()


# %% Cov / SE / corr from posterior samples
cov = np.cov(chain, rowvar=False)
se = compute_standard_errors(cov, warn_negative=True)
corr = compute_correlation_matrix(cov, handle_degenerate=True)
max_offdiag_corr = np.nanmax(np.abs(corr - np.eye(n_params)))

print(f"max |off-diag corr| = {max_offdiag_corr:.3f}")
print(f"posterior SE        = {np.round(se, 4)}")


# %% Save NPZ artifact + verify schema
np.savez(
    IMG_DIR / "run_results.npz",
    y_obs=y_obs,
    y_pred=model(theta_mean),
    theta_mean=theta_mean,
    theta_median=theta_median,
    theta_map=theta_map_post,
    theta_true=theta_true if has_ground_truth else np.full_like(theta_mean, np.nan),
    theta_zero=theta_zero,
    se=se,
    ci=np.stack([q025, q975], axis=1),
    cov=cov,
    converged=np.array([bool(converged)]),
    cov_is_calibrated=np.array([bool(cov_is_calibrated)]),
    acceptance_fraction=np.array([float(acc_frac)]),
    ess_total=np.array([float(eff_total) if np.isfinite(eff_total) else np.nan]),
    tr=np.array([float(spdcm.TR)]),
)
print(f"Wrote artifact to {IMG_DIR / 'run_results.npz'}")

with np.load(IMG_DIR / "run_results.npz") as d:
    print("NPZ keys:", sorted(d.files))
    print(f"  converged        = {bool(d['converged'][0])}")
    print(f"  cov_is_calibrated= {bool(d['cov_is_calibrated'][0])}")
    print(f"  acc_frac         = {d['acceptance_fraction'][0]:.3f}")
    print(f"  ess_total        = {d['ess_total'][0]}")
    print(f"  TR               = {float(d['tr'][0])}")

# %%
