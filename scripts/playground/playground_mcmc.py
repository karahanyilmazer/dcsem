"""Playground for ``mcmc_generic.py`` — step the emcee MCMC time-domain
pipeline cell by cell in VS Code's Interactive Window or Jupyter.

Mirrors the body of ``mcmc_generic.run_single_mcmc(cfg)`` at module
level with ``# %%`` cell markers. The model registry, RunConfig, and
the shared ``is_chain_converged`` helper are imported from the
production module so the source of truth stays single-place.

Default sample counts here are reduced (``n_walkers=24``, ``n_burn=200``,
``n_samples_mcmc=2000``) so cells finish in seconds rather than minutes.
Bump the constants below for production-grade convergence.

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

from dcsem import NOISE_CONFIG, to_latex_label  # noqa: E402
from dcsem.numerics import (  # noqa: E402
    compute_correlation_matrix,
    compute_standard_errors,
)
from dcsem.utils import is_chain_converged  # noqa: E402
from scripts.pipelines.mcmc_generic import MODEL_REGISTRY, RunConfig  # noqa: E402
from utils import get_out_dir  # noqa: E402

# %% Pick model + unpack
# Edit MODEL_NAME to switch model. ``dcm_2roi`` is slow with default
# sample counts; start with ``quadratic`` to verify the pipeline first.
MODEL_NAME = "quadratic"

cfg = RunConfig(
    model_name=MODEL_NAME,
    seed=42,
    n_walkers=24,
    n_burn=200,
    n_samples_mcmc=2000,
)
spec = MODEL_REGISTRY[cfg.model_name]

model = spec.func
model_name = spec.name
model_display_name = spec.display_name
param_names = spec.param_names
theta_true = spec.theta_true
theta_zero = spec.theta_zero
priors = spec.priors
IS_DCM_MODEL = spec.is_dcm
param_bounds = spec.param_bounds
n_params = len(theta_true)

n_walkers = cfg.n_walkers if cfg.n_walkers is not None else max(24, 2 * n_params)
n_burn = cfg.n_burn
n_samples_mcmc = cfg.n_samples_mcmc

print(f"Model: {model_display_name}")
print(f"  n_walkers = {n_walkers},  n_burn = {n_burn},  n_samples = {n_samples_mcmc}")
print(f"  priors    = {priors}")


# %% Output paths
opt_method = "MCMC"
IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=[opt_method, model_name],
)
IMG_DIR.mkdir(parents=True, exist_ok=True)
print(f"NPZ + plots → {IMG_DIR}")


# %% Generate noisy synthetic data
rng = np.random.default_rng(cfg.seed)

if not IS_DCM_MODEL:
    x_data = np.linspace(spec.x_min, spec.x_max, spec.n_samples)
    y_true = model(theta_true, x_data)
    noise_sigma = NOISE_CONFIG.get_noise_std(np.std(y_true))
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=spec.n_samples)
else:
    x_data = None
    y_true = model(theta_true, x_data)  # (T, R)
    noise_sigma = NOISE_CONFIG.get_noise_std(np.std(y_true))
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=y_true.shape)

print(f"y_obs shape = {y_obs.shape},  noise σ = {noise_sigma:.4e}")


# %% Define log-prior, log-likelihood, log-posterior (closures over priors / bounds / model)
def _in_bounds(theta):
    if param_bounds is None:
        return True
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
        if y_pred.shape != y.shape:
            return -np.inf
        r = (y - y_pred) / sigma
        return -0.5 * (np.sum(r * r) + r.size * np.log(2.0 * np.pi * sigma * sigma))
    except Exception:
        return -np.inf


def log_posterior(theta, x, y, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, sigma)


# Smoke check: log_prior at theta_zero should be finite
print(f"log_prior(theta_zero)  = {log_prior(theta_zero):+.4f}")
print(f"log_prior(theta_true)  = {log_prior(theta_true):+.4f}")


# %% MAP estimate via L-BFGS-B (used to seed the walkers)
def _neg_logpost(th):
    return -log_posterior(th, x_data, y_obs, noise_sigma)


_map_res = optimize.minimize(_neg_logpost, theta_zero, method="L-BFGS-B")
theta_est = _map_res.x

print(f"MAP success: {_map_res.success}")
print(f"theta_zero: {np.round(theta_zero, 4)}")
print(f"theta_MAP : {np.round(theta_est, 4)}")
print(f"theta_true: {np.round(theta_true, 4)}")


# %% Initialise walkers (Gaussian perturbation around MAP, clipped to bounds)
scale = np.maximum(0.05 * np.ones(n_params), 0.05 * np.abs(theta_est))
p0 = theta_est + rng.normal(0.0, scale, size=(n_walkers, n_params))
if param_bounds is not None:
    _lb = np.array([b[0] for b in param_bounds])
    _ub = np.array([b[1] for b in param_bounds])
    p0 = np.clip(p0, _lb, _ub)

print(f"p0 shape: {p0.shape}")
print(f"p0 range per dim: {p0.min(axis=0).round(3)} → {p0.max(axis=0).round(3)}")


# %% Run the sampler (burn-in + production)
sampler = emcee.EnsembleSampler(
    n_walkers, n_params, log_posterior, args=(x_data, y_obs, noise_sigma)
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

print(f"Total samples (after filter): {chain.shape[0]}")
print("Posterior summary:")
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

print(f"acceptance_fraction (mean): {acc_frac:.3f}  (healthy band [0.15, 0.80])")
print(f"autocorr time            : {tau}")
print(
    f"effective sample size    : {'n/a' if not np.isfinite(eff_total) else int(eff_total)}"
)
print(f"converged                : {converged}  (need ESS > {50 * n_params})")
print(f"cov_is_calibrated        : {cov_is_calibrated}")


# %% Trace plot per parameter — visual eyeball of mixing
samples_full = sampler.get_chain()  # (n_steps, n_walkers, n_params)
fig, axes = plt.subplots(n_params, figsize=(8, 2 * n_params), sharex=True)
if n_params == 1:
    axes = [axes]
for i in range(n_params):
    axes[i].plot(samples_full[:, :, i], "k", alpha=0.3, lw=0.5)
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
    y_pred=model(theta_mean, x_data),
    theta_mean=theta_mean,
    theta_median=theta_median,
    theta_map=theta_map_post,
    theta_true=theta_true,
    theta_zero=theta_zero,
    se=se,
    ci=np.stack([q025, q975], axis=1),
    cov=cov,
    converged=np.array([bool(converged)]),
    cov_is_calibrated=np.array([bool(cov_is_calibrated)]),
    acceptance_fraction=np.array([float(acc_frac)]),
    ess_total=np.array([float(eff_total) if np.isfinite(eff_total) else np.nan]),
)
print(f"Wrote artifact to {IMG_DIR / 'run_results.npz'}")

with np.load(IMG_DIR / "run_results.npz") as d:
    print("NPZ keys:", sorted(d.files))
    print(f"  converged        = {bool(d['converged'][0])}")
    print(f"  cov_is_calibrated= {bool(d['cov_is_calibrated'][0])}")
    print(f"  acc_frac         = {d['acceptance_fraction'][0]:.3f}")
    print(f"  ess_total        = {d['ess_total'][0]}")

# %%
