# %%
# Single-parameter MCMC estimation example.
#
# Estimates a01 (with other params fixed at ground truth) using emcee,
# with a proper Gaussian likelihood and Normal prior (matching mcmc_generic.py).
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from dcsem import PARAM_BOUNDS
from dcsem.plotting import set_style
from dcsem.utils import stim_boxcar
from utils import add_noise, simulate_bold

set_style()

# %%
# =============================================================================
# Configuration
# =============================================================================

time = np.arange(100)
u = stim_boxcar([[10, 20, 1]])

NUM_ROIS = 2

# Ground truth
true_params = {"a01": 0.6, "a10": 0.4, "c0": 0.5, "c1": 0.5}

# Only estimate a01; others are fixed at ground truth
params_to_est = ["a01"]
fixed_params = {k: v for k, v in true_params.items() if k not in params_to_est}

# Bounds from central config
bounds_dict = PARAM_BOUNDS.get_bounds_dict()
est_bounds = [bounds_dict[p] for p in params_to_est]

# Prior: Normal(mu=0.5, sigma=0.3) for a01 (centered on mid-range, fairly wide)
priors = [(0.5, 0.3)]

# MCMC settings
n_dim = len(params_to_est)
n_walkers = 32
n_burn = 500
n_samples = 1000

# %%
# =============================================================================
# Generate observed data
# =============================================================================
bold_true = simulate_bold(true_params, time=time, u=u, num_rois=NUM_ROIS)

# Add noise using SNR in dB (matching the pipeline convention)
bold_noisy, noise_sigma = add_noise(bold_true, snr_db=20.0)


# %%
# =============================================================================
# Likelihood, prior, posterior  (matching mcmc_generic.py)
# =============================================================================


def log_prior(theta):
    """Independent Normal priors."""
    logp = 0.0
    for i, (mu, sigma) in enumerate(priors):
        z = (theta[i] - mu) / sigma
        logp += -0.5 * (z * z + np.log(2.0 * np.pi * sigma * sigma))
    return logp


def log_likelihood(theta, y_obs, sigma):
    """Gaussian likelihood with known noise std."""
    if sigma <= 0 or not np.isfinite(sigma):
        return -np.inf
    # Build full param dict (free + fixed) without mutating shared state
    params = dict(fixed_params)
    for name, val in zip(params_to_est, theta):
        params[name] = val
    try:
        y_pred = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)
        if y_pred.shape != y_obs.shape:
            return -np.inf
        r = (y_obs - y_pred) / sigma
        return -0.5 * (np.sum(r * r) + r.size * np.log(2.0 * np.pi * sigma * sigma))
    except Exception:
        return -np.inf


def log_posterior(theta, y_obs, sigma):
    """Unnormalized log posterior = log prior + log likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, y_obs, sigma)
    return lp + ll


# %%
# =============================================================================
# Run MCMC
# =============================================================================

# Initialise walkers around a sensible starting point
initial_values = np.array([np.mean(bounds_dict[p]) for p in params_to_est])
p0 = initial_values + 0.01 * np.random.randn(n_walkers, n_dim)

sampler = emcee.EnsembleSampler(
    n_walkers, n_dim, log_posterior, args=(bold_noisy, noise_sigma)
)

print(f"Running MCMC: {n_burn} burn-in + {n_samples} production samples...")
state = sampler.run_mcmc(p0, n_burn, progress=True)
sampler.reset()
sampler.run_mcmc(state, n_samples, progress=True)

# %%
# =============================================================================
# Posterior summary
# =============================================================================
samples = sampler.get_chain(flat=True)
logp = sampler.get_log_prob(flat=True)

# Filter invalid samples
mask = np.isfinite(logp)
samples = samples[mask]
logp = logp[mask]

means = np.mean(samples, axis=0)
stds = np.std(samples, axis=0)
q025, q975 = np.percentile(samples, [2.5, 97.5], axis=0)

print("\nTrue Parameters:", {p: true_params[p] for p in params_to_est})
print("Estimated (mean):", dict(zip(params_to_est, means)))
print("Posterior std:", dict(zip(params_to_est, stds)))
print(f"95% CI: [{q025[0]:.4f}, {q975[0]:.4f}]")

# %%
# =============================================================================
# Corner plot
# =============================================================================
if samples.shape[1] > 1:
    corner.corner(
        samples,
        labels=params_to_est,
        truths=[true_params[p] for p in params_to_est],
        show_titles=True,
        title_fmt=".4f",
        quantiles=[0.16, 0.5, 0.84],
    )
else:
    # corner library crashes with 1D data; use a simple histogram instead
    fig, ax = plt.subplots()
    ax.hist(samples[:, 0], bins=50, density=True, alpha=0.7, label="posterior")
    ax.axvline(true_params[params_to_est[0]], color="red", ls="--", label="true")
    ax.axvline(means[0], color="blue", ls="-", label="mean")
    ax.set_xlabel(params_to_est[0])
    ax.set_ylabel("Density")
    ax.legend()
plt.suptitle("MCMC Posterior — Single Parameter Estimation")
plt.tight_layout()
plt.show(block=False)

# %%
