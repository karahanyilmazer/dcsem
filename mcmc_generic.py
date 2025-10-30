# %% Imports and config
import warnings
from pathlib import Path

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy import optimize

from utils import log_run, to_latex_label

# =============================================================================
# MODEL DEFINITIONS - Choose one or define your own
# =============================================================================


# 1️⃣ Quadratic (baseline, convex, well-conditioned)
def model(theta, x):
    a, b, c = theta
    return a * x**2 + b * x + c


model_name = "quadratic"
model_display_name = "Quadratic Model"
param_names = ["a", "b", "c"]
theta_true = np.array([1.0, -12.0, 20.0])
theta_zero = np.array([0.5, 0.0, 0.0])

# Prior specification: [(mu, sigma), ...]
# Broad, weakly-informative Normal priors
priors = [(0.0, 3.0), (0.0, 20.0), (0.0, 40.0)]


# 2️⃣ Product degeneracy (structural non-identifiability)
# def model(theta, x):
#     a, b, c = theta
#     return (a * b) * x + c


# model_name = "product_degen"
# model_display_name = "Product Model (Degenerate)"
# param_names = ["a", "b", "c"]
# theta_true = np.array([2.0, 3.0, 5.0])  # slope = a*b = 6
# theta_zero = np.array([1.0, 1.0, 0.0])
# priors = [(0.0, 5.0), (0.0, 5.0), (0.0, 20.0)]


# 3️⃣ Product reparametrized (identifiable)
# def model(theta, x):
#     alpha, c = theta
#     return alpha * x + c
# model_name = "product_reparam"
# model_display_name = "Product Model (Reparametrized)"
# param_names = ["alpha", "c"]
# theta_true = np.array([6.0, 5.0])
# theta_zero = np.array([1.0, 0.0])
# priors = [(0.0, 10.0), (0.0, 20.0)]


# 4️⃣ Sum of exponentials (sloppy model, huge condition number)
# def model(theta, x):
#     A1, k1, A2, k2 = theta
#     return A1 * np.exp(-k1 * x) + A2 * np.exp(-k2 * x)


# model_name = "sum_of_exponentials"
# model_display_name = "Sum of Exponentials Model"
# param_names = ["A1", "k1", "A2", "k2"]
# theta_true = np.array([5.0, 0.5, 3.0, 0.1])
# theta_zero = np.array([4.0, 0.4, 2.0, 0.15])
# priors = [(0.0, 10.0), (0.0, 2.0), (0.0, 10.0), (0.0, 2.0)]


# 5️⃣ Michaelis-Menten (nonlinear but identifiable)
# def model(theta, x):
#     Vmax, KM = theta
#     return Vmax * x / (KM + x)


# model_name = "michaelis_menten"
# model_display_name = "Michaelis-Menten Model"
# param_names = ["Vmax", "KM"]
# theta_true = np.array([10.0, 2.0])
# theta_zero = np.array([8.0, 1.5])
# priors = [(0.0, 20.0), (0.0, 5.0)]


# 6️⃣ Logistic / Sigmoid (nonlinear, correlated parameters)
# def model(theta, x):
#     L, k, x0 = theta
#     return L / (1 + np.exp(-k * (x - x0)))


# model_name = "logistic_sigmoid"
# model_display_name = "Logistic Sigmoid Model"
# param_names = ["L", "k", "x0"]
# theta_true = np.array([1.0, 1.0, 5.0])
# theta_zero = np.array([0.8, 0.8, 4.0])
# priors = [(0.0, 2.0), (0.0, 3.0), (0.0, 20.0)]


# 7️⃣ Power law
# def model(theta, x):
#     a, b = theta
#     return a * x**b


# model_name = "power_law"
# model_display_name = "Power Law Model"
# param_names = ["a", "b"]
# theta_true = np.array([2.0, 1.5])
# theta_zero = np.array([1.5, 1.2])
# priors = [(0.0, 5.0), (0.0, 3.0)]


# =============================================================================
# SETTINGS
# =============================================================================

# Reproducibility and data settings
SEED = 42
n_samples = 50
x_min, x_max = -5.0, 15.0
noise_sigma = 3.0  # known observation noise std

# Guard against zero/invalid noise
if not np.isfinite(noise_sigma) or noise_sigma <= 0:
    warnings.warn("noise_sigma <= 0 detected. Clamping to 1e-6 for stability.")
    noise_sigma = 1e-6

# Auto-detect number of parameters
n_params = len(theta_true)

# MCMC settings
n_walkers = max(24, 2 * n_params)  # should be >= 2 * n_params
n_burn = 5000
n_samples_mcmc = 10000

# Plot settings
cmap = load_cmap("Blues", cmap_type="continuous")
plot_dir = Path("img") / "inversion" / "MCMC" / model_name
plot_dir.mkdir(parents=True, exist_ok=True)
print(f"Plots will be saved to: {plot_dir}")

# Plot toggles
PLOT_CORNER = True
PLOT_POSTERIOR_BANDS = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def mse(theta, x, y):
    """Mean squared error loss."""
    y_pred = model(theta, x)
    return np.mean((y_pred - y) ** 2)


def log_prior(theta):
    """Independent Normal priors specified in priors list."""
    logp = 0.0
    for i, (mu, sigma) in enumerate(priors):
        z = (theta[i] - mu) / sigma
        logp += -0.5 * (z * z + np.log(2.0 * np.pi * sigma * sigma))
    return logp


def log_likelihood(theta, x, y, sigma):
    """Gaussian likelihood with known noise std."""
    if sigma <= 0 or not np.isfinite(sigma):
        return -np.inf
    y_pred = model(theta, x)
    r = (y - y_pred) / sigma
    return -0.5 * (np.sum(r * r) + y.size * np.log(2.0 * np.pi * sigma * sigma))


def log_posterior(theta, x, y, sigma):
    """Unnormalized log posterior."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, x, y, sigma)
    return lp + ll


def map_estimate(theta0, x, y, sigma):
    """Find MAP estimate via optimization."""

    def neg_logpost(th):
        return -log_posterior(th, x, y, sigma)

    res = optimize.minimize(neg_logpost, theta0, method="L-BFGS-B")
    return res.x


# =============================================================================
# DATA GENERATION
# =============================================================================

rng = np.random.default_rng(SEED)
x_data = np.linspace(x_min, x_max, n_samples)
y_clean = model(theta_true, x_data)
y_data = y_clean + rng.normal(0.0, noise_sigma, size=n_samples)


# =============================================================================
# MCMC SAMPLING
# =============================================================================

# Initialize walkers around MAP estimate
theta_est = map_estimate(theta_zero, x_data, y_data, noise_sigma)
scale = np.maximum(0.05 * np.ones(n_params), 0.05 * np.abs(theta_est))
p0 = theta_est + rng.normal(0.0, scale, size=(n_walkers, n_params))

# Run sampler
sampler = emcee.EnsembleSampler(
    n_walkers, n_params, log_posterior, args=(x_data, y_data, noise_sigma)
)

print(f"Running MCMC: {n_burn} burn-in + {n_samples_mcmc} production samples...")
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
    print(f"    {name}: [{q025[i]:.4f}, {q975[i]:.4f}] (True: {theta_true[i]:.4f})")


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
    print("  ⚠️  Low acceptance (<0.05) - walkers may be stuck!")
elif acc_frac > 0.8:
    print("  ⚠️  High acceptance (>0.8) - proposal may be too narrow!")


# =============================================================================
# PLOT: DATA AND FITTED CURVE
# =============================================================================

x_plot = np.linspace(x_data.min(), x_data.max(), 400)
y_mean = model(theta_mean, x_plot)
y_true = model(theta_true, x_plot)

plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, s=20, alpha=0.7, label="data")
plt.plot(x_plot, y_mean, color="tomato", label="posterior mean")
plt.plot(x_plot, y_true, color="gray", linestyle="--", label="true")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"{model_display_name} - Data and Fit")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "data_fit.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# POSTERIOR PREDICTIVE BANDS
# =============================================================================

if PLOT_POSTERIOR_BANDS:
    nsamp = min(400, chain.shape[0])
    idx = rng.choice(chain.shape[0], size=nsamp, replace=False)
    thetas = chain[idx]

    # Compute predictions for each posterior sample
    Y = np.array([model(th, x_plot) for th in thetas])
    y_lo = np.percentile(Y, 2.5, axis=0)
    y_hi = np.percentile(Y, 97.5, axis=0)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_data, y_data, s=18, alpha=0.6, label="data")
    plt.plot(x_plot, y_true, color="gray", linestyle="--", label="true")
    plt.plot(x_plot, y_mean, color="tomato", label="posterior mean")
    plt.fill_between(
        x_plot, y_lo, y_hi, color="tomato", alpha=0.2, label="95% posterior band"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{model_display_name} - Posterior Predictive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "posterior_predictive.png", dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# CORNER PLOT
# =============================================================================

if PLOT_CORNER:
    latex_labels = [to_latex_label(name) for name in param_names]
    fig = corner.corner(
        chain,
        labels=latex_labels,
        truths=theta_true,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        bins=50,
        smooth=0.8,
    )
    fig.suptitle(f"{model_display_name} - Posterior Distributions", y=1.0)
    plt.savefig(plot_dir / "corner_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# CORRELATION PLOT
# =============================================================================

cov = np.cov(chain, rowvar=False)
# Check for negative variances
diag_cov = np.diag(cov)
if np.any(diag_cov < 0):
    print("⚠️  Negative variance detected - Hessian may not be positive definite!")
    diag_cov = np.clip(diag_cov, 0, None)

se = np.sqrt(diag_cov)

# 95% confidence intervals
ci = np.vstack([theta_est - 1.96 * se, theta_est + 1.96 * se]).T

# Correlation matrix
denom = np.outer(se, se)
with np.errstate(invalid="ignore", divide="ignore"):
    corr = np.where(denom > 0, cov / denom, 0)
max_offdiag_corr = np.nanmax(np.abs(corr - np.eye(n_params)))

# Plot correlation matrix
latex_labels = [to_latex_label(name) for name in param_names]
fig, ax = plt.subplots()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    xticklabels=latex_labels,
    yticklabels=latex_labels,
    ax=ax,
)
ax.set_title(f"{model_display_name} - Parameter Correlation Matrix")
plt.tight_layout()
plt.savefig(plot_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# LOGGING
# =============================================================================

# Compute posterior standard deviations
theta_std = np.std(chain, axis=0)

log_run(
    model_name=model_name,
    method="MCMC",
    seed=SEED,
    settings={
        "n_samples": n_samples,
        "noise_sigma": noise_sigma,
        "n_walkers": n_walkers,
        "n_burn": n_burn,
        "n_samples_mcmc": n_samples_mcmc,
    },
    params={
        "names": param_names,
        "true": theta_true.tolist(),
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
        "mse_mean": float(mse(theta_mean, x_data, y_data)),
        "mse_map": float(mse(theta_est_post, x_data, y_data)),
    },
    hessian=None,
    correlation=corr.tolist() if corr is not None else None,
)

# %%
