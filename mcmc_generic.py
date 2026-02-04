# %% Imports and config
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy import optimize
from sklearn.metrics import mean_squared_error

from dcsem.utils import stim_boxcar
from utils import (
    get_colormap,
    get_out_dir,
    get_width_height_latex,
    log_run,
    set_style,
    simulate_bold,
    to_latex_label,
)

set_style()
width, height = get_width_height_latex()
cmap = get_colormap("YlGnBu_r")
conf_cmap = load_cmap("Revolucion", cmap_type="continuous")
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Reproducibility and data settings
SEED = 42

# =============================================================================
# MODEL DEFINITIONS - Choose one or define your own
# =============================================================================


# 1️⃣ Quadratic (baseline, convex, well-conditioned)
# def model(theta, x):
#     a, b, c = theta
#     return a * x**2 + b * x + c


# model_name = "quadratic"
# model_display_name = "Quadratic Model"
# param_names = ["a", "b", "c"]
# theta_true = np.array([1.0, -12.0, 20.0])
# theta_zero = np.array([0.5, 0.0, 0.0])

# # Prior specification: [(mu, sigma), ...]
# # Broad, weakly-informative Normal priors
# priors = [(0.0, 3.0), (0.0, 20.0), (0.0, 40.0)]
# IS_DCM_MODEL = False


# 2️⃣ Product degeneracy (structural non-identifiability)
def model(theta, x):
    a, b, c = theta
    return (a * b) * x + c


model_name = "product_degen"
model_display_name = "Product Model (Degenerate)"
param_names = ["a", "b", "c"]
theta_true = np.array([2.0, 3.0, 5.0])  # slope = a*b = 6
theta_zero = np.array([1.0, 1.0, 0.0])
priors = [(0.0, 5.0), (0.0, 5.0), (0.0, 20.0)]
IS_DCM_MODEL = False


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
# IS_DCM_MODEL = False


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
# IS_DCM_MODEL = False


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
# IS_DCM_MODEL = False


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
# IS_DCM_MODEL = False


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
# IS_DCM_MODEL = False


# # 8️⃣ DCM - 2 ROI BOLD model (requires different setup)
# # This model uses BOLD simulation instead of analytical functions

# IS_DCM_MODEL = True
# NUM_ROIS = 2
# time = np.arange(100)
# u = stim_boxcar([[10, 20, 1]])
# ODE_METHOD = "BDF"  # Stiff solver; use None for default RK45


# def model(theta, x):
#     """
#     For DCM: theta contains [a01, a10, c0, c1]
#     x is ignored (time and u are used instead)
#     Returns BOLD signals of shape (T, R)
#     """
#     params = dict(zip(param_names, theta))
#     bold = simulate_bold(
#         params, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
#     )
#     return bold  # Shape: (T, R)


# model_name = "dcm_2roi"
# model_display_name = "2-ROI DCM"
# param_names = ["a01", "a10", "c0", "c1"]
# theta_true = np.array([0.4, 0.6, 0.9, 0.2])
# theta_zero = np.array([0.1, 0.1, 0.1, 0.1])

# # Prior specification for DCM: [(mu, sigma), ...]
# # A-matrix connections: can be negative (inhibitory) or positive (excitatory)
# # C-matrix inputs: non-negative
# priors = [
#     (0, 0.5),  # a01: centered at 0, wide range for excitatory/inhibitory
#     (0, 0.5),  # a10: centered at 0, wide range for excitatory/inhibitory
#     (0.75, 0.25),  # c0: centered at 0.5, moderate positive range
#     (0.75, 0.25),  # c1: centered at 0.5, moderate positive range
# ]

# # Randomly choose true parameters from priors
# # theta_true = np.array([np.random.normal(mu, sigma) for (mu, sigma) in priors])
# print(f"True DCM parameters: {theta_true}")

# =============================================================================
# SETTINGS
# =============================================================================


# Auto-detect number of parameters
n_params = len(theta_true)

# MCMC settings
n_walkers = max(24, 2 * n_params)  # should be >= 2 * n_params
n_burn = 5000
n_samples_mcmc = 10000

# Optimization method name
opt_method = "MCMC"

# Plot settings
IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=[opt_method, model_name],
)
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
IMG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using model: {model_display_name}")
print(f"Plots will be saved to: {IMG_DIR}")
print(f"Plots will be saved to: {LATEX_DIR}")

# Plot toggles
PLOT_CORNER = True
PLOT_POSTERIOR_BANDS = True


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


def log_likelihood(theta, x, y, sigma):
    """Gaussian likelihood with known noise std."""
    if sigma <= 0 or not np.isfinite(sigma):
        return -np.inf
    try:
        y_pred = model(theta, x)
        # Ensure shapes match
        if y_pred.shape != y.shape:
            return -np.inf
        r = (y - y_pred) / sigma
        # Handle both 1D and multi-dimensional arrays
        return -0.5 * (np.sum(r * r) + r.size * np.log(2.0 * np.pi * sigma * sigma))
    except Exception:
        # If simulation fails, return -inf log likelihood
        return -np.inf


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

# Data settings
x_min, x_max = -20.0, 20.0  # Not used for DCM
n_samples = 50  # Not used for DCM

if not IS_DCM_MODEL:
    # Standard analytical models
    x_data = np.linspace(x_min, x_max, n_samples)
    # Add some variability to x_data
    # x_data += rng.normal(0.0, 0.5 * (x_max - x_min) / n_samples, size=n_samples)
    y_true = model(theta_true, x_data)
    noise_sigma = 0.10 * np.std(y_true)  # 10% of signal std
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=n_samples)
    noise_std_actual = noise_sigma  # Store actual noise level used
else:
    # DCM BOLD model
    x_data = None  # Not used for DCM
    y_true = model(theta_true, x_data)  # Shape: (T, R)
    noise_sigma = 0.10 * np.std(y_true)  # 10% of signal std
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=y_true.shape)
    noise_std_actual = noise_sigma  # Store actual noise level used


# =============================================================================
# MCMC SAMPLING
# =============================================================================

# Initialize walkers around MAP estimate
theta_est = map_estimate(theta_zero, x_data, y_obs, noise_sigma)
scale = np.maximum(0.05 * np.ones(n_params), 0.05 * np.abs(theta_est))
p0 = theta_est + rng.normal(0.0, scale, size=(n_walkers, n_params))

# Run sampler
sampler = emcee.EnsembleSampler(
    n_walkers, n_params, log_posterior, args=(x_data, y_obs, noise_sigma)
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

# %%
# =============================================================================
# PLOT: DATA AND FITTED CURVE
# =============================================================================

if not IS_DCM_MODEL:
    # Standard 1D analytical model plot
    x_plot = np.linspace(x_data.min(), x_data.max(), 400)
    y_mean = model(theta_mean, x_plot)
    y_true = model(theta_true, x_plot)

    # Plot
    plt.figure(figsize=(width, height / 1.5))
    plt.scatter(x_data, y_obs, s=20, alpha=0.7, label="data")
    plt.plot(x_plot, y_mean, color=default_colors[2], label="posterior mean")
    plt.plot(x_plot, y_true, color=default_colors[1], linestyle="--", label="true")

    # Adjust
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(rf"\textbf{{{model_display_name} - Data and Fit}}")
    plt.legend()

    # Save
    plt.tight_layout()
    plt.savefig(IMG_DIR / "data_fit.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_data_fit.pdf")
    plt.show()

else:
    # DCM BOLD model: plot each ROI's time series
    y_mean = model(theta_mean, None)  # Shape: (T, R)
    y_true = model(theta_true, None)  # Shape: (T, R)

    # Get time vector from globals (defined in DCM model section)
    time_vec = globals().get("time", np.arange(y_obs.shape[0]))
    num_rois = y_obs.shape[1]
    fig, axes = plt.subplots(1, num_rois, sharex=True, figsize=(width, height / 1.5))

    if num_rois == 1:
        axes = [axes]

    for r in range(num_rois):
        axes[r].plot(
            time_vec,
            y_obs[:, r],
            alpha=0.7,
            color=default_colors[0],
            label="observed",
        )
        axes[r].plot(
            time_vec,
            y_mean[:, r],
            color=default_colors[2],
            label="posterior mean",
        )
        axes[r].plot(
            time_vec,
            y_true[:, r],
            linestyle="--",
            color=default_colors[1],
            label="true",
        )
        axes[r].set_title(f"ROI {r + 1}")
        axes[r].set_xlabel("Time (s)")
        axes[r].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
        frameon=True,
    )

    axes[0].set_ylabel("BOLD amplitude (a.u.)")
    # axes[0].legend()
    fig.suptitle(rf"\textbf{{{model_display_name} - Best Fit ({opt_method})}}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "data_fit.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_data_fit.pdf")
    plt.show()

# %%
# =============================================================================
# POSTERIOR PREDICTIVE BANDS
# =============================================================================

if PLOT_POSTERIOR_BANDS:
    nsamp = min(400, chain.shape[0])
    idx = rng.choice(chain.shape[0], size=nsamp, replace=False)
    thetas = chain[idx]

    if not IS_DCM_MODEL:
        # Standard 1D analytical model
        # Compute predictions for each posterior sample
        Y = np.array([model(th, x_plot) for th in thetas])
        y_lo = np.percentile(Y, 2.5, axis=0)
        y_hi = np.percentile(Y, 97.5, axis=0)

        plt.figure()
        plt.scatter(x_data, y_obs, s=18, alpha=0.6, label="data")
        plt.plot(x_plot, y_true, color=default_colors[1], linestyle="--", label="true")
        plt.plot(x_plot, y_mean, color=default_colors[2], label="posterior mean")
        plt.fill_between(
            x_plot,
            y_lo,
            y_hi,
            color=default_colors[2],
            alpha=0.2,
            label="95% posterior band",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{model_display_name} - Best Fit")
        plt.legend()

        plt.tight_layout()
        plt.savefig(IMG_DIR / "posterior_predictive.png")
        plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_posterior_predictive.pdf")
        plt.show()
    else:
        # DCM BOLD model: plot each ROI with uncertainty bands
        time_vec = globals().get("time", np.arange(y_obs.shape[0]))
        num_rois = y_obs.shape[1]

        # Compute predictions for each posterior sample
        Y = np.array([model(th, None) for th in thetas])  # Shape: (nsamp, T, R)
        y_lo = np.percentile(Y, 2.5, axis=0)  # Shape: (T, R)
        y_hi = np.percentile(Y, 97.5, axis=0)  # Shape: (T, R)

        fig, axes = plt.subplots(
            1, num_rois, sharex=True, figsize=(width, height * 0.8)
        )
        if num_rois == 1:
            axes = [axes]

        for r in range(num_rois):
            axes[r].plot(time_vec, y_obs[:, r], label="observed", alpha=0.7)
            axes[r].plot(
                time_vec, y_mean[:, r], color=default_colors[2], label="posterior mean"
            )
            axes[r].fill_between(
                time_vec,
                y_lo[:, r],
                y_hi[:, r],
                color=default_colors[2],
                alpha=0.2,
                label="95\% posterior band",
            )
            axes[r].plot(
                time_vec,
                y_true[:, r],
                color=default_colors[1],
                linestyle="--",
                label="true",
            )
            axes[r].set_title(f"ROI {r}")
            axes[r].set_xlabel("Time (s)")
            axes[r].grid(True, alpha=0.3)

        # Create a single legend below all subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.07),
            frameon=True,
        )

        axes[0].set_ylabel("BOLD Amplitude (a.u.)")
        # axes[0].legend()
        fig.suptitle(
            rf"\textbf{{{model_display_name} - Posterior Predictive ({opt_method})}}"
        )

        plt.tight_layout()
        plt.savefig(IMG_DIR / "posterior_predictive.png")
        plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_posterior_predictive.pdf")
        plt.show()

# %%
# =============================================================================
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
        rf"\textbf{{{model_display_name} - Posterior Distributions ({opt_method})}}"
    )
    plt.tight_layout()
    plt.savefig(IMG_DIR / "corner_plot.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_corner_plot.pdf")
    plt.show()


# %%
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

# Remove ticks from heatmap
ax.tick_params(which="both", left=False, bottom=False)
# Remove ticks from colorbar
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(which="both", size=0)

ax.set_title(
    rf"\textbf{{{model_display_name} - Parameter Correlation Matrix ({opt_method})}}"
)
plt.tight_layout()
plt.savefig(IMG_DIR / "correlation_matrix.png")
plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_correlation_matrix.pdf")
plt.show()

# %%
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
        "n_samples": n_samples if not IS_DCM_MODEL else y_obs.shape[0],
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
# Get the full chain (not flattened)
samples = sampler.get_chain()  # Shape: (n_steps, n_walkers, n_params)

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
fig.suptitle(rf"\textbf{{MCMC Trace Plots - {model_display_name}}}")
plt.tight_layout()
plt.savefig(IMG_DIR / "trace_plots.png")
plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_trace_plots.pdf")
plt.show()
# %%
