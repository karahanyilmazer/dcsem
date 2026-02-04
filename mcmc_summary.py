# %% Imports and config
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score

from utils import (
    get_colormap,
    get_out_dir,
    get_width_height_latex,
    set_style,
    to_latex_label,
)

set_style()
width, height = get_width_height_latex()
cmap = get_colormap("YlGnBu_r")
conf_cmap = load_cmap("Revolucion", cmap_type="continuous")
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

models = {
    "quadratic": {
        "func": lambda theta, x: theta[0] * x**2 + theta[1] * x + theta[2],
        "display_name": "Quadratic Model",
        "param_names": ["a", "b", "c"],
        "theta_true": np.array([1.0, -12.0, 20.0]),
        "theta_zero": np.array([0.5, 0.0, 0.0]),
        "param_bounds": None,
        "priors": [
            (0.0, 2.0),  # a: mean=0, std=2
            (-15.0, 5.0),  # b: mean=-15, std=5
            (15.0, 10.0),  # c: mean=15, std=10
        ],
    },
    "product_degen": {
        "func": lambda theta, x: (theta[0] * theta[1]) * x + theta[2],
        "display_name": "Product Model (Degenerate)",
        "param_names": ["a", "b", "c"],
        "theta_true": np.array([17.0, 3.0, 5.0]),
        "theta_zero": np.array([1.0, 1.0, 0.0]),
        "param_bounds": None,
        "priors": [
            (10.0, 10.0),  # a: mean=10, std=10
            (0.0, 5.0),  # b: mean=0, std=5
            (0.0, 10.0),  # c: mean=0, std=10
        ],
    },
    "product_reparam": {
        "func": lambda theta, x: theta[0] * x + theta[1],
        "display_name": "Product Model (Reparametrized)",
        "param_names": [r"\alpha", "c"],
        "theta_true": np.array([51.0, 5.0]),
        "theta_zero": np.array([1.0, 0.0]),
        "param_bounds": None,
        "priors": [
            (50.0, 20.0),  # alpha: mean=50, std=20
            (0.0, 10.0),  # c: mean=0, std=10
        ],
    },
}

# =============================================================================
# SETTINGS
# =============================================================================

opt_method = "MCMC"
loss_function = mean_squared_error

# Data settings
x_min, x_max = -20.0, 20.0
n_samples = 30

# Plot settings
IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=["MCMC", "summary"],
)
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
IMG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Plots will be saved to: {IMG_DIR}")
print(f"Plots will be saved to: {LATEX_DIR}")

# MCMC settings
n_walkers = 32
n_burn = 500
n_samples_mcmc = 1000

# =============================================================================
# MCMC HELPER FUNCTIONS
# =============================================================================


def log_prior(theta, priors):
    """Gaussian priors with (mean, std) for each parameter."""
    logp = 0.0
    for i, (mu, sigma) in enumerate(priors):
        z = (theta[i] - mu) / sigma
        logp += -0.5 * (z * z + np.log(2.0 * np.pi * sigma * sigma))
    return logp


def log_likelihood(theta, x, y, sigma, model_func):
    """Gaussian likelihood."""
    if sigma <= 0 or not np.isfinite(sigma):
        return -np.inf
    try:
        y_pred = model_func(theta, x)
        if y_pred.shape != y.shape:
            return -np.inf
        r = (y - y_pred) / sigma
        return -0.5 * (np.sum(r * r) + r.size * np.log(2.0 * np.pi * sigma * sigma))
    except Exception:
        return -np.inf


def log_posterior(theta, x, y, sigma, priors, model_func):
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, x, y, sigma, model_func)
    return lp + ll


def map_estimate(theta_init, x, y, sigma, priors, model_func):
    """Find MAP estimate using optimization."""
    obj = lambda th: -log_posterior(th, x, y, sigma, priors, model_func)
    res = minimize(obj, theta_init, method="L-BFGS-B")
    return res.x


# =============================================================================
# MCMC INVERSION FOR ALL MODELS
# =============================================================================

results = {}

for model_name, model_config in models.items():
    print(f"\n{'=' * 60}")
    print(f"Processing: {model_config['display_name']}")
    print(f"{'=' * 60}")

    model_func = model_config["func"]
    param_names = model_config["param_names"]
    theta_true = model_config["theta_true"]
    theta_zero = model_config["theta_zero"]
    priors = model_config["priors"]
    n_params = len(theta_true)

    # Reset RNG for reproducibility across scripts
    model_rng = np.random.default_rng(SEED + hash(model_name) % 1000)

    # Generate data
    x_data = np.linspace(x_min, x_max, n_samples)
    y_true = model_func(theta_true, x_data)
    noise_sigma = 0.50 * np.std(y_true)
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=n_samples)

    # Get MAP estimate for initialization
    theta_map = map_estimate(theta_zero, x_data, y_obs, noise_sigma, priors, model_func)
    print(f"  MAP estimate: {np.round(theta_map, 4)}")

    # Initialize walkers around MAP
    scale = np.maximum(0.05 * np.ones(n_params), 0.05 * np.abs(theta_map))
    p0 = theta_map + rng.normal(0.0, scale, size=(n_walkers, n_params))

    # Run MCMC
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_params,
        log_posterior,
        args=(x_data, y_obs, noise_sigma, priors, model_func),
    )

    print(f"  Running MCMC: {n_burn} burn-in + {n_samples_mcmc} production samples...")
    state = sampler.run_mcmc(p0, n_burn, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, n_samples_mcmc, progress=True)

    # Extract samples
    samples = sampler.get_chain(flat=True)

    # Compute statistics
    theta_mean = np.mean(samples, axis=0)
    theta_std = np.std(samples, axis=0)

    # Compute correlation matrix from samples
    corr = np.corrcoef(samples.T)

    # Compute MSE at mean estimate
    mse_mean = loss_function(y_obs, model_func(theta_mean, x_data))

    # Store results
    results[model_name] = {
        "config": model_config,
        "x_data": x_data,
        "y_obs": y_obs,
        "y_true": y_true,
        "theta_true": theta_true,
        "theta_map": theta_map,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "samples": samples,
        "mse": mse_mean,
        "corr": corr,
        "noise_sigma": noise_sigma,
    }

    print(f"  True params   : {np.round(theta_true, 4)}")
    print(f"  Mean posterior: {np.round(theta_mean, 4)}")
    print(f"  Std posterior : {np.round(theta_std, 4)}")
    print(f"  MSE (at mean) : {mse_mean:.4f}")

# %%
# =============================================================================
# AGGREGATED PLOT 1: MODEL FITS (3 x 1)
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(width, height * 1.5))

for idx, (model_name, result) in enumerate(results.items()):
    ax = axes[idx]

    model_func = result["config"]["func"]
    x_data = result["x_data"]
    y_obs = result["y_obs"]
    theta_mean = result["theta_mean"]
    theta_true = result["theta_true"]
    samples = result["samples"]

    # Plot data
    x_plot = np.linspace(x_data.min(), x_data.max(), 400)
    y_true_plot = model_func(theta_true, x_plot)

    # Plot uncertainty band using samples
    n_plot_samples = min(200, len(samples))
    sample_indices = rng.choice(len(samples), n_plot_samples, replace=False)

    for i in sample_indices:
        y_sample = model_func(samples[i], x_plot)
        ax.plot(x_plot, y_sample, color=default_colors[2], alpha=0.02, linewidth=0.5)

    # Plot mean prediction
    y_pred = model_func(theta_mean, x_plot)

    ax.scatter(
        x_data, y_obs, s=20, alpha=0.7, label="data", color=default_colors[0], zorder=5
    )
    ax.plot(
        x_plot,
        y_pred,
        color=default_colors[2],
        label="posterior mean",
        linewidth=2,
        zorder=4,
    )
    ax.plot(
        x_plot,
        y_true_plot,
        color=default_colors[1],
        linestyle="--",
        label="true",
        linewidth=2,
        zorder=3,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"\textbf{{{result['config']['display_name']}}}", fontsize=11)
    ax.grid(True, alpha=0.3)

# Create a single legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.04),
    frameon=True,
)

fig.suptitle(r"\textbf{Model Fits Comparison (MCMC)}", y=0.995)
plt.tight_layout()
plt.savefig(IMG_DIR / "summary_MCMC_aggregated_fits.png")
plt.savefig(LATEX_DIR / "summary_MCMC_aggregated_fits.pdf")
plt.show()

# %%
# =============================================================================
# AGGREGATED PLOT 2: CORRELATION MATRICES (1 x 3)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(width * 1.2, height / 1.5))

for idx, (model_name, result) in enumerate(results.items()):
    ax = axes[idx]

    param_names = result["config"]["param_names"]
    corr = result["corr"]

    # Create latex labels
    latex_labels = [to_latex_label(name) for name in param_names]

    # Plot heatmap
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
        cbar=False,
    )

    # Remove ticks
    ax.tick_params(which="both", left=False, bottom=False)

    ax.set_title(rf"\textbf{{{result['config']['display_name']}}}", fontsize=10, pad=10)

# Add a single colorbar for all subplots
cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap=conf_cmap, norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Correlation", rotation=270, labelpad=10)
cbar.ax.tick_params(which="both", size=0)

fig.suptitle(r"\textbf{Parameter Correlation Matrices (MCMC)}", y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.savefig(IMG_DIR / "summary_MCMC_aggregated_correlations.png")
plt.savefig(LATEX_DIR / "summary_MCMC_aggregated_correlations.pdf")
plt.show()

# %%
# =============================================================================
# AGGREGATED PLOT 3: CORNER PLOTS (3 separate plots)
# =============================================================================

for idx, (model_name, result) in enumerate(results.items()):
    samples = result["samples"]
    param_names = result["config"]["param_names"]
    theta_true = result["theta_true"]

    # Create latex labels
    latex_labels = [to_latex_label(name) for name in param_names]

    # Create individual corner plot for this model
    fig = plt.figure(figsize=(width, width))
    fig = corner.corner(
        samples,
        labels=latex_labels,
        truths=theta_true,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        # title_kwargs={"fontsize": 10},
        # label_kwargs={"fontsize": 11},
        truth_color=default_colors[1],
        color=default_colors[0],
        hist_kwargs={"density": True},
    )

    # Add title
    fig.suptitle(
        rf"\textbf{{{result['config']['display_name']} - Posterior Distributions}}",
        y=0.97,
    )
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"summary_MCMC_corner_{model_name}.png")
    plt.savefig(LATEX_DIR / f"summary_MCMC_corner_{model_name}.pdf")
    plt.show()

# %%
# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY TABLE (MCMC)")
print("=" * 80)

for model_name, result in results.items():
    print(f"\n{result['config']['display_name']}:")
    print("-" * 60)

    param_names = result["config"]["param_names"]
    theta_true = result["theta_true"]
    theta_map = result["theta_map"]
    theta_mean = result["theta_mean"]
    theta_std = result["theta_std"]
    corr = result["corr"]

    model_func = result["config"]["func"]
    x_data = result["x_data"]
    y_obs = result["y_obs"]
    noise_sigma = result["noise_sigma"]
    priors = result["config"]["priors"]

    print(f"{'Parameter':<15} {'True':<12} {'MAP':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 60)
    for i, name in enumerate(param_names):
        print(
            f"{name:<15} {theta_true[i]:<12.4f} {theta_map[i]:<12.4f} "
            f"{theta_mean[i]:<12.4f} {theta_std[i]:<12.4f}"
        )

    # Compute R² score
    y_pred_mean = model_func(theta_mean, x_data)
    r2 = r2_score(y_obs, y_pred_mean)

    # Largest absolute off-diagonal correlation
    n_params = len(param_names)
    off_diag_mask = ~np.eye(n_params, dtype=bool)
    max_off_diag_corr = np.max(np.abs(corr[off_diag_mask]))

    # Eigenvalues of correlation matrix
    corr_eigenvalues = np.linalg.eigvalsh(corr)
    corr_eigenvalues = np.sort(corr_eigenvalues)[::-1]  # Sort descending

    # Condition number of correlation matrix
    corr_condition = np.linalg.cond(corr)

    # Compute Hessian at MAP estimate
    def neg_log_posterior(theta):
        return -log_posterior(theta, x_data, y_obs, noise_sigma, priors, model_func)

    # Numerical Hessian using finite differences
    eps = 1e-5
    hessian = np.zeros((n_params, n_params))
    f0 = neg_log_posterior(theta_map)

    for i in range(n_params):
        for j in range(i, n_params):
            theta_ij = theta_map.copy()
            theta_i = theta_map.copy()
            theta_j = theta_map.copy()

            theta_i[i] += eps
            theta_j[j] += eps
            theta_ij[i] += eps
            theta_ij[j] += eps

            f_ij = neg_log_posterior(theta_ij)
            f_i = neg_log_posterior(theta_i)
            f_j = neg_log_posterior(theta_j)

            hessian[i, j] = (f_ij - f_i - f_j + f0) / (eps * eps)
            hessian[j, i] = hessian[i, j]

    # Eigenvalues of Hessian
    hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    hessian_eigenvalues = np.sort(hessian_eigenvalues)[::-1]  # Sort descending

    print(f"\nMSE (at mean):              {result['mse']:.6f}")
    print(f"R² score:                   {r2:.6f}")
    print(f"Noise σ:                    {noise_sigma:.4f}")
    print(f"MCMC samples:               {len(result['samples'])}")
    print(f"\nMax |off-diag| correlation: {max_off_diag_corr:.6f}")
    print(f"Correlation condition #:    {corr_condition:.6f}")
    print(
        f"Correlation eigenvalues:    {np.array2string(corr_eigenvalues, precision=6, separator=', ')}"
    )
    print(
        f"Hessian eigenvalues:        {np.array2string(hessian_eigenvalues, precision=6, separator=', ')}"
    )

print("\n" + "=" * 80)

# %%
# =============================================================================
# LATEX TABLE OUTPUT
# =============================================================================

# print("\n" + "=" * 80)
# print("LATEX TABLE")
# print("=" * 80)
# print("\n")

# Start LaTeX table
latex_lines = []
latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"    \centering")
latex_lines.append(
    r"    \caption{Summary of MCMC parameter estimation and model diagnostics for all models. Reported values are posterior means with standard deviations.}"
)
latex_lines.append(r"    \label{tab:mcmc_summary_full}")
latex_lines.append(r"    \small")
latex_lines.append(r"    \begin{tabular}{lcccc}")
latex_lines.append(r"        \toprule")
latex_lines.append(
    r"        \textbf{Parameter} & \textbf{True} & \textbf{MAP} & \textbf{Mean} & \textbf{Std} \\"
)
latex_lines.append(r"        \midrule")

# Store diagnostics for each model
model_diagnostics = {}

for model_name, result in results.items():
    param_names = result["config"]["param_names"]
    theta_true = result["theta_true"]
    theta_map = result["theta_map"]
    theta_mean = result["theta_mean"]
    theta_std = result["theta_std"]
    corr = result["corr"]

    model_func = result["config"]["func"]
    x_data = result["x_data"]
    y_obs = result["y_obs"]
    noise_sigma = result["noise_sigma"]
    priors = result["config"]["priors"]

    # Compute diagnostics
    y_pred_mean = model_func(theta_mean, x_data)
    r2 = r2_score(y_obs, y_pred_mean)

    n_params = len(param_names)
    off_diag_mask = ~np.eye(n_params, dtype=bool)
    max_off_diag_corr = np.max(np.abs(corr[off_diag_mask]))

    corr_eigenvalues = np.linalg.eigvalsh(corr)
    corr_eigenvalues = np.sort(corr_eigenvalues)[::-1]

    corr_condition = np.linalg.cond(corr)

    # Compute Hessian at MAP estimate
    def neg_log_posterior(theta):
        return -log_posterior(theta, x_data, y_obs, noise_sigma, priors, model_func)

    eps = 1e-5
    hessian = np.zeros((n_params, n_params))
    f0 = neg_log_posterior(theta_map)

    for i in range(n_params):
        for j in range(i, n_params):
            theta_ij = theta_map.copy()
            theta_i = theta_map.copy()
            theta_j = theta_map.copy()

            theta_i[i] += eps
            theta_j[j] += eps
            theta_ij[i] += eps
            theta_ij[j] += eps

            f_ij = neg_log_posterior(theta_ij)
            f_i = neg_log_posterior(theta_i)
            f_j = neg_log_posterior(theta_j)

            hessian[i, j] = (f_ij - f_i - f_j + f0) / (eps * eps)
            hessian[j, i] = hessian[i, j]

    hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    hessian_eigenvalues = np.sort(hessian_eigenvalues)[::-1]

    model_diagnostics[model_name] = {
        "r2": r2,
        "max_off_diag_corr": max_off_diag_corr,
        "corr_condition": corr_condition,
        "corr_eigenvalues": corr_eigenvalues,
        "hessian_eigenvalues": hessian_eigenvalues,
    }

for model_idx, (model_name, result) in enumerate(results.items()):
    param_names = result["config"]["param_names"]
    theta_true = result["theta_true"]
    theta_map = result["theta_map"]
    theta_mean = result["theta_mean"]
    theta_std = result["theta_std"]

    # Model header
    latex_lines.append(
        rf"        \multicolumn{{5}}{{l}}{{\textbf{{{result['config']['display_name']}}}}} \\"
    )
    latex_lines.append(r"        \midrule")

    # Parameter rows
    for i, name in enumerate(param_names):
        # Handle special characters in parameter names
        param_latex = f"${name}$" if name != r"\alpha" else r"$\alpha$"
        latex_lines.append(
            f"        {param_latex:<20} & {theta_true[i]:<10.4f} & {theta_map[i]:<10.4f} & {theta_mean[i]:<10.4f} & {theta_std[i]:<10.4f} \\\\"
        )

    latex_lines.append(r"        \midrule")

    # Format eigenvalues for LaTeX
    diag = model_diagnostics[model_name]
    corr_eig_str = ", ".join([f"{x:.3f}" for x in diag["corr_eigenvalues"]])
    hess_eig_str = ", ".join(
        [
            f"{x:.2e}" if abs(x) < 0.01 or abs(x) > 1000 else f"{x:.2f}"
            for x in diag["hessian_eigenvalues"]
        ]
    )

    # Diagnostics in minipage
    latex_lines.append(r"        \multicolumn{5}{l}{")
    latex_lines.append(
        r"        \begin{minipage}{0.95\linewidth}\vspace{2pt}\footnotesize"
    )
    latex_lines.append(f"                MSE (at mean): {result['mse']:.2f} \\quad")
    latex_lines.append(f"                $R^2$: {diag['r2']:.3f} \\quad")
    latex_lines.append(
        f"                Noise $\\sigma$: {result['noise_sigma']:.2f} \\quad"
    )
    latex_lines.append(f"                Samples: {len(result['samples'])} \\\\")
    latex_lines.append(
        f"                Max $|\\mathrm{{corr}}_{{ij}}|$: {diag['max_off_diag_corr']:.3f} \\quad"
    )
    latex_lines.append(
        f"                Cond.\\# (corr.): {diag['corr_condition']:.2f} \\quad"
    )
    latex_lines.append(f"                Eigenvalues (corr.): [{corr_eig_str}] \\\\")
    latex_lines.append(f"                Eigenvalues (Hessian): [{hess_eig_str}]")
    latex_lines.append(r"                \vspace{2pt}\end{minipage}} \\")

    # Add spacing between models except for the last one
    if model_idx < len(results) - 1:
        latex_lines.append(r"        \addlinespace[0.7em]")
        latex_lines.append("")

latex_lines.append(r"        \bottomrule")
latex_lines.append(r"    \end{tabular}")
latex_lines.append(r"\end{table}")

# Print the table
for line in latex_lines:
    print(line)

# print("\n" + "=" * 80)

# %%
