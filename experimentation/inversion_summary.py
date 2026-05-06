# %% Imports and config
import itertools

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

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
    },
    "product_degen": {
        "func": lambda theta, x: (theta[0] * theta[1]) * x + theta[2],
        "display_name": "Product Model (Degenerate)",
        "param_names": ["a", "b", "c"],
        "theta_true": np.array([17.0, 3.0, 5.0]),
        "theta_zero": np.array([1.0, 1.0, 0.0]),
        "param_bounds": None,
    },
    "product_reparam": {
        "func": lambda theta, x: theta[0] * x + theta[1],
        "display_name": "Product Model (Reparametrized)",
        "param_names": [r"\alpha", "c"],
        "theta_true": np.array([51.0, 5.0]),
        "theta_zero": np.array([1.0, 0.0]),
        "param_bounds": None,
    },
}

# =============================================================================
# SETTINGS
# =============================================================================

algorithm = "L-BFGS-B"
opt_method = "Least Squares"
loss_function = mean_squared_error

# Data settings
x_min, x_max = -20.0, 20.0
n_samples = 30

# Plot settings
IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=[opt_method, "summary"],
)
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
LATEX_TEX_DIR = get_out_dir(type="latex", subfolder="tables")
IMG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Plots will be saved to: {IMG_DIR}")
print(f"Plots will be saved to: {LATEX_DIR}")

# =============================================================================
# INVERSION FOR ALL MODELS
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
    param_bounds = model_config["param_bounds"]
    n_params = len(theta_true)

    # Reset RNG for reproducibility across scripts
    model_rng = np.random.default_rng(SEED + hash(model_name) % 1000)

    # Generate data
    x_data = np.linspace(x_min, x_max, n_samples)
    y_true = model_func(theta_true, x_data)
    noise_sigma = 0.50 * np.std(y_true)
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=n_samples)

    # Fit model
    obj = lambda th: loss_function(y_obs, model_func(th, x_data))

    res = minimize(
        obj,
        theta_zero,
        method=algorithm,
        bounds=param_bounds,
    )

    theta_est = res.x
    mse_est = obj(theta_est)

    # Compute Hessian diagnostics
    hess_func = nd.Hessian(
        lambda theta: loss_function(y_obs, model_func(theta, x_data))
    )
    H = hess_func(theta_est)
    eigvals = np.linalg.eigvalsh(H)

    residuals = y_obs - model_func(theta_est, x_data)
    sigma_sq_est = np.var(residuals, ddof=n_params)

    rank_deficient = np.any(eigvals < 1e-8)

    if rank_deficient:
        print("⚠️  Hessian is rank-deficient - skipping correlation matrix")
        corr = np.eye(n_params)
        se = np.full(n_params, np.nan)
    else:
        try:
            H_inv = np.linalg.inv(H)
            cov = sigma_sq_est * H_inv
            diag_cov = np.clip(np.diag(cov), 0, None)
            se = np.sqrt(diag_cov)

            denom = np.outer(se, se)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.where(denom > 0, cov / denom, 0)
        except np.linalg.LinAlgError:
            print("⚠️  Failed to invert Hessian - using identity")
            corr = np.eye(n_params)
            se = np.full(n_params, np.nan)

    # Compute R² score
    y_pred = model_func(theta_est, x_data)
    r2 = r2_score(y_obs, y_pred)

    # Largest absolute off-diagonal correlation
    off_diag_mask = ~np.eye(n_params, dtype=bool)
    max_off_diag_corr = (
        np.max(np.abs(corr[off_diag_mask])) if not rank_deficient else np.nan
    )

    # Eigenvalues of correlation matrix
    corr_eigenvalues = np.linalg.eigvalsh(corr)
    corr_eigenvalues = np.sort(corr_eigenvalues)[::-1]  # Sort descending

    # Condition number of correlation matrix
    corr_condition = np.linalg.cond(corr) if not rank_deficient else np.nan

    # Hessian eigenvalues (already computed as eigvals)
    hessian_eigenvalues = np.sort(eigvals)[::-1]  # Sort descending

    # Store results
    results[model_name] = {
        "config": model_config,
        "x_data": x_data,
        "y_obs": y_obs,
        "y_true": y_true,
        "theta_est": theta_est,
        "theta_true": theta_true,
        "mse": mse_est,
        "r2": r2,
        "corr": corr,
        "se": se,
        "noise_sigma": noise_sigma,
        "max_off_diag_corr": max_off_diag_corr,
        "corr_condition": corr_condition,
        "corr_eigenvalues": corr_eigenvalues,
        "hessian_eigenvalues": hessian_eigenvalues,
    }

    print(f"  True params: {np.round(theta_true, 4)}")
    print(f"  Estimated  : {np.round(theta_est, 4)}")
    print(f"  MSE: {mse_est:.4f}")

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
    theta_est = result["theta_est"]
    theta_true = result["theta_true"]

    # Plot data
    x_plot = np.linspace(x_data.min(), x_data.max(), 400)
    y_pred = model_func(theta_est, x_plot)
    y_true_plot = model_func(theta_true, x_plot)

    ax.scatter(x_data, y_obs, s=20, alpha=0.7, label="data", color=default_colors[0])
    ax.plot(x_plot, y_pred, color=default_colors[2], label="fitted", linewidth=2)
    ax.plot(
        x_plot,
        y_true_plot,
        color=default_colors[1],
        linestyle="--",
        label="true",
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

fig.suptitle(rf"\textbf{{Model Fits Comparison ({opt_method})}}", y=0.995)
plt.tight_layout()
plt.savefig(IMG_DIR / "aggregated_fits.png")
plt.savefig(LATEX_DIR / f"summary_{opt_method}_aggregated_fits.pdf")
plt.show()

# %%

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
        cbar=False,  # We'll add a single colorbar later
    )

    # Remove ticks
    ax.tick_params(which="both", left=False, bottom=False)

    ax.set_title(rf"\textbf{{{result['config']['display_name']}}}", fontsize=10, pad=10)

# Add a single colorbar for all subplots
cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.7])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=conf_cmap, norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Correlation", rotation=270, labelpad=10)
cbar.ax.tick_params(which="both", size=0)

fig.suptitle(rf"\textbf{{Parameter Correlation Matrices ({opt_method})}}", y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.savefig(IMG_DIR / "aggregated_correlations.png")
plt.savefig(LATEX_DIR / f"summary_{opt_method}_aggregated_correlations.pdf")
plt.show()

# %%
# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

for model_name, result in results.items():
    print(f"\n{result['config']['display_name']}:")
    print("-" * 60)

    param_names = result["config"]["param_names"]
    theta_true = result["theta_true"]
    theta_est = result["theta_est"]
    se = result["se"]

    print(f"{'Parameter':<15} {'True':<12} {'Estimated':<12} {'Std. Error':<12}")
    print("-" * 60)
    for i, name in enumerate(param_names):
        se_str = f"{se[i]:.4f}" if np.isfinite(se[i]) else "N/A"
        print(f"{name:<15} {theta_true[i]:<12.4f} {theta_est[i]:<12.4f} {se_str:<12}")

    print(f"\nMSE:                        {result['mse']:.6f}")
    print(f"R² score:                   {result['r2']:.6f}")
    print(f"Noise σ:                    {result['noise_sigma']:.4f}")
    print(f"\nMax |off-diag| correlation: {result['max_off_diag_corr']:.6f}")
    print(f"Correlation condition #:    {result['corr_condition']:.6f}")
    print(
        f"Correlation eigenvalues:    {np.array2string(result['corr_eigenvalues'], precision=6, separator=', ')}"
    )
    print(
        f"Hessian eigenvalues:        {np.array2string(result['hessian_eigenvalues'], precision=6, separator=', ')}"
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
    r"    \caption{Summary of least-squares parameter estimation and model diagnostics for all models. Reported values are point estimates with standard errors.}"
)
latex_lines.append(r"    \label{tab:inversion_summary_full}")
latex_lines.append(r"    \small")
latex_lines.append(r"    \begin{tabular}{lcccc}")
latex_lines.append(r"        \toprule")
latex_lines.append(
    r"        \textbf{Parameter} & \textbf{True} & \textbf{Estimated} & \textbf{Std. Error} \\"
)
latex_lines.append(r"        \midrule")

for model_idx, (model_name, result) in enumerate(results.items()):
    param_names = result["config"]["param_names"]
    theta_true = result["theta_true"]
    theta_est = result["theta_est"]
    se = result["se"]

    # Model header
    latex_lines.append(
        rf"        \multicolumn{{4}}{{l}}{{\textbf{{{result['config']['display_name']}}}}} \\"
    )
    latex_lines.append(r"        \midrule")

    # Parameter rows
    for i, name in enumerate(param_names):
        se_str = f"{se[i]:.4f}" if np.isfinite(se[i]) else "N/A"
        # Handle special characters in parameter names
        param_latex = f"${name}$" if name != r"\alpha" else r"$\alpha$"
        latex_lines.append(
            f"        {param_latex:<20} & {theta_true[i]:<10.4f} & {theta_est[i]:<10.4f} & {se_str:<10} \\\\"
        )

    latex_lines.append(r"        \midrule")

    # Format eigenvalues for LaTeX
    corr_eig_str = ", ".join([f"{x:.2f}" for x in result["corr_eigenvalues"]])
    hess_eig_str = ", ".join(
        [
            f"{x:.2e}" if abs(x) < 0.01 or abs(x) > 1000 else f"{x:.2f}"
            for x in result["hessian_eigenvalues"]
        ]
    )

    # Diagnostics in minipage
    latex_lines.append(r"        \multicolumn{4}{l}{")
    latex_lines.append(
        r"        \begin{minipage}{0.95\linewidth}\vspace{2pt}\footnotesize"
    )
    latex_lines.append(f"                MSE: {result['mse']:.2f} \\quad")
    latex_lines.append(f"                $R^2$: {result['r2']:.3f} \\quad")
    latex_lines.append(
        f"                Noise $\\sigma$: {result['noise_sigma']:.2f} \\\\"
    )
    latex_lines.append(
        f"                Max $|\\mathrm{{corr}}_{{ij}}|$: {result['max_off_diag_corr']:.3f} \\quad"
    )
    latex_lines.append(
        f"                Cond.\\# (corr.): {result['corr_condition']:.2f} \\\\"
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

# Write the LaTeX table to a .tex file in LATEX_DIR
latex_output_path = LATEX_TEX_DIR / "summary_inversion_results.tex"
with open(latex_output_path, "w") as f:
    for line in latex_lines:
        f.write(line + "\n")
print(f"\nLaTeX table saved to: {latex_output_path}")

# %%
