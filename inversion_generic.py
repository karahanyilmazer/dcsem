# %% Imports and config
import itertools

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy.optimize import minimize
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from tqdm import tqdm

from dcsem.utils import stim_boxcar
from utils import (
    add_noise,
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
rng = np.random.default_rng(SEED)

# =============================================================================
# MODEL DEFINITIONS - All models defined as functions
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
# param_bounds = None
# IS_DCM_MODEL = False


# 2️⃣ Product degeneracy (structural non-identifiability)
# def model(theta, x):
#     a, b, c = theta
#     return (a * b) * x + c


# model_name = "product_degen"
# model_display_name = "Product Model (Degenerate)"
# param_names = ["a", "b", "c"]
# theta_true = np.array([2.0, 3.0, 5.0])  # slope = a*b = 6
# theta_zero = np.array([1.0, 1.0, 0.0])
# param_bounds = None
# IS_DCM_MODEL = False


# 3️⃣ Product reparametrized (identifiable)
# def model(theta, x):
#     alpha, c = theta
#     return alpha * x + c


# model_name = "product_reparam"
# model_display_name = "Product Model (Reparametrized)"
# param_names = ["alpha", "c"]
# theta_true = np.array([6.0, 5.0])  # slope = a*b = 6
# theta_zero = np.array([1.0, 0.0])
# param_bounds = None
# IS_DCM_MODEL = False


# 4️⃣ Sum of exponentials (sloppy model, huge condition number)
# def model(theta, x):
#     A1, k1, A2, k2 = theta
#     return A1 * np.exp(-k1 * x) + A2 * np.exp(-k2 * x)


# model_name = "sum_of_exponentials"
# model_display_name = "Sum of Exponentials"
# param_names = ["A1", "k1", "A2", "k2"]
# theta_true = np.array([5.0, 0.5, 3.0, 0.1])
# theta_zero = np.array([4.0, 0.4, 2.0, 0.15])
# param_bounds = None
# IS_DCM_MODEL = False


# 5️⃣ Michaelis-Menten (nonlinear but identifiable)
# def model(theta, x):
#     Vmax, KM = theta
#     return Vmax * x / (KM + x)


# model_name = "michaelis_menten"
# model_display_name = "Michaelis-Menten"
# param_names = ["Vmax", "KM"]
# theta_true = np.array([10.0, 2.0])
# theta_zero = np.array([8.0, 1.5])
# param_bounds = None
# IS_DCM_MODEL = False


# 6️⃣ Logistic / Sigmoid (nonlinear, correlated parameters)
# def model(theta, x):
#     L, k, x0 = theta
#     return L / (1 + np.exp(-k * (x - x0)))


# model_name = "logistic_sigmoid"
# model_display_name = "Logistic Sigmoid"
# param_names = ["L", "k", "x0"]
# theta_true = np.array([1.0, 1.0, 5.0])
# theta_zero = np.array([0.8, 0.8, 4.0])
# param_bounds = None
# IS_DCM_MODEL = False


# 7️⃣ Power law
# def model(theta, x):
#     a, b = theta
#     return a * x**b


# model_name = "power_law"
# model_display_name = "Power Law"
# param_names = ["a", "b"]
# theta_true = np.array([2.0, 1.5])
# theta_zero = np.array([1.5, 1.2])
# param_bounds = None
# IS_DCM_MODEL = False

# 8️⃣ DCM - 2 ROI BOLD model (requires different setup)
# This model uses BOLD simulation instead of analytical functions

IS_DCM_MODEL = True
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[10, 20, 1]])
ODE_METHOD = "BDF"  # Stiff solver; use None for default RK45


def model(theta, x):
    """
    For DCM: theta contains [a01, a10, c0, c1]
    x is ignored (time and u are used instead)
    Returns BOLD signals of shape (T, R)
    """
    params = dict(zip(param_names, theta))
    bold = simulate_bold(
        params, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
    )
    return bold  # Shape: (T, R)


model_name = "dcm_2roi"
model_display_name = "DCM 2-ROI Model"
param_names = ["a01", "a10", "c0", "c1"]

# DCM-specific bounds for optimization
param_bounds = [
    (0, 1.5),  # a01 (A matrix: can be inhibitory/excitatory)
    (0, 1.5),  # a10 (A matrix: can be inhibitory/excitatory)
    (0.0, 1.5),  # c0  (C matrix: non-negative input strength)
    (0.0, 1.5),  # c1  (C matrix: non-negative input strength)
]

# Define the true parameter set within bounds
# theta_true = np.array([0.4, 0.6, 0.9, 0.2])
# theta_true = np.array([rng.uniform(low, high) for (low, high) in param_bounds])
# theta_true = np.array([0.1, 0.3, 0.9, 0.2])
theta_true = np.array([0.4, 0.6, 0.9, 0.2])
theta_zero = np.array([rng.uniform(low, high) for (low, high) in param_bounds])
# theta_true = np.array([0.82646088, 0.91726585, 0.71238939, 0.91609269])


# =============================================================================
# SETTINGS
# =============================================================================

loss_function = r2_score
loss_function = mean_squared_error

# Auto-detect number of parameters
n_params = len(theta_zero)

# Optimization settings
opt_method = "L-BFGS-B"  # DCM needs bounds

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
PLOT_1D = True
PLOT_2D = True
PLOT_3D = False  # only if n_params == 3

# Landscape resolution
if not IS_DCM_MODEL:
    N_1D = 100
    N_2D = 100
else:
    N_1D = 20
    N_2D = 20

# Loss plot span overrides (None = auto, or dict with param indices)
SPAN_OVERRIDE = None  # e.g., {0: 2.0, 1: 5.0} for custom spans


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def make_range(center, span, n):
    """Create linearly spaced range around center."""
    return np.linspace(center - span, center + span, n)


def auto_span(value, default_factor=1.5, min_span=1.0):
    """Compute automatic span for parameter."""
    return default_factor * max(min_span, abs(value))


# =============================================================================
# DATA GENERATION
# =============================================================================


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
# FIT
# =============================================================================
loss_history = []


def callback(theta):
    loss = obj(theta)  # your objective function
    loss_history.append(loss)
    print(f"Iteration {len(loss_history)}: loss = {loss:.6e}")


def mse_norm(theta, x_data, y_obs_norm, y_mean, y_std):
    y_pred = model(theta, x_data)
    y_pred_norm = (y_pred - y_mean) / y_std
    return np.mean((y_pred_norm - y_obs_norm) ** 2)


# Apply z-score normalization
y_mean = y_obs.mean(axis=0, keepdims=True)
y_std = y_obs.std(axis=0, keepdims=True) + 1e-12  # avoid div by zero
y_true_norm = (y_true - y_mean) / y_std
y_obs_norm = (y_obs - y_mean) / y_std

obj = lambda th: loss_function(y_obs_norm, (model(th, x_data) - y_mean) / y_std)
# obj = lambda th: loss_function(y_obs, model(th, x_data))
# obj = lambda th: mse_norm(th, x_data, y_obs_norm, y_mean, y_std)

# Run the optimization
res = minimize(
    obj,
    theta_zero,
    method=opt_method,
    callback=callback,
    bounds=param_bounds,
)

# Check convergence
if not res.success:
    print(f"⚠️  Optimization did not converge: {res.message}")

# Extract estimated parameters and final MSE
theta_est = res.x
mse_est = obj(theta_est)

# Print fit results
print("Fit results:")
print(f"  True params: {np.round(theta_true, 4)}")
print(f"  Estimated  : {np.round(theta_est, 4)}")
print(f"  Loss: {mse_est:.4f}  (noise std = {noise_std_actual:.4f})")


# =============================================================================
# PLOT: DATA AND FITTED CURVE
# =============================================================================

if not IS_DCM_MODEL:
    # Standard 1D analytical model plot
    x_plot = np.linspace(x_data.min(), x_data.max(), 400)
    y_pred = model(theta_est, x_plot)
    y_true_plot = model(theta_true, x_plot)

    # Plot
    plt.figure(figsize=(width, height))
    plt.scatter(x_data, y_obs, s=20, alpha=0.7, label="data")
    plt.plot(x_plot, y_pred, color=default_colors[2], label="fitted")
    plt.plot(x_plot, y_true_plot, color=default_colors[1], linestyle="--", label="true")

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
    y_pred = model(theta_est, None)  # Shape: (T, R)
    y_true_plot = model(theta_true, None)  # Shape: (T, R)

    # Get time vector from globals (defined in DCM model section)
    time_vec = globals().get("time", np.arange(y_obs.shape[0]))
    num_rois = y_obs.shape[1]
    fig, axes = plt.subplots(1, num_rois, sharex=True, figsize=(width, height))

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
            y_pred[:, r],
            color=default_colors[2],
            label="fitted",
        )
        axes[r].plot(
            time_vec,
            y_true_plot[:, r],
            linestyle="--",
            color=default_colors[1],
            label="true",
        )
        axes[r].set_title(f"ROI {r+1}")
        axes[r].set_xlabel("Time (s)")
        axes[r].grid(True, alpha=0.3)

    # Create a single legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.07),
        frameon=True,
    )

    axes[0].set_ylabel("BOLD Amplitude")
    fig.suptitle(rf"\textbf{{{model_display_name} - Data and Fit}}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "data_fit.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_data_fit.pdf")
    plt.show()


# %%

# %%
# =============================================================================
# LOSS LANDSCAPES - 1D
# =============================================================================

if PLOT_1D:
    # Compute spans
    spans = []
    for i in range(n_params):
        if SPAN_OVERRIDE and i in SPAN_OVERRIDE:
            spans.append(SPAN_OVERRIDE[i])
        else:
            spans.append(auto_span(theta_est[i], min_span=0.0))

    fig, axes = plt.subplots(1, n_params, figsize=(width, height / 1.5))
    if n_params == 1:
        axes = [axes]

    for i, (ax, name, span) in enumerate(zip(axes, param_names, spans)):
        grid = make_range(theta_est[i], span, N_1D)
        losses = []

        for v in tqdm(grid, desc=f"1D loss {name}"):
            th = theta_est.copy()
            th[i] = v
            losses.append(loss_function(y_obs, model(th, x_data)))

        ax.plot(grid, losses)
        ax.axvline(theta_est[i], color=default_colors[2], label="estimate")
        ax.axvline(theta_true[i], color=default_colors[1], linestyle="--", label="true")
        ax.set_xlabel(to_latex_label(name))
        ax.set_title(f"MSE vs {to_latex_label(name)}")

    axes[0].set_ylabel("MSE")

    # Create a single legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
    )

    fig.suptitle(rf"\textbf{{{model_display_name} - 1D Loss Landscape}}")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.savefig(IMG_DIR / "loss_landscape_1d.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_loss_landscape_1d.pdf")
    plt.show()

# %%
# =============================================================================
# LOSS LANDSCAPES - 2D CONTOURS
# =============================================================================
if PLOT_2D and n_params >= 2:
    # Compute spans
    spans = []
    for i in range(n_params):
        if SPAN_OVERRIDE and i in SPAN_OVERRIDE:
            spans.append(SPAN_OVERRIDE[i])
        else:
            spans.append(auto_span(theta_est[i], min_span=0.0))

    # Generate all pairs
    pairs = list(itertools.combinations(range(n_params), 2))
    n_pairs = len(pairs)

    nrows = min(3, n_pairs)
    ncols = int(np.ceil(n_pairs / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height / 1.5))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for plot_idx, (i, j) in enumerate(pairs):
        ax = axes[plot_idx]

        # Create grids for parameters i and j
        grid_i = make_range(theta_est[i], spans[i], N_2D)
        grid_j = make_range(theta_est[j], spans[j], N_2D)

        Grid_i, Grid_j = np.meshgrid(grid_i, grid_j, indexing="ij")

        # Compute loss over grid
        Z = np.zeros_like(Grid_i)
        for ii in tqdm(
            range(len(grid_i)), desc=f"2D loss {param_names[i]} vs {param_names[j]}"
        ):
            for jj in range(len(grid_j)):
                th = theta_est.copy()
                th[i] = Grid_i[ii, jj]
                th[j] = Grid_j[ii, jj]
                Z[ii, jj] = loss_function(y_obs, model(th, x_data))

        # Plot contour
        cont = ax.contourf(Grid_i, Grid_j, Z, levels=30, cmap=cmap)
        ax.scatter(
            [theta_true[i]],
            [theta_true[j]],
            s=100,
            color=default_colors[1],
            edgecolors="black",
            label="true",
        )
        ax.scatter(
            [theta_est[i]],
            [theta_est[j]],
            marker="*",
            s=50,
            color=default_colors[2],
            edgecolors="black",
            label="estimate",
        )
        ax.set_xlabel(to_latex_label(param_names[i]))
        ax.set_ylabel(to_latex_label(param_names[j]))

        # Build title showing fixed params
        fixed_params = [k for k in range(n_params) if k not in [i, j]]
        fixed_str = ", ".join(
            [
                f"{to_latex_label(param_names[k])}={theta_est[k]:.3g}"
                for k in fixed_params
            ]
        )
        ax.set_title(
            f"({to_latex_label(param_names[i])}, {to_latex_label(param_names[j])}) | {fixed_str}"
            if fixed_str
            else f"({to_latex_label(param_names[i])}, {to_latex_label(param_names[j])})"
        )

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis("off")

    # Create a single legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=n_pairs,
        bbox_to_anchor=(0.5, -0.15),
        frameon=True,
    )

    fig.suptitle(rf"\textbf{{{model_display_name} - 2D Loss Landscape Contours}}")
    # fig.colorbar(cont, ax=axes[:n_pairs].tolist(), shrink=0.8, label="MSE", pad=0.02)
    plt.tight_layout()

    plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_loss_landscape_2d.pdf")
    plt.savefig(IMG_DIR / "loss_landscape_2d.png")
    plt.show()


# %%
# =============================================================================
# LOSS LANDSCAPE - 3D SURFACE (only if n_params == 3)
# =============================================================================

if PLOT_3D and n_params == 3:
    # Use first two parameters for 3D plot
    i, j = 0, 1

    spans = []
    for idx in range(n_params):
        if SPAN_OVERRIDE and idx in SPAN_OVERRIDE:
            spans.append(SPAN_OVERRIDE[idx])
        else:
            spans.append(auto_span(theta_est[idx]))

    grid_i = make_range(theta_est[i], spans[i], N_2D)
    grid_j = make_range(theta_est[j], spans[j], N_2D)
    Grid_i, Grid_j = np.meshgrid(grid_i, grid_j, indexing="ij")

    Z = np.zeros_like(Grid_i)
    for ii in tqdm(range(len(grid_i)), desc="3D surface computation"):
        for jj in range(len(grid_j)):
            th = theta_est.copy()
            th[i] = Grid_i[ii, jj]
            th[j] = Grid_j[ii, jj]
            Z[ii, jj] = loss_function(y_obs, model(th, x_data))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        Grid_i, Grid_j, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95
    )
    ax.set_xlabel(to_latex_label(param_names[i]))
    ax.set_ylabel(to_latex_label(param_names[j]))
    ax.set_zlabel("MSE")

    fixed_str = f"{to_latex_label(param_names[2])}={theta_est[2]:.3g}"
    ax.set_title(
        f"{model_display_name} | MSE({to_latex_label(param_names[i])}, {to_latex_label(param_names[j])}) | {fixed_str}"
    )

    z_est = loss_function(y_obs, model(theta_est, x_data))
    th_true_proj = theta_est.copy()
    th_true_proj[i] = theta_true[i]
    th_true_proj[j] = theta_true[j]
    z_true_proj = loss_function(y_obs, model(th_true_proj, x_data))

    ax.scatter(
        [theta_est[i]],
        [theta_est[j]],
        [z_est],
        s=50,
        color=default_colors[2],
        label="estimate",
    )
    ax.scatter(
        [theta_true[i]],
        [theta_true[j]],
        [z_true_proj],
        color=default_colors[1],
        edgecolors="black",
        s=45,
        label=f"true ({to_latex_label(param_names[2])} fixed)",
    )
    fig.colorbar(surf, ax=ax, shrink=0.7, aspect=12, pad=0.1, label="MSE")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "loss_landscape_3d.png")
    plt.show()


# %%
# =============================================================================
# HESSIAN-BASED DIAGNOSTICS
# =============================================================================
# Compute numerical Hessian
hess_func = nd.Hessian(lambda theta: loss_function(y_obs, model(theta, x_data)))
H = hess_func(theta_est)

# Eigenvalue analysis
eigvals = np.linalg.eigvalsh(H)
eps = 1e-12
eigvals_clipped = np.clip(eigvals, eps, None)
cond = np.max(eigvals_clipped) / np.min(eigvals_clipped)

# Parameter covariance
residuals = y_obs - model(theta_est, x_data)
sigma_sq_est = np.var(residuals, ddof=n_params)

# Check for degeneracy
rank_deficient = np.any(eigvals < 1e-8)

if rank_deficient:
    print("\n⚠️  Hessian is rank-deficient or nearly singular. Skipping inversion.")
    se = np.full(n_params, np.nan)
    max_offdiag_corr = np.nan
    corr = None
else:
    try:
        H_inv = np.linalg.inv(H)
        cov = sigma_sq_est * H_inv

        # Check for negative variances
        diag_cov = np.diag(cov)
        if np.any(diag_cov < 0):
            print(
                "⚠️  Negative variance detected - Hessian may not be positive definite!"
            )
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
            ax=ax,
            square=True,
            cbar_kws={"label": "Correlation"},
        )
        # Remove ticks from heatmap
        ax.tick_params(which="both", left=False, bottom=False)
        # Remove ticks from colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(which="both", size=0)

        ax.set_title(f"{model_display_name} - Parameter Correlation Matrix")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "correlation_matrix.png")
        plt.show()

    except np.linalg.LinAlgError:
        print("⚠️  Failed to invert Hessian - matrix is singular!")
        se = np.full(n_params, np.nan)
        max_offdiag_corr = np.nan
        corr = None

# Print diagnostics
print("\nHessian diagnostics:")
print(f"  Eigenvalues: {np.round(eigvals, 4)}")
print(f"  Condition number: {cond:.2e}")

if cond > 1e6:
    print(f"  ⚠️  High condition number - numerical instability likely!")
if np.min(eigvals) < 1e-6:
    print(
        f"  ⚠️  Near-zero eigenvalue ({np.min(eigvals):.2e}) - model may be degenerate!"
    )

print(
    f"  Estimated noise variance: {sigma_sq_est:.4f}, "
    f"True: {noise_std_actual**2:.4f} (std={noise_std_actual:.4f})"
)
print(f"  Standard errors: {np.round(se, 4)}")

if np.isfinite(max_offdiag_corr):
    print(f"  Max. off-diagonal correlation: {max_offdiag_corr:.3f}")
    if max_offdiag_corr > 0.95:
        print(f"  ⚠️  High parameter correlation - identifiability issues!")

if not rank_deficient:
    print("  95% Confidence intervals:")
    for i, name in enumerate(param_names):
        print(
            f"    {name}: [{ci[i, 0]:.4f}, {ci[i, 1]:.4f}] (True: {theta_true[i]:.4f})"
        )


# %%
# =============================================================================
# LOGGING
# =============================================================================

log_run(
    model_name=model_name,
    method=opt_method,
    seed=SEED,
    settings={
        "n_samples": n_samples if not IS_DCM_MODEL else y_obs.shape[0],
        "noise_sigma": noise_std_actual,
        # "noise_tsnr": noise_tsnr if IS_DCM_MODEL else None,
    },
    params={
        "names": param_names,
        "true": theta_true.tolist(),
        "init": theta_zero.tolist(),
        "est": theta_est.tolist(),
        "se": se.tolist() if not rank_deficient else [float("nan")] * n_params,
        "corr_max": float(max_offdiag_corr) if np.isfinite(max_offdiag_corr) else None,
    },
    hessian={"cond": float(cond), "eigvals": eigvals.tolist()},
    performance={"mse": float(mse_est)},
    correlation=corr.tolist() if corr is not None else None,
    overwrite=False,
)

# %%
