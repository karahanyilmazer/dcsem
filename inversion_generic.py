# %% Imports and config
import itertools
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from dcsem import NOISE_CONFIG, PARAM_BOUNDS, get_colormap, set_style, to_latex_label
from dcsem.diagnostics import compute_hessian_diagnostics
from dcsem.numerics import (
    compute_confidence_intervals,
    compute_correlation_matrix,
    compute_standard_errors,
    safe_hessian_inversion,
)
from dcsem.utils import stim_boxcar
from utils import (
    get_out_dir,
    get_width_height_latex,
    log_run,
    simulate_bold,
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
    is_dcm: bool = False
    param_bounds: Optional[list[tuple[float, float]]] = None
    num_rois: Optional[int] = None
    time: Optional[np.ndarray] = None
    u: Optional[Callable] = None
    ode_method: Optional[str] = None
    x_min: float = -20.0
    x_max: float = 20.0
    n_samples: int = 50


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================


def _quadratic(theta, x):
    a, b, c = theta
    return a * x**2 + b * x + c


def _product_degen(theta, x):
    a, b, c = theta
    return (a * b) * x + c


def _product_reparam(theta, x):
    alpha, c = theta
    return alpha * x + c


def _sum_of_exponentials(theta, x):
    A1, k1, A2, k2 = theta
    return A1 * np.exp(-k1 * x) + A2 * np.exp(-k2 * x)


def _michaelis_menten(theta, x):
    Vmax, KM = theta
    return Vmax * x / (KM + x)


def _logistic_sigmoid(theta, x):
    L, k, x0 = theta
    return L / (1 + np.exp(-k * (x - x0)))


def _power_law(theta, x):
    a, b = theta
    return a * x**b


def _make_dcm_func(param_names, time, u, num_rois, ode_method):
    """Factory returning a closure over DCM config."""

    def _dcm_func(theta, x):
        params = dict(zip(param_names, theta))
        bold = simulate_bold(
            params, time=time, u=u, num_rois=num_rois, ode_method=ode_method
        )
        return bold  # Shape: (T, R)

    return _dcm_func


# =============================================================================
# BUILD REGISTRY
# =============================================================================

_dcm_time = np.arange(100)
_dcm_u = stim_boxcar([[10, 20, 1]])
_dcm_param_names = ["a01", "a10", "c0", "c1"]
_dcm_bounds = PARAM_BOUNDS.get_bounds_list(_dcm_param_names)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "quadratic": ModelSpec(
        name="quadratic",
        display_name="Quadratic Model",
        func=_quadratic,
        param_names=["a", "b", "c"],
        theta_true=np.array([1.0, -12.0, 20.0]),
        theta_zero=np.array([0.5, 0.0, 0.0]),
    ),
    "product_degen": ModelSpec(
        name="product_degen",
        display_name="Product Model (Degenerate)",
        func=_product_degen,
        param_names=["a", "b", "c"],
        theta_true=np.array([2.0, 3.0, 5.0]),
        theta_zero=np.array([1.0, 1.0, 0.0]),
    ),
    "product_reparam": ModelSpec(
        name="product_reparam",
        display_name="Product Model (Reparametrized)",
        func=_product_reparam,
        param_names=["alpha", "c"],
        theta_true=np.array([6.0, 5.0]),
        theta_zero=np.array([1.0, 0.0]),
    ),
    "sum_of_exponentials": ModelSpec(
        name="sum_of_exponentials",
        display_name="Sum of Exponentials",
        func=_sum_of_exponentials,
        param_names=["A1", "k1", "A2", "k2"],
        theta_true=np.array([5.0, 0.5, 3.0, 0.1]),
        theta_zero=np.array([4.0, 0.4, 2.0, 0.15]),
    ),
    "michaelis_menten": ModelSpec(
        name="michaelis_menten",
        display_name="Michaelis-Menten",
        func=_michaelis_menten,
        param_names=["Vmax", "KM"],
        theta_true=np.array([10.0, 2.0]),
        theta_zero=np.array([8.0, 1.5]),
    ),
    "logistic_sigmoid": ModelSpec(
        name="logistic_sigmoid",
        display_name="Logistic Sigmoid",
        func=_logistic_sigmoid,
        param_names=["L", "k", "x0"],
        theta_true=np.array([1.0, 1.0, 5.0]),
        theta_zero=np.array([0.8, 0.8, 4.0]),
    ),
    "power_law": ModelSpec(
        name="power_law",
        display_name="Power Law",
        func=_power_law,
        param_names=["a", "b"],
        theta_true=np.array([2.0, 1.5]),
        theta_zero=np.array([1.5, 1.2]),
    ),
    "dcm_2roi": ModelSpec(
        name="dcm_2roi",
        display_name="2-ROI DCM",
        func=_make_dcm_func(
            _dcm_param_names, _dcm_time, _dcm_u, num_rois=2, ode_method="BDF"
        ),
        param_names=_dcm_param_names,
        theta_true=np.array([0.4, 0.6, 0.9, 0.2]),
        theta_zero=np.array([rng.uniform(low, high) for (low, high) in _dcm_bounds]),
        is_dcm=True,
        param_bounds=_dcm_bounds,
        num_rois=2,
        time=_dcm_time,
        u=_dcm_u,
        ode_method="BDF",
    ),
}

# =============================================================================
# UNPACK SELECTED MODEL
# =============================================================================

ACTIVE_MODEL = {
    1: "quadratic",
    2: "product_degen",
    3: "product_reparam",
    4: "sum_of_exponentials",
    5: "michaelis_menten",
    6: "logistic_sigmoid",
    7: "power_law",
    8: "dcm_2roi",
}[1]

spec = MODEL_REGISTRY[ACTIVE_MODEL]
model = spec.func
model_name = spec.name
model_display_name = spec.display_name
param_names = spec.param_names
theta_true = spec.theta_true
theta_zero = spec.theta_zero
IS_DCM_MODEL = spec.is_dcm
param_bounds = spec.param_bounds


# =============================================================================
# SETTINGS
# =============================================================================

# Loss function must return a value to MINIMIZE (lower = better fit).
loss_function = mean_squared_error

# Auto-detect number of parameters
n_params = len(theta_zero)

# Optimization settings
opt_method = "L-BFGS-B"  # DCM needs bounds
title_suffix = "Least Squares"

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
PLOT_1D = False
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


def make_objective(model_func, y_obs, x_data, loss_fn, normalize=True):
    """Define the objective once; reuse for optimization, Hessian, and landscapes."""
    if normalize:
        y_mean = y_obs.mean(axis=0, keepdims=True)
        y_std = y_obs.std(axis=0, keepdims=True) + 1e-12

        y_obs_norm = (y_obs - y_mean) / y_std

        def objective(theta):
            y_pred_norm = (model_func(theta, x_data) - y_mean) / y_std
            return loss_fn(y_obs_norm, y_pred_norm)

        return objective, y_mean, y_std, y_obs_norm
    else:

        def objective(theta):
            return loss_fn(y_obs, model_func(theta, x_data))

        return objective, None, None, None


# =============================================================================
# DATA GENERATION
# =============================================================================

# Data settings
x_min, x_max = spec.x_min, spec.x_max
n_samples = spec.n_samples

if not IS_DCM_MODEL:
    # Standard analytical models
    x_data = np.linspace(x_min, x_max, n_samples)
    y_true = model(theta_true, x_data)
    noise_sigma = NOISE_CONFIG.get_noise_std(np.std(y_true))
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=n_samples)
    noise_std_actual = noise_sigma  # Store actual noise level used
else:
    # DCM BOLD model
    x_data = None  # Not used for DCM
    y_true = model(theta_true, x_data)  # Shape: (T, R)
    noise_sigma = NOISE_CONFIG.get_noise_std(np.std(y_true))
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=y_true.shape)
    noise_std_actual = noise_sigma  # Store actual noise level used

# =============================================================================
# FIT
# =============================================================================
loss_history = []

# Build the single objective used everywhere
obj, y_mean, y_std, y_obs_norm = make_objective(
    model, y_obs, x_data, loss_function, normalize=True
)


def callback(theta):
    loss = obj(theta)
    loss_history.append(loss)
    print(f"Iteration {len(loss_history)}: loss = {loss:.6e}")


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


# %%
# =============================================================================
# PLOT: DATA AND FITTED CURVE
# =============================================================================

if not IS_DCM_MODEL:
    # Standard 1D analytical model plot
    x_plot = np.linspace(x_data.min(), x_data.max(), 400)
    y_pred = model(theta_est, x_plot)
    y_true_plot = model(theta_true, x_plot)

    # Plot
    plt.figure(figsize=(width, height / 1.5))
    plt.scatter(x_data, y_obs, s=20, alpha=0.7, label="data")
    plt.plot(x_plot, y_pred, color=default_colors[2], label="fitted")
    plt.plot(x_plot, y_true_plot, color=default_colors[1], linestyle="--", label="true")

    # Adjust
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(rf"\textbf{{{model_display_name} - Best Fit ({title_suffix})}}")
    plt.legend()

    # Save
    plt.tight_layout()
    plt.savefig(IMG_DIR / "data_fit.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{title_suffix}_data_fit_new.pdf")
    plt.show()

else:
    # DCM BOLD model: plot each ROI's time series
    y_pred = model(theta_est, None)  # Shape: (T, R)
    y_true_plot = model(theta_true, None)  # Shape: (T, R)

    # Get time vector from spec
    time_vec = spec.time if spec.time is not None else np.arange(y_obs.shape[0])
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
        axes[r].set_title(f"ROI {r + 1}")
        axes[r].set_xlabel("Time (s)")
        axes[r].grid(True, alpha=0.3)

    # Create a single legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.1),
        frameon=True,
    )

    axes[0].set_ylabel("BOLD Amplitude")
    fig.suptitle(
        rf"\textbf{{{model_display_name} - Best Fit ({title_suffix})}}", y=0.95
    )
    plt.tight_layout()
    plt.savefig(IMG_DIR / "data_fit.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{title_suffix}_data_fit_new.pdf")
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
            losses.append(obj(th))

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
        bbox_to_anchor=(0.5, -0.1),
        frameon=True,
    )

    fig.suptitle(rf"\textbf{{{model_display_name} - 1D Loss Landscape}}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "loss_landscape_1d.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{title_suffix}_loss_landscape_1d_new.pdf")
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

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height * 2))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for plot_idx, (i, j) in enumerate(pairs):
        ax = axes[plot_idx]

        # Unified grid for all models, centered on theta_est
        grid_i = make_range(theta_est[i], spans[i], N_2D)
        grid_j = make_range(theta_est[j], spans[j], N_2D)
        if param_bounds is not None:
            grid_i = np.clip(grid_i, param_bounds[i][0], param_bounds[i][1])
            grid_j = np.clip(grid_j, param_bounds[j][0], param_bounds[j][1])

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
                Z[ii, jj] = obj(th)

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
        ax.set_title(
            rf"{to_latex_label(param_names[i])} vs. {to_latex_label(param_names[j])} (others fixed)"
        )

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis("off")

    # Create a single legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=n_pairs,
        bbox_to_anchor=(0.5, 0.03),
        frameon=True,
    )

    # Create a single shared colorbar below all subplots
    cbar_ax = fig.add_axes([0.25, -0.04, 0.5, 0.02])  # [left, bottom, width, height]
    cb = fig.colorbar(
        cont,
        cax=cbar_ax,
        orientation="horizontal",
    )

    # Replace numeric ticks with qualitative labels
    cb.set_ticks([])
    cb.set_label("MSE Loss", labelpad=5)

    # Optional gradient labels for clarity
    cb.ax.text(0.0, -0.3, "Low", va="top", ha="left", fontsize="small", color="black")
    cb.ax.text(
        0.0004, -0.3, "High", va="top", ha="left", fontsize="small", color="black"
    )

    fig.suptitle(
        rf"\textbf{{{model_display_name} - 2D Loss Landscape Contours}}", y=0.98
    )
    plt.tight_layout()

    plt.savefig(LATEX_DIR / f"{model_name}_{title_suffix}_loss_landscape_2d_new.pdf")
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
            Z[ii, jj] = obj(th)

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

    z_est = obj(theta_est)
    th_true_proj = theta_est.copy()
    th_true_proj[i] = theta_true[i]
    th_true_proj[j] = theta_true[j]
    z_true_proj = obj(th_true_proj)

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
# Canonical NLL Hessian in scaled parameter space for numerical stability.
# Cov = H_NLL^{-1} directly (sigma_sq=1.0, absorbed into NLL).
HESS_STEP = 1e-3  # step size for numerical Hessian

# Parameter scaling: map param_bounds → [0,1]^n if bounds available
if param_bounds is not None:
    _lowers_h = np.array([b[0] for b in param_bounds])
    _scales_h = np.array([b[1] - b[0] for b in param_bounds])
else:
    _lowers_h = np.zeros(n_params)
    _scales_h = np.ones(n_params)


def _to_scaled_h(theta):
    return (theta - _lowers_h) / _scales_h


def _from_scaled_h(s):
    return s * _scales_h + _lowers_h


# Build canonical NLL objective (0.5 * SSE / sigma2 in residual space)
if y_mean is not None:
    _r_est = (y_obs_norm - (model(theta_est, x_data) - y_mean) / y_std).ravel()
    _resid_scale = 1.0  # already normalized; division is by sigma2_est below
else:
    _r_raw = (y_obs - model(theta_est, x_data)).ravel()
    _resid_scale = float(np.std(_r_raw)) + 1e-12
    _r_est = _r_raw / _resid_scale

_sigma2_est = float(np.dot(_r_est, _r_est)) / max(_r_est.size - n_params, 1)


def _nll_obj(theta):
    y_pred = model(theta, x_data)
    if not np.all(np.isfinite(y_pred)):
        return 1e10
    if y_mean is not None:
        r = (y_obs_norm - (y_pred - y_mean) / y_std).ravel()
    else:
        r = ((y_obs - y_pred).ravel()) / _resid_scale
    return 0.5 * float(np.dot(r, r)) / _sigma2_est


def _nll_scaled_h(s):
    return _nll_obj(_from_scaled_h(s))


theta_s = _to_scaled_h(theta_est)

# Initialise fallback values
se = np.full(n_params, np.nan)
ci = np.full((n_params, 2), np.nan)
corr = None
max_offdiag_corr = np.nan
hess_diag = {}

try:
    H_nll_s = nd.Hessian(_nll_scaled_h, step=HESS_STEP)(theta_s)
    H_nll_s = 0.5 * (H_nll_s + H_nll_s.T)
    H_nll = H_nll_s / np.outer(_scales_h, _scales_h)

    # Hessian diagnostics using positive-spectrum condition number
    hess_diag = compute_hessian_diagnostics(H_nll)

    # Cov = H_NLL^{-1} via pinvh (zeros flat/degenerate directions)
    cov, _ = safe_hessian_inversion(H_nll, 1.0, regularization=1e-6, method="pinvh")
    se = compute_standard_errors(cov, warn_negative=True)
    ci = compute_confidence_intervals(theta_est, se, alpha=0.05)
    corr = compute_correlation_matrix(cov, handle_degenerate=True)
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
    ax.tick_params(which="both", left=False, bottom=False)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(which="both", size=0)
    ax.set_title(rf"\textbf{{{model_display_name} - Parameter Correlation Matrix}}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "correlation_matrix.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{title_suffix}_correlation_matrix_new.pdf")
    plt.show()

except (np.linalg.LinAlgError, Exception) as _hess_err:
    print(f"⚠️  Hessian inversion failed: {type(_hess_err).__name__}: {_hess_err}")

# Print diagnostics
cond = hess_diag.get("condition_number", np.inf)
rank_deficient = hess_diag.get("is_near_singular", True)
print("\nHessian diagnostics:")
if hess_diag:
    print(f"  Eigenvalues (min/med/max): "
          f"{hess_diag['eigvals_min']:.3e} / "
          f"{hess_diag['eigvals_med']:.3e} / "
          f"{hess_diag['eigvals_max']:.3e}")
    print(f"  Condition number (pos. spectrum): {cond:.2e}")
    print(f"  Negative eigenvalues: {hess_diag['n_negative_eigvals']}")
    if hess_diag["is_near_singular"]:
        print("  ⚠️  Near-singular Hessian — model may be degenerate!")
else:
    print("  (Hessian computation failed)")

print(f"  Standard errors: {np.round(se, 4)}")

if np.isfinite(max_offdiag_corr):
    print(f"  Max. off-diagonal correlation: {max_offdiag_corr:.3f}")
    if max_offdiag_corr > 0.95:
        print("  ⚠️  High parameter correlation - identifiability issues!")

if not rank_deficient:
    print("  95% Confidence intervals (local quadratic approx):")
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
    },
    params={
        "names": param_names,
        "true": theta_true.tolist(),
        "init": theta_zero.tolist(),
        "est": theta_est.tolist(),
        "se": se.tolist() if not rank_deficient else [float("nan")] * n_params,
        "corr_max": float(max_offdiag_corr) if np.isfinite(max_offdiag_corr) else None,
    },
    hessian={
        "cond": float(cond),
        "eigval_min": hess_diag.get("eigvals_min", float("nan")),
        "eigval_max": hess_diag.get("eigvals_max", float("nan")),
        "n_negative": hess_diag.get("n_negative_eigvals", 0),
    },
    performance={"mse": float(mse_est)},
    correlation=corr.tolist() if corr is not None else None,
    overwrite=False,
)

# %%
