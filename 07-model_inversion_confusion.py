# %%
# !%load_ext autoreload
# !%autoreload 2
import pickle
from random import choice

import matplotlib.pyplot as plt
import numpy as np

try:
    from IPython.display import Markdown, display
except ImportError:
    display = print
    Markdown = lambda s: s  # pass through raw string
from scipy.optimize import minimize
from seaborn import heatmap
from sklearn.metrics import confusion_matrix, mean_squared_error
from tqdm import tqdm

from dcsem import NOISE_CONFIG, PARAM_BOUNDS, get_colormap, set_style
from dcsem.utils import stim_boxcar
from utils import (
    get_out_dir,
    get_width_height_latex,
    simulate_bold,
)

set_style()
IMG_DIR = get_out_dir(type="img", subfolder="bench_final")
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
MODEL_DIR = get_out_dir(type="model", subfolder="bench")
cmap = get_colormap("YlGnBu")
width, height = get_width_height_latex()

SEED = 42
rng = np.random.default_rng(SEED)

# %%
# ======================================================================================
# DCM Model Configuration (matching inversion_generic.py)
# ======================================================================================

NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[10, 20, 1]])  # Input stimulus
ODE_METHOD = "BDF"  # Stiff solver

# Parameters to estimate
param_names = ["a01", "a10", "c0", "c1"]
n_params = len(param_names)

# DCM-specific bounds for optimization (using central config for consistency)
param_bounds = PARAM_BOUNDS.get_bounds_list(param_names)


def model(theta, x):
    """
    DCM model: theta contains [a01, a10, c0, c1]
    x is ignored (time and u are used instead)
    Returns BOLD signals of shape (T, R)
    """
    params = dict(zip(param_names, theta))
    bold = simulate_bold(
        params, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
    )
    return bold  # Shape: (T, R)


# %%
# ======================================================================================
# Inversion Helper Functions
# ======================================================================================


def invert_model(y_obs, initial_guess, param_bounds, loss_function=mean_squared_error):
    """
    Invert the DCM model using L-BFGS-B optimizer with normalization.

    Args:
        y_obs: Observed BOLD signal (T, R)
        initial_guess: Initial parameter guess
        param_bounds: Parameter bounds
        loss_function: Loss function to minimize

    Returns:
        theta_est: Estimated parameters
        loss_val: Final loss value
    """
    # Raw MSE objective (no normalisation — matching inversion_generic.py)
    def obj(th):
        y_pred = model(th, None)
        return loss_function(y_obs, y_pred)

    # Run optimization
    res = minimize(
        obj,
        initial_guess,
        method="L-BFGS-B",
        bounds=param_bounds,
    )

    theta_est = res.x
    loss_val = obj(theta_est)

    return theta_est, loss_val


# %%
# ======================================================================================
# Confusion Matrix Simulation
# ======================================================================================

display(Markdown("## Running Parameter Change Detection Simulation"))

n_samples = 500
change_amount = 0.3  # Match BENCH test effect size (0.1 is below detection at 10% noise)
change_thr = 0.15  # ~half the change amount

true_change = []  # Ground truth: which parameter changed (0=none, 1-4=param index)
inferred_change = []  # Inferred: which parameter changed

# Initial guess for optimization (middle of bounds)
initial_guess = np.array([0.5, 0.5, 0.5, 0.5])

for sample_i in tqdm(range(n_samples), desc="Simulating parameter changes"):
    # ==================================================================================
    # Step 1: Generate random baseline parameters
    # ==================================================================================
    theta_baseline = np.array([rng.uniform(low, high) for (low, high) in param_bounds])

    # Generate baseline BOLD signal
    params_baseline = dict(zip(param_names, theta_baseline))
    bold_baseline = simulate_bold(
        params_baseline, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
    )

    # Add noise
    noise_sigma = NOISE_CONFIG.get_noise_std(np.std(bold_baseline))
    bold_baseline_obs = bold_baseline + rng.normal(
        0.0, noise_sigma, size=bold_baseline.shape
    )

    # ==================================================================================
    # Step 2: Invert baseline model to get first estimate
    # ==================================================================================
    theta_first_est, loss_first = invert_model(
        bold_baseline_obs, initial_guess, param_bounds
    )

    # ==================================================================================
    # Step 3: Randomly choose which parameter to change (0 = no change)
    # ==================================================================================
    change_idx = choice([0, 1, 2, 3, 4])  # 0=no change, 1-4=param index
    true_change.append(change_idx)

    if change_idx == 0:
        # No change: use same BOLD signal
        bold_perturbed_obs = bold_baseline_obs
    else:
        # Change one parameter
        param_idx = change_idx - 1
        theta_perturbed = theta_baseline.copy()
        theta_perturbed[param_idx] = theta_perturbed[param_idx] + change_amount

        # Clip to bounds
        low, high = param_bounds[param_idx]
        theta_perturbed[param_idx] = np.clip(theta_perturbed[param_idx], low, high)

        # Generate perturbed BOLD signal
        params_perturbed = dict(zip(param_names, theta_perturbed))
        bold_perturbed = simulate_bold(
            params_perturbed, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
        )

        # Add noise
        noise_sigma = NOISE_CONFIG.get_noise_std(np.std(bold_perturbed))
        bold_perturbed_obs = bold_perturbed + rng.normal(
            0.0, noise_sigma, size=bold_perturbed.shape
        )

    # ==================================================================================
    # Step 4: Invert perturbed model to get second estimate
    # ==================================================================================
    theta_second_est, loss_second = invert_model(
        bold_perturbed_obs, initial_guess, param_bounds
    )

    # ==================================================================================
    # Step 5: Detect which parameter changed the most
    # ==================================================================================
    diff = np.abs(theta_second_est - theta_first_est)
    largest_change = np.max(diff)

    if largest_change > change_thr:
        # Infer that the parameter with largest change is the one that changed
        inferred_change.append(np.argmax(diff) + 1)  # +1 because 0 is "no change"
    else:
        # No significant change detected
        inferred_change.append(0)

# %%
# ======================================================================================
# Plot Confusion Matrix
# ======================================================================================

labels = ["No Change", "$a_{01}$", "$a_{10}$", "$c_0$", "$c_1$"]
conf_mat = confusion_matrix(true_change, inferred_change, normalize="true")

fig, ax = plt.subplots(1, 1, figsize=(width / 1.2, height))
heatmap(
    conf_mat,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    cbar=True,
    square=True,
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
    cbar_kws={"label": "Proportion"},
)
ax.set_xlabel("Inferred Change")
ax.set_ylabel("Actual Change")
ax.set_title(r"\textbf{Model Inversion - Parameter Change Detection}")
ax.tick_params(axis="x", which="minor", bottom=False, top=False)
ax.tick_params(axis="y", which="minor", left=False, right=False)
ax.tick_params(which="both", left=False, bottom=False)

# Remove ticks from colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(which="both", size=0)

plt.tight_layout()
plt.savefig(IMG_DIR / "confusion_matrix_model_inversion.png")
plt.savefig(LATEX_DIR / "confusion_matrix_model_inversion.pdf")
plt.show(block=False)

# %%
# ======================================================================================
# Save Results
# ======================================================================================


# Save confusion matrix
with open(MODEL_DIR / "conf_inversion.pkl", "wb") as f:
    pickle.dump(conf_mat, f)

# Save detailed results
results = {
    "true_change": true_change,
    "inferred_change": inferred_change,
    "conf_mat": conf_mat,
    "labels": labels,
    "settings": {
        "n_samples": n_samples,
        "change_amount": change_amount,
        "change_thr": change_thr,
        "param_bounds": param_bounds,
        "seed": SEED,
    },
}

with open(MODEL_DIR / "conf_inversion_results.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"\nResults saved to {MODEL_DIR}")
print(f"Confusion matrix shape: {conf_mat.shape}")
print(f"\nOverall accuracy: {np.trace(conf_mat) / len(labels):.2%}")

# %%
# ======================================================================================
# Print Summary Statistics
# ======================================================================================

display(Markdown("## Summary Statistics"))

print("\nConfusion Matrix (row-normalized):")
print("=" * 60)
for i, label in enumerate(labels):
    print(f"{label:12s}: {conf_mat[i, :]}")

print("\n" + "=" * 60)
print("Per-class metrics:")
print("=" * 60)

for i, label in enumerate(labels):
    # True positive rate (recall/sensitivity)
    tpr = conf_mat[i, i]
    # Precision
    col_sum = conf_mat[:, i].sum()
    precision = conf_mat[i, i] / col_sum if col_sum > 0 else 0.0

    print(f"{label:12s}: Recall={tpr:.2%}, Precision={precision:.2%}")

print("=" * 60)

# %%
