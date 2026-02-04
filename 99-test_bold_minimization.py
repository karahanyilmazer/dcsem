# %%
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pypalettes import load_cmap
from scipy.optimize import minimize
from tqdm import tqdm

from dcsem.utils import stim_boxcar
from utils import add_underscore, get_colormap, set_style, simulate_bold

set_style()

# -----------------------------
# Function definitions
# -----------------------------


def theta_to_params(theta):
    return dict(zip(param_names, map(float, theta)))


def mse(theta):
    params = theta_to_params(theta)
    bold_pred = simulate_bold(
        params, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
    )
    return float(np.mean((bold_pred - bold_obs) ** 2))


def _callback(xk):
    theta_hist.append(np.array(xk, dtype=float))


def make_range(center, span, n, name=None):
    if name is not None:
        low_bound, high_bound = param_bounds[name]
    else:
        low_bound, high_bound = 0.0, 2.0  # fallback
    low = max(low_bound, center - span)
    high = min(high_bound, center + span)
    return np.linspace(low, high, n)


# -----------------------------
# Data generation (true signals)
# -----------------------------
OUT_DIR = Path(__file__).parent / "results" / "bold_estimation" / "wip"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])

true_params = {
    "a01": 0.4,
    "a10": 0.6,
    "c0": 0.9,
    "c1": 0.2,
}
param_names = list(true_params.keys())

# Prefer stiff solver for large/unstable couplings
ODE_METHOD = "BDF"  # or "Radau"; set None to use default RK45

bold_true = simulate_bold(
    true_params, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
)

# Optional observation noise (set to 0.0 for clean)
noise_sigma = 0.01
if noise_sigma > 0:
    rng = np.random.default_rng(0)
    bold_obs = bold_true + rng.normal(0, noise_sigma, bold_true.shape)
else:
    bold_obs = bold_true

# -----------------------------
# Model + MSE loss
# -----------------------------


# -----------------------------
# Estimate parameters via minimize
# -----------------------------
theta0 = np.array([0.3, 0.8, 0.7, 0.3], dtype=float)
theta0 = np.array([0.1, 0.3, 0.2, 0.5], dtype=float)
theta_hist = [theta0.copy()]


# Use literature-informed bounds:
# A-matrix: allow negative (inhibitory) and positive (excitatory) connections
# C-matrix: non-negative, typically less than ~1.5
bounds = [
    (-1.5, 1.5),  # a01 (A matrix)
    (-1.5, 1.5),  # a10 (A matrix)
    (0.0, 1.5),  # c0  (C matrix)
    (0.0, 1.5),  # c1  (C matrix)
]

res = minimize(
    mse,
    theta0,
    method="L-BFGS-B",
    bounds=bounds,
    callback=_callback,
    # options={"disp": True, "maxiter": 200},
)
theta_hat = res.x

print("Estimated [a01, a10, c0, c1]:", theta_hat)
print("True      [a01, a10, c0, c1]:", [true_params[k] for k in param_names])


# -----------------------------
# Plot observed vs fitted BOLD
# -----------------------------
bold_fit = simulate_bold(
    theta_to_params(theta_hat), time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
)

CMAP = load_cmap("X78")
COLORS = CMAP.colors

# %%
fig, axes = plt.subplots(1, NUM_ROIS, sharex=True, figsize=(12, 5))
for r in range(NUM_ROIS):
    axes[r].plot(
        time,
        bold_obs[:, r],
        label="Observed",
        lw=2,
        color=COLORS[0],
    )
    axes[r].plot(
        time,
        bold_fit[:, r],
        label="Fitted",
        lw=3,
        color=COLORS[1],
    )
    axes[r].plot(
        time,
        bold_true[:, r],
        linestyle="--",
        label="True",
        lw=3,
        color=COLORS[2],
    )
    axes[r].set_title(f"ROI {r}")
    axes[r].set_xlabel("time")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
fig.suptitle("Model Inversion Fit to Simulated BOLD Signals")
plt.tight_layout()
plt.savefig(OUT_DIR / "bold_fit.png")
plt.show()

# %%
# -------------------------------------------------------------
# Loss landscapes: 1D for each param, 2D for each pair
# -------------------------------------------------------------

# Choose spans per-parameter
spans = {
    "a01": max(0.4, abs(theta_hat[0])) * 1.5,
    "a10": max(0.4, abs(theta_hat[1])) * 1.5,
    "c0": max(0.5, abs(theta_hat[2])) * 1.2,
    "c1": max(0.5, abs(theta_hat[3])) * 1.2,
}


param_bounds = dict(zip(param_names, bounds))


# 1D: vary one parameter, fix others
grids_1d = {
    name: make_range(theta_hat[i], spans[name], 150, name=name)
    for i, name in enumerate(param_names)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=True)
axes = axes.flatten()

for ax, name in tqdm(zip(axes, param_names), total=len(axes)):
    i = param_names.index(name)
    grid = grids_1d[name]

    # Build parameter vectors; use simulate_bold batching to speed up
    params = theta_to_params(theta_hat)
    params[name] = grid  # array -> batched simulations
    bold_grid = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS, squeeze=False)
    # bold_grid shape: (N, T, R)
    losses = np.mean((bold_grid - bold_obs[None, ...]) ** 2, axis=(1, 2))

    ax.plot(grid, losses, lw=3, color=COLORS[0])
    ax.axvline(theta_hat[i], lw=3, label="Estimate", color=COLORS[1])
    ax.axvline(true_params[name], linestyle="--", lw=3, label="True", color=COLORS[2])
    ax.set_xlabel(add_underscore(name))
    # ax.set_title(f"MSE vs {name}")
    ax.set_ylim(None, 0.0002)
    ax.grid(True)

axes[0].set_ylabel("MSE Loss Value")
axes[2].set_ylabel("MSE Loss Value")
axes[0].legend(loc="best")

fig.suptitle("1D MSE Loss Landscape\nVary One Parameter, Fix Others")
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout()
plt.savefig(OUT_DIR / "loss_1d_all.png")
plt.show()

# %%

# 2D: vary parameter pairs, fix the other two
CMAP = get_colormap()

pairs = list(combinations(true_params.keys(), 2))
Na, Nb = 10, 10

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, (p, q) in tqdm(enumerate(pairs), total=len(pairs)):
    p_grid = make_range(theta_hat[param_names.index(p)], spans[p], Na, name=p)
    q_grid = make_range(theta_hat[param_names.index(q)], spans[q], Nb, name=q)
    PP, QQ = np.meshgrid(p_grid, q_grid, indexing="ij")

    # Flatten to batch through simulate_bold
    P_flat = PP.ravel()
    Q_flat = QQ.ravel()
    params = theta_to_params(theta_hat)
    params[p] = P_flat
    params[q] = Q_flat

    # Batched simulation over grid
    bold_grid = simulate_bold(
        params,
        time=time,
        u=u,
        num_rois=NUM_ROIS,
        squeeze=False,
        ode_method=ODE_METHOD,
    )

    # Compute MSE per grid point and reshape
    Z = np.mean((bold_grid - bold_obs[None, ...]) ** 2, axis=(1, 2)).reshape(PP.shape)

    ax = axes[idx]
    cont = ax.contourf(PP, QQ, Z, levels=28, cmap=CMAP)
    ax.scatter(
        [theta_hat[param_names.index(p)]],
        [theta_hat[param_names.index(q)]],
        color="tomato",
        s=35,
        label="Estimate",
    )
    ax.scatter(
        [true_params[p]],
        [true_params[q]],
        color="white",
        edgecolors="black",
        s=30,
        label="True",
    )
    ax.set_xlabel(add_underscore(p))
    ax.set_ylabel(add_underscore(q))
    ax.set_title(f"({add_underscore(p)}, {add_underscore(q)})")

axes[0].legend(loc="upper left")
fig.suptitle(
    "2D MSE Loss Landscape\nVary Parameter Pairs, Fix the Other Two",
)
# plt.tight_layout()
cbar = plt.colorbar(cont, ax=axes.tolist(), shrink=0.9, label="MSE Loss Value")
plt.savefig(OUT_DIR / "loss_2d_all.png")
plt.show()

# %%
