"""Playground for ``inversion_generic.py`` — step the L-BFGS-B time-domain
pipeline cell by cell in VS Code's Interactive Window or Jupyter.

Mirrors the body of ``inversion_generic.run_single(cfg)`` at module level
with ``# %%`` cell markers so each stage can be executed and inspected
independently. The model registry, RunConfig, and pure helpers are
imported from ``inversion_generic`` so the source of truth stays
single-place.

Usage:
    Open in VS Code → click "Run Cell" above each ``# %%`` block.
    Variables persist across cells in the Interactive Window.

NOTE: artifacts land in the same ``results/images/...`` path as a
production run, so this will overwrite the most recent NPZ for the
chosen model. That's fine for testing — re-run any production script to
refresh.
"""

# %% Imports + style
import os

# Set a temp latex dir so utils.get_out_dir(type="latex", ...) does not raise
# even if the user has not configured DCSEM_LATEX_DIR in their .env.
os.environ.setdefault("DCSEM_LATEX_DIR", "/tmp/dcsem_latex_playground")
os.makedirs("/tmp/dcsem_latex_playground", exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numdifftools as nd  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402

from dcsem import NOISE_CONFIG, to_latex_label  # noqa: E402
from dcsem.diagnostics import compute_hessian_diagnostics  # noqa: E402
from dcsem.numerics import (  # noqa: E402
    compute_confidence_intervals,
    compute_correlation_matrix,
    compute_standard_errors,
    safe_hessian_inversion,
)
from inversion_generic import (  # noqa: E402
    MODEL_REGISTRY,
    RunConfig,
    auto_span,
    make_objective,
    make_range,
)
from utils import get_out_dir  # noqa: E402


# %% Pick model + unpack spec
# Edit MODEL_NAME to switch model:
#   "quadratic", "product_degen", "sum_of_exponentials",
#   "michaelis_menten", "logistic_sigmoid", "power_law", "dcm_2roi"
MODEL_NAME = "quadratic"

cfg = RunConfig(model_name=MODEL_NAME, seed=42, hess_step=1e-3)
spec = MODEL_REGISTRY[cfg.model_name]

model = spec.func
model_name = spec.name
model_display_name = spec.display_name
param_names = spec.param_names
theta_true = spec.theta_true
theta_zero = spec.theta_zero
IS_DCM_MODEL = spec.is_dcm
param_bounds = spec.param_bounds
n_params = len(theta_zero)

print(f"Model: {model_display_name}")
print(f"  param_names = {param_names}")
print(f"  theta_true  = {theta_true}")
print(f"  theta_zero  = {theta_zero}")
print(f"  bounds      = {param_bounds}")
print(f"  is_dcm      = {IS_DCM_MODEL}")


# %% Output paths
opt_method = "L-BFGS-B"
IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=[opt_method, model_name],
)
IMG_DIR.mkdir(parents=True, exist_ok=True)
print(f"NPZ + plots → {IMG_DIR}")


# %% Generate noisy synthetic data
rng = np.random.default_rng(cfg.seed)
loss_function = mean_squared_error

if not IS_DCM_MODEL:
    x_data = np.linspace(spec.x_min, spec.x_max, spec.n_samples)
    y_true = model(theta_true, x_data)
    noise_sigma = NOISE_CONFIG.get_noise_std(np.std(y_true))
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=spec.n_samples)
else:
    x_data = None
    y_true = model(theta_true, x_data)  # (T, R) BOLD time series
    noise_sigma = NOISE_CONFIG.get_noise_std(np.std(y_true))
    y_obs = y_true + rng.normal(0.0, noise_sigma, size=y_true.shape)

print(f"y_obs shape = {y_obs.shape},  noise σ = {noise_sigma:.4e}")


# %% Quick visual sanity check on the data
plt.figure()
if not IS_DCM_MODEL:
    plt.scatter(x_data, y_obs, s=20, alpha=0.6, label="observed")
    plt.plot(x_data, y_true, "k--", lw=1, label="true")
    plt.xlabel("x")
    plt.ylabel("y")
else:
    for r in range(y_obs.shape[1]):
        plt.plot(y_obs[:, r], alpha=0.7, label=f"ROI {r + 1}")
    plt.xlabel("time")
    plt.ylabel("BOLD")
plt.title(f"{model_display_name} — observed data")
plt.legend()
plt.tight_layout()
plt.show()


# %% Build the optimisation objective + run L-BFGS-B
obj, _ = make_objective(model, y_obs, x_data, loss_function, noise_sigma=noise_sigma)
loss_history = []


def callback(theta):
    loss_history.append(obj(theta))


res = minimize(obj, theta_zero, method=opt_method, callback=callback, bounds=param_bounds)
theta_est = res.x
mse_est = obj(theta_est)

print(f"Converged: {res.success}  ({res.message})")
print(f"Iterations: {len(loss_history)}")
print(f"Final MSE : {mse_est:.6e}")
print(f"theta_true: {np.round(theta_true, 4)}")
print(f"theta_est : {np.round(theta_est, 4)}")


# %% Plot loss history
plt.figure()
plt.plot(loss_history, "o-")
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.title("Optimiser loss history")
plt.yscale("log")
plt.tight_layout()
plt.show()


# %% (Optional, slow for DCM) 1D loss landscapes around the estimate
PLOT_1D = True
if PLOT_1D:
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3))
    if n_params == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        span = auto_span(theta_est[i], min_span=0.0)
        N_1D = 100 if not IS_DCM_MODEL else 20
        grid = make_range(theta_est[i], span, N_1D)
        losses = []
        for v in grid:
            th = theta_est.copy()
            th[i] = v
            losses.append(obj(th))
        ax.plot(grid, losses)
        ax.axvline(theta_est[i], color="C2", label="estimate")
        ax.axvline(theta_true[i], color="C1", linestyle="--", label="true")
        ax.set_xlabel(to_latex_label(name))
        ax.set_title(f"MSE vs {name}")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    plt.tight_layout()
    plt.show()


# %% Set up scaled NLL objective for the Hessian
HESS_STEP = cfg.hess_step

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


_r_est = (y_obs - model(theta_est, x_data)).ravel()
_sigma2_est = float(np.dot(_r_est, _r_est)) / max(_r_est.size - n_params, 1)


def _nll_obj(theta):
    y_pred = model(theta, x_data)
    if not np.all(np.isfinite(y_pred)):
        return 1e10
    r = (y_obs - y_pred).ravel()
    return 0.5 * float(np.dot(r, r)) / _sigma2_est


def _nll_scaled_h(s):
    return _nll_obj(_from_scaled_h(s))


print(f"sigma2_est = {_sigma2_est:.4e}")


# %% Compute the Hessian and inspect its eigenvalue spectrum
theta_s = _to_scaled_h(theta_est)
H_nll_s = nd.Hessian(_nll_scaled_h, step=HESS_STEP)(theta_s)
H_nll_s = 0.5 * (H_nll_s + H_nll_s.T)
H_nll = H_nll_s / np.outer(_scales_h, _scales_h)

hess_diag = compute_hessian_diagnostics(H_nll)
print("Hessian diagnostics:")
print(f"  eigvals (min/med/max): {hess_diag['eigvals_min']:.3e} / "
      f"{hess_diag['eigvals_med']:.3e} / {hess_diag['eigvals_max']:.3e}")
print(f"  condition number:      {hess_diag['condition_number']:.2e}")
print(f"  is_near_singular:      {hess_diag['is_near_singular']}")
print(f"  n_negative_eigvals:    {hess_diag['n_negative_eigvals']}")


# %% Invert the Hessian to get cov + cov_is_calibrated
cov, cov_diag = safe_hessian_inversion(
    H_nll, 1.0, regularization=1e-6, method="adaptive_ridge"
)
cov_is_calibrated = not (
    cov_diag.get("regularization_warning", False)
    or hess_diag.get("is_near_singular", False)
)
se = compute_standard_errors(cov, warn_negative=True)
ci = compute_confidence_intervals(theta_est, se, alpha=0.05)
corr = compute_correlation_matrix(cov, handle_degenerate=True)
max_offdiag_corr = np.nanmax(np.abs(corr - np.eye(n_params)))

print(f"cov_is_calibrated = {cov_is_calibrated}")
print(f"max |off-diag corr| = {max_offdiag_corr:.3f}")
print("95% CIs:")
for i, name in enumerate(param_names):
    print(f"  {name}: [{ci[i, 0]:+.4f}, {ci[i, 1]:+.4f}]  (true: {theta_true[i]:+.4f})")


# %% Inspect the correlation matrix
fig, ax = plt.subplots()
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n_params), [to_latex_label(n) for n in param_names])
ax.set_yticks(range(n_params), [to_latex_label(n) for n in param_names])
for i in range(n_params):
    for j in range(n_params):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")
fig.colorbar(im, ax=ax, label="correlation")
ax.set_title(f"{model_display_name} — parameter correlation")
plt.tight_layout()
plt.show()


# %% Save NPZ artifact + verify schema
np.savez(
    IMG_DIR / "run_results.npz",
    y_obs=y_obs,
    y_pred=model(theta_est, x_data),
    theta_est=theta_est,
    theta_true=theta_true,
    theta_zero=theta_zero,
    se=se,
    ci=ci,
    cov=cov,
    hess_cond=np.array([float(hess_diag["condition_number"])]),
    cov_is_calibrated=np.array([bool(cov_is_calibrated)]),
    hess_is_near_singular=np.array([bool(hess_diag["is_near_singular"])]),
    converged=np.array([bool(res.success)]),
)
print(f"Wrote artifact to {IMG_DIR / 'run_results.npz'}")

with np.load(IMG_DIR / "run_results.npz") as d:
    print("NPZ keys:", sorted(d.files))
    print(f"  cov_is_calibrated = {bool(d['cov_is_calibrated'][0])}")
    print(f"  converged         = {bool(d['converged'][0])}")
    print(f"  theta_est shape   = {d['theta_est'].shape}")
