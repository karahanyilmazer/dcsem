"""Playground for ``spdcm_generic.py`` — step the L-BFGS-B spectral DCM
pipeline cell by cell in VS Code's Interactive Window or Jupyter.

Mirrors the body of ``spdcm_generic.run_single(cfg)`` at module level
with ``# %%`` cell markers. Helpers (``_resolve_effective_tr``,
``_resolve_param_spec``, ``estimate_log_sigma_e``, ``make_objective``)
are imported from the production module so the source of truth stays
single-place.

Default mode is ``synthetic_csd`` — the fastest path. Switch to
``synthetic_bold`` (slow, needs sdeint) or ``empirical`` (needs a
``bold_path``) by editing the cfg below.

NOTE: artifacts land in the same ``results/images/...`` path as a
production run.
"""

# %% Imports + style
import os

os.environ.setdefault("DCSEM_LATEX_DIR", "/tmp/dcsem_latex_playground")
os.makedirs("/tmp/dcsem_latex_playground", exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numdifftools as nd  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from dcsem import SpectralDCM, to_latex_label  # noqa: E402
from dcsem.diagnostics import compute_hessian_diagnostics  # noqa: E402
from dcsem.numerics import (  # noqa: E402
    compute_confidence_intervals,
    compute_correlation_matrix,
    compute_standard_errors,
    safe_hessian_inversion,
)
from pipelines.spdcm_generic import (  # noqa: E402
    RunConfig,
    _resolve_effective_tr,
    _resolve_param_spec,
)
from utils import get_out_dir  # noqa: E402

# %% Configure the run
cfg = RunConfig(
    n_rois=2,
    TR=1.0,
    data_mode="synthetic_csd",  # "synthetic_csd" | "synthetic_bold" | "empirical"
    snr=10.0,
    seed=42,
    profile_params=[],  # [] = skip profile sweep; None = all params
)


# %% Resolve effective TR + build SpectralDCM
tr_effective, _bold_pre_loaded = _resolve_effective_tr(cfg)
spdcm = SpectralDCM(
    n_rois=cfg.n_rois,
    TR=tr_effective,
    self_connection=cfg.self_connection,
    freq_lo=cfg.freq_lo,
    freq_hi=cfg.freq_hi,
    n_freqs=cfg.n_freqs,
)
print(f"SpectralDCM(n_rois={spdcm.n_rois}, TR={spdcm.TR}, n_freqs={cfg.n_freqs})")
print(f"freq band: [{cfg.freq_lo:.3f}, {cfg.freq_hi:.3f}] Hz")


# %% Resolve parameter spec
param_names, theta_true, theta_zero, param_bounds = _resolve_param_spec(
    spdcm,
    cfg.theta_true,
    cfg.theta_zero,
    cfg.param_names,
    cfg.param_bounds,
)
n_params = len(theta_zero)


def model(theta, _x=None):
    return spdcm.predict_csd(theta)


print(f"param_names: {param_names}")
print(f"theta_true : {np.round(theta_true, 4)}")
print(f"theta_zero : {np.round(theta_zero, 4)}")
print(f"bounds     : {param_bounds}")


# %% Output paths
opt_method = "L-BFGS-B"
model_name = f"spdcm_{cfg.n_rois}roi"
model_display_name = f"{cfg.n_rois}-ROI Spectral DCM (LS-spDCM)"
IMG_DIR = get_out_dir(
    type="img",
    subfolder="inversion",
    extra_subfolders=[opt_method, model_name],
)
IMG_DIR.mkdir(parents=True, exist_ok=True)
print(f"NPZ + plots → {IMG_DIR}")


# %% Generate / load CSD data
rng = np.random.default_rng(cfg.seed)
bold_ref = None
tvec_ref = None

if cfg.data_mode == "synthetic_csd":
    y_obs = spdcm.generate_noisy_csd(theta_true, snr=cfg.snr, rng=rng)
    has_ground_truth = True
elif cfg.data_mode == "synthetic_bold":
    bold_ref, tvec_ref = spdcm.simulate_bold(theta_true, T=cfg.T_sim, rng=rng)
    y_obs = spdcm.observed_csd(bold_ref, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    has_ground_truth = True
elif cfg.data_mode == "empirical":
    bold_ref = _bold_pre_loaded
    y_obs = spdcm.observed_csd(bold_ref, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    tvec_ref = np.arange(bold_ref.shape[0]) * spdcm.TR
    has_ground_truth = False
else:
    raise ValueError(f"Unknown data_mode: {cfg.data_mode!r}")

print(f"CSD vector length: {len(y_obs)}")
if has_ground_truth:
    y_true = model(theta_true)
    noise_std_actual = float(np.std(y_obs - y_true))
    print(f"noise std (obs - true): {noise_std_actual:.4e}")
else:
    y_true = None
    noise_std_actual = np.nan


# %% Visual sanity check on the observed CSD vector
# y_obs is the flattened real-valued CSD (n_freqs × components); plot
# element-wise rather than against frequency to avoid shape mismatches.
plt.figure()
plt.plot(np.abs(y_obs), label="|y_obs|")
if has_ground_truth:
    plt.plot(np.abs(y_true), "k--", lw=1, label="|y_true|")
plt.xlabel("CSD vector index")
plt.ylabel("|component|")
plt.yscale("log")
plt.title(f"{model_display_name} — observed CSD ({cfg.data_mode})")
plt.legend()
plt.tight_layout()
plt.show()


# %% Build absorbed-σ NLL objective + run L-BFGS-B
def obj(theta):
    y_pred = model(theta)
    if not np.all(np.isfinite(y_pred)):
        return 1e10
    r = (y_obs - y_pred).ravel()
    return float(np.dot(r, r))  # raw SSE, scale-equivalent to MSE for fitting


loss_history = []


def callback(theta):
    loss_history.append(obj(theta))


# Clip theta_zero to bounds (defensive)
theta_zero_clipped = np.clip(
    theta_zero, [b[0] for b in param_bounds], [b[1] for b in param_bounds]
)

res = minimize(
    obj, theta_zero_clipped, method=opt_method, callback=callback, bounds=param_bounds
)
theta_est = res.x

print(f"Converged: {res.success}  ({res.message})")
print(f"Iterations: {len(loss_history)}")
print(f"Final SSE : {obj(theta_est):.6e}")
if has_ground_truth:
    print(f"theta_true: {np.round(theta_true, 4)}")
print(f"theta_est : {np.round(theta_est, 4)}")


# %% Plot loss history
plt.figure()
plt.plot(loss_history, "o-")
plt.xlabel("iteration")
plt.ylabel("SSE")
plt.title("Optimiser loss history (L-BFGS-B)")
plt.yscale("log")
plt.tight_layout()
plt.show()


# %% Build canonical absorbed-σ NLL for the Hessian
HESS_STEP = 1e-3

_lowers_h = np.array([b[0] for b in param_bounds])
_scales_h = np.array([b[1] - b[0] for b in param_bounds])


def _to_scaled_h(theta):
    return (theta - _lowers_h) / _scales_h


def _from_scaled_h(s):
    return s * _scales_h + _lowers_h


_r_est = (y_obs - model(theta_est)).ravel()
_sigma2_est = float(np.dot(_r_est, _r_est)) / max(_r_est.size - n_params, 1)


def _nll_obj(theta):
    y_pred = model(theta)
    if not np.all(np.isfinite(y_pred)):
        return 1e10
    r = (y_obs - y_pred).ravel()
    return 0.5 * float(np.dot(r, r)) / _sigma2_est


def _nll_scaled_h(s):
    return _nll_obj(_from_scaled_h(s))


print(f"sigma2_est = {_sigma2_est:.4e}")


# %% Compute Hessian + diagnostics
theta_s = _to_scaled_h(theta_est)
H_nll_s = nd.Hessian(_nll_scaled_h, step=HESS_STEP)(theta_s)
H_nll_s = 0.5 * (H_nll_s + H_nll_s.T)
H_nll = H_nll_s / np.outer(_scales_h, _scales_h)

hess_diag = compute_hessian_diagnostics(H_nll)
print("Hessian diagnostics:")
print(
    f"  eigvals (min/med/max): {hess_diag['eigvals_min']:.3e} / "
    f"{hess_diag['eigvals_med']:.3e} / {hess_diag['eigvals_max']:.3e}"
)
print(f"  condition number:      {hess_diag['condition_number']:.2e}")
print(f"  is_near_singular:      {hess_diag['is_near_singular']}")


# %% Invert Hessian → cov + cov_is_calibrated (pinvh — diagnostic, matches production)
cov, cov_diag = safe_hessian_inversion(H_nll, 1.0, regularization=1e-6, method="pinvh")
cov_is_calibrated = not (
    cov_diag.get("rank_deficient", False) or hess_diag.get("is_near_singular", False)
)
se = compute_standard_errors(cov, warn_negative=True)
ci = compute_confidence_intervals(theta_est, se, alpha=0.05)
corr = compute_correlation_matrix(cov, handle_degenerate=True)
max_offdiag_corr = np.nanmax(np.abs(corr - np.eye(n_params)))

print(f"cov_is_calibrated = {cov_is_calibrated}")
print(f"max |off-diag corr| = {max_offdiag_corr:.3f}")
print("95% CIs:")
for i, name in enumerate(param_names):
    if has_ground_truth:
        print(
            f"  {name}: [{ci[i, 0]:+.4f}, {ci[i, 1]:+.4f}]  (true: {theta_true[i]:+.4f})"
        )
    else:
        print(f"  {name}: [{ci[i, 0]:+.4f}, {ci[i, 1]:+.4f}]")


# %% Correlation matrix
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
    y_pred=model(theta_est),
    theta_est=theta_est,
    theta_true=theta_true if has_ground_truth else np.full_like(theta_est, np.nan),
    theta_zero=theta_zero,
    se=se,
    ci=ci,
    cov=cov,
    hess_cond=np.array([float(hess_diag["condition_number"])]),
    cov_is_calibrated=np.array([bool(cov_is_calibrated)]),
    hess_is_near_singular=np.array([bool(hess_diag["is_near_singular"])]),
    converged=np.array([bool(res.success)]),
    tr=np.array([float(spdcm.TR)]),
)
print(f"Wrote artifact to {IMG_DIR / 'run_results.npz'}")

with np.load(IMG_DIR / "run_results.npz") as d:
    print("NPZ keys:", sorted(d.files))
    print(f"  cov_is_calibrated = {bool(d['cov_is_calibrated'][0])}")
    print(f"  converged         = {bool(d['converged'][0])}")
    print(f"  TR                = {float(d['tr'][0])}")

# %%
