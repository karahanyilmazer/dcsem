# %% Imports and config
from dataclasses import dataclass, field
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import pandas as pd
import seaborn as sns
from pypalettes import load_cmap
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from dcsem import SpectralDCM, get_colormap, set_style, to_latex_label
from dcsem.diagnostics import compute_hessian_diagnostics
from dcsem.numerics import (
    compute_confidence_intervals,
    compute_correlation_matrix,
    compute_standard_errors,
    safe_hessian_inversion,
)
from dcsem.spectral import _build_A_matrix
from utils import get_out_dir, get_width_height_latex, log_run

set_style()
width, height = get_width_height_latex()
cmap = get_colormap("YlGnBu_r")
conf_cmap = load_cmap("Revolucion", cmap_type="continuous")
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class RunConfig:
    # SpectralDCM constructor
    n_rois: int = 2
    TR: float = 1.0
    self_connection: float = -1.0
    freq_lo: float = 0.01
    freq_hi: float = 0.1
    n_freqs: int = 32
    # Parameters
    theta_true: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 0.6, np.log(0.05)])
    )
    theta_zero: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, np.log(0.1)])
    )
    param_names: Optional[list] = None
    param_bounds: Optional[list] = None
    # Data
    data_mode: Literal["synthetic_csd", "synthetic_bold", "empirical"] = "synthetic_csd"
    snr: float = 10.0  # used in synthetic_csd mode
    bold_path: Optional[str] = None  # NPZ path for empirical mode; keys: "bold", "TR"
    T_sim: int = 300  # seconds; used in synthetic_bold mode
    nperseg: Optional[int] = None  # Welch segment length (None → T//4)
    noverlap: Optional[int] = None  # Welch overlap (None → nperseg//2)
    # Reproducibility & output
    seed: int = 42
    # Profile likelihood
    profile_params: Optional[list] = None  # None = all params; [] = skip
    # Fixed parameters: {param_index: fixed_value} — held constant during optimization
    fixed_params: Optional[dict] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _load_bold_csv(path: str, time_col: str = "time_s") -> np.ndarray:
    """Load BOLD time series from CSV. Drops time column. Returns (T, R) float32."""
    df = pd.read_csv(path)
    if time_col in df.columns:
        df = df.drop(columns=[time_col])
    return df.to_numpy(dtype=np.float32)


def _resolve_effective_tr(cfg: "RunConfig") -> tuple[float, Optional[np.ndarray]]:
    """Resolve the effective TR for SpectralDCM construction.

    For empirical-mode runs, an NPZ ``"TR"`` key overrides ``cfg.TR`` so the
    Welch sampling rate inside ``observed_csd`` and the HRF FFT scaling inside
    ``predict_csd`` use the data's actual TR. A warning is printed when the
    config and file disagree, so the override is never silent.

    Returns
    -------
    tr_effective : float
        TR to pass to ``SpectralDCM(...)``.
    bold_loaded : np.ndarray or None
        Pre-loaded BOLD time series for empirical mode (avoids loading twice);
        ``None`` for synthetic modes.
    """
    if cfg.data_mode != "empirical":
        return float(cfg.TR), None

    if cfg.bold_path is None:
        raise ValueError("data_mode='empirical' requires cfg.bold_path to be set.")

    bold_path_str = str(cfg.bold_path)
    if bold_path_str.endswith(".csv"):
        # CSV format carries no TR metadata; fall back to cfg.TR.
        bold = _load_bold_csv(bold_path_str)
        return float(cfg.TR), bold

    data = np.load(bold_path_str)
    if "bold" not in data.files:
        raise KeyError(
            f"NPZ at {bold_path_str!r} is missing required key 'bold'; "
            f"available keys: {list(data.files)}"
        )
    bold = data["bold"]
    if "TR" not in data.files:
        return float(cfg.TR), bold

    tr_loaded = float(data["TR"])
    if not np.isclose(tr_loaded, cfg.TR, rtol=1e-6):
        print(
            f"⚠️  Empirical NPZ TR={tr_loaded:.4g} overrides cfg.TR={cfg.TR:.4g} "
            f"(observed_csd and HRF FFT now use TR={tr_loaded:.4g})."
        )
    return tr_loaded, bold


def _resolve_param_spec(
    spdcm: SpectralDCM,
    theta_true: np.ndarray,
    theta_zero: np.ndarray,
    param_names: Optional[list] = None,
    param_bounds: Optional[list] = None,
) -> tuple[list, np.ndarray, np.ndarray, list]:
    """Resolve and validate spectral parameter metadata against the model."""
    default_names = spdcm.get_param_names()
    n_params = len(default_names)

    theta_true = np.asarray(theta_true, dtype=float)
    theta_zero = np.asarray(theta_zero, dtype=float)
    param_names = default_names if param_names is None else list(param_names)
    param_bounds = spdcm.get_bounds() if param_bounds is None else list(param_bounds)

    if theta_true.shape != (n_params,):
        raise ValueError(
            f"theta_true has shape {theta_true.shape}, expected ({n_params},)."
        )
    if theta_zero.shape != (n_params,):
        raise ValueError(
            f"theta_zero has shape {theta_zero.shape}, expected ({n_params},)."
        )
    if len(param_names) != n_params:
        raise ValueError(
            f"param_names has length {len(param_names)}, expected {n_params}."
        )
    if len(param_bounds) != n_params:
        raise ValueError(
            f"param_bounds has length {len(param_bounds)}, expected {n_params}."
        )
    return param_names, theta_true, theta_zero, param_bounds


def estimate_log_sigma_e(spdcm: SpectralDCM, y_obs: np.ndarray) -> float:
    """Rough estimate of log(sigma_e) from observed CSD diagonal power.

    Assumes A ≈ diag(self_connection), giving H_neural ≈ diag(1/(jω - λ)).
    Uses the mean diagonal power across frequencies to back out sigma_e.

    Parameters
    ----------
    spdcm : SpectralDCM
        Fitted SpectralDCM instance (provides HRF spectrum and freqs).
    y_obs : ndarray
        Vectorized observed CSD from spdcm.observed_csd or generate_noisy_csd.

    Returns
    -------
    float
        log(sigma_e) estimate, clamped to [-8, 2].
    """
    S_stack = spdcm._unvectorize_csd(y_obs)  # (n_freqs, R, R)
    lam = spdcm.self_connection  # negative diagonal eigenvalue

    sigma2_estimates = []
    for k, f in enumerate(spdcm.freqs):
        omega = 2 * np.pi * f
        h_neural_mag2 = 1.0 / (omega**2 + lam**2)  # |1/(jω - λ)|²
        h_hrf_mag2 = abs(spdcm.hrf_spectrum[k]) ** 2
        h_tot_mag2 = h_hrf_mag2 * h_neural_mag2
        mean_diag = float(np.mean(S_stack[k].diagonal().real))
        if mean_diag > 0 and h_tot_mag2 > 1e-20:
            sigma2_estimates.append(mean_diag / h_tot_mag2)

    if not sigma2_estimates:
        return np.log(0.1)

    sigma2 = float(np.median(sigma2_estimates))
    log_sigma_e = 0.5 * np.log(max(sigma2, 1e-16))
    return float(np.clip(log_sigma_e, -8.0, 2.0))


def make_objective(model_func, y_obs, x_data, loss_fn, normalize=True):
    """Define objective once; reuse for optimization, Hessian, and landscapes.

    When predict_csd returns np.inf (unstable A), the objective returns 1e10
    to guide the optimizer away from unstable regions.
    """
    if normalize:
        y_mean = y_obs.mean(axis=0, keepdims=True)
        y_std = y_obs.std(axis=0, keepdims=True) + 1e-12
        y_obs_norm = (y_obs - y_mean) / y_std

        def objective(theta):
            y_pred = model_func(theta, x_data)
            if not np.all(np.isfinite(y_pred)):
                return 1e10
            y_pred_norm = (y_pred - y_mean) / y_std
            return loss_fn(y_obs_norm, y_pred_norm)

        return objective, y_mean, y_std, y_obs_norm
    else:

        def objective(theta):
            y_pred = model_func(theta, x_data)
            if not np.all(np.isfinite(y_pred)):
                return 1e10
            return loss_fn(y_obs, y_pred)

        return objective, None, None, None


# =============================================================================
# SINGLE RUN
# =============================================================================


def run_single(cfg: RunConfig = RunConfig()):
    """Single-run spDCM: simulate (or load) → fit → diagnostics → plots."""

    # --- Model identity ---
    model_name = f"spdcm_{cfg.n_rois}roi"
    model_display_name = f"{cfg.n_rois}-ROI Spectral DCM (LS-spDCM)"
    opt_method = "L-BFGS-B"
    title_suffix = "Least Squares"

    # --- Resolve TR before constructing SpectralDCM ---
    # Empirical NPZ files may specify their own TR; the Welch sampling rate
    # in observed_csd and the HRF FFT scaling in predict_csd both depend on
    # spdcm.TR, so we must build the model with the data's actual TR.
    tr_effective, _bold_pre_loaded = _resolve_effective_tr(cfg)

    # --- SpectralDCM instance ---
    spdcm = SpectralDCM(
        n_rois=cfg.n_rois,
        TR=tr_effective,
        self_connection=cfg.self_connection,
        freq_lo=cfg.freq_lo,
        freq_hi=cfg.freq_hi,
        n_freqs=cfg.n_freqs,
    )
    param_names, theta_true, theta_zero, param_bounds = _resolve_param_spec(
        spdcm,
        cfg.theta_true,
        cfg.theta_zero,
        cfg.param_names,
        cfg.param_bounds,
    )
    n_params = len(theta_zero)
    model = lambda theta, _x: spdcm.predict_csd(theta)
    x_data = None

    # Fixed-parameter support: reduce the optimised vector to free params only
    _fixed = cfg.fixed_params or {}
    if any(idx < 0 or idx >= n_params for idx in _fixed):
        raise ValueError("fixed_params contains an out-of-range parameter index.")
    _free_indices = [i for i in range(n_params) if i not in _fixed]
    _theta_zero_full = theta_zero.copy()  # snapshot before any slicing

    def _embed(theta_free):
        """Embed free-param vector back into full theta."""
        theta_full = _theta_zero_full.copy()
        for val_idx, param_idx in enumerate(_free_indices):
            theta_full[param_idx] = theta_free[val_idx]
        for param_idx, val in _fixed.items():
            theta_full[param_idx] = val
        return theta_full

    if _fixed:
        _model_full = model

        def model(theta_free, _x, _mf=_model_full):
            return _mf(_embed(theta_free), _x)

        param_names = [param_names[i] for i in _free_indices]
        theta_true = theta_true[_free_indices]
        theta_zero = theta_zero[_free_indices]
        param_bounds = [param_bounds[i] for i in _free_indices]
        n_params = len(_free_indices)

    def _to_full(theta_free):
        """Return full-length theta regardless of whether params are fixed."""
        return _embed(theta_free) if _fixed else theta_free

    # Resolve profile param indices into the (possibly reduced) free-param space
    if cfg.profile_params is not None:
        if any(idx < 0 or idx >= len(theta_zero) for idx in cfg.profile_params):
            raise ValueError("profile_params contains an out-of-range parameter index.")
        _profile_idxs = [
            _free_indices.index(i) for i in cfg.profile_params if i in _free_indices
        ]
    else:
        _profile_idxs = list(range(n_params))

    IMG_DIR = get_out_dir(
        type="img",
        subfolder="inversion",
        extra_subfolders=[opt_method, model_name],
    )
    LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {model_display_name}")
    print(f"Data mode  : {cfg.data_mode}")
    print(f"Plots will be saved to: {IMG_DIR}")

    PLOT_SPECTRAL = True
    PLOT_BOLD = True

    # =============================================================================
    # DATA LOADING / GENERATION
    # =============================================================================

    rng = np.random.default_rng(cfg.seed)
    bold_ref = None  # (T, R) array if loaded/simulated; None for synthetic_csd
    tvec_ref = None  # matching time axis

    if cfg.data_mode == "synthetic_csd":
        y_obs = spdcm.generate_noisy_csd(cfg.theta_true, snr=cfg.snr, rng=rng)
        has_ground_truth = True

    elif cfg.data_mode == "synthetic_bold":
        # Requires sdeint; mirrors the real-data Welch path exactly
        bold_ref, tvec_ref = spdcm.simulate_bold(cfg.theta_true, T=cfg.T_sim, rng=rng)
        y_obs = spdcm.observed_csd(bold_ref, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
        has_ground_truth = True

    elif cfg.data_mode == "empirical":
        # _resolve_effective_tr already loaded the BOLD and resolved TR;
        # spdcm.TR is now correct, so observed_csd uses the right fs.
        bold_ref = _bold_pre_loaded
        y_obs = spdcm.observed_csd(bold_ref, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
        tvec_ref = np.arange(bold_ref.shape[0]) * spdcm.TR
        has_ground_truth = False

    else:
        raise ValueError(f"Unknown data_mode: {cfg.data_mode!r}")

    if has_ground_truth:
        y_true = model(theta_true, x_data)
        noise_std_actual = float(np.std(y_obs - y_true))
    else:
        y_true = None
        noise_std_actual = np.nan

    print(f"CSD vector length: {len(y_obs)}")
    if has_ground_truth:
        print(f"Noise std (obs - true): {noise_std_actual:.4e}")

    # =============================================================================
    # FIT
    # =============================================================================

    loss_history = []

    obj, y_mean, y_std, y_obs_norm = make_objective(
        model, y_obs, x_data, mean_squared_error, normalize=True
    )

    def callback(theta):
        loss = obj(theta)
        loss_history.append(loss)
        print(f"Iteration {len(loss_history)}: loss = {loss:.6e}")

    theta_zero = np.clip(
        theta_zero,
        [b[0] for b in param_bounds],
        [b[1] for b in param_bounds],
    )

    res = minimize(
        obj,
        theta_zero,
        method=opt_method,
        callback=callback,
        bounds=param_bounds,
    )

    if not res.success:
        print(f"Optimization did not converge: {res.message}")

    theta_est = res.x
    mse_est = obj(theta_est)

    print("\nFit results:")
    if has_ground_truth:
        print(f"  True params: {np.round(theta_true, 4)}")
    print(f"  Estimated  : {np.round(theta_est, 4)}")
    print(f"  Loss: {mse_est:.4f}")

    # =============================================================================
    # STABILITY DIAGNOSTICS
    # =============================================================================

    R = spdcm.n_rois
    n_A = R * (R - 1)
    A_est = _build_A_matrix(theta_est[:n_A], R, spdcm.self_connection)
    eigvals_A = np.linalg.eigvals(A_est)

    print("\nStability diagnostics (estimated A):")
    print(f"  Eigenvalues of A: {np.round(eigvals_A, 4)}")
    if np.any(eigvals_A.real >= 0):
        print("  WARNING: A has non-negative eigenvalue — system may be unstable!")
    else:
        print("  A is stable (all eigenvalues have negative real part).")

    max_cond = 0.0
    for f in spdcm.freqs:
        omega = 2 * np.pi * f
        cond_f = np.linalg.cond(1j * omega * np.eye(R) - A_est)
        if cond_f > max_cond:
            max_cond = cond_f
    print(f"  Max condition number of (jωI − A) across freqs: {max_cond:.2e}")
    if max_cond > 1e6:
        print("  WARNING: High condition number — numerical instability possible!")

    # =============================================================================
    # SPECTRAL PLOTS
    # =============================================================================

    if PLOT_SPECTRAL:
        print("\nGenerating spectral plots...")
        freqs = spdcm.freqs

        S_obs = spdcm._unvectorize_csd(y_obs)
        S_pred = spdcm._unvectorize_csd(model(theta_est, None))
        if has_ground_truth:
            S_true = spdcm._unvectorize_csd(y_true)

        # --- Auto-spectra (log-log) ---
        fig, axes = plt.subplots(1, R, figsize=(width, height / 1.5), sharey=False)
        if R == 1:
            axes = [axes]

        for r in range(R):
            ax = axes[r]
            ax.semilogy(
                freqs,
                S_obs[:, r, r].real,
                color=default_colors[0],
                alpha=0.8,
                label="observed",
                lw=1.5,
            )
            ax.semilogy(
                freqs,
                S_pred[:, r, r].real,
                color=default_colors[2],
                label="fitted",
                lw=2,
            )
            if has_ground_truth:
                ax.semilogy(
                    freqs,
                    S_true[:, r, r].real,
                    color=default_colors[1],
                    linestyle="--",
                    label="true",
                    lw=1.5,
                )
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power (a.u.)")
            ax.set_title(f"ROI {r + 1} Auto-spectrum $S_{{{r + 1}{r + 1}}}(\\omega)$")
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        ncol = 3 if has_ground_truth else 2
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            bbox_to_anchor=(0.5, -0.08),
            frameon=True,
        )
        fig.suptitle(rf"\textbf{{{model_display_name} — Auto-Spectra}}", y=1.02)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "auto_spectra.png", bbox_inches="tight")
        plt.savefig(LATEX_DIR / f"{model_name}_autospectra.pdf", bbox_inches="tight")
        # plt.close("all")

        # --- Cross-spectrum magnitude and coherence ---
        if R >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(width, height / 1.5))

            # |S12(ω)|
            ax = axes[0]
            ax.semilogy(
                freqs,
                np.abs(S_obs[:, 0, 1]),
                color=default_colors[0],
                alpha=0.8,
                label="observed",
                lw=1.5,
            )
            ax.semilogy(
                freqs,
                np.abs(S_pred[:, 0, 1]),
                color=default_colors[2],
                label="fitted",
                lw=2,
            )
            if has_ground_truth:
                ax.semilogy(
                    freqs,
                    np.abs(S_true[:, 0, 1]),
                    color=default_colors[1],
                    linestyle="--",
                    label="true",
                    lw=1.5,
                )
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(r"$|S_{12}(\omega)|$")
            ax.set_title(r"Cross-spectrum $|S_{12}(\omega)|$")
            ax.grid(True, alpha=0.3)

            # Coherence
            ax = axes[1]
            series = [
                (S_obs, "observed", default_colors[0], "-"),
                (S_pred, "fitted", default_colors[2], "-"),
            ]
            if has_ground_truth:
                series.append((S_true, "true", default_colors[1], "--"))
            for S, lbl, col, ls in series:
                denom = S[:, 0, 0].real * S[:, 1, 1].real
                coh = np.abs(S[:, 0, 1]) ** 2 / (denom + 1e-30)
                ax.plot(freqs, coh, color=col, linestyle=ls, label=lbl, lw=1.5)

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Coherence")
            ax.set_title(r"Coherence $|S_{12}|^2 / (S_{11} S_{22})$")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            handles, labels = axes[0].get_legend_handles_labels()
            ncol = 3 if has_ground_truth else 2
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=ncol,
                bbox_to_anchor=(0.5, -0.08),
                frameon=True,
            )
            fig.suptitle(
                rf"\textbf{{{model_display_name} — Cross-Spectrum \& Coherence}}",
                y=1.02,
            )
            plt.tight_layout()
            plt.savefig(IMG_DIR / "cross_spectrum_coherence.png", bbox_inches="tight")
            plt.savefig(
                LATEX_DIR / f"{model_name}_cross_coherence.pdf", bbox_inches="tight"
            )
            # plt.close("all")
            # plt.show()

    # =============================================================================
    # BOLD TIME SERIES
    # =============================================================================

    BOLD_SEED = cfg.seed + 1

    if PLOT_BOLD:
        print("\nGenerating BOLD time series plots...")
        if cfg.data_mode == "synthetic_csd":
            # Simulate with pinned seed so differences reflect only parameter mismatch
            bold_ref_display, tvec_display = spdcm.simulate_bold(
                _to_full(theta_true), T=200, rng=np.random.default_rng(BOLD_SEED)
            )
            bold_est_display, _ = spdcm.simulate_bold(
                _to_full(theta_est), T=200, rng=np.random.default_rng(BOLD_SEED)
            )
            bold_ref_label = "true"
        elif cfg.data_mode == "synthetic_bold":
            # Re-simulate from theta_true with pinned seed for fair comparison
            bold_ref_display, tvec_display = spdcm.simulate_bold(
                _to_full(theta_true), T=cfg.T_sim, rng=np.random.default_rng(BOLD_SEED)
            )
            bold_est_display, _ = spdcm.simulate_bold(
                _to_full(theta_est), T=cfg.T_sim, rng=np.random.default_rng(BOLD_SEED)
            )
            bold_ref_label = "true"
        else:  # empirical
            # Cap display simulation at 120 s to avoid slow sdeint with near-unstable params
            T_display = min(int(bold_ref.shape[0] * spdcm.TR), 120)
            n_display = int(T_display / spdcm.TR)
            bold_ref_display = bold_ref[:n_display]
            tvec_display = tvec_ref[:n_display]
            bold_est_display, _ = spdcm.simulate_bold(
                _to_full(theta_est), T=T_display, rng=np.random.default_rng(BOLD_SEED)
            )
            # Align lengths
            T_min = min(bold_ref_display.shape[0], bold_est_display.shape[0])
            bold_ref_display = bold_ref_display[:T_min]
            bold_est_display = bold_est_display[:T_min]
            tvec_display = tvec_display[:T_min]
            bold_ref_label = "observed"

        fig, axes = plt.subplots(
            1, R, figsize=(width, height / 1.5), sharey=False, sharex=True
        )
        if R == 1:
            axes = [axes]

        for r in range(R):
            ax = axes[r]
            ax.plot(
                tvec_display,
                bold_ref_display[:, r],
                color=default_colors[1],
                linestyle="--",
                label=bold_ref_label,
                lw=1.5,
            )
            ax.plot(
                tvec_display,
                bold_est_display[:, r],
                color=default_colors[2],
                label="fitted",
                lw=1.5,
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("BOLD (a.u.)")
            ax.set_title(f"ROI {r + 1}")
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.08),
            frameon=True,
        )
        fig.suptitle(rf"\textbf{{{model_display_name} — BOLD Time Series}}", y=1.02)
        plt.tight_layout()
        plt.savefig(IMG_DIR / "bold_timeseries.png", bbox_inches="tight")
        plt.savefig(LATEX_DIR / f"{model_name}_bold.pdf", bbox_inches="tight")
        # plt.close("all")

    # =============================================================================
    # HESSIAN-BASED DIAGNOSTICS
    # =============================================================================

    print("\nComputing Hessian-based diagnostics...")
    # Parameter scaling helpers (map param_bounds to [0,1]^n for numerical stability)
    _lowers_h = np.array([b[0] for b in param_bounds])
    _scales_h = np.array([b[1] - b[0] for b in param_bounds])

    def _to_scaled_h(theta):
        return (theta - _lowers_h) / _scales_h

    def _from_scaled_h(s):
        return s * _scales_h + _lowers_h

    HESS_STEP = 1e-3
    theta_s = _to_scaled_h(theta_est)

    # NLL in normalized residual space; sigma2_est absorbed so Cov = H_NLL^{-1} exactly
    r_norm = y_obs_norm - (model(theta_est, x_data) - y_mean) / y_std
    sigma2_est = float(np.sum(r_norm**2) / max(len(r_norm.ravel()) - n_params, 1))

    def _nll_norm(theta, _ym=y_mean, _ys=y_std, _yo=y_obs_norm, _s2=sigma2_est):
        y_pred = model(theta, None)
        if not np.all(np.isfinite(y_pred)):
            return 1e10
        r = (_yo - (y_pred - _ym) / _ys).ravel()
        return 0.5 * float(np.dot(r, r)) / _s2

    def _nll_scaled(s):
        return _nll_norm(_from_scaled_h(s))

    try:
        H_nll_s = nd.Hessian(_nll_scaled, step=HESS_STEP)(theta_s)
        H_nll_s = 0.5 * (H_nll_s + H_nll_s.T)
        H_nll = H_nll_s / np.outer(_scales_h, _scales_h)
    except Exception:
        H_nll = nd.Hessian(obj)(theta_est)
        H_nll = 0.5 * (H_nll + H_nll.T)

    hess_diag = compute_hessian_diagnostics(H_nll)
    cond = hess_diag["condition_number"]
    eigvals_H = np.linalg.eigvalsh(H_nll)

    try:
        # Cov = H_NLL^{-1} via adaptive_ridge (minimum ridge to restore pos-def).
        # The diagnostics dict surfaces ``regularization_warning`` whenever the
        # ridge had to lift indefiniteness beyond the epsilon floor — that case
        # means the reported covariance is regularised, not the asymptotic
        # inverse-Hessian, and CIs are not calibrated.
        cov, cov_diag = safe_hessian_inversion(
            H_nll, 1.0, regularization=1e-6, method="adaptive_ridge"
        )
        cov_is_calibrated = not (
            cov_diag.get("regularization_warning", False)
            or hess_diag.get("is_near_singular", False)
        )
        if not cov_is_calibrated:
            print(
                "⚠️  Covariance is regularised (adaptive_ridge lifted indefinite "
                "Hessian); reported CIs are diagnostic, not asymptotic."
            )
        se = compute_standard_errors(cov, warn_negative=True)
        ci = compute_confidence_intervals(theta_est, se, alpha=0.05)
        corr = compute_correlation_matrix(cov, handle_degenerate=True)
        max_offdiag_corr = np.nanmax(np.abs(corr - np.eye(n_params)))
        rank_deficient = hess_diag["is_near_singular"]

        # Correlation heatmap
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
        ax.set_title(rf"\textbf{{{model_display_name} — Parameter Correlation Matrix}}")
        plt.tight_layout()
        plt.savefig(IMG_DIR / "correlation_matrix.png")
        plt.savefig(LATEX_DIR / f"{model_name}_{title_suffix}_correlation_matrix.pdf")
        # plt.close("all")

    except np.linalg.LinAlgError:
        print("Failed to invert Hessian — matrix is singular!")
        se = np.full(n_params, np.nan)
        ci = np.full((n_params, 2), np.nan)
        max_offdiag_corr = np.nan
        corr = None
        cov = np.full((n_params, n_params), np.nan)
        rank_deficient = True
        cov_is_calibrated = False

    print("\nHessian diagnostics:")
    print(f"  Eigenvalues: {np.round(eigvals_H, 4)}")
    print(f"  Condition number: {cond:.2e}")

    if np.isinf(cond) or cond > 1e6:
        print("  WARNING: High condition number — numerical instability likely!")
    if eigvals_H[0] < 1e-6:
        print(
            f"  WARNING: Near-zero eigenvalue ({eigvals_H[0]:.2e})"
            " — model may be degenerate!"
        )

    print(f"  Estimated noise variance: {sigma2_est:.4e}")
    print(f"  Standard errors: {np.round(se, 4)}")

    if np.isfinite(max_offdiag_corr):
        print(f"  Max. off-diagonal correlation: {max_offdiag_corr:.3f}")
        if max_offdiag_corr > 0.95:
            print("  WARNING: High parameter correlation — identifiability issues!")

    if not rank_deficient and has_ground_truth:
        print("  95% Confidence intervals:")
        for i, name in enumerate(param_names):
            print(
                f"    {name}: [{ci[i, 0]:.4f}, {ci[i, 1]:.4f}]  (True: {theta_true[i]:.4f})"
            )
    elif not rank_deficient:
        print("  95% Confidence intervals:")
        for i, name in enumerate(param_names):
            print(f"    {name}: [{ci[i, 0]:.4f}, {ci[i, 1]:.4f}]")

    # =============================================================================
    # PROFILE LIKELIHOOD
    # =============================================================================

    PLOT_PROFILE = True

    if PLOT_PROFILE:
        print("\nGenerating profile likelihood plots...")
        from dcsem.diagnostics import profile_likelihood_1d

        profile_idxs = _profile_idxs

        if profile_idxs:
            n_profile = len(profile_idxs)
            ncols = min(4, n_profile)
            nrows = (n_profile + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(width, height / 2 * nrows), squeeze=False
            )
            axes_flat = axes.ravel()

            for ax_idx, k in enumerate(profile_idxs):
                lo, hi = param_bounds[k]
                grid = np.linspace(lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo), 30)
                free_bounds = param_bounds[:k] + param_bounds[k + 1 :]
                grid_vals, profile_vals = profile_likelihood_1d(
                    obj, theta_est, k, grid, free_bounds
                )
                ax = axes_flat[ax_idx]
                ax.plot(grid_vals, profile_vals, color=default_colors[2], lw=1.5)
                ax.axvline(
                    theta_est[k],
                    color=default_colors[0],
                    linestyle="--",
                    lw=1,
                    label="est",
                )
                if has_ground_truth:
                    ax.axvline(
                        theta_true[k],
                        color=default_colors[1],
                        linestyle="--",
                        lw=1,
                        label="true",
                    )
                ax.set_xlabel(to_latex_label(param_names[k]))
                ax.set_ylabel("MSE")
                ax.set_title(f"Profile: {to_latex_label(param_names[k])}")
                ax.grid(True, alpha=0.3)

            for ax_idx in range(n_profile, len(axes_flat)):
                axes_flat[ax_idx].set_visible(False)

            handles, labels_leg = axes_flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels_leg,
                    loc="lower center",
                    ncol=len(handles),
                    bbox_to_anchor=(0.5, -0.04),
                    frameon=True,
                )
            fig.suptitle(
                rf"\textbf{{{model_display_name} — Profile Likelihood}}", y=1.02
            )
            plt.tight_layout()
            plt.savefig(IMG_DIR / "profile_likelihood.png", bbox_inches="tight")
            plt.savefig(
                LATEX_DIR / f"{model_name}_profile_likelihood.pdf", bbox_inches="tight"
            )
            # plt.close("all")

    # =============================================================================
    # NPZ ARTIFACT SAVE
    # =============================================================================

    np.savez(
        IMG_DIR / "run_results.npz",
        y_obs=y_obs,
        y_pred=model(theta_est, None),
        theta_est=theta_est,
        theta_true=theta_true if has_ground_truth else np.full_like(theta_est, np.nan),
        theta_zero=theta_zero,
        se=se,
        ci=ci,
        cov=cov,
        hess_cond=np.array([hess_diag.get("condition_number", np.nan)]),
        # Stage-2 additions: downstream analyses can filter on cov_is_calibrated
        # to know whether reported CIs are asymptotic or diagnostic-only.
        cov_is_calibrated=np.array([bool(cov_is_calibrated)]),
        hess_is_near_singular=np.array([bool(hess_diag.get("is_near_singular", False))]),
        converged=np.array([res.success]),
        tr=np.array([float(spdcm.TR)]),
    )
    print(f"\nArtifacts saved to: {IMG_DIR}")

    # =============================================================================
    # LOGGING
    # =============================================================================

    log_run(
        model_name=model_name,
        method=opt_method,
        seed=cfg.seed,
        settings={
            "data_mode": cfg.data_mode,
            "n_freqs": len(spdcm.freqs),
            "csd_dim": len(y_obs),
            "snr": cfg.snr if cfg.data_mode == "synthetic_csd" else None,
            "noise_std_actual": noise_std_actual
            if np.isfinite(noise_std_actual)
            else None,
        },
        params={
            "names": param_names,
            "true": theta_true.tolist() if has_ground_truth else None,
            "init": theta_zero.tolist(),
            "est": theta_est.tolist(),
            "se": se.tolist() if not rank_deficient else [float("nan")] * n_params,
            "corr_max": float(max_offdiag_corr)
            if np.isfinite(max_offdiag_corr)
            else None,
        },
        hessian={
            "cond": float(cond) if np.isfinite(cond) else None,
            "eigvals": eigvals_H.tolist(),
        },
        performance={"mse": float(mse_est)},
        correlation=corr.tolist() if corr is not None else None,
        overwrite=False,
    )


# %% Run
if __name__ == "__main__":
    _BOLD_PATH = "data/sub-karahan_run-01_DMN_timeseries.csv"
    _TR = 0.8
    _N_ROIS = 4
    # Estimate log_sigma_e from data before building the RunConfig
    _spdcm_init = SpectralDCM(n_rois=_N_ROIS, TR=_TR)
    _bold_init = _load_bold_csv(_BOLD_PATH)
    _y_obs_init = _spdcm_init.observed_csd(_bold_init)
    _log_sigma_e_init = estimate_log_sigma_e(_spdcm_init, _y_obs_init)
    print(
        f"Estimated log_sigma_e from data: {_log_sigma_e_init:.3f}  (sigma_e ≈ {np.exp(_log_sigma_e_init):.4f})"
    )

    run_single(
        RunConfig(
            n_rois=_N_ROIS,
            TR=_TR,
            data_mode="empirical",
            bold_path=_BOLD_PATH,
            theta_true=np.concatenate(
                [np.zeros(_N_ROIS * (_N_ROIS - 1)), [_log_sigma_e_init]]
            ),
            theta_zero=np.concatenate(
                [np.zeros(_N_ROIS * (_N_ROIS - 1)), [_log_sigma_e_init]]
            ),
            param_bounds=_spdcm_init.get_bounds()[:-1]
            + [(_log_sigma_e_init - 3.0, _log_sigma_e_init + 3.0)],
            profile_params=list(range(_N_ROIS * (_N_ROIS - 1))),
        )
    )

# %%
