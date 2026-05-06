# %% Imports and config
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy import optimize
from sklearn.metrics import mean_squared_error

from dcsem import SpectralDCM, get_colormap, set_style, to_latex_label
from dcsem.utils import is_chain_converged
from pipelines.spdcm_generic import _resolve_effective_tr, estimate_log_sigma_e
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
    priors: Optional[list] = None
    # Data
    data_mode: Literal["synthetic_csd", "synthetic_bold", "empirical"] = "synthetic_csd"
    snr: float = 10.0  # used in synthetic_csd mode
    bold_path: Optional[str] = None  # NPZ path for empirical mode; keys: "bold", "TR"
    T_sim: int = 300  # seconds; used in synthetic_bold mode
    nperseg: Optional[int] = None  # Welch segment length (None → T//4)
    noverlap: Optional[int] = None  # Welch overlap (None → nperseg//2)
    # MCMC settings
    n_walkers: Optional[int] = None  # None → max(24, 2*n_params)
    n_burn: int = 5000
    n_samples_mcmc: int = 10000
    # Reproducibility & output
    seed: int = 42


# =============================================================================
# SINGLE MCMC RUN
# =============================================================================


def _resolve_param_spec(
    spdcm: SpectralDCM,
    theta_true: np.ndarray,
    theta_zero: np.ndarray,
    param_names: Optional[list] = None,
    param_bounds: Optional[list] = None,
    priors: Optional[list] = None,
) -> tuple[list, np.ndarray, np.ndarray, list, list]:
    """Resolve and validate spectral parameter metadata for MCMC workflows."""
    default_names = spdcm.get_param_names()
    n_params = len(default_names)

    theta_true = np.asarray(theta_true, dtype=float)
    theta_zero = np.asarray(theta_zero, dtype=float)
    param_names = default_names if param_names is None else list(param_names)
    param_bounds = spdcm.get_bounds() if param_bounds is None else list(param_bounds)

    if priors is None:
        priors = []
        for low, high in param_bounds[:-1]:
            sigma = max((high - low) / 4.0, 0.25)
            priors.append((0.0, sigma))
        priors.append((float(theta_zero[-1]), 2.0))
    else:
        priors = list(priors)

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
    if len(priors) != n_params:
        raise ValueError(f"priors has length {len(priors)}, expected {n_params}.")
    return param_names, theta_true, theta_zero, param_bounds, priors


def run_single_mcmc(cfg: RunConfig = RunConfig()):
    """Single-run spDCM MCMC: simulate (or load) → sample → diagnostics → plots."""

    # --- Model identity ---
    model_name = f"spdcm_{cfg.n_rois}roi"
    model_display_name = f"{cfg.n_rois}-ROI Spectral DCM (LS-spDCM)"
    opt_method = "MCMC"

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
    param_names, theta_true, theta_zero, param_bounds, priors = _resolve_param_spec(
        spdcm,
        cfg.theta_true,
        cfg.theta_zero,
        cfg.param_names,
        cfg.param_bounds,
        cfg.priors,
    )
    n_params = len(theta_true)
    n_walkers = cfg.n_walkers if cfg.n_walkers is not None else max(24, 2 * n_params)
    n_burn = cfg.n_burn
    n_samples_mcmc = cfg.n_samples_mcmc
    model = lambda theta, _x: spdcm.predict_csd(theta)
    x_data = None

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

    PLOT_CORNER = True
    PLOT_POSTERIOR_BANDS = True
    PLOT_BOLD_BANDS = True

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

    # Use noise std as the likelihood sigma
    noise_sigma = (
        noise_std_actual
        if np.isfinite(noise_std_actual) and noise_std_actual > 1e-12
        else 1e-3
    )

    print(f"CSD vector length: {len(y_obs)}")
    if has_ground_truth:
        print(f"Noise std (obs - true): {noise_std_actual:.4e}")
    print(f"Likelihood sigma: {noise_sigma:.4e}")

    # Data-driven log_sigma_e prior: replace the default theta_zero-centred
    # prior with one centred on a moment estimator of log(sigma_e) from the
    # observed CSD diagonal. This avoids biasing the posterior toward a poor
    # initial guess and matches the L-BFGS-B script's data-driven approach.
    # Only applies when the user did not supply explicit priors.
    if cfg.priors is None and "log_sigma_e" in param_names:
        log_sigma_e_data = estimate_log_sigma_e(spdcm, y_obs)
        sigma_idx = param_names.index("log_sigma_e")
        priors[sigma_idx] = (float(log_sigma_e_data), 2.0)
        print(
            f"log_sigma_e prior: data-driven, centred at "
            f"{log_sigma_e_data:.3f} (was theta_zero[-1])."
        )

    # =============================================================================
    # MCMC HELPER FUNCTIONS
    # =============================================================================

    def in_bounds(theta):
        for val, (low, high) in zip(theta, param_bounds):
            if val < low or val > high:
                return False
        return True

    def log_prior(theta):
        """Independent Normal priors specified in priors list."""
        if not in_bounds(theta):
            return -np.inf
        logp = 0.0
        for i, (mu, sigma) in enumerate(priors):
            z = (theta[i] - mu) / sigma
            logp += -0.5 * (z * z + np.log(2.0 * np.pi * sigma * sigma))
        return logp

    def log_likelihood(theta, x, y, sigma):
        """Gaussian likelihood on flattened CSD vector.

        NOTE: Uses a Gaussian likelihood on the real-valued CSD vector.
        TODO: Can be upgraded to a Wishart spectral likelihood for statistically
              principled inference.

        Returns -inf if the A matrix is unstable (predict_csd returns np.inf).
        """
        if sigma <= 0 or not np.isfinite(sigma):
            return -np.inf
        try:
            y_pred = model(theta, x)
            if not np.all(np.isfinite(y_pred)):  # unstable A
                return -np.inf
            r = (y - y_pred) / sigma
            return -0.5 * (np.sum(r * r) + r.size * np.log(2.0 * np.pi * sigma**2))
        except Exception:
            return -np.inf

    def log_posterior(theta, x, y, sigma):
        """Unnormalized log posterior."""
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta, x, y, sigma)
        return lp + ll

    def map_estimate(theta0, x, y, sigma):
        """Find MAP estimate via L-BFGS-B optimization."""

        def neg_logpost(th):
            return -log_posterior(th, x, y, sigma)

        theta0 = np.clip(theta0, [b[0] for b in param_bounds], [b[1] for b in param_bounds])
        res = optimize.minimize(
            neg_logpost, theta0, method="L-BFGS-B", bounds=param_bounds
        )
        return res.x

    # =============================================================================
    # MCMC SAMPLING
    # =============================================================================

    print("\nFinding MAP estimate for walker initialization...")
    theta_est = map_estimate(theta_zero, x_data, y_obs, noise_sigma)
    print(f"  MAP estimate: {np.round(theta_est, 4)}")

    scale = np.maximum(0.05 * np.ones(n_params), 0.05 * np.abs(theta_est))
    p0 = theta_est + rng.normal(0.0, scale, size=(n_walkers, n_params))
    p0 = np.clip(
        p0,
        [b[0] for b in param_bounds],
        [b[1] for b in param_bounds],
    )

    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, log_posterior, args=(x_data, y_obs, noise_sigma)
    )

    print(f"\nRunning MCMC: {n_burn} burn-in + {n_samples_mcmc} production samples...")
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
    if has_ground_truth:
        print(f"  True params: {np.round(theta_true, 4)}")
    print(f"  Mean       : {np.round(theta_mean, 4)}")
    print(f"  Median     : {np.round(theta_median, 4)}")
    print(f"  MAP        : {np.round(theta_est_post, 4)}")
    print("  95% Credible intervals:")
    for i, name in enumerate(param_names):
        if has_ground_truth:
            print(
                f"    {name}: [{q025[i]:.4f}, {q975[i]:.4f}]  (True: {theta_true[i]:.4f})"
            )
        else:
            print(f"    {name}: [{q025[i]:.4f}, {q975[i]:.4f}]")

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
        print("  WARNING: Low acceptance (<0.05) — walkers may be stuck!")
    elif acc_frac > 0.8:
        print("  WARNING: High acceptance (>0.8) — proposal may be too narrow!")

    # Real convergence flag: acceptance in healthy band AND enough effective samples.
    # Asymptotic-Gaussian credible intervals are reliable only when the chain has
    # actually converged, so cov_is_calibrated tracks the same criterion.  The
    # threshold logic lives in dcsem.utils.is_chain_converged so a future tweak
    # changes one place; acc_ok/ess_ok stay inline for the diagnostic message.
    acc_ok = 0.15 <= acc_frac <= 0.80
    ess_ok = bool(np.isfinite(eff_total)) and eff_total > 50 * n_params
    converged = is_chain_converged(acc_frac, eff_total, n_params)
    cov_is_calibrated = converged
    if not converged:
        reasons = []
        if not acc_ok:
            reasons.append(f"acceptance={acc_frac:.3f} outside [0.15, 0.80]")
        if not ess_ok:
            reasons.append(
                f"ESS={'n/a' if not np.isfinite(eff_total) else int(eff_total)} "
                f"≤ 50 × n_params={50 * n_params}"
            )
        print(
            "⚠️  MCMC chain not converged ({}); credible intervals are not "
            "calibrated.".format("; ".join(reasons))
        )

    # =============================================================================
    # PLOT: POSTERIOR PREDICTIVE SPECTRAL BANDS
    # =============================================================================

    if PLOT_POSTERIOR_BANDS:
        nsamp = min(400, chain.shape[0])
        idx = rng.choice(chain.shape[0], size=nsamp, replace=False)
        thetas_sample = chain[idx]

        freqs = spdcm.freqs
        R = spdcm.n_rois

        S_samples = []
        for th in thetas_sample:
            y_pred = model(th, None)
            if np.all(np.isfinite(y_pred)):
                S_samples.append(spdcm._unvectorize_csd(y_pred))
        S_samples = np.array(S_samples)  # (nsamp_valid, n_freqs, R, R)

        S_obs = spdcm._unvectorize_csd(y_obs)
        S_mean = spdcm._unvectorize_csd(model(theta_mean, None))
        if has_ground_truth:
            S_true = spdcm._unvectorize_csd(y_true)

        # --- Auto-spectra with 95% posterior bands ---
        fig, axes = plt.subplots(1, R, figsize=(width, height / 1.5))
        if R == 1:
            axes = [axes]

        for r in range(R):
            ax = axes[r]
            psd_samples = S_samples[:, :, r, r].real
            lo = np.percentile(psd_samples, 2.5, axis=0)
            hi = np.percentile(psd_samples, 97.5, axis=0)

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
                S_mean[:, r, r].real,
                color=default_colors[2],
                label="posterior mean",
                lw=2,
            )
            ax.fill_between(
                freqs,
                np.maximum(lo, 1e-20),
                hi,
                color=default_colors[2],
                alpha=0.2,
                label="95% posterior band",
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
        ncol = 4 if has_ground_truth else 3
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            bbox_to_anchor=(0.5, -0.10),
            frameon=True,
        )
        fig.suptitle(
            rf"\textbf{{{model_display_name} — Posterior Predictive Auto-Spectra ({opt_method})}}",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(IMG_DIR / "posterior_auto_spectra.png", bbox_inches="tight")
        plt.savefig(
            LATEX_DIR / f"{model_name}_{opt_method}_posterior_auto_spectra.pdf",
            bbox_inches="tight",
        )
        plt.show()

        # --- Cross-spectrum magnitude with 95% bands ---
        if R >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(width, height / 1.5))

            ax = axes[0]
            cross_samples = np.abs(S_samples[:, :, 0, 1])
            lo12 = np.percentile(cross_samples, 2.5, axis=0)
            hi12 = np.percentile(cross_samples, 97.5, axis=0)

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
                np.abs(S_mean[:, 0, 1]),
                color=default_colors[2],
                label="posterior mean",
                lw=2,
            )
            ax.fill_between(
                freqs,
                np.maximum(lo12, 1e-20),
                hi12,
                color=default_colors[2],
                alpha=0.2,
                label="95% posterior band",
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
                (S_mean, "posterior mean", default_colors[2], "-"),
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
            ncol = 4 if has_ground_truth else 3
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=ncol,
                bbox_to_anchor=(0.5, -0.10),
                frameon=True,
            )
            fig.suptitle(
                rf"\textbf{{{model_display_name} — Cross-Spectrum \& Coherence ({opt_method})}}",
                y=1.02,
            )
            plt.tight_layout()
            plt.savefig(IMG_DIR / "posterior_cross_coherence.png", bbox_inches="tight")
            plt.savefig(
                LATEX_DIR / f"{model_name}_{opt_method}_cross_coherence.pdf",
                bbox_inches="tight",
            )
            plt.show()

    # =============================================================================
    # BOLD POSTERIOR PREDICTIVE BANDS
    # =============================================================================

    BOLD_SEED = cfg.seed + 1
    R = spdcm.n_rois

    if PLOT_BOLD_BANDS:
        # Reference BOLD: ground-truth simulation or empirical BOLD
        if has_ground_truth:
            bold_gt, tvec_bold = spdcm.simulate_bold(
                theta_true, T=200, rng=np.random.default_rng(BOLD_SEED)
            )
            bold_true = bold_gt
            bold_ref_label = "true"
        else:
            bold_true = bold_ref
            tvec_bold = tvec_ref
            bold_ref_label = "observed"

        T_bold = bold_true.shape[0]
        T_sim_bold = int(T_bold * spdcm.TR)

        nsamp = min(50, chain.shape[0])
        idx = rng.choice(chain.shape[0], size=nsamp, replace=False)
        bolds = []
        for i, th in enumerate(chain[idx]):
            b, _ = spdcm.simulate_bold(
                th, T=T_sim_bold, rng=np.random.default_rng(BOLD_SEED + 10 + i)
            )
            T_min_b = min(b.shape[0], T_bold)
            bolds.append(b[:T_min_b])
        bolds = np.array(bolds)  # (nsamp, T_min, R)

        bold_mean, _ = spdcm.simulate_bold(
            theta_mean, T=T_sim_bold, rng=np.random.default_rng(BOLD_SEED)
        )
        T_min = min(bold_mean.shape[0], T_bold, bolds.shape[1])
        bold_mean = bold_mean[:T_min]
        bold_true = bold_true[:T_min]
        tvec_bold = tvec_bold[:T_min]
        bolds = bolds[:, :T_min, :]

        fig, axes = plt.subplots(
            1, R, figsize=(width, height / 1.5), sharey=False, sharex=True
        )
        if R == 1:
            axes = [axes]

        for r in range(R):
            ax = axes[r]
            lo = np.percentile(bolds[:, :, r], 2.5, axis=0)
            hi = np.percentile(bolds[:, :, r], 97.5, axis=0)
            ax.fill_between(
                tvec_bold,
                lo,
                hi,
                color=default_colors[2],
                alpha=0.2,
                label="95% posterior band",
            )
            ax.plot(
                tvec_bold,
                bold_mean[:, r],
                color=default_colors[2],
                label="posterior mean",
                lw=1.5,
            )
            ax.plot(
                tvec_bold,
                bold_true[:, r],
                color=default_colors[1],
                linestyle="--",
                label=bold_ref_label,
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
            ncol=3,
            bbox_to_anchor=(0.5, -0.08),
            frameon=True,
        )
        fig.suptitle(
            rf"\textbf{{{model_display_name} — Posterior Predictive BOLD ({opt_method})}}",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(IMG_DIR / "posterior_bold.png", bbox_inches="tight")
        plt.savefig(
            LATEX_DIR / f"{model_name}_{opt_method}_posterior_bold.pdf",
            bbox_inches="tight",
        )
        plt.show()

    # =============================================================================
    # CORNER PLOT
    # =============================================================================

    if PLOT_CORNER:
        latex_labels = [to_latex_label(name) for name in param_names]
        fig = plt.figure(figsize=(width, width))
        fig = corner.corner(
            chain,
            labels=latex_labels,
            truths=theta_true if has_ground_truth else None,
            show_titles=True,
            title_fmt=".3f",
            quantiles=[0.16, 0.5, 0.84],
            bins=50,
            smooth=0.8,
            truth_color=default_colors[1],
            fig=fig,
        )
        fig.suptitle(
            rf"\textbf{{{model_display_name} — Posterior Distributions ({opt_method})}}"
        )
        plt.tight_layout()
        plt.savefig(IMG_DIR / "corner_plot.png")
        plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_corner_plot.pdf")
        plt.show()

    # =============================================================================
    # CORRELATION PLOT
    # =============================================================================

    cov_post = np.cov(chain, rowvar=False)
    diag_cov = np.diag(cov_post)
    if np.any(diag_cov < 0):
        print("WARNING: Negative variance in posterior covariance!")
        diag_cov = np.clip(diag_cov, 0, None)

    se_post = np.sqrt(diag_cov)
    denom = np.outer(se_post, se_post)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, cov_post / denom, 0)
    max_offdiag_corr = np.nanmax(np.abs(corr - np.eye(n_params)))

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
    ax.tick_params(which="both", left=False, bottom=False)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(which="both", size=0)
    ax.set_title(
        rf"\textbf{{{model_display_name} — Parameter Correlation Matrix ({opt_method})}}"
    )
    plt.tight_layout()
    plt.savefig(IMG_DIR / "correlation_matrix.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_correlation_matrix.pdf")
    plt.show()

    # =============================================================================
    # TRACE PLOTS
    # =============================================================================

    samples = sampler.get_chain()  # (n_steps, n_walkers, n_params)

    fig, axes = plt.subplots(n_params, figsize=(width, 2 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i in range(n_params):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3, linewidth=0.5)
        if has_ground_truth:
            ax.axhline(
                theta_true[i], color=default_colors[1], linestyle="--", label="true"
            )
        ax.axhline(theta_mean[i], color=default_colors[2], linestyle="-", label="mean")
        ax.set_ylabel(to_latex_label(param_names[i]))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Step number")
    fig.suptitle(rf"\textbf{{MCMC Trace Plots — {model_display_name}}}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "trace_plots.png")
    plt.savefig(LATEX_DIR / f"{model_name}_{opt_method}_trace_plots.pdf")
    plt.show()

    # =============================================================================
    # NPZ ARTIFACT SAVE
    # =============================================================================

    np.savez(
        IMG_DIR / "run_results.npz",
        y_obs=y_obs,
        y_pred=model(theta_mean, None),
        theta_mean=theta_mean,
        theta_median=theta_median,
        theta_map=theta_est_post,
        theta_true=cfg.theta_true
        if has_ground_truth
        else np.full_like(theta_mean, np.nan),
        theta_zero=theta_zero,
        se=se_post,
        ci=np.stack([q025, q975], axis=1),
        cov=cov_post,
        converged=np.array([converged]),
        cov_is_calibrated=np.array([bool(cov_is_calibrated)]),
        acceptance_fraction=np.array([acc_frac]),
        ess_total=np.array([float(eff_total)]),
        tr=np.array([float(spdcm.TR)]),
    )
    print(f"\nArtifacts saved to: {IMG_DIR}")

    # =============================================================================
    # LOGGING
    # =============================================================================

    theta_std = np.std(chain, axis=0)

    log_run(
        model_name=model_name,
        method=opt_method,
        seed=cfg.seed,
        settings={
            "data_mode": cfg.data_mode,
            "n_freqs": len(spdcm.freqs),
            "csd_dim": len(y_obs),
            "snr": cfg.snr if cfg.data_mode == "synthetic_csd" else None,
            "noise_sigma": noise_sigma,
            "n_walkers": n_walkers,
            "n_burn": n_burn,
            "n_samples_mcmc": n_samples_mcmc,
        },
        params={
            "names": param_names,
            "true": theta_true.tolist() if has_ground_truth else None,
            "init": theta_zero.tolist(),
            "mean": theta_mean.tolist(),
            "median": theta_median.tolist(),
            "map": theta_est_post.tolist(),
            "std": theta_std.tolist(),
            "ci_lower": q025.tolist(),
            "ci_upper": q975.tolist(),
            "corr_max": float(max_offdiag_corr)
            if np.isfinite(max_offdiag_corr)
            else None,
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


# %% Run
if __name__ == "__main__":
    run_single_mcmc()

# %%
