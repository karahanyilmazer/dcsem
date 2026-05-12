"""Forward-simulation demo — single-layer Spectral DCM, 2 ROIs.

Cell-by-cell script. Open in VS Code's Interactive Window or Jupyter
and step through the ``# %%`` blocks. Same bidirectional 2-ROI
connectivity as ``dcm_1layer_2roi.py``, but the system is driven by
internal neuronal noise (no exogenous stimulus) and the forward model
returns the cross-spectral density (CSD) instead of a BOLD time course.

What you should see (LS-spDCM forward model)
--------------------------------------------
The CSD has the closed form
    S(ω) = σ_e² · H_hrf(ω) · (jωI − A)⁻¹ · (jωI − A)⁻ᴴ · H_hrf(ω)*
i.e. white innovations passed through a multi-ROI neural filter and a
shared HRF.  Concretely:

(a) ``|S_ii(f)|`` is a Lorentzian × HRF² — monotonically decreasing
    in 0.01–0.1 Hz with **no peaks**, because A's eigenvalues are real
    here.  Peaks appear only when A has complex eigenvalues with
    non-trivial imaginary part — see the resonant-case demo cell at
    the bottom of this file.
(b) ``|S_ij(f)| ≤ √(S_ii · S_jj)`` (Cauchy–Schwarz).
(c) Magnitude-squared coherence γ²(f) ∈ [0, 1].  Each ROI has an
    independent white-noise drive, so coherence reflects only what
    couples through A; for moderate coupling and σ_e = 0.1 the values
    here are O(0.1), not near unity.
(d) ``∠ S_ij(f)`` is non-zero (asymmetric A → frequency-dependent lag).
(e) Empirical Welch CSD from ``simulate_bold`` should track the
    analytical ``predict_csd`` to within ~1 dex; a constant-factor
    offset would point at a normalisation bug in the HRF kernel.
"""

# %% Imports + style + output dirs
import matplotlib.pyplot as plt
import numpy as np

from dcsem import SpectralDCM, plot_dcm_graph
from dcsem.models import DCM
from utils import get_out_dir, set_style

set_style()
IMG_DIR = get_out_dir(type="img", subfolder="spdcm")
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
FIG_NAME = "spdcm_1layer_2roi"


# %% Build SpectralDCM
spdcm = SpectralDCM(
    n_rois=2,
    TR=1.0,
    self_connection=-1.0,
    freq_lo=0.01,
    freq_hi=0.1,
    n_freqs=32,
)
print("param order:", spdcm.get_param_names())
# expected: ['a01', 'a10', 'log_sigma_e']


# %% Ground-truth parameters
# Topology mirrors dcm_1layer_2roi.py:
#   R0 → R1 (a01 = 0.8), R1 → R0 (a10 = 0.4), self = -1.
# log_sigma_e = log(0.1) → moderate neuronal-noise std.
theta_true = np.array(
    [
        0.8,  # a01: R0 → R1
        0.4,  # a10: R1 → R0
        np.log(0.1),  # log_sigma_e
    ]
)

# Reconstruct A by hand (matches what SpectralDCM does internally) and
# print the max real-part eigenvalue as a stability check.
A = np.array(
    [
        [-1.0, 0.4],
        [0.8, -1.0],
    ]
)
print("A:\n", A)
eigvals = np.linalg.eigvals(A)
print(f"max Re(eig(A)) = {eigvals.real.max():+.4f}  (must be < 0 for stability)")


# %% Diagram of the spDCM connectivity (no input arrows — spDCM is noise-driven)
# Build a dummy DCM with C = 0 purely so plot_dcm_graph can render the
# same connectivity layout used by the DCM demos.
num_rois = spdcm.n_rois
dcm_for_graph = DCM(num_rois, params={"A": A, "C": np.zeros(num_rois)})

fig, _ = plot_dcm_graph(
    dcm_for_graph,
    show_self_connections=True,
    show_inputs=False,
    threshold=1e-12,
    figsize=None,
    node_color=None,
    node_radius=0.30,
    fontsize=14,
)
for ext in ("svg", "png", "pdf"):
    target = (LATEX_DIR if ext == "pdf" else IMG_DIR) / f"{FIG_NAME}_graph.{ext}"
    fig.savefig(target, bbox_inches="tight")
plt.show(block=False)


# %% Forward predict CSD
y_vec = spdcm.predict_csd(theta_true)
assert np.all(np.isfinite(y_vec)), "predict_csd returned non-finite (unstable A?)"
S = spdcm._unvectorize_csd(y_vec)  # complex, shape (n_freqs, R, R)
print(f"CSD shape: {S.shape}")


# %% Plot CSD: auto-PSD per ROI + cross-PSD magnitude
fig, axs = plt.subplots(1, 2, figsize=(9, 4))

freqs = spdcm.freqs
for r in range(num_rois):
    axs[0].plot(freqs, S[:, r, r].real, label=f"ROI {r}")
axs[0].set_yscale("log")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel(r"$|S_{ii}(f)|$")
axs[0].set_title("Auto-PSD")
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(freqs, np.abs(S[:, 0, 1]), color="k")
axs[1].set_yscale("log")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel(r"$|S_{01}(f)|$")
axs[1].set_title("Cross-PSD")
axs[1].grid(alpha=0.3)

fig.suptitle(r"\textbf{Spectral DCM CSD — 2-ROI}")
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}.pdf")
plt.show(block=False)


# %% Eigenvalues of A — explains the smooth, peakless CSD shape
# Real eigenvalues  ⇒  each pole is a Lorentzian (monotonic, no peak).
# Complex eigenvalues with |Im(λ)| > 0  ⇒  resonance at f ≈ |Im(λ)|/(2π).
eigvals_A = np.linalg.eigvals(A)
print("eig(A):", eigvals_A)
print(
    f"max |Im(eig)| = {np.max(np.abs(eigvals_A.imag)):.4f}   "
    "(=0 ⇒ no resonance peak expected)"
)

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(eigvals_A.real, eigvals_A.imag, s=80, color="C0", zorder=3)
ax.axvline(0, color="k", lw=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
ax.set_title(r"Eigenvalues of $A$ (stable iff $\mathrm{Re}<0$)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}_eigs.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}_eigs.pdf")
plt.show(block=False)


# %% HRF transfer function — the haemodynamic kernel as a low-pass filter
# spdcm.hrf_spectrum is the *neural-state-to-BOLD* kernel after stripping
# the implicit 1-ROI neural filter (see spectral.py:_compute_hrf_spectrum).
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
axs[0].plot(spdcm.freqs, np.abs(spdcm.hrf_spectrum), color="C2")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel(r"$|H_{\mathrm{hrf}}(f)|$")
axs[0].set_title("HRF magnitude")
axs[0].grid(alpha=0.3)

axs[1].plot(spdcm.freqs, np.unwrap(np.angle(spdcm.hrf_spectrum)), color="C2")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel(r"$\angle H_{\mathrm{hrf}}(f)$ [rad]")
axs[1].set_title("HRF phase (unwrapped)")
axs[1].grid(alpha=0.3)

fig.suptitle("Haemodynamic kernel")
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}_hrf.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}_hrf.pdf")
plt.show(block=False)


# %% Time-domain BOLD realisation
# `simulate_bold` runs the *full nonlinear* Balloon model with stochastic
# neural drive (sdeint).  predict_csd is the linearised, ensemble-mean CSD;
# a single realisation is one draw from that ensemble.
T_sim = 600  # seconds
rng = np.random.default_rng(0)
bold, tvec = spdcm.simulate_bold(theta_true, T=T_sim, rng=rng)
print(f"bold shape: {bold.shape},  std per ROI: {bold.std(axis=0)}")

fig, ax = plt.subplots(figsize=(9, 3))
for r in range(num_rois):
    ax.plot(tvec, bold[:, r], lw=0.8, label=f"ROI {r}", alpha=0.85)
ax.set_xlabel("Time (s)")
ax.set_ylabel("BOLD")
ax.set_title(rf"Simulated BOLD ($T={T_sim}$ s, noise-driven)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}_bold.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}_bold.pdf")
plt.show(block=False)


# %% Empirical (Welch) vs analytical CSD — headline self-consistency check
# observed_csd Welch-estimates the CSD from `bold`. As T → ∞ the empirical
# curve should track the analytical predict_csd. A persistent constant-
# factor offset would suggest a normalisation bug in _compute_hrf_spectrum.
y_emp = spdcm.observed_csd(bold)
S_emp = spdcm._unvectorize_csd(y_emp)

fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for r in range(num_rois):
    axs[0].plot(freqs, S[:, r, r].real, lw=2, color=f"C{r}",
                label=f"analytical ROI {r}")
    axs[0].plot(freqs, S_emp[:, r, r].real, "o", ms=4, color=f"C{r}",
                alpha=0.55, label=f"empirical ROI {r}")
axs[0].set_yscale("log")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel(r"$|S_{ii}(f)|$")
axs[0].set_title("Auto-PSD: predict\\_csd vs Welch")
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)

axs[1].plot(freqs, np.abs(S[:, 0, 1]), color="k", lw=2, label="analytical")
axs[1].plot(freqs, np.abs(S_emp[:, 0, 1]), "o", ms=4, color="C3",
            alpha=0.6, label="empirical")
axs[1].set_yscale("log")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel(r"$|S_{01}(f)|$")
axs[1].set_title("Cross-PSD: predict\\_csd vs Welch")
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)

fig.suptitle("Forward-model self-consistency (linearised vs nonlinear sim)")
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}_csd_overlay.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}_csd_overlay.pdf")
plt.show(block=False)


# %% Cross-spectrum phase + magnitude-squared coherence
# Phase of S_ij encodes lead/lag: asymmetric A (a01 ≠ a10) → non-zero phase.
# Coherence γ²(ω) = |S_ij|² / (S_ii · S_jj) ∈ [0, 1] is the frequency-domain
# analogue of correlation.  Expect γ² ≈ 1 at low f, decreasing toward Nyquist.
S_ii = S[:, 0, 0].real
S_jj = S[:, 1, 1].real
S_ij = S[:, 0, 1]
coh_sq = (np.abs(S_ij) ** 2) / (S_ii * S_jj)
phase = np.unwrap(np.angle(S_ij))

fig, axs = plt.subplots(1, 2, figsize=(9, 4))
axs[0].plot(freqs, phase, color="C3")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel(r"$\angle S_{01}(f)$ [rad]")
axs[0].set_title("Cross-spectrum phase")
axs[0].grid(alpha=0.3)

axs[1].plot(freqs, coh_sq, color="C4")
axs[1].set_ylim(-0.05, 1.05)
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel(r"$\gamma^2_{01}(f)$")
axs[1].set_title("Magnitude-squared coherence")
axs[1].grid(alpha=0.3)

fig.suptitle("Directionality + shared-variance diagnostics")
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}_phase_coh.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}_phase_coh.pdf")
plt.show(block=False)


# %% Resonant-case sanity demo — peaks emerge for complex eigenvalues
# Antisymmetric coupling gives complex eigenvalues at -1 ± 1.5j with
# resonance frequency  ω_r = √(b² − a²)  where A's eigs are −a ± jb.
# Two takeaways:
#   1. The **neural** CSD (before HRF) has an obvious peak.
#   2. The **BOLD** CSD (after HRF) often masks that peak because the HRF
#      is strongly low-pass — so a peakless BOLD spectrum does NOT rule
#      out neural resonance.  This is a real limitation of fMRI, not a
#      bug in the forward model.
A_res = np.array([[-1.0, 1.5], [-1.5, -1.0]])
eigs_res = np.linalg.eigvals(A_res)
a_re, b_im = -eigs_res[0].real, abs(eigs_res[0].imag)
f_neural_peak = np.sqrt(max(b_im**2 - a_re**2, 0.0)) / (2 * np.pi)
print(
    f"resonant eig(A): {eigs_res}   "
    f"(neural peak at f ≈ {f_neural_peak:.3f} Hz)"
)

theta_res = np.array([1.5, -1.5, np.log(0.1)])  # a01, a10, log_sigma_e
freqs_res = np.linspace(0.005, 0.4, 256)
omega_res = 2 * np.pi * freqs_res
sigma_e_res = np.exp(theta_res[2])

# Neural CSD: σ_e² · (jωI − A)⁻¹ · (jωI − A)⁻ᴴ  (no HRF)
S_neural_res = np.zeros((len(freqs_res), 2, 2), dtype=complex)
S_neural_orig = np.zeros_like(S_neural_res)
A_orig = np.array([[-1.0, 0.4], [0.8, -1.0]])
for k, w in enumerate(omega_res):
    Hr = np.linalg.solve(1j * w * np.eye(2) - A_res, np.eye(2))
    Ho = np.linalg.solve(1j * w * np.eye(2) - A_orig, np.eye(2))
    S_neural_res[k] = sigma_e_res**2 * (Hr @ Hr.conj().T)
    S_neural_orig[k] = sigma_e_res**2 * (Ho @ Ho.conj().T)

# BOLD CSD via the package (uses HRF)
spdcm_res = SpectralDCM(
    n_rois=2, TR=1.0, self_connection=-1.0,
    freq_lo=0.005, freq_hi=0.4, n_freqs=128,
)
S_bold_res = spdcm_res._unvectorize_csd(spdcm_res.predict_csd(theta_res))
S_bold_orig = spdcm_res._unvectorize_csd(spdcm_res.predict_csd(theta_true))

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for r in range(num_rois):
    axs[0].plot(freqs_res, S_neural_orig[:, r, r].real, lw=1.4, ls="--",
                color=f"C{r}", label=f"original ROI {r}")
    axs[0].plot(freqs_res, S_neural_res[:, r, r].real, lw=2,
                color=f"C{r}", label=f"resonant ROI {r}")
axs[0].axvline(f_neural_peak, color="k", lw=0.8, ls=":",
               label=r"$f_{\mathrm{peak}}$")
axs[0].set_yscale("log")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel(r"$|S^{\mathrm{neural}}_{ii}(f)|$")
axs[0].set_title("Neural CSD (pre-HRF) — peak is visible")
axs[0].legend(fontsize=7, ncol=2)
axs[0].grid(alpha=0.3)

for r in range(num_rois):
    axs[1].plot(spdcm_res.freqs, S_bold_orig[:, r, r].real, lw=1.4, ls="--",
                color=f"C{r}", label=f"original ROI {r}")
    axs[1].plot(spdcm_res.freqs, S_bold_res[:, r, r].real, lw=2,
                color=f"C{r}", label=f"resonant ROI {r}")
axs[1].axvline(f_neural_peak, color="k", lw=0.8, ls=":",
               label=r"$f_{\mathrm{peak}}$")
axs[1].set_yscale("log")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel(r"$|S^{\mathrm{BOLD}}_{ii}(f)|$")
axs[1].set_title("BOLD CSD (post-HRF) — HRF masks the peak")
axs[1].legend(fontsize=7, ncol=2)
axs[1].grid(alpha=0.3)

fig.suptitle("Real vs complex eigenvalues of A — neural peak vs BOLD masking")
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}_resonance.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}_resonance.pdf")
plt.show(block=False)

# %%
