"""Forward-simulation demo — single-layer Spectral DCM, 3 ROIs (chain + feedback).

Cell-by-cell script. Same connectivity as ``dcm_1layer_3roi.py``:

    ROI 0 ──→ ROI 1 ──→ ROI 2
                          │
                          ▼ (inhibitory feedback)
                       ROI 0

Driven by internal neuronal noise (no exogenous stimulus); the forward
model returns the cross-spectral density (CSD) at each frequency.

What you should see (LS-spDCM forward model)
--------------------------------------------
The CSD has the closed form
    S(ω) = σ_e² · H_hrf(ω) · (jωI − A)⁻¹ · (jωI − A)⁻ᴴ · H_hrf(ω)*

(a) ``|S_ii(f)|`` is a Lorentzian × HRF² — monotonically decreasing in
    0.01–0.1 Hz with **no peaks**, because A's eigenvalues are real here.
(b) ``|S_ij(f)| ≤ √(S_ii · S_jj)`` (Cauchy–Schwarz).
(c) Magnitude-squared coherence γ²(f) ∈ [0, 1]. Independent white-
    noise drives mean γ² is small overall; expect direct edges
    (0–1 and 1–2) to be higher than the indirect pair (0–2), which
    only "sees" through ROI 1 and the inhibitory feedback.
(d) ``∠ S_ij(f)`` is non-zero where coupling is asymmetric — the
    feedback edge ROI 2 → ROI 0 introduces a frequency-dependent lag.
(e) Empirical Welch CSD from ``simulate_bold`` should track the
    analytical ``predict_csd`` to within ~1 dex.
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
FIG_NAME = "spdcm_1layer_3roi"


# %% Build SpectralDCM
spdcm = SpectralDCM(
    n_rois=3,
    TR=1.0,
    self_connection=-1.0,
    freq_lo=0.01,
    freq_hi=0.1,
    n_freqs=32,
)
print("param order:", spdcm.get_param_names())
# expected: ['a01', 'a02', 'a10', 'a12', 'a20', 'a21', 'log_sigma_e']


# %% Ground-truth parameters
# Topology mirrors dcm_1layer_3roi.py:
#   R0 → R1 (a01 = 0.5), R1 → R2 (a12 = 0.4), R2 → R0 (a20 = -0.2),
#   all other off-diagonals zero, self = -1.
theta_true = np.array(
    [
        0.5,  # a01: R0 → R1
        0.0,  # a02
        0.0,  # a10
        0.4,  # a12: R1 → R2
        -0.2,  # a20: R2 → R0  (inhibitory feedback)
        0.0,  # a21
        np.log(0.1),  # log_sigma_e
    ]
)

A = np.array(
    [
        [-1.0, 0.0, -0.2],
        [0.5, -1.0, 0.0],
        [0.0, 0.4, -1.0],
    ]
)
print("A:\n", A)
eigvals = np.linalg.eigvals(A)
print(f"max Re(eig(A)) = {eigvals.real.max():+.4f}  (must be < 0 for stability)")


# %% Diagram of the spDCM connectivity (no input arrows — spDCM is noise-driven)
num_rois = spdcm.n_rois
dcm_for_graph = DCM(num_rois, params={"A": A, "C": np.zeros(num_rois)})

fig, _ = plot_dcm_graph(
    dcm_for_graph,
    show_self_connections=True,
    show_inputs=False,
    threshold=1e-12,
    figsize=None,
    node_color=None,
    node_radius=0.22,
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


# %% Plot CSD: auto-PSD per ROI + cross-PSD magnitudes for each ROI pair
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

for i in range(num_rois):
    for j in range(i + 1, num_rois):
        axs[1].plot(freqs, np.abs(S[:, i, j]), label=rf"$|S_{{{i}{j}}}|$")
axs[1].set_yscale("log")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel(r"$|S_{ij}(f)|$")
axs[1].set_title("Cross-PSD")
axs[1].legend()
axs[1].grid(alpha=0.3)

fig.suptitle(r"\textbf{Spectral DCM CSD — 3-ROI chain + feedback}")
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
axs[0].legend(fontsize=7, ncol=2)
axs[0].grid(alpha=0.3)

pair_idx = 0
for i in range(num_rois):
    for j in range(i + 1, num_rois):
        c = f"C{pair_idx + 3}"
        axs[1].plot(freqs, np.abs(S[:, i, j]), lw=2, color=c,
                    label=rf"analytical $|S_{{{i}{j}}}|$")
        axs[1].plot(freqs, np.abs(S_emp[:, i, j]), "o", ms=4, color=c,
                    alpha=0.55, label=rf"empirical $|S_{{{i}{j}}}|$")
        pair_idx += 1
axs[1].set_yscale("log")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel(r"$|S_{ij}(f)|$")
axs[1].set_title("Cross-PSD: predict\\_csd vs Welch")
axs[1].legend(fontsize=7, ncol=2)
axs[1].grid(alpha=0.3)

fig.suptitle("Forward-model self-consistency (linearised vs nonlinear sim)")
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}_csd_overlay.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}_csd_overlay.pdf")
plt.show(block=False)


# %% Cross-spectrum phase + magnitude-squared coherence (per ROI pair)
# Coherence γ²(ω) = |S_ij|² / (S_ii · S_jj) ∈ [0, 1].  Directly-coupled
# pairs (0–1 and 1–2) should show higher coherence than the indirectly-
# coupled pair (0–2) which only "sees" through ROI 1.
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for i in range(num_rois):
    for j in range(i + 1, num_rois):
        S_ij = S[:, i, j]
        S_ii = S[:, i, i].real
        S_jj = S[:, j, j].real
        coh_sq = (np.abs(S_ij) ** 2) / (S_ii * S_jj)
        phase = np.unwrap(np.angle(S_ij))
        axs[0].plot(freqs, phase, label=rf"$\angle S_{{{i}{j}}}$")
        axs[1].plot(freqs, coh_sq, label=rf"$\gamma^2_{{{i}{j}}}$")

axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel(r"$\angle S_{ij}(f)$ [rad]")
axs[0].set_title("Cross-spectrum phase")
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)

axs[1].set_ylim(-0.05, 1.05)
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel(r"$\gamma^2_{ij}(f)$")
axs[1].set_title("Magnitude-squared coherence")
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)

fig.suptitle("Directionality + shared-variance diagnostics")
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}_phase_coh.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}_phase_coh.pdf")
plt.show(block=False)

# %%
