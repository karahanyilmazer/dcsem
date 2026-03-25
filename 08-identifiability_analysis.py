# %%
# Fisher Information and Identifiability Analysis for 2-ROI DCM
#
# Computes the forward-model Jacobian ∂BOLD/∂θ at multiple baseline points,
# the Fisher information matrix (FIM), and reports which parameter combinations
# are identifiable.  Also projects the Jacobian into the PCA subspace used by
# BENCH to assess whether PCA preserves discriminative information.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm

from dcsem import NOISE_CONFIG, PARAM_BOUNDS
from dcsem.plotting import set_style, to_latex_label
from dcsem.utils import stim_boxcar
from utils import get_out_dir, get_width_height_latex, simulate_bold

set_style()
width, height = get_width_height_latex()
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

IMG_DIR = get_out_dir(type="img", subfolder="identifiability")

# %%
# =============================================================================
# Configuration
# =============================================================================

NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[10, 20, 1]])

param_names = ["a01", "a10", "c0", "c1"]
n_params = len(param_names)
bounds_dict = PARAM_BOUNDS.get_bounds_dict()

SEED = 42
rng = np.random.default_rng(SEED)

# Finite-difference step for Jacobian
FD_STEP = 1e-4


# %%
# =============================================================================
# Jacobian computation
# =============================================================================


def compute_jacobian(theta_dict, param_names, fd_step=FD_STEP):
    """Compute ∂BOLD/∂θ at a single baseline point via central differences.

    Returns J of shape (T*R, n_params).
    """
    bold_0 = simulate_bold(theta_dict, time=time, u=u, num_rois=NUM_ROIS).ravel()
    n_obs = bold_0.size
    J = np.zeros((n_obs, len(param_names)))

    for i, name in enumerate(param_names):
        theta_plus = dict(theta_dict)
        theta_minus = dict(theta_dict)
        theta_plus[name] = theta_dict[name] + fd_step
        theta_minus[name] = theta_dict[name] - fd_step

        # Clip to bounds
        lo, hi = bounds_dict[name]
        theta_plus[name] = np.clip(theta_plus[name], lo, hi)
        theta_minus[name] = np.clip(theta_minus[name], lo, hi)

        bold_plus = simulate_bold(theta_plus, time=time, u=u, num_rois=NUM_ROIS).ravel()
        bold_minus = simulate_bold(
            theta_minus, time=time, u=u, num_rois=NUM_ROIS
        ).ravel()

        actual_step = theta_plus[name] - theta_minus[name]
        if actual_step > 0:
            J[:, i] = (bold_plus - bold_minus) / actual_step
        else:
            J[:, i] = 0.0

    return J


def compute_fim(J, noise_sigma):
    """Fisher information matrix: FIM = J' J / σ²."""
    return J.T @ J / noise_sigma**2


# %%
# =============================================================================
# Analyse at multiple baseline points
# =============================================================================

N_BASELINES = 200

print(f"Computing Jacobians at {N_BASELINES} random baseline points...")

eigenvalue_spectra = []
condition_numbers = []
fim_list = []

for _ in tqdm(range(N_BASELINES)):
    theta = {name: rng.uniform(*bounds_dict[name]) for name in param_names}

    bold_base = simulate_bold(theta, time=time, u=u, num_rois=NUM_ROIS)
    noise_sigma = NOISE_CONFIG.get_noise_std(np.std(bold_base))

    J = compute_jacobian(theta, param_names)
    fim = compute_fim(J, noise_sigma)

    eigvals = np.linalg.eigvalsh(fim)
    eigenvalue_spectra.append(eigvals)
    condition_numbers.append(eigvals[-1] / max(eigvals[0], 1e-30))
    fim_list.append(fim)

eigenvalue_spectra = np.array(eigenvalue_spectra)  # (N_BASELINES, n_params)
condition_numbers = np.array(condition_numbers)

# %%
# =============================================================================
# Summary statistics
# =============================================================================

print("\n" + "=" * 60)
print("FISHER INFORMATION — IDENTIFIABILITY ANALYSIS")
print("=" * 60)

print(f"\nAcross {N_BASELINES} random baseline parameter sets:")
print("  FIM eigenvalue ranges (min across baselines):")
for i in range(n_params):
    lo = np.percentile(eigenvalue_spectra[:, i], 5)
    med = np.median(eigenvalue_spectra[:, i])
    hi = np.percentile(eigenvalue_spectra[:, i], 95)
    print(
        f"    λ_{i + 1}: [{lo:.2e}, {med:.2e}, {hi:.2e}]  (5th, 50th, 95th percentile)"
    )

print(
    f"\n  Condition number: median={np.median(condition_numbers):.1f}, "
    f"max={np.max(condition_numbers):.1f}"
)

n_degenerate = np.sum(eigenvalue_spectra[:, 0] < 1e-6)
print(
    f"  Baselines with near-zero smallest eigenvalue (<1e-6): "
    f"{n_degenerate}/{N_BASELINES} ({100 * n_degenerate / N_BASELINES:.0f}%)"
)

if n_degenerate > 0:
    print("  ⚠️  Some parameter combinations are near-unidentifiable!")
else:
    print(
        "  ✓  All 4 parameters appear structurally identifiable at all tested baselines."
    )

# Average FIM and its eigenvectors
fim_avg = np.mean(fim_list, axis=0)
eigvals_avg, eigvecs_avg = np.linalg.eigh(fim_avg)

print(f"\n  Average FIM eigenvalues: {np.round(eigvals_avg, 2)}")
print("  Least identifiable direction (eigenvector of smallest eigenvalue):")
least_ident = eigvecs_avg[:, 0]
for i, name in enumerate(param_names):
    print(f"    {name}: {least_ident[i]:+.3f}")

# %%
# =============================================================================
# Plot 1: Eigenvalue spectrum across baselines
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(width, height * 0.8))

# Box plot of eigenvalues
axes[0].boxplot(
    eigenvalue_spectra,
    tick_labels=[rf"$\lambda_{{{i + 1}}}$" for i in range(n_params)],
)
axes[0].set_yscale("log")
axes[0].set_ylabel("Eigenvalue")
axes[0].set_title("FIM Eigenvalue Distribution")
axes[0].axhline(1e-6, color="red", ls="--", alpha=0.5, label="Near-singular threshold")
axes[0].legend(fontsize="small")

# Histogram of condition numbers
axes[1].hist(np.log10(condition_numbers), bins=30, color=default_colors[0], alpha=0.7)
axes[1].set_xlabel(r"$\log_{10}$  Condition Number")
axes[1].set_ylabel("Count")
axes[1].set_title("FIM Condition Number")
axes[1].axvline(
    np.log10(1e8), color="red", ls="--", alpha=0.5, label="Ill-conditioned threshold"
)
axes[1].legend(fontsize="small")

plt.tight_layout()
plt.savefig(IMG_DIR / "fim_eigenvalue_spectrum.png")
plt.show(block=False)

# %%
# =============================================================================
# Plot 2: Average FIM correlation structure
# =============================================================================

# Convert average FIM to correlation (normalise by diagonal)
diag_sqrt = np.sqrt(np.diag(fim_avg))
fim_corr = fim_avg / np.outer(diag_sqrt + 1e-30, diag_sqrt + 1e-30)


fig, ax = plt.subplots(figsize=(width * 0.6, width * 0.6))
latex_labels = [to_latex_label(name) for name in param_names]
sns.heatmap(
    fim_corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    xticklabels=latex_labels,
    yticklabels=latex_labels,
    square=True,
    ax=ax,
)
ax.set_title("Average FIM Correlation Structure")
plt.tight_layout()
plt.savefig(IMG_DIR / "fim_correlation.png")
plt.show(block=False)

# %%
# =============================================================================
# PCA Subspace Validation for BENCH
# =============================================================================
# Check whether PCA components preserve the Jacobian's column space,
# i.e., whether the projected Jacobian still has full column rank.

print("\n" + "=" * 60)
print("PCA SUBSPACE VALIDATION FOR BENCH")
print("=" * 60)

# Generate training data for PCA (same as 04-extract_summary_measures.py)
N_PCA_SAMPLES = 5000
bold_all = []
for _ in tqdm(range(N_PCA_SAMPLES), desc="Generating PCA training data"):
    theta = {name: rng.uniform(*bounds_dict[name]) for name in param_names}
    bold = simulate_bold(theta, time=time, u=u, num_rois=NUM_ROIS)
    bold_all.append(bold.ravel())

bold_matrix = np.array(bold_all)  # (N_PCA_SAMPLES, T*R)
bold_centered = bold_matrix - bold_matrix.mean(axis=1, keepdims=True)

for n_components in [3, 4, 5, 6]:
    pca = PCA(n_components=n_components)
    pca.fit(bold_centered)

    # Project Jacobians into PCA space and check rank
    ranks = []
    cond_projected = []
    for i in range(min(50, N_BASELINES)):
        theta = {name: rng.uniform(*bounds_dict[name]) for name in param_names}
        bold_base = simulate_bold(theta, time=time, u=u, num_rois=NUM_ROIS)
        J = compute_jacobian(theta, param_names)

        # Centre J columns (matching PCA centering)
        J_centered = J - J.mean(axis=0, keepdims=True)

        # Project: J_pca = V' @ J, where V is PCA loadings (n_features, n_components)
        J_pca = pca.components_ @ J  # (n_components, n_params)

        rank = np.linalg.matrix_rank(J_pca, tol=1e-6)
        ranks.append(rank)

        sv = np.linalg.svd(J_pca, compute_uv=False)
        cond_projected.append(sv[0] / max(sv[-1], 1e-30))

    ranks = np.array(ranks)
    cond_projected = np.array(cond_projected)

    full_rank_pct = 100 * np.mean(ranks == n_params)
    print(f"\n  n_components={n_components}:")
    print(f"    Projected Jacobian full-rank: {full_rank_pct:.0f}% of baselines")
    print(
        f"    Projected Jacobian condition number: "
        f"median={np.median(cond_projected):.1f}, max={np.max(cond_projected):.1f}"
    )

    if full_rank_pct < 100:
        print(f"    ⚠️  PCA with {n_components} components loses parameter information!")
    else:
        print(
            f"    ✓  PCA with {n_components} components preserves full parameter discriminability."
        )

# %%
# =============================================================================
# Plot 3: Sensitivity plot — ∂BOLD/∂θ for each parameter
# =============================================================================

# Compute at a representative baseline
theta_ref = {"a01": 0.4, "a10": 0.4, "c0": 0.5, "c1": 0.5}
J_ref = compute_jacobian(theta_ref, param_names)
J_ref_2d = J_ref.reshape(len(time), NUM_ROIS, n_params)

fig, axes = plt.subplots(n_params, NUM_ROIS, figsize=(width, height * 2), sharex=True)

for i, name in enumerate(param_names):
    for r in range(NUM_ROIS):
        axes[i, r].plot(time, J_ref_2d[:, r, i], color=default_colors[i])
        axes[i, r].set_ylabel(rf"$\partial$BOLD / $\partial${to_latex_label(name)}")
        if i == 0:
            axes[i, r].set_title(f"ROI {r + 1}")
        if i == n_params - 1:
            axes[i, r].set_xlabel("Time (s)")

fig.suptitle("BOLD Sensitivity to DCM Parameters", y=1.01)
plt.tight_layout()
plt.savefig(IMG_DIR / "bold_sensitivity.png")
plt.show(block=False)

print("\nPlots saved to:", IMG_DIR)

# %%
