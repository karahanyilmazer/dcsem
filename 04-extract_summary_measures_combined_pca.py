# %%
# !%load_ext autoreload
# !%autoreload 2
import pickle

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from sklearn.decomposition import PCA
from tqdm import tqdm

from dcsem.utils import stim_boxcar
from utils import (
    get_out_dir,
    get_width_height_latex,
    initialize_parameters,
    set_style,
    simulate_bold,
)

set_style()
width, height = get_width_height_latex()
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

n_comps = 3
# setting = f"no_noise_{n_comps}"
setting = f"with_noise_{n_comps}"

LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
IMG_DIR = get_out_dir(type="img", subfolder=f"bench_{setting}")
MODEL_DIR = get_out_dir(type="model", subfolder=f"bench_{setting}")

SEED = 42
rng = np.random.default_rng(SEED)

# %%
# ======================================================================================
# Bilinear neural model parameters
NUM_LAYERS = 1
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[10, 20, 1]])  # Input stimulus

# Parameters to set and estimate
params_to_set = ["a01", "a10", "c0", "c1"]

# Ground truth parameter values
bounds = {
    "a01": (0.0, 1.0),
    "a10": (0.0, 1.0),
    "c0": (0.0, 1.0),
    "c1": (0.0, 1.0),
}

# ======================================================================================
# %%
display(Markdown("## Data Generation"))

n_samples = 10000
bolds_roi0 = []
bolds_roi1 = []
noise_sigmas = []
for _ in tqdm(range(n_samples)):
    initial_values = initialize_parameters(bounds, params_to_set, random=True)

    # Initialize the BOLD signals
    bold_true = simulate_bold(
        dict(zip(params_to_set, initial_values)),
        time=time,
        u=u,
        num_rois=NUM_ROIS,
    )

    if setting == "no_noise":
        bold_obsv = bold_true
    else:
        noise_sigma = 0.10 * np.std(bold_true)  # 10% of signal std
        noise_sigmas.append(noise_sigma)
        bold_obsv = bold_true + rng.normal(0, noise_sigma, size=bold_true.shape)

    bolds_roi0.append(bold_obsv[:, 0])
    bolds_roi1.append(bold_obsv[:, 1])

bolds_roi0 = np.array(bolds_roi0)
bolds_roi1 = np.array(bolds_roi1)
bold_concat = np.concatenate([bolds_roi0, bolds_roi1], axis=1)
bold_concat_c = bold_concat - np.mean(bold_concat, axis=1, keepdims=True)  # Mean center

# Check for zero variance (happens with no_noise and limited parameter variation)
if np.std(bold_concat_c) < 1e-10:
    print(
        "⚠️  Warning: Data has near-zero variance. Consider using 'with_noise' setting."
    )
    print(f"   Standard deviation: {np.std(bold_concat_c):.2e}")

# %%
display(Markdown("## Fitting PCA"))
errors = []
n_vals = np.arange(1, 10)
elbow_pca = n_comps

for n in n_vals:
    pca_temp = PCA(n_components=n)
    try:
        bold_pca_temp = pca_temp.fit_transform(bold_concat_c)
        bold_recon = pca_temp.inverse_transform(bold_pca_temp)
        error = np.mean((bold_concat_c - bold_recon) ** 2)
        errors.append(error)
    except ValueError as e:
        print(f"⚠️  PCA with {n} components failed: {e}")
        errors.append(np.nan)

# Fit final PCA with chosen number of components
pca = PCA(n_components=elbow_pca)
bold_pca = pca.fit_transform(bold_concat_c)

# %%
display(Markdown("## Reconstruction Example"))
initial_values = initialize_parameters(bounds, params_to_set, random=True)

# Initialize the BOLD signals
bold_true = simulate_bold(
    dict(zip(params_to_set, initial_values)),
    time=time,
    u=u,
    num_rois=NUM_ROIS,
)

# Add noise based on setting
if setting == "no_noise":
    bold_obsv = bold_true
else:
    # Use mean of noise sigmas from training data
    avg_noise = np.mean(noise_sigmas) if noise_sigmas else 0.0
    bold_obsv = bold_true + rng.normal(0, avg_noise, size=bold_true.shape)

tmp_bold = np.concatenate([bold_obsv[:, 0], bold_obsv[:, 1]]).reshape(1, -1)
tmp_bold_c = tmp_bold - np.mean(tmp_bold, axis=1)

# Check for NaN or constant values before transformation
if np.any(np.isnan(tmp_bold_c)) or np.all(tmp_bold_c == 0):
    print("⚠️  Warning: Data contains NaN or all zeros after centering")
    bold_pca_recon = np.zeros((1, elbow_pca))
    bold_recon_pca = tmp_bold_c
    recon_error_pca = 0.0
else:
    bold_pca_recon = pca.transform(tmp_bold_c)
    bold_recon_pca = pca.inverse_transform(bold_pca_recon)
    recon_error_pca = np.mean((tmp_bold_c - bold_recon_pca) ** 2)

# %%
# ======================================================================================
# COMBINED FIGURE
# ======================================================================================

fig, axes = plt.subplots(4, 1, figsize=(width, height * 2.1), constrained_layout=True)

# First row: Concatenated BOLD signals (spanning both columns)
ax1 = axes[0]
ax1.plot(bold_concat.T, lw=0.7, alpha=0.3)
ax1.set_title(
    r"\textbf{(a)} Concatenated BOLD Signals as Input", fontsize=10, loc="left"
)
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Amplitude (a.u.)")
ax1.set_ylim(-0.003, 0.07)

# Second row: PCA elbow plot (spanning both columns)
ax2 = axes[1]
ax2.plot(n_vals, errors, color=default_colors[0])
ax2.axvline(
    elbow_pca,
    color=default_colors[1],
    linestyle="--",
    label=f"Elbow (n={elbow_pca})",
)
ax2.set_title(r"\textbf{(b)} PCA Component Selection", fontsize=10, loc="left")
ax2.set_xlabel("Number of PCA Components")
ax2.set_ylabel("Reconstruction Error (MSE)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Third row, first column: Principal components
ax3 = axes[2]
for i in range(elbow_pca):
    ax3.plot(pca.components_[i], label=f"PC {i + 1}")
ax3.set_title(r"\textbf{(c)} Principal Components", fontsize=10, loc="left")
ax3.set_xlabel("Sample Index")
ax3.set_ylabel("Amplitude (a.u.)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)


# Fourth row: Reconstructed BOLD (spanning both columns)
ax5 = axes[3]
ax5.plot(tmp_bold_c.T, label="Original", color=default_colors[0])
ax5.plot(
    bold_recon_pca.T,
    label="Reconstructed",
    linestyle="--",
    color=default_colors[1],
)
ax5.set_title(
    r"\textbf{(d)} BOLD Reconstruction",
    fontsize=10,
    loc="left",
)
ax5.set_xlabel("Sample Index")
ax5.set_ylabel("Amplitude (a.u.)")
ax5.legend()
ax5.grid(True, alpha=0.3)

fig.suptitle(r"\textbf{Extracting Summary Measures with PCA}")

# plt.savefig(IMG_DIR / f"pca_summary_combined_{setting}.png")
# plt.savefig(LATEX_DIR / f"pca_summary_combined_{setting}.pdf")
plt.show()

print(f"PCA Reconstruction Error: {recon_error_pca:.6f}")

# %%
display(Markdown("## Save the Fitted PCA Model"))

# Dump the PCA object
with open(MODEL_DIR / f"pca_{setting}.pkl", "wb") as f:
    pickle.dump(pca, f)

with open(MODEL_DIR / f"noise_sigmas_pca_{setting}.pkl", "wb") as f:
    pickle.dump(noise_sigmas, f)

print(f"PCA model saved to {MODEL_DIR / f'pca_{setting}.pkl'}")
print(f"Noise sigmas saved to {MODEL_DIR / f'noise_sigmas_pca_{setting}.pkl'}")

# %%
