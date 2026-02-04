# %%
# !%load_ext autoreload
# !%autoreload 2
import pickle

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from sklearn.decomposition import FastICA
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

LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
IMG_DIR = get_out_dir(type="img", subfolder="bench")
MODEL_DIR = get_out_dir(type="model", subfolder="bench")

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

    noise_sigma = 0.10 * np.std(bold_true)  # 10% of signal std
    noise_sigmas.append(noise_sigma)
    bold_obsv = bold_true + rng.normal(0, noise_sigma, size=bold_true.shape)

    bolds_roi0.append(bold_obsv[:, 0])
    bolds_roi1.append(bold_obsv[:, 1])

bolds_roi0 = np.array(bolds_roi0)
bolds_roi1 = np.array(bolds_roi1)
bold_concat = np.concatenate([bolds_roi0, bolds_roi1], axis=1)
bold_concat_c = bold_concat - np.mean(bold_concat, axis=1, keepdims=True)  # Mean center

# %%
display(Markdown("## Fitting ICA"))
errors = []
n_vals = np.arange(1, 10)
elbow_ica = 3

for n in n_vals:
    ica_temp = FastICA(n_components=n)
    bold_ica_temp = ica_temp.fit_transform(bold_concat_c)
    bold_recon = ica_temp.inverse_transform(bold_ica_temp)

    error = np.mean((bold_concat_c - bold_recon) ** 2)
    errors.append(error)

# Fit final ICA with chosen number of components
ica = FastICA(n_components=elbow_ica)
bold_ica = ica.fit_transform(bold_concat_c)

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
bold_obsv = bold_true + rng.normal(0, np.mean(noise_sigmas), size=bold_true.shape)
tmp_bold = np.concatenate([bold_obsv[:, 0], bold_obsv[:, 1]]).reshape(1, -1)
tmp_bold_c = tmp_bold - np.mean(tmp_bold, axis=1)

bold_ica_recon = ica.transform(tmp_bold_c)
bold_recon_ica = ica.inverse_transform(bold_ica_recon)
recon_error_ica = np.mean((tmp_bold_c - bold_recon_ica) ** 2)

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

# Second row: ICA elbow plot (spanning both columns)
ax2 = axes[1]
ax2.plot(n_vals, errors, color=default_colors[0])
ax2.axvline(
    elbow_ica,
    color=default_colors[1],
    linestyle="--",
    label=f"Elbow (n={elbow_ica})",
)
ax2.set_title(r"\textbf{(b)} ICA Component Selection", fontsize=10, loc="left")
ax2.set_xlabel("Number of ICA Components")
ax2.set_ylabel("Reconstruction Error (MSE)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Third row, first column: Independent components
ax3 = axes[2]
for i in range(elbow_ica):
    ax3.plot(ica.components_[i], label=f"IC {i + 1}")
ax3.set_title(r"\textbf{(c)} Independent Components", fontsize=10, loc="left")
ax3.set_xlabel("Sample Index")
ax3.set_ylabel("Amplitude (a.u.)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)


# Fourth row: Reconstructed BOLD (spanning both columns)
ax5 = axes[3]
ax5.plot(tmp_bold_c.T, label="Original", color=default_colors[0])
ax5.plot(
    bold_recon_ica.T,
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

fig.suptitle(r"\textbf{Extracting Summary Measures with ICA}")

plt.savefig(IMG_DIR / "ica_summary_combined.png")
plt.savefig(LATEX_DIR / "ica_summary_combined.pdf")
plt.show()

print(f"ICA Reconstruction Error: {recon_error_ica:.6f}")

# %%
display(Markdown("## Save the Fitted ICA Model"))

# Dump the ICA object
with open(MODEL_DIR / "ica.pkl", "wb") as f:
    pickle.dump(ica, f)

print(f"ICA model saved to {MODEL_DIR / 'ica.pkl'}")

# %%
