# %%
# !%load_ext autoreload
# !%autoreload 2
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

try:
    from IPython.display import Markdown, display
except ImportError:
    display = print
    Markdown = str
from sklearn.decomposition import PCA, FastICA
from tqdm import tqdm

from dcsem import PARAM_BOUNDS
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

LATEX_DIR = get_out_dir(type="latex")
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
# u = stim_boxcar([[0, 10, 1], [40, 10, 0.5], [50, 20, 1]])

# Parameters to set and estimate
params_to_set = ["a01", "a10", "c0", "c1"]

# Parameter bounds from central config
bounds = PARAM_BOUNDS.get_bounds_dict()

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

    # bold_obsv = bold_true
    # bold_obsv = add_noise(bold_true, snr_db=30)
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
fig, axs = plt.subplots(1, 2)
axs[0].plot(bolds_roi0.T, lw=0.8)
axs[0].set_title("ROI 1")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude (a.u.)")
axs[0].set_ylim(-0.003, 0.07)

axs[1].plot(bolds_roi1.T, lw=0.8)
axs[1].set_title("ROI 2")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylim(-0.003, 0.07)

plt.savefig(IMG_DIR / "bold_signals.png")
plt.savefig(LATEX_DIR / "bold_signals.pdf")
plt.show()

# %%
display(Markdown("## Concatenated BOLD Signal"))
plt.figure()
plt.plot(bold_concat.T)
plt.title("Concatenated BOLD Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.savefig(IMG_DIR / "bold_concat.png")
plt.savefig(LATEX_DIR / "bold_concat.pdf")
plt.show()

# %%
display(Markdown("## Fitting PCA"))
errors = []
n_vals = np.arange(1, 21)
elbow_pca = 3

for n in n_vals:
    pca = PCA(n_components=n)
    bold_pca = pca.fit_transform(bold_concat_c)
    bold_recon = pca.inverse_transform(bold_pca)

    error = np.mean((bold_concat_c - bold_recon) ** 2)
    errors.append(error)

plt.figure()
plt.plot(n_vals, errors)
plt.axvline(elbow_pca, color=default_colors[1], linestyle="--", label="Elbow")
plt.xlabel("Number of PCA Components")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid()
plt.savefig(IMG_DIR / "pca_elbow.png")
plt.savefig(LATEX_DIR / "pca_elbow.pdf")
plt.show()

# %%
pca = PCA(n_components=elbow_pca)
bold_pca = pca.fit_transform(bold_concat_c)

fig, axs = plt.subplots(2, 1, figsize=(width, height * 1.2))
# fig.suptitle("PCA Reconstruction")

axs[0].plot(pca.components_.T)
axs[0].set_title("Principal Components")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Amplitude")
axs[0].legend([f"Component {i + 1}" for i in range(elbow_pca)])

axs[1].plot(bold_pca)
axs[1].set_title("Transformed Data")
axs[1].set_xlabel("Sample")
axs[1].set_ylabel("PCA Value")

plt.tight_layout()
plt.savefig(IMG_DIR / "pca_components.png")
plt.savefig(LATEX_DIR / "pca_components.pdf")
plt.show()

# %%
display(Markdown("## Fitting ICA"))
errors = []
n_vals = np.arange(1, 21)
elbow_ica = 3

for n in n_vals:
    ica = FastICA(n_components=n)
    bold_ica = ica.fit_transform(bold_concat_c)
    bold_recon = ica.inverse_transform(bold_ica)

    error = np.mean((bold_concat_c - bold_recon) ** 2)
    errors.append(error)

plt.figure()
plt.plot(n_vals, errors)
plt.axvline(elbow_ica, color="red", linestyle="--", label="Elbow")
plt.xlabel("Number of ICA Components")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid()
plt.savefig(IMG_DIR / "ica_elbow.png")
plt.savefig(LATEX_DIR / "ica_elbow.pdf")
plt.show()

# %%
ica = FastICA(n_components=elbow_ica)
bold_ica = ica.fit_transform(bold_concat_c)

# Create a figure and a GridSpec layout
fig = plt.figure(figsize=(width, height * 1.7))
gs = fig.add_gridspec(2, 2)

# Top left plot (Mixing Matrix)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ica.mixing_)
ax1.set_title("Mixing Matrix")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")

# Top right plot (Components)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ica.components_.T)
ax2.set_title("Components")
ax2.set_xlabel("Time")
ax2.legend([f"Component {i + 1}" for i in range(elbow_ica)])

# Bottom plot spanning both columns (IC Value)
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(bold_ica)
ax3.set_title("Transformed Data")
ax3.set_xlabel("Parameter Combination")
ax3.set_ylabel("IC Value")

# Adjust layout and display
plt.tight_layout()
plt.savefig(IMG_DIR / "ica_components.png")
plt.savefig(LATEX_DIR / "ica_components.pdf")
plt.show()

# %%
display(Markdown("## Reconstruction"))
initial_values = initialize_parameters(bounds, params_to_set, random=True)

# Initialize the BOLD signals
bold_true = simulate_bold(
    dict(zip(params_to_set, initial_values)),
    time=time,
    u=u,
    num_rois=NUM_ROIS,
)
bold_obsv = bold_true
tmp_bold = np.concatenate([bold_obsv[:, 0], bold_obsv[:, 1]]).reshape(1, -1)
tmp_bold_c = tmp_bold - np.mean(tmp_bold, axis=1)

# %%
bold_pca = pca.transform(tmp_bold_c)
bold_recon_pca = pca.inverse_transform(bold_pca)
recon_error_pca = np.mean((tmp_bold_c - bold_recon_pca) ** 2)

bold_ica = ica.transform(tmp_bold_c)
bold_recon_ica = ica.inverse_transform(bold_ica)
recon_error_ica = np.mean((tmp_bold_c - bold_recon_ica) ** 2)

fig, axs = plt.subplots(2, 1, figsize=(width, height * 1.2))
axs[0].plot(tmp_bold_c.T, label="True")
axs[0].plot(bold_recon_pca.T, label="Reconstructed")
axs[1].plot(tmp_bold_c.T, label="True")
axs[1].plot(bold_recon_ica.T, label="Reconstructed")

axs[0].set_title("PCA Reconstruction")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[1].set_title("ICA Reconstruction")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Amplitude")
axs[1].legend()

plt.tight_layout()

plt.show()

print(f"PCA Reconstruction Error: {recon_error_pca}")
print(f"ICA Reconstruction Error: {recon_error_ica}")

# %%
display(Markdown("## Save the Fitted Models"))

# Dump the PCA and ICA objects
with open(MODEL_DIR / "pca.pkl", "wb") as f:
    pickle.dump(pca, f)

with open(MODEL_DIR / "ica.pkl", "wb") as f:
    pickle.dump(ica, f)


# %%
