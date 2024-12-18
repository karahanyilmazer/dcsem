# %%
# !%load_ext autoreload
# !%autoreload 2
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from IPython.display import Markdown, display
from sklearn.decomposition import PCA, FastICA
from tqdm import tqdm

from dcsem.utils import stim_boxcar
from utils import initialize_parameters, set_style, simulate_bold

set_style()

# %%
# ======================================================================================
# Bilinear neural model parameters
NUM_LAYERS = 1
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])  # Input stimulus
# u = stim_boxcar([[0, 10, 1], [40, 10, 0.5], [50, 20, 1]])

# Parameters to set and estimate
params_to_set = ['a01', 'a10', 'c0', 'c1']

# Ground truth parameter values
bounds = {
    'a01': (0.0, 1.0),
    'a10': (0.0, 1.0),
    'c0': (0.0, 1.0),
    'c1': (0.0, 1.0),
}

# ======================================================================================
# %%
display(Markdown('## Data Generation'))

n_samples = 10000
bolds_roi0 = []
bolds_roi1 = []
for _ in tqdm(range(n_samples)):
    initial_values = initialize_parameters(bounds, params_to_set, random=True)

    # Initialize the BOLD signals
    bold_true = simulate_bold(
        dict(zip(params_to_set, initial_values)),
        time=time,
        u=u,
        num_rois=NUM_ROIS,
    )
    bold_obsv = bold_true
    # bold_obsv = add_noise(bold_true, snr_db=30)

    bolds_roi0.append(bold_obsv[:, 0])
    bolds_roi1.append(bold_obsv[:, 1])

bolds_roi0 = np.array(bolds_roi0)
bolds_roi1 = np.array(bolds_roi1)
bold_concat = np.concatenate([bolds_roi0, bolds_roi1], axis=1)
bold_concat_c = bold_concat - np.mean(bold_concat, axis=1, keepdims=True)  # Mean center

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(bolds_roi0.T)
axs[0].set_title('ROI 0')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].set_ylim(-0.003, 0.07)

axs[1].plot(bolds_roi1.T)
axs[1].set_title('ROI 1')
axs[1].set_xlabel('Time')
axs[1].set_ylim(-0.003, 0.07)

plt.savefig('img/presentation/bold_signals.png')
plt.show()

# %%
display(Markdown('## Fitting PCA'))
errors = []
n_vals = np.arange(1, 21)
elbow_pca = 4

for n in n_vals:
    pca = PCA(n_components=n)
    bold_pca = pca.fit_transform(bold_concat_c)
    bold_recon = pca.inverse_transform(bold_pca)

    error = np.mean((bold_concat_c - bold_recon) ** 2)
    errors.append(error)

plt.figure(figsize=(8, 5))
plt.plot(n_vals, errors)
plt.axvline(elbow_pca, color='red', linestyle='--', label='Elbow')
plt.xlabel('Number of PCA Components')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.grid()
plt.savefig('img/presentation/pca_elbow.png')
plt.show()

# %%
pca = PCA(n_components=elbow_pca)
bold_pca = pca.fit_transform(bold_concat_c)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(pca.components_.T)
axs[0].set_title('Components')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].legend([f'Component {i+1}' for i in range(elbow_pca)])

axs[1].plot(bold_pca)
axs[1].set_title('Transformed Data')
axs[1].set_xlabel('Sample')
axs[1].set_ylabel('PCA Value')

plt.savefig('img/presentation/pca_components.png')
plt.show()

# %%
display(Markdown('## Fitting ICA'))
errors = []
n_vals = np.arange(1, 21)
elbow_ica = 4

for n in n_vals:
    ica = FastICA(n_components=n)
    bold_ica = ica.fit_transform(bold_concat_c)
    bold_recon = ica.inverse_transform(bold_ica)

    error = np.mean((bold_concat_c - bold_recon) ** 2)
    errors.append(error)

plt.figure(figsize=(8, 5))
plt.plot(n_vals, errors)
plt.axvline(elbow_ica, color='red', linestyle='--', label='Elbow')
plt.xlabel('Number of ICA Components')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.grid()
plt.show()

# %%
ica = FastICA(n_components=elbow_ica)
bold_ica = ica.fit_transform(bold_concat_c)

# Create a figure and a GridSpec layout
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2)

# Top left plot (Mixing Matrix)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ica.mixing_)
ax1.set_title('Mixing Matrix')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')

# Top right plot (Components)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ica.components_.T)
ax2.set_title('Components')
ax2.set_xlabel('Time')
ax2.legend([f'Component {i+1}' for i in range(elbow_ica)])

# Bottom plot spanning both columns (IC Value)
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(bold_ica)
ax3.set_title('Transformed Data')
ax3.set_xlabel('Parameter Combination')
ax3.set_ylabel('IC Value')

# Adjust layout and display
plt.tight_layout()
plt.show()

# %%
display(Markdown('## Reconstruction'))
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

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(tmp_bold_c.T, label='True')
axs[0].plot(bold_recon_pca.T, label='Reconstructed')
axs[1].plot(tmp_bold_c.T, label='True')
axs[1].plot(bold_recon_ica.T, label='Reconstructed')

axs[0].set_title('PCA Reconstruction')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].legend()
axs[1].set_title('ICA Reconstruction')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
plt.show()

print(f'PCA Reconstruction Error: {recon_error_pca}')
print(f'ICA Reconstruction Error: {recon_error_ica}')

# %%
display(Markdown('## Save the Fitted Models'))

# Create a models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Dump the PCA and ICA objects
with open('models/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

with open('models/ica.pkl', 'wb') as f:
    pickle.dump(ica, f)


# %%
