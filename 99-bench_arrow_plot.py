# %%
# !%load_ext autoreload
# !%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
from bench import acquisition, change_model, diffusion_models, dti
from bench import plot as bench_plot
from tqdm import tqdm

from dcsem.utils import stim_boxcar
from utils import get_summary_measures, initialize_parameters, simulate_bold_multi

# %%
# ======================================================================================
# Bilinear neural model parameters
NUM_LAYERS = 1
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])  # Input stimulus

# Parameters to set and estimate
params_to_set = ['w01', 'w10', 'i0', 'i1']

# Ground truth parameter values
bounds = {
    'w01': (0.0, 1.0),
    'w10': (0.0, 1.0),
    'i0': (0.0, 1.0),
    'i1': (0.0, 1.0),
}


# Define the parameters
params = {}
params['w01'] = 0.5
params['w10'] = 0.5
params['i0'] = 1
params['i1'] = 0
bold_base = simulate_bold_multi(params, time=time, u=u, num_rois=NUM_ROIS)
summ_base = get_summary_measures('PCA', time, u, NUM_ROIS, **params)

# Increase alpha
params['w01'] += 0.5
bold_w01 = simulate_bold_multi(params, time=time, u=u, num_rois=NUM_ROIS)
summ_w01 = get_summary_measures('PCA', time, u, NUM_ROIS, **params)

# Increase gamma
params['w01'] -= 0.5
params['w10'] += 0.5
bold_w10 = simulate_bold_multi(params, time=time, u=u, num_rois=NUM_ROIS)
summ_w10 = get_summary_measures('PCA', time, u, NUM_ROIS, **params)

# ======================================================================================
# %%
# Plot the BOLD signals
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(time, bold_base[:, 0], label='Base')
axs[0].plot(time, bold_w01[:, 0], label='Increased W01')
axs[0].plot(time, bold_w10[:, 0], label='Increased W10')

axs[1].plot(time, bold_base[:, 1], label='Base')
axs[1].plot(time, bold_w01[:, 1], label='Increased W01')
axs[1].plot(time, bold_w10[:, 1], label='Increased W10')

axs[0].set_title('DCM Simulation')
axs[0].set_ylabel('BOLD Signal')
axs[0].legend()
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('BOLD Signal')
axs[1].legend()

tmp = axs[0].twinx()
tmp.set_ylabel('Layer 1', rotation=0, labelpad=25)
tmp.set_yticks([])
tmp = axs[1].twinx()
tmp.set_ylabel('Layer 2', rotation=0, labelpad=25)
tmp.set_yticks([])

plt.show()

# %%
bold_base_comb = np.r_[bold_base[:, 0], bold_base[:, 1]]
bold_w01_comb = np.r_[bold_w01[:, 0], bold_w01[:, 1]]
bold_w10_comb = np.r_[bold_w10[:, 0], bold_w10[:, 1]]

fig, axs = plt.subplots(1, figsize=(10, 3))
axs.plot(bold_base_comb, label='Base')
axs.plot(bold_w01_comb, label='Increased W01')
axs.plot(bold_w10_comb, label='Increased W10')

plt.legend()
plt.show()

# %%
arrowprops = {}
default_arrowprops = dict(arrowstyle='<-', color='black')
default_arrowprops.update(arrowprops)
names = ['Base', 'Alpha', 'Gamma']
max_name_length = max([len(name) for name in names])
offset_x = np.maximum(0.01 * max_name_length, 0.005)
offset_y = 0.1
offset = 0.005

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(summ_base[0, 0], summ_base[0, 1], 'o', label='Base')
ax.plot(summ_w01[0, 0], summ_w01[0, 1], 'o', label='Increased W01')
ax.plot(summ_w10[0, 0], summ_w10[0, 1], 'o', label='Increased W10')

ax.annotate(
    '',
    xy=(summ_base[0, 0], summ_base[0, 1]),
    xytext=(summ_w01[0, 0], summ_w01[0, 1]),
    arrowprops=default_arrowprops,
)
ax.annotate(
    '',
    xy=(summ_base[0, 0], summ_base[0, 1]),
    xytext=(summ_w10[0, 0], summ_w10[0, 1]),
    arrowprops=default_arrowprops,
)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
plt.show()

# %%
n_samples = 1000
param_vals = []
summs = []

for sample_i in tqdm(range(n_samples)):
    # Randomly initialize the parameters
    params = initialize_parameters(bounds, params_to_set, random=True)
    params = dict(zip(params_to_set, params))

    # Get the summary measures
    summ = get_summary_measures('PCA', time, u, NUM_ROIS, **params)[0]

    # Store the results
    param_vals.append(params)
    summs.append(summ)

# %%
# Get the first two PCs
summs = np.array(summs)
pc1 = summs[:, 0]
pc2 = summs[:, 1]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.ravel()

for i, param in enumerate(params_to_set):
    # Extract the parameter values
    param_values = np.array([p[param] for p in param_vals])

    # Scatter plot, coloring by the current parameter
    scatter = axs[i].scatter(pc1, pc2, c=param_values, cmap='viridis', s=10)
    axs[i].set_title(f'Color by {param}')
    axs[i].set_xlabel('PC1')
    axs[i].set_ylabel('PC2')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axs[i])
    cbar.set_label(f'{param} value')

plt.tight_layout()
plt.show()

# %%
