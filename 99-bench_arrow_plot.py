# %%
# !%load_ext autoreload
# !%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

params['w10'] -= 0.5
params['i0'] += 0.5
bold_i0 = simulate_bold_multi(params, time=time, u=u, num_rois=NUM_ROIS)
summ_i0 = get_summary_measures('PCA', time, u, NUM_ROIS, **params)

params['i0'] -= 0.5
params['i1'] += 0.5
bold_i1 = simulate_bold_multi(params, time=time, u=u, num_rois=NUM_ROIS)
summ_i1 = get_summary_measures('PCA', time, u, NUM_ROIS, **params)

# ======================================================================================
# %%
# Plot the BOLD signals
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(time, bold_base[:, 0], label='Base')
axs[0].plot(time, bold_w01[:, 0], label='Increased W01')
axs[0].plot(time, bold_w10[:, 0], label='Increased W10')
axs[0].plot(time, bold_i0[:, 0], label='Increased I0')
axs[0].plot(time, bold_i1[:, 0], label='Increased I1')

axs[1].plot(time, bold_base[:, 1], label='Base')
axs[1].plot(time, bold_w01[:, 1], label='Increased W01')
axs[1].plot(time, bold_w10[:, 1], label='Increased W10')
axs[1].plot(time, bold_i0[:, 1], label='Increased I0')
axs[1].plot(time, bold_i1[:, 1], label='Increased I1')

axs[0].set_title('DCM Simulation')
axs[0].set_ylabel('BOLD Signal')
axs[0].legend()
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('BOLD Signal')
axs[1].legend()

tmp = axs[0].twinx()
tmp.set_ylabel('ROI 1', rotation=0, labelpad=25)
tmp.set_yticks([])
tmp = axs[1].twinx()
tmp.set_ylabel('ROI 2', rotation=0, labelpad=25)
tmp.set_yticks([])

plt.show()

# %%
bold_base_comb = np.r_[bold_base[:, 0], bold_base[:, 1]]
bold_w01_comb = np.r_[bold_w01[:, 0], bold_w01[:, 1]]
bold_w10_comb = np.r_[bold_w10[:, 0], bold_w10[:, 1]]
bold_i0_comb = np.r_[bold_i0[:, 0], bold_i0[:, 1]]
bold_i1_comb = np.r_[bold_i1[:, 0], bold_i1[:, 1]]

fig, axs = plt.subplots(1, figsize=(10, 3))
axs.plot(bold_base_comb, label='Base')
axs.plot(bold_w01_comb, label='Increased W01')
axs.plot(bold_w10_comb, label='Increased W10')
axs.plot(bold_i0_comb, label='Increased I0')
axs.plot(bold_i1_comb, label='Increased I1')

plt.legend()
plt.show()

# %%
comp1, comp2 = 3, 4
arrowprops = {}
default_arrowprops = dict(arrowstyle='<-', color='black')
default_arrowprops.update(arrowprops)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1], 'o', label='Base')
ax.plot(summ_w01[0, comp1 - 1], summ_w01[0, comp2 - 1], 'o', label='Increased W01')
ax.plot(summ_w10[0, comp1 - 1], summ_w10[0, comp2 - 1], 'o', label='Increased W10')
ax.plot(summ_i0[0, comp1 - 1], summ_i0[0, comp2 - 1], 'o', label='Increased I0')
ax.plot(summ_i1[0, comp1 - 1], summ_i1[0, comp2 - 1], 'o', label='Increased I1')

ax.annotate(
    '',
    xy=(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1]),
    xytext=(summ_w01[0, comp1 - 1], summ_w01[0, comp2 - 1]),
    arrowprops=default_arrowprops,
)
ax.annotate(
    '',
    xy=(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1]),
    xytext=(summ_w10[0, comp1 - 1], summ_w10[0, comp2 - 1]),
    arrowprops=default_arrowprops,
)
ax.annotate(
    '',
    xy=(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1]),
    xytext=(summ_i0[0, comp1 - 1], summ_i0[0, comp2 - 1]),
    arrowprops=default_arrowprops,
)
ax.annotate(
    '',
    xy=(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1]),
    xytext=(summ_i1[0, comp1 - 1], summ_i1[0, comp2 - 1]),
    arrowprops=default_arrowprops,
)

ax.set_title('BENCH Arrow Plot')
ax.set_xlabel(f'PC{comp1}')
ax.set_ylabel(f'PC{comp2}')
ax.legend()
plt.savefig(f'img/bench/arrow_plot-pc{comp1}and{comp2}.png')
plt.show()

# %%
n_samples = 5000
param_vals = []
summs_pca = []
summs_ica = []

for sample_i in tqdm(range(n_samples)):
    # Randomly initialize the parameters
    params = initialize_parameters(bounds, params_to_set, random=True)
    params = dict(zip(params_to_set, params))

    # Get the summary measures
    summ_pca = get_summary_measures('PCA', time, u, NUM_ROIS, **params)[0]
    summ_ica = get_summary_measures('ICA', time, u, NUM_ROIS, **params)[0]


    # Store the results
    param_vals.append(params)
    summs_pca.append(summ_pca)
    summs_ica.append(summ_pca)

# %%
# Get the first two PCs
summs_pca = np.array(summs_pca)
pc1 = summs_pca[:, 0]
pc2 = summs_pca[:, 1]


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
plt.savefig(f'img/change_by_param-pc{comp1}_{comp2}.png')
plt.show()

# %%
# Get the first two ICs
summs_ica = np.array(summs_ica)
ic1 = summs_ica[:, 0]
ic2 = summs_ica[:, 1]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()

for i, param in enumerate(params_to_set):
    # Extract the parameter values
    param_values = np.array([p[param] for p in param_vals])

    # Scatter plot, coloring by the current parameter
    scatter = axs[i].scatter(pc1, pc2, c=param_values, s=10)
    axs[i].set_title(f'{param} Changes')
    axs[i].set_xlabel('IC1')
    axs[i].set_ylabel('IC2')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axs[i])
    # cbar.set_label(f'{param} value')

axs[0].set_xlabel('')
axs[1].set_xlabel('')
axs[1].set_ylabel('')
axs[3].set_ylabel('')

plt.tight_layout()
plt.savefig(f'img/change_by_param-ic{comp1}_{comp2}.png')
plt.show()

# %%
w01_vals = [list(d.values())[0] for d in param_vals]
w10_vals = [list(d.values())[1] for d in param_vals]
i0_vals = [list(d.values())[2] for d in param_vals]
i1_vals = [list(d.values())[3] for d in param_vals]

df = pd.DataFrame(summs_ica, columns=['IC1', 'IC2', 'IC3', 'IC4'])
df['w01'] = w01_vals
df['w10'] = w10_vals
df['i0'] = i0_vals
df['i1'] = i1_vals

df.head()

# %%
sns.pairplot(
    df,
    hue='w01',
    palette='viridis',
    vars=['IC1', 'IC2', 'IC3', 'IC4'],
    diag_kind='kde',  # hist
    diag_kws=dict(hue=None, color='black', alpha=0.5),
)
plt.show()

# %%
