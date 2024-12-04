# %%
# !%load_ext autoreload
# !%autoreload 2
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm import tqdm

from dcsem.utils import stim_boxcar
from utils import get_summary_measures, initialize_parameters, simulate_bold_multi

plt.rcParams['font.family'] = 'Times New Roman'

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
params['i0'] = 0.5
params['i1'] = 0.5
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
def add_underscore(param):
    # Use regex to insert an underscore before a digit sequence and group digits for LaTeX
    latex_param = re.sub(r'(\D)(\d+)', r'\1_{\2}', param)
    return r"${" + latex_param + r"}$"


param_labels = {param: add_underscore(param) for param in params_to_set}

# Plot the BOLD signals
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(time, bold_base[:, 0], label='Base')
axs[0].plot(time, bold_w01[:, 0], label=f'Increased {param_labels["w01"]}')
axs[0].plot(time, bold_w10[:, 0], label=f'Increased {param_labels["w10"]}')
axs[0].plot(time, bold_i0[:, 0], label=f'Increased {param_labels["i0"]}')
axs[0].plot(time, bold_i1[:, 0], label=f'Increased {param_labels["i1"]}')

axs[1].plot(time, bold_base[:, 1], label='Base')
axs[1].plot(time, bold_w01[:, 1], label=f'Increased {param_labels["w01"]}')
axs[1].plot(time, bold_w10[:, 1], label=f'Increased {param_labels["w10"]}')
axs[1].plot(time, bold_i0[:, 1], label=f'Increased {param_labels["i0"]}')
axs[1].plot(time, bold_i1[:, 1], label=f'Increased {param_labels["i1"]}')

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
axs.plot(bold_w01_comb, label=f'Increased {param_labels["w01"]}')
axs.plot(bold_w10_comb, label=f'Increased {param_labels["w10"]}')
axs.plot(bold_i0_comb, label=f'Increased {param_labels["i0"]}')
axs.plot(bold_i1_comb, label=f'Increased {param_labels["i1"]}')

plt.legend()
plt.show()

# %%
comp1, comp2 = 1, 2
arrowprops = {}
default_arrowprops = dict(arrowstyle='<-', color='black')
default_arrowprops.update(arrowprops)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1], 'o', label='Base')
ax.plot(
    summ_w01[0, comp1 - 1],
    summ_w01[0, comp2 - 1],
    'o',
    label=f'Increased {param_labels["w01"]}',
)
ax.plot(
    summ_w10[0, comp1 - 1],
    summ_w10[0, comp2 - 1],
    'o',
    label=f'Increased {param_labels["w10"]}',
)
ax.plot(
    summ_i0[0, comp1 - 1],
    summ_i0[0, comp2 - 1],
    'o',
    label=f'Increased {param_labels["i0"]}',
)
ax.plot(
    summ_i1[0, comp1 - 1],
    summ_i1[0, comp2 - 1],
    'o',
    label=f'Increased {param_labels["i1"]}',
)

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
display(Markdown('## Run the simulation'))
n_samples = 5000
change_amount = 0.1
param_vals = []
summs_ica = []
summs_pca = []
summs_change_ica = {key: [] for key in params_to_set}
summs_change_pca = {key: [] for key in params_to_set}
diffs_ica = {key: [] for key in params_to_set}
diffs_pca = {key: [] for key in params_to_set}

for sample_i in tqdm(range(n_samples)):
    # Initialize the parameters
    param_vals.append(initialize_parameters(bounds, params_to_set, random=True))

    # Get the latest sample
    sample = param_vals[-1]

    # Create a dictionary of unchanged parameters
    params = dict(zip(params_to_set, sample))

    # Get the summary measures for unchanged parameters
    summ_pca = get_summary_measures('PCA', time, u, NUM_ROIS, **params)[0]
    # summ_ica = get_summary_measures('ICA', time, u, NUM_ROIS, **params)[0]

    summs_pca.append(summ_pca)
    # summs_ica.append(summ_ica)

    for i, param in enumerate(params_to_set):
        # Get the latest sample again
        sample = param_vals[-1]

        # Introduce a change in one parameter
        sample[i] = sample[i] + change_amount

        # Check if the parameter is still within bounds
        if sample[i] > bounds[param][1]:
            sample[i] = bounds[param][1]

        # Create a new dictionary for the changed parameters
        params = dict(zip(params_to_set, sample))

        # Get the summary measures after the change
        summ_pca_change = get_summary_measures('PCA', time, u, NUM_ROIS, **params)[0]
        # summ_ica_change = get_summary_measures('ICA', time, u, NUM_ROIS, **params)[0]

        # Calculate the difference between unchanged and changed summary measures
        summ_pca_diff = summ_pca - summ_pca_change
        # summ_ica_diff = summ_ica - summ_ica_change

        # Store the results
        summs_change_pca[param].append(summ_pca_change)
        diffs_pca[param].append(summ_pca_diff)
        # summs_change_ica[param].append(summ_ica_change)
        # diffs_ica[param].append(summ_ica_diff)


# %%
method = 'PCA'
comp_to_plot1, comp_to_plot2 = 1, 2

if method == 'PCA':
    summs_arr = np.array(summs_pca)
    labels = [f'PC{comp_to_plot1}', f'PC{comp_to_plot2}']
elif method == 'ICA':
    summs_arr = np.array(summs_ica)
    labels = [f'PC{comp_to_plot1}', f'PC{comp_to_plot2}']

# Get the first two components
comp1 = summs_arr[:, comp_to_plot1 - 1]
comp2 = summs_arr[:, comp_to_plot2 - 1]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.ravel()

for i, param in enumerate(params_to_set):
    # Extract the parameter values
    param_values = np.array([p[i] for p in param_vals])

    # Scatter plot, coloring by the current parameter
    scatter = axs[i].scatter(comp1, comp2, c=param_values, s=10)
    axs[i].set_title(f'Effect of {param_labels[param]}')
    axs[i].set_xlabel(labels[0])
    axs[i].set_ylabel(labels[1])

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axs[i])
    cbar.set_label(f'{param_labels[param]} value')

# Remove unnecessary labels
axs[0].set_xlabel('')
axs[1].set_xlabel('')
axs[1].set_ylabel('')
axs[3].set_ylabel('')

plt.tight_layout()
plt.savefig(f'img/bench/change_by_param-{method}_{comp_to_plot1}&{comp_to_plot2}.png')
plt.show()

# %%
method = 'PCA'
param_to_plot = 'i1'

if method == 'PCA':
    arr = np.array(summs_pca)
    columns = ['PC1', 'PC2', 'PC3', 'PC4']
elif method == 'ICA':
    arr = np.array(summs_ica)
    columns = ['IC1', 'IC2', 'IC3', 'IC4']

df = pd.DataFrame(arr, columns=columns)
df['w01'] = [val[0] for val in param_vals]
df['w10'] = [val[1] for val in param_vals]
df['i0'] = [val[2] for val in param_vals]
df['i1'] = [val[3] for val in param_vals]

df.head()

# Normalize the param values for continuous coloring
norm = Normalize(vmin=df[param_to_plot].min(), vmax=df[param_to_plot].max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)

# Map normalized colors for the scatter plot
g = sns.PairGrid(df, vars=columns)
g.map_diag(sns.histplot, color='black', alpha=0.5)
g.map_offdiag(
    sns.scatterplot,
    hue=df[param_to_plot],
    palette='viridis',
    edgecolor=None,
    s=15,
)

# Add a colorbar for the continuous colormap
g.figure.suptitle(
    f'Effect of {param_labels[param_to_plot]} on {method} Summary Measures', y=1
)
g.figure.subplots_adjust(top=0.95)

cbar = g.figure.colorbar(
    sm, ax=g.axes, location='bottom', shrink=0.8, aspect=50, pad=0.08
)
cbar.set_label(f'{param_labels[param_to_plot]} Value')

plt.savefig(f'img/bench/change_by_param_{param_to_plot}-{method}_pairplot.png')
plt.show()

# %%
method = 'PCA'

if method == 'PCA':
    data = summs_change_pca
    columns = ['PC1', 'PC2', 'PC3', 'PC4']
elif method == 'ICA':
    data = summs_change_ica
    columns = ['IC1', 'IC2', 'IC3', 'IC4']

dfs = []
for param, values in data.items():
    df = pd.DataFrame(values, columns=columns)
    df['Parameter'] = param_labels[param]  # Add a column for parameter type
    dfs.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.head()

# Create a PairGrid
g = sns.PairGrid(
    combined_df,
    hue='Parameter',
    vars=columns,
    palette='muted',
)

# Map plots
g.map_diag(sns.histplot, alpha=0.5)
g.map_offdiag(sns.scatterplot, edgecolor=None, s=10)

# Add legend and title
g.add_legend()
g.figure.suptitle(f'Effect of Parameter Changes on {method} Summary Measures', y=1.02)

plt.savefig(f'img/bench/bench_param_change-{method}_pairplot.png')
plt.show()

# %%
