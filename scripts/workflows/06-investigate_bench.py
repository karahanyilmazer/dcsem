# %%
# !%load_ext autoreload
# !%autoreload 2
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from IPython.display import Markdown, display
except ImportError:
    display = print
    Markdown = lambda s: s
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm import tqdm

from dcsem import PARAM_BOUNDS
from dcsem.utils import stim_boxcar
from utils import (
    add_underscore,
    get_colormap,
    get_out_dir,
    get_param_colors,
    get_summary_measures,
    initialize_parameters,
    set_style,
    simulate_bold,
)

set_style()
n_comps = 4
# setting = f"no_noise_{n_comps}"
setting = f"with_noise_{n_comps}"
IMG_DIR = get_out_dir(type="img", subfolder=f"bench_{setting}")
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
MODEL_DIR = get_out_dir(type="model", subfolder=f"bench_{setting}")
cmap = get_colormap("YlGnBu")
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# %%
# ======================================================================================
# Bilinear neural model parameters
NUM_LAYERS = 1
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[10, 20, 1]])  # Input stimulus

param_colors = get_param_colors()

# Parameters to set and estimate
params_to_set = ["a01", "a10", "c0", "c1"]

# Parameter bounds from central config
bounds = PARAM_BOUNDS.get_bounds_dict()

# Define the parameters
params = {}
params["a01"] = 0.5
params["a10"] = 0.5
params["c0"] = 0.5
params["c1"] = 0
bold_base = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)
summ_base = get_summary_measures("PCA", time, u, NUM_ROIS, MODEL_DIR, setting, **params)

# Increase alpha
params["a01"] += 0.5
bold_a01 = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)
summ_a01 = get_summary_measures("PCA", time, u, NUM_ROIS, MODEL_DIR, setting, **params)

# Increase gamma
params["a01"] -= 0.5
params["a10"] += 0.5
bold_a10 = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)
summ_a10 = get_summary_measures("PCA", time, u, NUM_ROIS, MODEL_DIR, setting, **params)

params["a10"] -= 0.5
params["c0"] += 0.5
bold_c0 = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)
summ_c0 = get_summary_measures("PCA", time, u, NUM_ROIS, MODEL_DIR, setting, **params)

params["c0"] -= 0.5
params["c1"] += 0.5
bold_c1 = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)
summ_c1 = get_summary_measures("PCA", time, u, NUM_ROIS, MODEL_DIR, setting, **params)


# ======================================================================================
# %%
param_labels = {param: add_underscore(param) for param in params_to_set}

# Plot the BOLD signals
fig, axs = plt.subplots(2, 1)
axs[0].plot(time, bold_base[:, 0], color="black", label="Base")
axs[0].plot(
    time,
    bold_a01[:, 0],
    color=param_colors["a01"],
    label=f"Increased {param_labels['a01']}",
)
axs[0].plot(
    time,
    bold_a10[:, 0],
    color=param_colors["a10"],
    label=f"Increased {param_labels['a10']}",
)
axs[0].plot(
    time,
    bold_c0[:, 0],
    color=param_colors["c0"],
    label=f"Increased {param_labels['c0']}",
)
axs[0].plot(
    time,
    bold_c1[:, 0],
    color=param_colors["c1"],
    label=f"Increased {param_labels['c1']}",
)

axs[1].plot(time, bold_base[:, 1], color="black", label="Base")
axs[1].plot(
    time,
    bold_a01[:, 1],
    color=param_colors["a01"],
    label=f"Increased {param_labels['a01']}",
)
axs[1].plot(
    time,
    bold_a10[:, 1],
    color=param_colors["a10"],
    label=f"Increased {param_labels['a10']}",
)
axs[1].plot(
    time,
    bold_c0[:, 1],
    color=param_colors["c0"],
    label=f"Increased {param_labels['c0']}",
)
axs[1].plot(
    time,
    bold_c1[:, 1],
    color=param_colors["c1"],
    label=f"Increased {param_labels['c1']}",
)

axs[0].set_title("DCM Simulation")
axs[0].set_ylabel("BOLD Signal")
axs[0].legend()
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("BOLD Signal")
axs[1].legend()

tmp = axs[0].twinx()
tmp.set_ylabel("ROI 1", rotation=0, labelpad=20)
tmp.set_yticks([])
tmp = axs[1].twinx()
tmp.set_ylabel("ROI 2", rotation=0, labelpad=20)
tmp.set_yticks([])

plt.show(block=False)

# %%
bold_base_comb = np.r_[bold_base[:, 0], bold_base[:, 1]]
bold_a01_comb = np.r_[bold_a01[:, 0], bold_a01[:, 1]]
bold_a10_comb = np.r_[bold_a10[:, 0], bold_a10[:, 1]]
bold_c0_comb = np.r_[bold_c0[:, 0], bold_c0[:, 1]]
bold_c1_comb = np.r_[bold_c1[:, 0], bold_c1[:, 1]]

fig, ax = plt.subplots()
ax.plot(bold_base_comb, c="black", label="Base")
ax.plot(bold_a01_comb, c=param_colors["a01"], label=f"Increased {param_labels['a01']}")
ax.plot(bold_a10_comb, c=param_colors["a10"], label=f"Increased {param_labels['a10']}")
ax.plot(bold_c0_comb, c=param_colors["c0"], label=f"Increased {param_labels['c0']}")
ax.plot(bold_c1_comb, c=param_colors["c1"], label=f"Increased {param_labels['c1']}")
ax.set_title("Concatenated BOLD Signals")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")

plt.legend()
plt.show(block=False)

# %%
comp1, comp2 = 3, 4
arrowprops = {}
default_arrowprops = dict(arrowstyle="<-", color="black")
default_arrowprops.update(arrowprops)

fig, ax = plt.subplots()
ax.plot(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1], "o", c="black", label="Base")
ax.plot(
    summ_a01[0, comp1 - 1],
    summ_a01[0, comp2 - 1],
    "o",
    c=param_colors["a01"],
    label=f"Increased {param_labels['a01']}",
)
ax.plot(
    summ_a10[0, comp1 - 1],
    summ_a10[0, comp2 - 1],
    "o",
    c=param_colors["a10"],
    label=f"Increased {param_labels['a10']}",
)
ax.plot(
    summ_c0[0, comp1 - 1],
    summ_c0[0, comp2 - 1],
    "o",
    c=param_colors["c0"],
    label=f"Increased {param_labels['c0']}",
)
ax.plot(
    summ_c1[0, comp1 - 1],
    summ_c1[0, comp2 - 1],
    "o",
    c=param_colors["c1"],
    label=f"Increased {param_labels['c1']}",
)

ax.annotate(
    "",
    xy=(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1]),
    xytext=(summ_a01[0, comp1 - 1], summ_a01[0, comp2 - 1]),
    arrowprops=default_arrowprops,
)
ax.annotate(
    "",
    xy=(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1]),
    xytext=(summ_a10[0, comp1 - 1], summ_a10[0, comp2 - 1]),
    arrowprops=default_arrowprops,
)
ax.annotate(
    "",
    xy=(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1]),
    xytext=(summ_c0[0, comp1 - 1], summ_c0[0, comp2 - 1]),
    arrowprops=default_arrowprops,
)
ax.annotate(
    "",
    xy=(summ_base[0, comp1 - 1], summ_base[0, comp2 - 1]),
    xytext=(summ_c1[0, comp1 - 1], summ_c1[0, comp2 - 1]),
    arrowprops=default_arrowprops,
)

ax.set_title("BENCH Arrow Plot")
ax.set_xlabel(f"PC{comp1}")
ax.set_ylabel(f"PC{comp2}")
ax.legend()
plt.savefig(IMG_DIR / f"arrow_plot-pc{comp1}and{comp2}.png")
plt.savefig(LATEX_DIR / f"arrow_plot-pc{comp1}and{comp2}.pdf")
plt.show(block=False)

# %%
display(Markdown("## Run the simulation"))
n_samples = 5000
change_amount = 0.3
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
    summ_pca = get_summary_measures(
        "PCA", time, u, NUM_ROIS, MODEL_DIR, setting, **params
    )[0]
    # summ_ica = get_summary_measures('ICA', time, u, NUM_ROIS, MODEL_DIR, **params)[0]

    summs_pca.append(summ_pca)
    # summs_ica.append(summ_ica)

    for i, param in enumerate(params_to_set):
        # Fresh copy each time — perturbations must be independent per parameter
        perturbed = list(param_vals[-1])

        # Introduce a change in one parameter only
        perturbed[i] = perturbed[i] + change_amount

        # Check if the parameter is still within bounds
        if perturbed[i] > bounds[param][1]:
            perturbed[i] = bounds[param][1]

        # Create a new dictionary for the changed parameters
        params = dict(zip(params_to_set, perturbed))

        # Get the summary measures after the change
        summ_pca_change = get_summary_measures(
            "PCA", time, u, NUM_ROIS, MODEL_DIR, setting, **params
        )[0]
        # summ_ica_change = get_summary_measures(
        #     "ICA", time, u, NUM_ROIS, MODEL_DIR, **params
        # )[0]

        # Calculate the difference between unchanged and changed summary measures
        summ_pca_diff = summ_pca - summ_pca_change
        # summ_ica_diff = summ_ica - summ_ica_change

        # Store the results
        summs_change_pca[param].append(summ_pca_change)
        diffs_pca[param].append(summ_pca_diff)
        # summs_change_ica[param].append(summ_ica_change)
        # diffs_ica[param].append(summ_ica_diff)


# %%
method = "PCA"
comp_to_plot1, comp_to_plot2 = 1, 2

if method == "PCA":
    summs_arr = np.array(summs_pca)
    labels = [f"PC{comp_to_plot1}", f"PC{comp_to_plot2}"]
elif method == "ICA":
    summs_arr = np.array(summs_ica)
    labels = [f"PC{comp_to_plot1}", f"PC{comp_to_plot2}"]

# Get the first two components
comp1 = summs_arr[:, comp_to_plot1 - 1]
comp2 = summs_arr[:, comp_to_plot2 - 1]

fig, axs = plt.subplots(2, 2)
axs = axs.ravel()

for i, param in enumerate(params_to_set):
    # Extract the parameter values
    param_values = np.array([p[i] for p in param_vals])

    # Scatter plot, coloring by the current parameter
    scatter = axs[i].scatter(comp1, comp2, c=param_values, s=10, cmap=cmap)
    axs[i].set_title(f"Effect of {param_labels[param]}")
    axs[i].set_xlabel(labels[0])
    axs[i].set_ylabel(labels[1])

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axs[i])
    cbar.set_label(f"{param_labels[param]} value")

# Remove unnecessary labels
axs[0].set_xlabel("")
axs[1].set_xlabel("")
axs[1].set_ylabel("")
axs[3].set_ylabel("")

plt.tight_layout()
plt.savefig(IMG_DIR / f"change_by_param-{method}_{comp_to_plot1}&{comp_to_plot2}.png")
plt.savefig(LATEX_DIR / f"change_by_param-{method}_{comp_to_plot1}&{comp_to_plot2}.pdf")
plt.show(block=False)

# %%
method = "PCA"
param_to_plot = "c1"

if method == "PCA":
    arr = np.array(summs_pca)
    columns = [f"PC{i + 1}" for i in range(arr.shape[1])]
elif method == "ICA":
    arr = np.array(summs_ica)
    columns = [f"IC{i + 1}" for i in range(arr.shape[1])]

df = pd.DataFrame(arr, columns=columns)
df["a01"] = [val[0] for val in param_vals]
df["a10"] = [val[1] for val in param_vals]
df["c0"] = [val[2] for val in param_vals]
df["c1"] = [val[3] for val in param_vals]

df.head()

# Normalize the param values for continuous coloring
norm = Normalize(vmin=df[param_to_plot].min(), vmax=df[param_to_plot].max())
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

# Map normalized colors for the scatter plot
g = sns.PairGrid(df, vars=columns)
g.map_diag(sns.histplot, color="black", alpha=0.5)
g.map_offdiag(
    sns.scatterplot,
    hue=df[param_to_plot],
    palette=cmap,
    edgecolor=None,
    s=15,
)

# Add a colorbar for the continuous colormap
g.figure.suptitle(
    f"Effect of {param_labels[param_to_plot]} on {method} Summary Measures", y=1
)
g.figure.subplots_adjust(top=0.95)

cbar = g.figure.colorbar(
    sm, ax=g.axes, location="bottom", shrink=0.8, aspect=50, pad=0.08
)
cbar.set_label(f"{param_labels[param_to_plot]} Value")

plt.savefig(IMG_DIR / f"change_by_param_{param_to_plot}-{method}_pairplot.png")
plt.savefig(LATEX_DIR / f"change_by_param_{param_to_plot}-{method}_pairplot.pdf")
plt.show(block=False)

# %%
method = "PCA"

if method == "PCA":
    data = summs_change_pca
    columns = [f"PC{i + 1}" for i in range(arr.shape[1])]
elif method == "ICA":
    data = summs_change_ica
    columns = [f"IC{i + 1}" for i in range(arr.shape[1])]

dfs = []
for param, values in data.items():
    df = pd.DataFrame(values, columns=columns)
    df["Parameter"] = param_labels[param]  # Add a column for parameter type
    dfs.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.head()

# Create a PairGrid
g = sns.PairGrid(
    combined_df,
    hue="Parameter",
    vars=columns,
    palette="muted",
    # palette={
    #     param_labels[key]: default_colors[i] for i, key in enumerate(params_to_set)
    # },
)

# Map plots
g.map_diag(sns.histplot, alpha=0.5)
g.map_offdiag(sns.scatterplot, edgecolor=None, alpha=0.5, s=10)

# Add legend and title
g.add_legend()
g.figure.suptitle(f"Effect of Parameter Changes on {method} Summary Measures", y=1.02)

plt.savefig(IMG_DIR / f"bench_param_change-{method}_pairplot.png")
plt.savefig(LATEX_DIR / f"bench_param_change-{method}_pairplot.pdf")
plt.show(block=False)

# %%
