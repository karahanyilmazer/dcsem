# %%
# !%load_ext autoreload
# !%autoreload 2

import matplotlib.pyplot as plt
import numpy as np
from bench import plot as bench_plot

from dcsem import models, utils


# %%
def get_one_layer_A(L0=0.2):
    connections = [f'R0, L0 -> R1, L0 = {L0}']  # ROI0 -> ROI1 connection
    return utils.create_A_matrix(num_rois, num_layers, connections, self_connections=-1)


def get_one_layer_C(L0=1):
    connections = [f'R0, L0 = {L0}']  # Input --> ROI0 connection
    return utils.create_C_matrix(num_rois, num_layers, connections)


def get_bold(params):
    # Instantiate the DCM object
    dcm = models.DCM(
        num_rois,
        params={
            'A': params['A_L0'],
            'C': params['C_L0'],
            'alpha': params['alpha'],
            'gamma': params['gamma'],
        },
    )
    bold, _ = dcm.simulate(time, u)
    return bold


# Input
time = np.arange(100)  # Time vector (seconds)
u = utils.stim_boxcar([[0, 30, 1]])  # Stimulus function (onset, duration, amplitude)
num_rois, num_layers = 2, 1

# Define the parameters
params = {}
params['alpha'] = 0.2
params['gamma'] = 0.5
params['A_L0'] = get_one_layer_A()
params['C_L0'] = get_one_layer_C()
bold_base = get_bold(params)

# Increase alpha
params['alpha'] += 0.5
bold_alpha = get_bold(params)

# Increase gamma
params['alpha'] -= 0.5
params['gamma'] += 0.5
bold_gamma = get_bold(params)

# %%
# Plot the BOLD signals
fig, axs = plt.subplots(2, 1)
axs[0].plot(time, bold_base[:, 0], label='Base')
axs[0].plot(time, bold_alpha[:, 0], label='Increased Alpha')
axs[0].plot(time, bold_gamma[:, 0], label='Increased Gamma')

axs[1].plot(time, bold_base[:, 1], label='Base')
axs[1].plot(time, bold_alpha[:, 1], label='Increased Alpha')
axs[1].plot(time, bold_gamma[:, 1], label='Increased Gamma')

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
# Extract summary statistics from all BOLD signals
def get_summary_stats(bold):
    stats = {}
    for i, roi in enumerate(bold.T):
        stats[f'ROI_{i}'] = {
            'mean': np.mean(roi).astype(float),
            'std': np.std(roi).astype(float),
            'max': np.max(roi).astype(float),
            'min': np.min(roi).astype(float),
        }
    return stats  #


stats_base = get_summary_stats(bold_base)
stats_alpha = get_summary_stats(bold_alpha)
stats_gamma = get_summary_stats(bold_gamma)

# %%
stat_1, stat_2 = 'min', 'max'

max_base = stats_base['ROI_0'][stat_1]
std_base = stats_base['ROI_0'][stat_2]
max_alpha = stats_alpha['ROI_0'][stat_1]
std_alpha = stats_alpha['ROI_0'][stat_2]
max_gamma = stats_gamma['ROI_0'][stat_1]
std_gamma = stats_gamma['ROI_0'][stat_2]

base_vals = np.array([max_base, std_base])
alpha_vals = np.array([max_alpha, std_alpha])
gamma_vals = np.array([max_gamma, std_gamma])

# bench_plot.arrow_plot(
#     baseline=base_vals,
#     changes=[
#         alpha_vals - base_vals,
#         gamma_vals - base_vals,
#     ],
#     names=['Baseline', 'Alpha', 'Gamma'],
#     bvals=[0],
# )

arrowprops = {}
default_arrowprops = dict(arrowstyle='<-', color='black')
default_arrowprops.update(arrowprops)
names = ['Base', 'Alpha', 'Gamma']
max_name_length = max([len(name) for name in names])
offset_x = np.maximum(0.01 * max_name_length, 0.005)
offset_y = 0.1
offset = 0.005

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(base_vals[0], base_vals[1], label='Base')
ax.scatter(alpha_vals[0], alpha_vals[1], label='Alpha')
ax.scatter(gamma_vals[0], gamma_vals[1], label='Gamma')
ax.annotate(
    '',
    xy=(base_vals[0], base_vals[1]),
    xytext=(alpha_vals[0], alpha_vals[1]),
    arrowprops=default_arrowprops,
)
ax.annotate(
    '',
    xy=(base_vals[0], base_vals[1]),
    xytext=(gamma_vals[0], gamma_vals[1]),
    arrowprops=default_arrowprops,
)
x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
y_range = ax.get_xlim()[1] - ax.get_xlim()[0]
ax.text(
    base_vals[0] + x_range * 0.01,
    base_vals[1] + y_range * 10,
    'Base',
    ha='left',
    va='top',
    size=12,
    fontdict=None,
)
ax.text(
    gamma_vals[0] + x_range * 0.01,
    gamma_vals[1] + y_range * 10,
    'Gamma',
    ha='left',
    va='top',
    size=12,
    fontdict=None,
)
ax.text(
    alpha_vals[0] + x_range * 0.01,
    alpha_vals[1] + y_range * 10,
    'Alpha',
    ha='left',
    va='top',
    size=12,
    fontdict=None,
)
ax.set_xlabel(stat_1)
ax.set_ylabel(stat_2)
ax.legend()
plt.show()

# %%
