# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from tqdm import tqdm

from dcsem import models, utils

plt.rcParams['font.family'] = 'Times New Roman'


# Function to lighten colors progressively
def lighten_color(color, factor):
    white = np.array([1, 1, 1, 1])  # RGBA for white
    return (1 - factor) * np.array(to_rgba(color)) + factor * white


# Function to run simulations for different parameter types
def simulate_dcm(param_name, param_values):
    bold_signals = []
    for value in tqdm(param_values, desc=f"Simulating for {param_name}"):
        dcm = models.DCM(num_rois, params={'A': A, 'C': C, param_name: value})
        bold_signals.append(dcm.simulate(time, u)[0])
    return bold_signals


# Plotting function with color gradation
def plot_bold(axs, bold_tcs, param_name, param_values, row, base_color):
    for i, (bold, value) in enumerate(zip(bold_tcs, param_values)):
        # Lighter color for each successive line
        light_color = lighten_color(base_color, i / len(param_values))
        axs[row, 0].plot(time, bold[:, 0], label=f'{value:.2f}', color=light_color)
        axs[row, 1].plot(time, bold[:, 1], label=f'{value:.2f}', color=light_color)

    # Create legend and place it outside the plot to the right
    legend = axs[row, 1].legend(
        bbox_to_anchor=(1.05, 0.5),  # Coordinates for positioning the legend
        loc='center left',  # Location of the legend relative to bbox_to_anchor
        borderaxespad=0.0,  # Padding between the axes and the legend
        title=fr'$\{param}$',  # Title for the legend
        fontsize='small',  # Set the font size for legend entries
        ncol=1,  # Make sure legend entries are in one column
    )

    # Set the color of the legend title to the base color
    legend.get_title().set_color(base_color)


# %%
# Input
time = np.arange(100)  # Time vector (seconds)
u = utils.stim_boxcar([[0, 30, 1]])  # Stimulus function (onset, duration, amplitude)

# Connectivity parameters
num_rois = 2
num_layers = 1
connections = ['R0, L0 -> R1, L0 = 0.2']  # ROI0 -> ROI1 connection
A = utils.create_A_matrix(num_rois, num_layers, connections, self_connections=-1)
C = utils.create_C_matrix(num_rois, num_layers, ['R0, L0 = 1.0'])

# Parameters to vary
params = {
    'kappa': np.linspace(1, 2, 9),
    'gamma': np.linspace(0, 1, 9),
    'alpha': np.linspace(0.1, 1, 9),
}

# Run simulations for each parameter set
bold_tcs = {param: simulate_dcm(param, values) for param, values in params.items()}

# %%
# Create the plot
fig, axs = plt.subplots(len(params), num_rois, figsize=(10, 8))

# Get the default color cycle (tab10 colormap has 10 distinct colors)
cmap = plt.get_cmap('tab10')

# Plot for each parameter with the corresponding color from the cycle
for i, (param, values) in enumerate(params.items()):
    base_color = cmap(i)  # Get the base color from the colormap
    plot_bold(axs, bold_tcs[param], param, values, i, base_color)

# Set titles and labels
for i in range(num_rois):
    axs[0, i].set_title(f'ROI {i}')
    axs[len(params) - 1, i].set_xlabel('Time (s)')

for i in range(len(params)):
    axs[i, 0].set_ylabel('BOLD Signal')

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()


# %%
