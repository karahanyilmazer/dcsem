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


def get_one_layer_A(L0=0.2):
    connections = [f'R0, L0 -> R1, L0 = {L0}']  # ROI0 -> ROI1 connection
    return utils.create_A_matrix(num_rois, num_layers, connections, self_connections=-1)


def get_one_layer_C(L0=1):
    connections = [f'R0, L0 = {L0}']  # Input --> ROI0 connection
    return utils.create_C_matrix(num_rois, num_layers, connections)


# Function to run simulations for different parameter types
def simulate_dcm(param_name, param_values):
    bold_signals = []
    for value in tqdm(param_values, desc=f"Simulating for {param_name}"):
        if param_name == 'A_L0':
            A = get_one_layer_A(value)
            C = get_one_layer_C()
        elif param_name == 'C_L0':
            A = get_one_layer_A()
            C = get_one_layer_C(value)
        else:
            A = get_one_layer_A()
            C = get_one_layer_C()
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

    # Define the title of the legend
    if param_name in ['alpha', 'gamma', 'kappa']:
        title = fr'$\{param_name}$'
    else:
        title = 'A_L0'.split('_')
        title = fr'${title[0]}_{{{title[1]}}}$'

    # Create legend and place it outside the plot to the right
    legend = axs[row, 1].legend(
        bbox_to_anchor=(1.05, 0.5),
        loc='center left',
        borderaxespad=0.0,
        title=title,
        fontsize='small',
        ncol=1,
    )

    # Set the color of the legend title to the base color
    legend.get_title().set_color(base_color)


def plot_param_change(figsize=(10, 12)):
    # Create the plot
    _, axs = plt.subplots(len(params), num_rois, figsize=figsize)

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
# Input
time = np.arange(100)  # Time vector (seconds)
u = utils.stim_boxcar([[0, 30, 1]])  # Stimulus function (onset, duration, amplitude)

# Connectivity parameters
num_rois = 2
num_layers = 1

# Parameters to vary
params = {
    'kappa': np.linspace(1, 2, 9),
    'gamma': np.linspace(0, 1, 9),
    'alpha': np.linspace(0.1, 1, 9),
    'A_L0': np.linspace(0, 1, 9),
    'C_L0': np.linspace(0, 1, 9),
}

# Run simulations for each parameter set
bold_tcs = {param: simulate_dcm(param, values) for param, values in params.items()}

plot_param_change()

# %%
