# %%
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from dcsem.utils import stim_boxcar
from utils import simulate_bold

plt.rcParams['font.family'] = 'Times New Roman'


def add_underscore(param):
    # Use regex to insert an underscore before a digit sequence and group digits for LaTeX
    latex_param = re.sub(r'(\D)(\d+)', r'\1_{\2}', param)
    return r"${" + latex_param + r"}$"


# Function to lighten colors progressively
def lighten_color(color, factor):
    white = np.array([1, 1, 1, 1])  # RGBA for white
    return (1 - factor) * np.array(to_rgba(color)) + factor * white


# Plotting function with color gradation
def plot_bold(axs, bold_tcs, param_name, param_values, row, base_color):
    for i, (bold, value) in enumerate(zip(bold_tcs, param_values)):
        # Lighter color for each successive line
        light_color = lighten_color(base_color, 0.9 * (i / len(param_values)))
        axs[row, 0].plot(
            time, bold[:, 0], label=f'{value:.2f}', lw=1.5, color=light_color
        )
        axs[row, 1].plot(
            time, bold[:, 1], label=f'{value:.2f}', lw=1.5, color=light_color
        )

    # Define the title of the legend
    if param_name in ['alpha', 'gamma', 'kappa']:
        title = fr'$\{param_name}$'
    else:
        # title = param_name.split('_')
        # title = fr'${title[0]}_{{{title[1]}}}$'
        title = add_underscore(param_name)

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


def plot_param_change(figsize=(10, 12), save=False):
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
    if save:
        plt.savefig('img/presentation/param_change.png', dpi=600)
    plt.show()


# %%
if __name__ == '__main__':
    # Input
    time = np.arange(100)  # Time vector (seconds)
    # Stimulus function (onset, duration, amplitude)
    u = stim_boxcar([[0, 30, 1]])

    # Connectivity parameters
    num_rois = 2
    num_layers = 1

    # Parameters to vary
    params = {
        'a01': np.linspace(0, 1, 9),
        'a10': np.linspace(0, 1, 9),
        'c0': np.linspace(0, 1, 9),
        'c1': np.linspace(0, 1, 9),
    }

    # Run simulations for each parameter set
    bold_tcs = {
        param: simulate_bold({param: values}, 2, time, u)
        for param, values in params.items()
    }

    plot_param_change((13, 9), save=True)

# %%
