# %%
# !%load_ext autoreload
# !%autoreload 2
from random import choice

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from scipy.optimize import minimize
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
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


def loss(params, true_params, true_bold):
    a01, a10, c0, c1 = params
    current_params = dict(true_params)
    current_params['a01'] = a01
    current_params['a10'] = a10
    current_params['c0'] = c0
    current_params['c1'] = c1
    obs_bold = simulate_bold(current_params, time=time, u=u, num_rois=NUM_ROIS)

    return np.mean((true_bold - obs_bold) ** 2)


# %%#
display(Markdown('## Run the simulation'))
n_samples = 500
change_amount = 0.1
param_vals = []
true_change = []
inferred_change = []
change_thr = 0.1

initial_guess = [0.3, 0.4, 0.5, 0.6]

for sample_i in tqdm(range(n_samples)):
    # Initialize the parameters
    param_vals.append(initialize_parameters(bounds, params_to_set, random=True))

    # Get the latest sample
    sample = param_vals[-1]

    # Create a dictionary of unchanged parameters
    params = dict(zip(params_to_set, sample))
    true_bold = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)

    # Call the minimization routine
    result = minimize(
        loss,
        initial_guess,
        args=(params, true_bold),
        options={'gtol': 1e-12, 'ftol': 1e-12},
        method='L-BFGS-B',
        bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
    )

    # Extract the optimized parameter value
    first_est = np.array(result.x)

    # Make a random choice for the change
    true_change.append(choice([0, 1, 2, 3, 4]))

    # No change
    if true_change[-1] == 0:
        # Call the minimization routine
        result = minimize(
            loss,
            initial_guess,
            args=(params, true_bold),
            options={'gtol': 1e-12, 'ftol': 1e-12},
            method='L-BFGS-B',
            bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
        )

        # Extract the optimized parameter value
        second_est = np.array(result.x)

    # Change in one parameter
    else:
        i = true_change[-1] - 1
        param = params_to_set[i]

        # Introduce a change in one parameter
        sample[i] = sample[i] + change_amount

        # Check if the parameter is still within bounds
        if sample[i] > bounds[param][1]:
            sample[i] = bounds[param][1]

        # Create a new dictionary for the changed parameters
        params = dict(zip(params_to_set, sample))
        true_bold = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)

        # Call the minimization routine
        result = minimize(
            loss,
            initial_guess,
            args=(params, true_bold),
            options={'gtol': 1e-12, 'ftol': 1e-12},
            method='L-BFGS-B',
            bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
        )

        # Extract the optimized parameter value
        second_est = np.array(result.x)

    diff = np.abs(first_est - sample)
    largest_change = np.max(diff)
    if largest_change > change_thr:
        inferred_change.append(np.argmax(diff))
    else:
        inferred_change.append(0)


# %%
labels = ['No Change', 'a01', 'a10', 'c0', 'c1']
conf_mat = confusion_matrix(true_change, inferred_change, normalize='true')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
heatmap(
    conf_mat,
    annot=True,
    fmt='.2f',
    # cmap='coolwarm',
    cbar=False,
    square=True,
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
)
ax.set_xlabel('Inferred Change')
ax.set_ylabel('Actual Change')
plt.title('Model Inversion')
plt.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.tick_params(axis='y', which='minor', left=False, right=False)
plt.savefig('results/confusion_matrix_model_inversion.png')
plt.show()

# %%
labels = ['No Change', 'a01', 'a10', 'c0', 'c1']
conf_mat = confusion_matrix(true_change, inferred_change, normalize='true')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
heatmap(
    conf_mat,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    cbar=False,
    square=True,
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
)
ax.set_xlabel('Inferred Change')
ax.set_ylabel('Actual Change')
plt.title('Model Inversion')
plt.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.tick_params(axis='y', which='minor', left=False, right=False)
plt.savefig('results/confusion_matrix_model_inversion.png')
plt.show()
# %%
import pickle

with open('models/conf_inversion.pkl', 'wb') as f:
    pickle.dump(conf_mat, f)

# %%
