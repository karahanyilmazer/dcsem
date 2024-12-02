# %%
# !%load_ext autoreload
# !%autoreload 2
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from bench import acquisition, change_model, diffusion_models, model_inversion
from IPython.display import Markdown, display
from scipy.stats import uniform

from dcsem.utils import stim_boxcar
from utils import initialize_parameters, simulate_bold_multi

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


# ======================================================================================
# %%
def calc_comps(param_names, method, param_vals):
    # Ensure param_names is a list, even for a single key
    if isinstance(param_names, str):
        param_names = [param_names]

    # If param_vals is not iterable (e.g., scalar), map directly
    if not isinstance(param_vals, (list, np.ndarray)):
        param_vals = [param_vals]  # Convert scalar to a list for consistent mapping

    # Handle single key and single value case
    if len(param_names) == 1:
        param_dict = {param_names[0]: np.array(param_vals)}

    else:
        # Handle multi-key case
        param_dict = dict(zip(param_names, param_vals))

    # Initialize the BOLD signals
    bold_true = simulate_bold_multi(
        param_dict,
        time=time,
        u=u,
        num_rois=NUM_ROIS,
    )
    bold_obsv = bold_true
    tmp_bold = np.concatenate([bold_obsv[:, :, 0], bold_obsv[:, :, 1]], axis=1)
    tmp_bold_c = tmp_bold - np.mean(tmp_bold, axis=1, keepdims=True)

    if method == 'PCA':
        pca = pickle.load(open('models/pca.pkl', 'rb'))
        components = pca.transform(tmp_bold_c)
    elif method == 'ICA':
        ica = pickle.load(open('models/ica.pkl', 'rb'))
        components = ica.transform(tmp_bold_c)

    return components


initial_values = initialize_parameters(bounds, params_to_set, random=True)
comps = calc_comps(['w01'], 'PCA', np.array([0.5]))

# %%

priors = {
    'param_vals': uniform(
        loc=bounds['w01'][0], scale=bounds['w01'][1] - bounds['w01'][0]
    ),
}

tr = change_model.Trainer(
    forward_model=calc_comps,
    priors=priors,
    kwargs={'method': 'PCA', 'param_names': ['w01']},
    # param_prior_dists=priors_bench,
    # summary_names=summary_measures.summary_names(acq, shm_deg),
)
mdl = tr.train(n_samples=100, verbose=True)

# %%
