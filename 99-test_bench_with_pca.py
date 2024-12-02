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
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

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
def calc_comps(method, **kwargs):
    # Define the allowed parameters
    allowed_keys = ['w01', 'w10', 'i0', 'i1']

    # Find invalid keys
    invalid_keys = [key for key in kwargs.keys() if key not in allowed_keys]

    # Assert that all keys are allowed
    assert (
        not invalid_keys
    ), f'Invalid parameter keys: {invalid_keys}. Allowed keys are: {allowed_keys}.'
    # Filter all arguments that are not None
    params = {}
    for key, val in kwargs.items():
        if key == 'method':
            continue
        if val is not None:
            # Convert the values to a numpy array
            if not isinstance(val, (list, np.ndarray)):
                val = [val]
            if not isinstance(val, np.ndarray):
                val = np.array(val)

            params[key] = val

    # Assert that all values have the same length
    lengths = [len(v) for v in params.values()]
    assert all(
        length == lengths[0] for length in lengths
    ), 'All values must have the same length!'

    # Initialize the BOLD signals
    bold_true = simulate_bold_multi(
        params,
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


# Test the function
comps = calc_comps('PCA', w01=[0.5, 1.0], w10=[1.0, 0.7])
print(comps)

# %%
priors = {
    'w01': uniform(loc=bounds['w01'][0], scale=bounds['w01'][1] - bounds['w01'][0]),
    'w10': uniform(loc=bounds['w10'][0], scale=bounds['w10'][1] - bounds['w10'][0]),
    'i0': uniform(loc=bounds['i0'][0], scale=bounds['i0'][1] - bounds['i0'][0]),
    'i1': uniform(loc=bounds['i1'][0], scale=bounds['i1'][1] - bounds['i1'][0]),
}

tr = change_model.Trainer(
    forward_model=calc_comps,
    priors=priors,
    kwargs={'method': 'PCA'},
    measurement_names=['PC1', 'PC2', 'PC3', 'PC4'],
)
mdl = tr.train(n_samples=5000, verbose=True)

# %%
n_test_samples = 2000
noise_level = 0.02
effect_size = 0.1
n_repeats = 50

true_change, data, data2, sn = tr.generate_test_samples(
    n_samples=n_test_samples,
    n_repeats=n_repeats,
    effect_size=effect_size,
    noise_std=noise_level,
)

probs, infered_change_bench, amount, _ = mdl.infer(data, data2 - data, sn)
print('Accuracy:', np.mean(infered_change_bench == true_change))
# %%
conf_mat = confusion_matrix(true_change, infered_change_bench, normalize='true')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
heatmap(
    conf_mat,
    annot=True,
    fmt='.2f',
    # cmap='coolwarm',
    cbar=False,
    square=True,
    xticklabels=mdl.model_names,
    yticklabels=mdl.model_names,
    ax=ax,
)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.title('BENCH')
plt.savefig('results/confusion_matrix_bench.png')
plt.show()


# %%
