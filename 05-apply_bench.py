# %%
# !%load_ext autoreload
# !%autoreload 2
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from bench import change_model
from IPython.display import Markdown, display
from scipy.stats import uniform
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

from dcsem.utils import stim_boxcar
from utils import set_style, simulate_bold

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

METHOD = 'PCA'
if METHOD == 'PCA':
    pca = pickle.load(open('models/pca.pkl', 'rb'))
elif METHOD == 'ICA':
    ica = pickle.load(open('models/ica.pkl', 'rb'))


# ======================================================================================
# %%
def calc_comps(method, **kwargs):
    # Define the allowed parameters
    allowed_keys = ['a01', 'a10', 'c0', 'c1']

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
    bold_true = simulate_bold(
        params,
        time=time,
        u=u,
        num_rois=NUM_ROIS,
    )
    bold_obsv = bold_true
    tmp_bold = np.concatenate([bold_obsv[:, :, 0], bold_obsv[:, :, 1]], axis=1)
    tmp_bold_c = tmp_bold - np.mean(tmp_bold, axis=1, keepdims=True)

    if method == 'PCA':
        components = pca.transform(tmp_bold_c)
    elif method == 'ICA':
        components = ica.transform(tmp_bold_c)

    return components


# Check if the function works
comps = calc_comps('PCA', a01=[0.5, 1.0], a10=[1.0, 0.7])
print('PCA components:\n', comps)

# %%
priors = {
    'a01': uniform(loc=bounds['a01'][0], scale=bounds['a01'][1] - bounds['a01'][0]),
    'a10': uniform(loc=bounds['a10'][0], scale=bounds['a10'][1] - bounds['a10'][0]),
    'c0': uniform(loc=bounds['c0'][0], scale=bounds['c0'][1] - bounds['c0'][0]),
    'c1': uniform(loc=bounds['c1'][0], scale=bounds['c1'][1] - bounds['c1'][0]),
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
noise_level = 0.0001
effect_size = 0.3
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
    cmap='Blues',
    cbar=False,
    square=True,
    xticklabels=mdl.model_names,
    yticklabels=mdl.model_names,
    ax=ax,
)
ax.set_xlabel('Inferred Change')
ax.set_ylabel('Actual Change')
plt.title('BENCH')
plt.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.tick_params(axis='y', which='minor', left=False, right=False)
plt.savefig('results/confusion_matrix_bench_new.png')
plt.show()


# %%
import pickle

with open('models/conf_bench_new.pkl', 'wb') as f:
    pickle.dump(conf_mat, f)

with open('models/mdl.pkl', 'wb') as f:
    pickle.dump(mdl, f)
# %%
