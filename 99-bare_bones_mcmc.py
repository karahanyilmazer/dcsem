# %%
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from IPython.display import Math, display
from tqdm import tqdm

from dcsem import models
from dcsem.utils import create_A_matrix, create_C_matrix, stim_boxcar

plt.rcParams['font.family'] = 'Times New Roman'


# %%
def get_one_layer_A(w01=0.4, w10=0.4, self_connections=-1):
    connections = []
    connections.append(f'R0, L0 -> R1, L0 = {w01}')  # ROI0 -> ROI1 connection
    connections.append(f'R1, L0 -> R0, L0 = {w10}')  # ROI1 -> ROI0 connection
    return create_A_matrix(
        num_rois=2,
        num_layers=1,
        paired_connections=connections,
        self_connections=self_connections,
    )


def get_one_layer_C(i0=1, i1=0):
    connections = []
    connections.append(f'R0, L0 = {i0}')  # Input --> ROI0 connection
    connections.append(f'R1, L0 = {i1}')  # Input --> ROI1 connection
    return create_C_matrix(num_rois=2, num_layers=1, input_connections=connections)


def simulate_bold(params, num_rois, time, u):
    # Define input arguments for defining A and C matrices
    A_param_names = ['w01', 'w10', 'self_connections']
    C_param_names = ['i0', 'i1']

    # Extract relevant parameters for A and C matrices from params
    A_kwargs = {k: params[k] for k in A_param_names if k in params}
    C_kwargs = {k: params[k] for k in C_param_names if k in params}

    # Create A and C matrices using extracted parameters
    A = get_one_layer_A(**A_kwargs)
    C = get_one_layer_C(**C_kwargs)

    dcm = models.DCM(
        num_rois,
        params={
            'A': A,
            'C': C,
            **params,
        },
    )
    return dcm.simulate(time, u)[0]


def add_noise(bold_true, snr_db):
    snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear scale
    sigma = np.max(bold_true) / snr_linear
    return bold_true + np.random.normal(0, sigma, bold_true.shape)


def filter_params(params, keys):
    return {k: params[k] for k in keys}


def initialize_parameters(bounds, params_to_sim, random=False):
    initial_values = []
    for param in params_to_sim:
        if random:
            initial_values.append(np.random.uniform(*bounds[param]))
        else:
            initial_values.append(np.mean(bounds[param]))

    return initial_values


def mcmc(
    posterior: Callable,
    args: Tuple,
    p0: np.ndarray,
    cov=None,
    bounds=None,
    step_size=1e0,
    n_samples=1000,
    burnin=100,
    skips=10,
):
    """
    MCMC sampling of a distribution
    Args:

        posterior: The log pdf of distribution (Callable function like f(p0, *args)
        args: parameters of the distribution (tuple)
        p0: initial point (array like)
        cov: covariance of samples
        bounds: sample bounds (n_params, 2)

    Returns:
        Samples, pos
    """
    p0 = np.atleast_1d(p0)
    if cov is None:
        cov = np.eye(len(p0))

    cov = np.atleast_2d(cov)
    min_eig = np.linalg.eigvalsh(cov)[0]
    correction = np.maximum(1e-14 - min_eig, 0)
    L1 = np.linalg.cholesky(cov + correction * np.eye(cov.shape[0])) / np.sqrt(len(p0))

    if bounds is None:
        bounds = np.array([[-np.inf, np.inf]] * len(p0))

    assert (p0 >= bounds[:, 0]).all() and (
        p0 <= bounds[:, 1]
    ).all(), "P0 doesnt fit to the bounds"

    current = np.array(p0)
    prob_cur = posterior(current, *args)
    samples = []
    all_probs = []
    jumps = n_samples * skips + burnin
    for j in tqdm(range(jumps)):
        proposed = current + L1 @ np.random.randn(*current.shape) * step_size
        if (proposed >= bounds[:, 0]).all() and (proposed <= bounds[:, 1]).all():
            prob_next = posterior(proposed, *args)
            diff = np.clip(
                prob_next - prob_cur, -700, 700
            )  # to avoid overflow/underflow
            if np.exp(diff) > np.random.rand():
                current = proposed
                prob_cur = prob_next

        samples.append(current)
        all_probs.append(prob_cur)
    return (
        np.squeeze(np.stack(samples, axis=0))[burnin::skips],
        np.array(all_probs)[burnin::skips],
    )


def log_probability(param_vals, param_names, bounds, bold_observed, num_rois):

    params = dict(zip(param_names, param_vals))

    # Check if the proposed parameters are within bounds
    for key, val in params.items():
        if not bounds[key][0] < val < bounds[key][1]:  # Uniform prior
            return -np.inf

    bold_est = simulate_bold(params, time=time, u=u, num_rois=num_rois)
    likelihood = -0.5 * np.sum((bold_observed - bold_est) ** 2)  # Gaussian likelihood
    return likelihood


def plot_bold_signals(time, bold_true, bold_noisy, bold_estimated):
    num_rois = bold_true.shape[1]
    _, axs = plt.subplots(1, num_rois, figsize=(10, 4))
    for i in range(num_rois):
        axs[i].plot(time, bold_noisy[:, i], label='Observed', lw=2)
        axs[i].plot(time, bold_true[:, i], label='Ground Truth', lw=2)
        axs[i].plot(
            time,
            bold_estimated[:, i],
            label='Estimated',
            ls='--',
            lw=2,
            c='tomato',
        )
        axs[i].set_title(f'ROI {i}')
        axs[i].set_xlabel('Time (s)')
        axs[i].legend()

    axs[0].set_ylabel('BOLD Signal')

    plt.tight_layout()
    plt.show()


def plot_mcmc_results(params_to_est, samples, true_params):
    num_params = len(params_to_est)
    _, axs = plt.subplots(1, num_params, figsize=(15, 5))
    axs = [axs] if num_params == 1 else axs
    for i, param in enumerate(params_to_est):
        axs[i].hist(samples[:, i], bins=30, alpha=0.7, label=f"{param} Samples")
        axs[i].axvline(
            true_params[param], color="red", linestyle="--", label="True Value"
        )
        axs[i].set_title(param)
        axs[i].legend()

    plt.tight_layout()
    plt.show()


# %%
# ======================================================================================
# Bilinear neural model parameters
NUM_LAYERS = 1
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])  # Input stimulus

# Parameters to set and estimate
params_to_set = ['A_w01', 'A_w10']
params_to_est = ['A_w01']

# Ground truth parameter values
true_params = {
    'A_w01': 0.7,
    'A_w10': 0.7,
}
bounds = {
    'A_w01': (0.0, 1.0),
    'A_w10': (0.0, 1.0),
}

# Initial values for the parameters to estimate
initial_values = initialize_parameters(bounds, params_to_est)
print('Initial guesses:\t', dict(zip(params_to_est, initial_values)))
# ======================================================================================
# %%
snr_range = np.linspace(0.001, 50, 20)
estimated_vals = {key: [] for key in params_to_est}
step_size = 0.1  # Step size for `mcmc`
n_samples = 10000  # Sampling steps
n_burn = 1000  # Burn-in steps

bold_true = simulate_bold(true_params, time=time, u=u, num_rois=NUM_ROIS)
bold_obsv = bold_true
# bold_obsv = add_noise(bold_true, snr_db=snr_range[-1])

samples_mcmc, probs_mcmc = mcmc(
    posterior=log_probability,
    args=(params_to_est, bounds, bold_obsv, NUM_ROIS),
    p0=initial_values,
    bounds=np.array(list(bounds.values())),
    step_size=step_size,
    n_samples=n_samples,
    burnin=n_burn,
    skips=1,
)
samples_mcmc = samples_mcmc.reshape(-1, 1)

mean_mcmc = np.mean(samples_mcmc, axis=0)
std_mcmc = np.std(samples_mcmc, axis=0)

print(f'MCMC Results:\tMean: {mean_mcmc[0]:.2f}, Std: {std_mcmc[0]:.2f}')

# %%
plt.figure(figsize=(12, 6))
plt.hist(samples_mcmc, bins=30, alpha=0.5, label='MCMC', density=True)
plt.axvline(
    true_params[params_to_est[0]],
    color='r',
    linestyle='--',
    label=f'True {params_to_est[0]}',
)
plt.xlabel(params_to_est[0])
plt.ylabel('Density')
plt.legend()
plt.show()
# %%
fig = corner(
    samples_mcmc,
    labels=[params_to_est[0]],
    truths=[true_params[params_to_est[0]]],
)
plt.show()

# %%
inds = np.random.randint(len(samples_mcmc), size=100)
for ind in inds:
    sample = samples_mcmc[ind]
    sample = np.atleast_2d(sample)
    params = dict(zip(params_to_est, sample))
    plt.plot(
        time,
        simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)[:, 0],
        'C4',
        alpha=0.1,
    )
# plt.errorbar(x, y, yerr=yerr, fmt='.k', capsize=0)
plt.plot(time, bold_obsv[:, 0], color='k', label='observed')
# plt.plot(time, bold_true[:, 0], color='k', label='truth')
plt.legend()
plt.xlim(0, 100)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# %%
for i in range(len(params_to_est)):
    mcmc_est = np.percentile(samples_mcmc[:, i], [16, 50, 84])
    q = np.diff(mcmc_est)
    txt = rf"\mathrm{{{params_to_est[i]}}} = {mcmc_est[1]:.3f}_{{-{q[0]:.3f}}}^{{{q[1]:.3f}}}"
    display(Math(txt))

# %%
