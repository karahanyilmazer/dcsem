# %%
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Math, display
from tqdm import tqdm

from dcsem.utils import stim_boxcar
from utils import filter_params, initialize_parameters, simulate_bold

plt.rcParams['font.family'] = 'Times New Roman'


# %%
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

print('Estimating:\t\t', filter_params(true_params, params_to_est))
print('Initial guesses:\t', dict(zip(params_to_est, initial_values)))
# ======================================================================================
# %%
step_size = 0.001  # Step size for MCMC
n_samples = 300  # Sampling steps
n_burn = 100  # Burn-in steps

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
plt.figure()
plt.plot(samples_mcmc)
plt.axhline(
    true_params[params_to_est[0]],
    color='tomato',
    linestyle='--',
    label='True Value',
)
plt.xlabel('Step')
plt.ylabel('Parameter Value')
plt.legend()
plt.show()

# %%
plt.figure()
inds = np.random.randint(len(samples_mcmc), size=100)
for ind in inds:
    sample = samples_mcmc[ind]
    params = dict(zip(params_to_est, sample))
    plt.plot(
        time,
        simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)[:, 0],
        'C4',
        alpha=0.1,
    )
plt.plot(time, bold_obsv[:, 0], color='k', label='Observed')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(0, 100)
plt.legend()
plt.show()

# %%
for i in range(len(params_to_est)):
    mcmc_est = np.percentile(samples_mcmc[:, i], [16, 50, 84])
    q = np.diff(mcmc_est)
    txt = rf"\mathrm{{{params_to_est[i]}}} = {mcmc_est[1]:.3f}_{{-{q[0]:.3f}}}^{{{q[1]:.3f}}}"
    display(Math(txt))

# %%
