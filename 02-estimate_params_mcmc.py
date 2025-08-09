# %%
# !%load_ext autoreload
# !%autoreload 2
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.linalg import inv
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess

from dcsem.utils import stim_boxcar
from utils import (
    add_noise,
    add_underscore,
    filter_params,
    get_param_colors,
    initialize_parameters,
    set_style,
    simulate_bold,
)

set_style()


# %%
def plot_bold_signals(time, bold_true, bold_noisy, bold_estimated):
    """
    Plot the observed, true, and estimated BOLD signals for each region of
    interest (ROI).

    Args:
        time (ndarray): A 1D array representing time points.
        bold_true (ndarray): The ground truth BOLD signal, a 2D array where
                             each column corresponds to an ROI.
        bold_noisy (ndarray): The observed BOLD signal with added noise,
                              in the same shape as bold_true.
        bold_estimated (ndarray): The estimated BOLD signal, also a 2D array
                                  with the same shape as bold_true.
    """
    num_rois = bold_true.shape[1]
    _, axs = plt.subplots(1, num_rois, figsize=(10, 4))
    for i in range(num_rois):
        axs[i].plot(time, bold_noisy[:, i], label="Observed", lw=2)
        axs[i].plot(time, bold_true[:, i], label="Ground Truth", lw=2)
        axs[i].plot(
            time,
            bold_estimated[:, i],
            label="Estimated",
            ls="--",
            lw=2,
            c="tomato",
        )
        axs[i].set_title(f"ROI {i}")
        axs[i].set_xlabel("Time (s)")
        axs[i].legend()

    axs[0].set_ylabel("BOLD Signal")

    plt.tight_layout()
    plt.show()


def log_probability(param, bold_signal, est_name, all_params, bounds):
    """
    Log-probability function for MCMC.

    Args:
        param (list): Parameter values to evaluate.
        bold_signal (ndarray): Observed noisy BOLD signal.
        est_name (list): Names of parameters to estimate.
        all_params (dict): All parameters, including fixed ones.
        bounds (list of tuples): Bounds for each parameter.

    Returns:
        float: Log-probability value.
    """
    # Check if parameters are within bounds
    for p, v, (low, high) in zip(est_name, param, bounds):
        if not (low <= v <= high):
            return -np.inf  # Outside bounds → log-probability is -∞

    # Update parameter dictionary
    for p, v in zip(est_name, param):
        all_params[p] = v

    # Simulate BOLD signal
    bold_simulated = simulate_bold(
        all_params,
        time=time,
        u=u,
        num_rois=NUM_ROIS,
    )

    # Compute likelihood (negative squared error)
    residuals = bold_simulated - bold_signal
    likelihood = -0.5 * np.sum(residuals**2)

    return likelihood


time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])

# Model parameters
NUM_ROIS = 2
NUM_LAYERS = 1
RANDOM = False

# Parameters to use in the simulation and estimation
params_to_set = ["a01", "a10", "c0", "c1"]
params_to_est = ["a01"]

# Ground truth parameter values
true_params = {
    "a01": 0.6,
    "a10": 0.4,
    "c0": 0.5,
    "c1": 0.5,
}
true_params = filter_params(true_params, params_to_set)

# Bounds for the parameters
bounds = {
    "a01": (0, 1),
    "a10": (0, 1),
    "c0": (0, 1),
    "c1": (0, 1),
}
bounds = filter_params(bounds, params_to_est)

initial_values = initialize_parameters(bounds, params_to_est, random=RANDOM)
bounds = [(bounds[param]) for param in params_to_est]

bold_true = simulate_bold(
    true_params,
    time=time,
    u=u,
    num_rois=NUM_ROIS,
)
bold_noisy = add_noise(bold_true, snr_db=100)

# Number of dimensions (parameters to estimate)
n_dim = len(params_to_est)

# Number of walkers (chains)
n_walkers = 32

# Burn-in phase + Sampling
n_burn = 500  # Samples to discard
n_samples = 1000  # Samples to keep

# Initialize walkers around the initial guess
initial_guess = [
    initial_values + 0.01 * np.random.randn(n_dim) for _ in range(n_walkers)
]

# Run MCMC
sampler = emcee.EnsembleSampler(
    n_walkers,
    n_dim,
    log_probability,
    args=(bold_noisy, params_to_est, true_params.copy(), bounds),
)
sampler.run_mcmc(initial_guess, n_burn + n_samples, progress=True)

# %%
# Discard burn-in samples and reshape
samples = sampler.get_chain(discard=n_burn, flat=True)

# %%
# Compute mean and standard deviation for each parameter
means = np.mean(samples, axis=0)
stds = np.std(samples, axis=0)
estimated_params = dict(zip(params_to_est, means))

# Print results
print("True Parameters:", true_params)
print("Estimated Parameters (MCMC):", estimated_params)

# Visualize posterior distributions
corner.corner(
    samples,
    labels=params_to_est,
    truths=[true_params[p] for p in params_to_est],
)
plt.show()

# %%
