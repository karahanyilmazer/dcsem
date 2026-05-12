# %%
# !%load_ext autoreload
# !%autoreload 2
"""
MCMC Parameter Estimation for DCM Models

This script demonstrates Bayesian parameter estimation for Dynamic Causal Models
using Markov Chain Monte Carlo (MCMC) methods with the emcee package.
"""
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess

from dcsem import models, utils

plt.rcParams["font.family"] = "Helvetica"


# %%
# Configuration and Parameters
# =============================================================================

# Model parameters
NUM_ROIS = 2
NUM_LAYERS = 1

# Simulation parameters
TIME_POINTS = 100
STIMULUS_CONFIG = [[0, 30, 1]]  # [onset, duration, amplitude]
SNR_DB = 100  # Signal-to-noise ratio in dB

# Parameters to use in the simulation and estimation
PARAMS_TO_SIM = ["alpha", "kappa", "gamma", "A_L0", "C_L0"]
PARAMS_TO_EST = ["alpha"]

# Ground truth parameter values
TRUE_PARAMS = {
    "alpha": 0.2,
    "kappa": 1.5,
    "gamma": 0.5,
    "A_L0": 0.8,
    "C_L0": 0.7,
}

# Parameter bounds
PARAM_BOUNDS = {
    "alpha": (0.2, 1.0),
    "kappa": (1.0, 2.0),
    "gamma": (0.0, 1.0),
    "A_L0": (0.0, 1.0),
    "C_L0": (0.0, 1.0),
}

# MCMC parameters
N_WALKERS = 32  # Number of walkers (chains)
N_BURN = 500  # Burn-in samples to discard
N_SAMPLES = 1000  # Samples to keep

print("Configuration loaded:")
print(f"  - Model: {NUM_ROIS} ROIs, {NUM_LAYERS} layers")
print(f"  - Parameters to estimate: {PARAMS_TO_EST}")
print(f"  - MCMC: {N_WALKERS} walkers, {N_BURN} burn-in, {N_SAMPLES} samples")

# %%
# Utility Functions for Matrix Generation
# =============================================================================


def get_one_layer_A(L0=0.2):
    """
    Generate the connectivity matrix (A matrix) for one layer, representing
    the connections between regions of interest (ROIs).

    Args:
        L0 (float, optional): Strength of the connection from ROI 0, layer 0,
                              to ROI 1, layer 0. Defaults to 0.2.

    Returns:
        ndarray: The A matrix representing connectivity between different
                 regions (ROIs) in the model. This is typically a 2D array
                 where rows and columns correspond to regions and layers,
                 with the strength of connections specified.
    """
    connections = [f"R0, L0 -> R1, L0 = {L0}"]  # ROI0 -> ROI1 connection
    return utils.create_A_matrix(NUM_ROIS, NUM_LAYERS, connections, self_connections=-1)


def get_one_layer_C(L0=1):
    """
    Generate the input connectivity matrix (C matrix) for one layer.

    Args:
        L0 (int, optional): Input strength for layer 0 of region 0 (ROI0).
                            Defaults to 1.

    Returns:
        ndarray: The C matrix representing input connections for the model.
                 This is typically a 2D array where the strength of external
                 inputs to different regions or layers is specified.
    """
    connections = [f"R0, L0 = {L0}"]  # Input --> ROI0 connection
    return utils.create_C_matrix(NUM_ROIS, NUM_LAYERS, connections)


# %%
# Parameter Handling Functions
# =============================================================================


def initialize_parameters(bounds, params_to_sim):
    """
    Initialize parameters for the simulation and estimation.

    Args:
        bounds (dict): Bounds for the parameters.
        params_to_sim (list): Parameters used for simulation.

    Returns:
        dict: A dictionary of initial parameter values.
    """
    initial_values = []
    for param in params_to_sim:
        initial_values.append(np.random.uniform(*bounds[param]))

    return initial_values


def filter_params(params, keys):
    """
    Filter a dictionary to only include the specified keys.

    Args:
        params (dict): A dictionary of parameters.
        keys (list): A list of keys to keep.

    Returns:
        dict: A dictionary containing only the specified keys.
    """
    return {k: params[k] for k in keys}


# %%
# Simulation Functions
# =============================================================================


def simulate_bold(params, time, u, num_rois, **kwargs):
    """
    Simulate the BOLD signal using a DCM model with the given parameters.

    Args:
        params (dict): A dictionary of parameters for the simulation, including:
            - 'A_L0' (float): Connectivity parameter for layer 0 (optional, default=0.2).
            - 'C_L0' (float): Input parameter for layer 0 (optional, default=1.0).
            Other model-specific parameters may also be included.
        time (ndarray): Time points for the simulation.
        u (ndarray): External stimulus function.
        num_rois (int): Number of regions of interest.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: Simulated BOLD signal for each region of interest, typically
                 a 2D array (time points x ROIs).
    """
    dcm = models.DCM(
        num_rois,
        params={
            "A": get_one_layer_A(params.get("A_L0", 0.2)),
            "C": get_one_layer_C(params.get("C_L0", 1.0)),
            **params,
        },
    )
    return dcm.simulate(time, u)[0]


# %%
# Optimization Functions
# =============================================================================


def objective(param_vals, param_names, time, u, bold_signal):
    """
    Objective function to minimize the difference between simulated and observed BOLD
    signals.

    Args:
        param_vals (list or ndarray): Initial values of the parameters to be optimized.
        param_names (list of str): Names of the parameters corresponding to
                                   `param_vals`.
        time (ndarray): Time points for the BOLD signal.
        u (ndarray): Stimulus function, typically representing external input.
        bold_signal (ndarray): The observed BOLD signal with noise, typically a 2D
                               a 2D array (time points x ROIs).

    Returns:
        loss (float): The sum of squared errors between the simulated and real BOLD
                      signals.
    """

    # Map the parameter values to their names
    params = dict(zip(param_names, param_vals))
    A = get_one_layer_A(params.get("A_L0", 0.2))
    C = get_one_layer_C(params.get("C_L0", 1.0))
    num_rois = bold_signal.shape[1]
    bold_simulated = simulate_bold(params, time=time, u=u, A=A, C=C, num_rois=num_rois)

    # print('params:', params)
    # print(bold_simulated.shape, bold_signal.shape)

    loss = np.mean((bold_simulated - bold_signal) ** 2)

    # Compute the sum of squared errors
    return loss


def estimate_parameters(
    initial_values,
    param_names,
    bounds=None,
    normalize=True,
    **kwargs,
):
    """
    Estimate parameters by minimizing the objective function using the L-BFGS-B method.

    Args:
        initial_values (list or ndarray): Initial guesses for the parameter values to
                                          be optimized.
        param_names (list of str): Names of the parameters to be estimated.
        bounds (list of tuples): Bounds for each parameter, where each tuple contains
                                 the lower and upper bounds.
        normalize (bool): Whether to normalize the Hessian matrix with estimated sigma.
                          Defaults to True.
        **kwargs: Additional keyword arguments containing the following:
            - time (ndarray): Time points for the BOLD signal.
            - u (ndarray): Stimulus function.
            - bold_signal (ndarray): Observed BOLD signal with noise.

    Returns:
        tuple: A tuple containing:
            - estimated_params (dict): Dictionary mapping parameter names to their
                                       optimized values.
            - hessian (ndarray): Numerically approximated Hessian matrix using finite
                                 differences.
            - cov_mat (ndarray): Covariance matrix of the estimated parameters.
            - std (ndarray): Standard deviations of the estimated parameters.
    """
    # Perform the minimization
    opt = minimize(
        objective,
        x0=initial_values,
        args=(
            param_names,
            kwargs["time"],
            kwargs["u"],
            kwargs["bold_signal"],
        ),
        bounds=bounds,
        method="L-BFGS-B",
    )

    # Map the optimized parameter values back to their names
    estimated_params = dict(zip(param_names, opt.x.tolist()))

    # Define the objective function as a function of the parameters only
    def f(p):
        return objective(
            p,
            param_names,
            kwargs["time"],
            kwargs["u"],
            kwargs["bold_signal"],
        )

    # Compute the Hessian matrix using finite differences
    hessian = approx_hess(opt.x, f)

    if normalize:
        # Compute residuals and estimate the variance of the noise
        residuals = kwargs["bold_signal"] - simulate_bold(
            estimated_params,
            time=kwargs["time"],
            u=kwargs["u"],
            num_rois=kwargs["num_rois"],
        )
        n = residuals.size  # Total number of observations
        p = len(opt.x)  # Number of parameters
        sig_est = np.sum(residuals**2) / (n - p)  # Estimated variance of the residuals

        # Compute the covariance matrix as the scaled inverse Hessian
        cov_mat = inv(hessian) * sig_est

    else:
        cov_mat = inv(hessian)

    std = np.sqrt(np.diag(cov_mat))

    return estimated_params, hessian, cov_mat, std


# %%
# Noise and Visualization Functions
# =============================================================================


def add_noise(bold_true, snr_db):
    """
    Add Gaussian noise to the BOLD signal.

    Args:
        bold_true (ndarray): The true BOLD signal, typically a 2D array where each
                             column represents the BOLD signal for a different region
                             of interest (ROI).
        snr_db (float): Signal-to-noise ratio in decibels. Higher values correspond
                       to lower noise levels.

    Returns:
        ndarray: The noisy BOLD signal with added Gaussian noise.

    Raises:
        ValueError: If snr_db is negative or bold_true is empty.
    """
    if not isinstance(bold_true, np.ndarray):
        raise TypeError("bold_true must be a numpy array")
    if bold_true.size == 0:
        raise ValueError("bold_true cannot be empty")
    if snr_db < 0:
        raise ValueError("SNR in dB cannot be negative")

    snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear scale
    signal_max = np.max(np.abs(bold_true))
    if signal_max == 0:
        raise ValueError("Signal cannot be all zeros")

    sigma = signal_max / snr_linear
    return bold_true + np.random.normal(0, sigma, bold_true.shape)


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


def run_simulation(
    time,
    u,
    num_rois,
    num_layers,
    true_params,
    initial_values,
    params_to_est,
    snr,
    bounds=None,
    normalize=True,
    plot=True,
    verbose=True,
):
    """
    Runs a simulation, estimates parameters, and displays the results.

    Args:
        time (np.array): Time points for the simulation.
        u (np.array): External input to the system.
        num_rois (int): Number of regions of interest.
        num_layers (int): Number of layers in the system.
        true_params (dict): True parameter values for the simulation.
        initial_values (list): Initial guesses for parameter estimation.
        params_to_est (list): Parameters to estimate.
        snr (float): Signal-to-noise ratio for adding noise.
        bounds (list, optional): Parameter bounds for estimation. Defaults to None.
        normalize (bool, optional): Whether to normalize the Hessian. Defaults to True.
        plot (bool, optional): Whether to plot the results. Defaults to True.
        verbose (bool, optional): Whether to print the results. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - hessian (np.array): Hessian matrix from the estimation.
            - covariance (np.array): Covariance matrix from the estimation.
            - std (np.array): Standard deviations of the estimated parameters.
            - err (np.array): Error between the true and estimated parameters.
    """
    if bounds:
        bounds = [(bounds[param]) for param in params_to_est]

    # Simulate observed data
    bold_true = simulate_bold(
        true_params,
        time=time,
        u=u,
        num_rois=num_rois,
    )

    # Add noise to the observed data
    bold_noisy = add_noise(bold_true, snr_db=snr)

    # Estimate parameters
    est_params, hessian, covariance, std = estimate_parameters(
        initial_values,
        params_to_est,
        bounds,
        normalize,
        time=time,
        u=u,
        bold_signal=bold_noisy,
        num_rois=num_rois,
    )

    # Compute the error between the true and estimated parameters
    true_vals = np.array(list(filter_params(true_params, params_to_est).values()))
    est_vals = np.array(list(est_params.values()))
    err = true_vals - est_vals

    # Simulate data using estimated parameters
    bold_estimated = simulate_bold(est_params, time=time, u=u, num_rois=num_rois)

    if plot:
        # Plot results
        plot_bold_signals(time, bold_true, bold_noisy, bold_estimated)

    if verbose:
        # Print results
        print("\tTrue\tEstimated\tInitial Guess")
        for i, param in enumerate(params_to_est):
            print(f"{param}:\t", end="")
            print(f"{true_params[param]:.2f}\t", end="")
            print(f"{est_params[param]:.2f}\t\t", end="")
            print(f"{initial_values[i]:.2f}")

        print("\nHessian:")
        print(hessian)

        print("\nCovariance matrix:")
        print(covariance)

        print("\nVariances of the estimated parameters:")
        print(np.diag(covariance))

        print("\nStandard deviations of the estimated parameters:")
        print(std, "\n\n")

    return hessian, covariance, std, err


# %%
# MCMC Functions
# =============================================================================


def log_probability(
    params, bold_signal, param_names, all_params, bounds, time, u, num_rois
):
    """
    Log-probability function for MCMC.

    Args:
        params (list): Parameter values to evaluate.
        bold_signal (ndarray): Observed noisy BOLD signal.
        param_names (list): Names of parameters to estimate.
        all_params (dict): All parameters, including fixed ones.
        bounds (list of tuples): Bounds for each parameter.
        time (ndarray): Time points for simulation.
        u (callable): Stimulus function.
        num_rois (int): Number of regions of interest.

    Returns:
        float: Log-probability value.
    """
    # Check if parameters are within bounds
    for param_name, param_val, (low, high) in zip(param_names, params, bounds):
        if not (low <= param_val <= high):
            return -np.inf  # Outside bounds → log-probability is -∞

    # Update parameter dictionary
    updated_params = all_params.copy()
    for param_name, param_val in zip(param_names, params):
        updated_params[param_name] = param_val

    # Simulate BOLD signal
    try:
        bold_simulated = simulate_bold(
            updated_params,
            time=time,
            u=u,
            num_rois=num_rois,
        )
    except Exception:
        return -np.inf  # Return -inf if simulation fails

    # Compute likelihood (negative squared error)
    residuals = bold_simulated - bold_signal
    likelihood = -0.5 * np.sum(residuals**2)

    return likelihood


# %%
# Data Generation and Setup
# =============================================================================

# Setup time and stimulus
time = np.arange(TIME_POINTS)
u = utils.stim_boxcar(STIMULUS_CONFIG)

# Filter parameters for this run
true_params = filter_params(TRUE_PARAMS, PARAMS_TO_SIM)
param_bounds = filter_params(PARAM_BOUNDS, PARAMS_TO_SIM)

# Initialize parameter values
initial_values = initialize_parameters(param_bounds, PARAMS_TO_EST)
bounds_list = [param_bounds[param] for param in PARAMS_TO_EST]

print("Setup complete:")
print(f"  - Time points: {len(time)}")
print(f"  - True parameters: {true_params}")
print(f"  - Parameters to estimate: {PARAMS_TO_EST}")

# %%
# Generate Synthetic Data
# =============================================================================

# Generate synthetic data
bold_true = simulate_bold(
    true_params,
    time=time,
    u=u,
    num_rois=NUM_ROIS,
)

# Add noise to create observed data
bold_noisy = add_noise(bold_true, snr_db=SNR_DB)

print(f"Data generated:")
print(f"  - BOLD signal shape: {bold_true.shape}")
print(f"  - SNR: {SNR_DB} dB")
print(f"  - Signal range: [{bold_true.min():.3f}, {bold_true.max():.3f}]")

# Plot the true and noisy signals
fig, axes = plt.subplots(1, NUM_ROIS, figsize=(12, 4))
if NUM_ROIS == 1:
    axes = [axes]

for i in range(NUM_ROIS):
    axes[i].plot(time, bold_true[:, i], label="True BOLD", lw=2, alpha=0.8)
    axes[i].plot(time, bold_noisy[:, i], label="Noisy BOLD", lw=1, alpha=0.7)
    axes[i].set_title(f"ROI {i}")
    axes[i].set_xlabel("Time (s)")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

axes[0].set_ylabel("BOLD Signal")
plt.tight_layout()
plt.show()

# %%
# MCMC Setup and Execution
# =============================================================================

# Number of dimensions (parameters to estimate)
n_dim = len(PARAMS_TO_EST)

# Initialize walkers around the initial guess
initial_guess = [
    initial_values + 0.01 * np.random.randn(n_dim) for _ in range(N_WALKERS)
]

print(f"Starting MCMC with:")
print(f"  - {N_WALKERS} walkers")
print(f"  - {n_dim} dimensions")
print(f"  - {N_BURN} burn-in samples")
print(f"  - {N_SAMPLES} sampling iterations")
print(f"  - Initial values: {initial_values}")

# Run MCMC
sampler = emcee.EnsembleSampler(
    N_WALKERS,
    n_dim,
    log_probability,
    args=(
        bold_noisy,
        PARAMS_TO_EST,
        true_params.copy(),
        bounds_list,
        time,
        u,
        NUM_ROIS,
    ),
)

# Execute the sampling
sampler.run_mcmc(initial_guess, N_BURN + N_SAMPLES, progress=True)

print("MCMC sampling completed!")

# %%
# Results Analysis
# =============================================================================

# Discard burn-in samples and reshape
samples = sampler.get_chain(discard=N_BURN, flat=True)

print(f"Final sample shape: {samples.shape}")
print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

# %%
# Parameter Estimation Results
# =============================================================================

# Compute mean and standard deviation for each parameter
means = np.mean(samples, axis=0)
stds = np.std(samples, axis=0)
estimated_params = dict(zip(PARAMS_TO_EST, means))

# Print results
print("Parameter Estimation Results:")
print("=" * 50)
print(f"{'Parameter':<12} {'True':<8} {'Estimated':<12} {'Std Dev':<10}")
print("-" * 50)
for i, param in enumerate(PARAMS_TO_EST):
    true_val = true_params[param]
    est_val = means[i]
    std_val = stds[i]
    print(f"{param:<12} {true_val:<8.3f} {est_val:<12.3f} {std_val:<10.3f}")

print("\nEstimation Errors:")
for i, param in enumerate(PARAMS_TO_EST):
    error = abs(true_params[param] - means[i])
    rel_error = error / abs(true_params[param]) * 100
    print(f"  {param}: {error:.4f} ({rel_error:.1f}%)")

# %%
# Posterior Visualization
# =============================================================================

# Visualize posterior distributions
fig = corner.corner(
    samples,
    labels=PARAMS_TO_EST,
    truths=[true_params[p] for p in PARAMS_TO_EST],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    quantiles=[0.16, 0.5, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
    plot_density=False,
    plot_contours=True,
    fill_contours=True,
    smooth=True,
)

plt.suptitle("MCMC Posterior Distributions", fontsize=16, y=1.02)
plt.show()

# Show chain convergence
if len(PARAMS_TO_EST) <= 4:  # Only plot if not too many parameters
    fig, axes = plt.subplots(
        len(PARAMS_TO_EST), 1, figsize=(10, 3 * len(PARAMS_TO_EST))
    )
    if len(PARAMS_TO_EST) == 1:
        axes = [axes]

    for i, param in enumerate(PARAMS_TO_EST):
        chain = sampler.get_chain()[:, :, i]
        for walker in range(N_WALKERS):
            axes[i].plot(chain[:, walker], alpha=0.3, color="steelblue", lw=0.5)
        axes[i].axhline(
            true_params[param],
            color="red",
            linestyle="--",
            lw=2,
            label=f"True value: {true_params[param]:.3f}",
        )
        axes[i].axvline(
            N_BURN, color="black", linestyle=":", alpha=0.7, label="Burn-in end"
        )
        axes[i].set_ylabel(f"{param}")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("MCMC Step")
    plt.suptitle("MCMC Chain Traces", fontsize=14)
    plt.tight_layout()
    plt.show()

print("Analysis complete!")


# %%
