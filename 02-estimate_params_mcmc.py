# %%
# !%load_ext autoreload
# !%autoreload 2
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess

from dcsem import models, utils

plt.rcParams['font.family'] = 'Times New Roman'


# %%
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
    connections = [f'R0, L0 -> R1, L0 = {L0}']  # ROI0 -> ROI1 connection
    return utils.create_A_matrix(num_rois, num_layers, connections, self_connections=-1)


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
    connections = [f'R0, L0 = {L0}']  # Input --> ROI0 connection
    return utils.create_C_matrix(num_rois, num_layers, connections)


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


def simulate_bold(params, **kwargs):
    """
    Simulate the BOLD signal using a DCM model with the given parameters.

    Args:
        params (dict): A dictionary of parameters for the simulation, including:
            - 'A_L0' (float): Connectivity parameter for layer 0 of region of
                              interest 0 (optional, default=0.2).
            - 'C_L0' (float): Input parameter for layer 0 of region of interest
                              0 (optional, default=1.0).
            Other model-specific parameters may also be included in this dict.

        **kwargs: Additional keyword arguments, including:
            - time (ndarray): Time points for the simulation.
            - u (ndarray): External stimulus function.

    Returns:
        ndarray: Simulated BOLD signal for each region of interest, typically
                 a 2D array (time points x ROIs).
    """
    dcm = models.DCM(
        kwargs['num_rois'],
        params={
            'A': get_one_layer_A(params.get('A_L0', 0.2)),
            'C': get_one_layer_C(params.get('C_L0', 1.0)),
            **params,
        },
    )
    return dcm.simulate(kwargs['time'], kwargs['u'])[0]


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
    A = get_one_layer_A(params.get('A_L0', 0.2))
    C = get_one_layer_C(params.get('C_L0', 1.0))
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
            kwargs['time'],
            kwargs['u'],
            kwargs['bold_signal'],
        ),
        bounds=bounds,
        method='L-BFGS-B',
    )

    # Map the optimized parameter values back to their names
    estimated_params = dict(zip(param_names, opt.x.tolist()))

    # Define the objective function as a function of the parameters only
    def f(p):
        return objective(
            p,
            param_names,
            kwargs['time'],
            kwargs['u'],
            kwargs['bold_signal'],
        )

    # Compute the Hessian matrix using finite differences
    hessian = approx_hess(opt.x, f)

    if normalize:
        # Compute residuals and estimate the variance of the noise
        residuals = kwargs['bold_signal'] - simulate_bold(
            estimated_params,
            time=kwargs['time'],
            u=kwargs['u'],
            num_rois=kwargs['num_rois'],
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


def add_noise(bold_true, snr_db):
    """
    Add Gaussian noise to the BOLD signal.

    Args:
        bold_true (ndarray): The true BOLD signal, typically a 2D array where each
                             column represents the BOLD signal for a different region
                             of interest (ROI).
        snr (float): Signal-to-noise ratio. Higher values correspond to lower noise
                     levels.


    Returns:
        ndarray: The noisy BOLD signal with added Gaussian noise.
    """

    snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear scale
    sigma = np.max(bold_true) / snr_linear
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
        print('\tTrue\tEstimated\tInitial Guess')
        for i, param in enumerate(params_to_est):
            print(f'{param}:\t', end='')
            print(f'{true_params[param]:.2f}\t', end='')
            print(f'{est_params[param]:.2f}\t\t', end='')
            print(f'{initial_values[i]:.2f}')

        print('\nHessian:')
        print(hessian)

        print('\nCovariance matrix:')
        print(covariance)

        print('\nVariances of the estimated parameters:')
        print(np.diag(covariance))

        print('\nStandard deviations of the estimated parameters:')
        print(std, '\n\n')

    return hessian, covariance, std, err


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
        num_rois=num_rois,
    )

    # Compute likelihood (negative squared error)
    residuals = bold_simulated - bold_signal
    likelihood = -0.5 * np.sum(residuals**2)

    return likelihood


time = np.arange(100)
u = utils.stim_boxcar([[0, 30, 1]])

# Model parameters
num_rois = 2
num_layers = 1

# Parameters to use in the simulation and estimation
params_to_sim = ['alpha', 'kappa', 'gamma', 'A_L0', 'C_L0']
params_to_est = ['alpha']
# Ground truth parameter values
true_params = {
    'alpha': 0.2,
    'kappa': 1.5,
    'gamma': 0.5,
    'A_L0': 0.8,
    'C_L0': 0.7,
}
true_params = filter_params(true_params, params_to_sim)
bounds = {
    'alpha': (0.2, 1.0),
    'kappa': (1.0, 2.0),
    'gamma': (0.0, 1.0),
    'A_L0': (0.0, 1.0),
    'C_L0': (0.0, 1.0),
}
bounds = filter_params(bounds, params_to_sim)
initial_values = initialize_parameters(bounds, params_to_est)
bounds = [(bounds[param]) for param in params_to_est]

bold_true = simulate_bold(
    true_params,
    time=time,
    u=u,
    num_rois=num_rois,
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
