# %%
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import minimize

from dcsem import models, utils

plt.style.use(['science', 'no-latex'])
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
        float: The sum of squared errors between the simulated and real BOLD signals.
    """

    # Map the parameter values to their names
    params = dict(zip(param_names, param_vals))
    A = get_one_layer_A(params.get('A_L0', 0.2))
    C = get_one_layer_C(params.get('C_L0', 1.0))
    num_rois = bold_signal.shape[1]
    bold_simulated = simulate_bold(params, time=time, u=u, A=A, C=C, num_rois=num_rois)

    # Compute the sum of squared errors
    return np.sum((bold_simulated - bold_signal) ** 2)



def estimate_parameters(initial_values, bounds, param_names, **kwargs):
    """
    Estimate parameters by minimizing the objective function using the L-BFGS-B method.

    Args:
        initial_values (list or ndarray): Initial guesses for the parameter values to
                                          be optimized.
        bounds (list of tuples): Bounds for each parameter, where each tuple contains
                                 the lower and upper bounds.
        param_names (list of str): Names of the parameters to be estimated.
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
            kwargs['num_rois'],
        ),
        bounds=bounds,
        method='L-BFGS-B',
    )

    # Map the optimized parameter values back to their names
    estimated_params = dict(zip(param_names, opt.x.tolist()))

    return estimated_params, opt.hess_inv.todense()


def add_noise(bold_true, snr):
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
    sigma = np.max(bold_true) / snr
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


# %%
if __name__ == '__main__':
    # Set up the time vector and stimulus function
    time = np.arange(100)
    u = utils.stim_boxcar([[0, 30, 1]])

    # Connectivity parameters
    num_rois = 2
    num_layers = 1

    # ==================================================================================
    # Set the ground truth parameter values
    # ==================================================================================
    true_params = {
        'alpha': 0.5,
        'kappa': 1.5,
        'gamma': 0.5,
        'A_L0': 0.2,
        'C_L0': 1.0,
    }
    # ==================================================================================
    # Set the initial guesses and bounds for the parameter optimization
    # ==================================================================================
    initial_values = {
        'alpha': 0.5,
        'kappa': 1.5,
        'gamma': 0.5,
        'A_L0': 0,
        'C_L0': 0.3,
    }
    bounds = {
        'alpha': (0.1, 1.0),
        'kappa': (1.0, 2.0),
        'gamma': (0.0, 1.0),
        'A_L0': (0, 1),
        'C_L0': (0, 1),
    }
    # ==================================================================================
    # Choose the parameters to simulate and estimate
    # ==================================================================================
    params_to_sim = ['alpha', 'kappa', 'gamma', 'A_L0', 'C_L0']
    params_to_est = ['alpha', 'kappa', 'gamma', 'A_L0']
    # ==================================================================================

    # Filter the parameters to simulate and estimate
    true_params = {k: true_params[k] for k in params_to_sim}
    initial_values = [initial_values[k] for k in params_to_est]
    bounds = [bounds[k] for k in params_to_est]

    # Simulate observed data
    bold_true = simulate_bold(
        true_params,
        time=time,
        u=u,
        num_rois=num_rois,
    )

    # Add noise to the observed data
    bold_noisy = add_noise(bold_true, snr=30)

    # Estimate parameters
    estimated_params, hess_inv = estimate_parameters(
        initial_values,
        bounds,
        params_to_est,
        time=time,
        u=u,
        bold_signal=bold_noisy,
        num_rois=num_rois,
    )

    # Simulate data using estimated parameters
    bold_estimated = simulate_bold(estimated_params, time=time, u=u, num_rois=num_rois)

    # Plot results
    plot_bold_signals(time, bold_true, bold_noisy, bold_estimated, num_rois)

    print('\tTrue\tEstimated')
    for param in params_to_est:
        print(f'{param}:\t{true_params[param]:.2f}\t{estimated_params[param]:.6f}')

    print('\nHessian inverse:')
    print(hess_inv)

    print('\nVariances of the estimated parameters:')
    print(np.diag(hess_inv))

# %%
