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
    connections = [f'R0, L0 -> R1, L0 = {L0}']  # ROI0 -> ROI1 connection
    return utils.create_A_matrix(num_rois, num_layers, connections, self_connections=-1)


def get_one_layer_C(L0=1):
    connections = [f'R0, L0 = {L0}']  # Input --> ROI0 connection
    return utils.create_C_matrix(num_rois, num_layers, connections)


# Simulate observed data with specified parameters
def simulate_bold(params, **kwargs):
    dcm = models.DCM(
        kwargs['num_rois'],
        params={
            'A': get_one_layer_A(params.get('A_L0', 0.2)),
            'C': get_one_layer_C(params.get('C_L0', 1.0)),
            **params,
        },
    )
    return dcm.simulate(kwargs['time'], kwargs['u'])[0]


# Objective function to minimize, generalized for any parameters
def objective(param_vals, param_names, time, u, bold_signal, num_rois):

    # Map the parameter values to their names
    params = dict(zip(param_names, param_vals))
    A = get_one_layer_A(params.get('A_L0', 0.2))
    C = get_one_layer_C(params.get('C_L0', 1.0))
    bold_simulated = simulate_bold(params, time=time, u=u, A=A, C=C, num_rois=num_rois)

    # Compute the sum of squared errors
    return np.sum((bold_simulated - bold_signal) ** 2)


# Estimate parameters using optimization
def estimate_parameters(initial_values, bounds, param_names, **kwargs):
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
    sigma = np.max(bold_true) / snr
    return bold_true + np.random.normal(0, sigma, bold_true.shape)


# Plot observed and estimated BOLD signals
def plot_bold_signals(time, bold_true, bold_noisy, bold_estimated, num_rois):
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
