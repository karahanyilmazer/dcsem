# %%
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import minimize

from dcsem import models, utils

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'Times New Roman'


# %%
# Simulate observed data with specified parameters
def simulate_observed_data(params, **kwargs):
    dcm = models.DCM(
        kwargs['num_rois'],
        params={'A': kwargs['A'], 'C': kwargs['C'], **params},
    )
    return dcm.simulate(kwargs['time'], kwargs['u'])[0]


# Objective function to minimize, generalized for any parameters
def objective(params, param_names, time, u, A, C, bold_signal, num_rois):
    # Map the parameter values to their names
    params = dict(zip(param_names, params))
    # Run DCM simulation with the provided parameters
    dcm = models.DCM(
        num_rois,
        params={'A': A, 'C': C, **params},
    )
    bold_simulated = dcm.simulate(time, u)[0]

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
            kwargs['A'],
            kwargs['C'],
            kwargs['bold_signal'],
            kwargs['num_rois'],
        ),
        bounds=bounds,
        method='L-BFGS-B',
    )

    # Map the optimized parameter values back to their names
    estimated_params = dict(zip(param_names, opt.x.tolist()))

    return estimated_params, opt.hess_inv.todense()


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
    connections = ['R0, L0 -> R1, L0 = 0.2']
    A = utils.create_A_matrix(num_rois, num_layers, connections, self_connections=-1)
    C = utils.create_C_matrix(num_rois, num_layers, ['R0, L0 = 1.0'])

    # Ground truth values for the parameters
    true_params = {'alpha': 0.5, 'kappa': 1.5, 'gamma': 0.5}

    # Initial guesses and bounds for the optimization
    initial_values = {'alpha': 0.5, 'kappa': 1.5, 'gamma': 0.5}
    bounds = {'alpha': (0.1, 1.0), 'kappa': (1.0, 2.0), 'gamma': (0.0, 1.0)}

    # Parameter names to simulate and estimate
    param_names = ['alpha', 'kappa', 'gamma', 'A']
    param_names = ['alpha', 'kappa', 'gamma']
    # param_names = ['alpha', 'kappa']

    # Get a subset of the parameter dictionaries
    true_params = {k: true_params[k] for k in param_names}
    initial_values = [initial_values[k] for k in param_names]
    bounds = [bounds[k] for k in param_names]

    # Simulate observed data
    bold_true = simulate_observed_data(
        true_params,
        time=time,
        u=u,
        A=A,
        C=C,
        num_rois=num_rois,
    )

    # Add noise to the observed data
    snr = 30
    sigma = np.max(bold_true) / snr
    bold_noisy = bold_true + np.random.normal(0, sigma, bold_true.shape)

    # Estimate parameters
    estimated_params, hess_inv = estimate_parameters(
        initial_values,
        bounds,
        param_names,
        time=time,
        u=u,
        A=A,
        C=C,
        bold_signal=bold_noisy,
        num_rois=num_rois,
    )

    # Simulate data using estimated parameters
    bold_estimated = simulate_observed_data(
        estimated_params, time=time, u=u, A=A, C=C, num_rois=num_rois
    )

    # Plot results
    plot_bold_signals(time, bold_true, bold_noisy, bold_estimated, num_rois)

    print('\tTrue\tEstimated')
    for param in param_names:
        print(f'{param}:\t{true_params[param]:.2f}\t{estimated_params[param]:.6f}')

    print('\nHessian inverse:')
    print(hess_inv)

    print('\nVariances of the estimated parameters:')
    print(np.diag(hess_inv))

# %%
