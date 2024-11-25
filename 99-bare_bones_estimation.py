# %%
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.linalg import inv
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess
from tqdm import tqdm

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


def initialize_parameters(bounds, params_to_sim):
    initial_values = []
    for param in params_to_sim:
        initial_values.append(np.random.uniform(*bounds[param]))

    return initial_values


def filter_params(params, keys):
    return {k: params[k] for k in keys}


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


def add_noise(bold_true, snr_db):
    snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear scale
    sigma = np.max(bold_true) / snr_linear
    return bold_true + np.random.normal(0, sigma, bold_true.shape)


def objective(param_vals, param_names, bold_observed):
    # Map the parameter values to their names
    params = dict(zip(param_names, param_vals))

    # Simulate the estimated BOLD signal
    bold_simulated = simulate_bold(params, time=time, u=u, num_rois=num_rois)

    # Compute the sum of squared errors
    loss = np.mean((bold_simulated - bold_observed) ** 2)

    return loss


def estimate_parameters(
    initial_values,
    param_names,
    bounds=None,
    normalize=True,
    **kwargs,
):
    # Perform the minimization
    opt = minimize(
        objective,
        x0=initial_values,
        args=(
            param_names,
            kwargs['bold_signal'],
        ),
        bounds=bounds,
        method='L-BFGS-B',
    )

    # Map the optimized parameter values back to their names
    estimated_params = dict(zip(param_names, opt.x.tolist()))

    # Compute the Hessian matrix using finite differences
    hessian = approx_hess(opt.x, objective, args=(param_names, kwargs['bold_signal']))

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


def run_pipeline(
    true_params,
    initial_values,
    params_to_est,
    snr,
    bounds=None,
    normalize=True,
    plot=True,
    verbose=True,
):

    # Simulate observed data
    bold_true = simulate_bold(
        true_params,
        time=time,
        u=u,
        num_rois=num_rois,
    )

    # Add noise to the observed data
    bold_noisy = add_noise(bold_true, snr_db=snr)

    # Filter the relevant bounds if there are any
    if bounds:
        bounds = [bounds[param] for param in params_to_est]

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

    if plot:
        # Simulate data using estimated parameters
        bold_estimated = simulate_bold(est_params, time=time, u=u, num_rois=num_rois)
        # Plot results
        plot_bold_signals(time, bold_true, bold_noisy, bold_estimated)

    if verbose:
        # Print results
        print('\tTrue\tEstimated\tInitial Guess')
        for i, param in enumerate(params_to_est):
            print(f'{param}:\t', end='')
            print(f'{true_params[param]:.2f}\t', end='')
            print(f'{est_params[param]:.4f}\t\t', end='')
            print(f'{initial_values[i]:.2f}')

        print('\nHessian:')
        print(hessian)

        print('\nCovariance matrix:')
        print(covariance)

        print('\nVariances of the estimated parameters:')
        print(np.diag(covariance))

        print('\nStandard deviations of the estimated parameters:')
        print(std, '\n\n')

    return hessian, covariance, std, err, est_params


# %%
# DCM parameters
time = np.arange(100)
u = utils.stim_boxcar([[0, 30, 1]])
num_rois = 2
num_layers = 1

# Parameters to set and estimate
params_to_set = ['alpha', 'kappa', 'gamma', 'A_L0', 'C_L0']
params_to_est = ['C_L0', 'A_L0']

# Ground truth parameter values
true_params = {
    'alpha': 0.2,
    'kappa': 1.5,
    'gamma': 0.5,
    'A_L0': 0.8,
    'C_L0': 0.7,
}
true_params = filter_params(true_params, params_to_set)
bounds = {
    'alpha': (0.2, 1.0),
    'kappa': (1.0, 2.0),
    'gamma': (0.0, 1.0),
    'A_L0': (0.0, 1.0),
    'C_L0': (0.0, 1.0),
}
bounds = filter_params(bounds, params_to_set)

# Initial values for the parameters to estimate
initial_values = initialize_parameters(bounds, params_to_est)
print('Initial guesses:\t', dict(zip(params_to_est, initial_values)))

# %%
# Run the estimation pipeline
estimated_vals = {key: [] for key in params_to_est}
snr_range = np.linspace(0.001, 50, 20)

for snr in tqdm(snr_range):
    hess, cov, std, err, est = run_pipeline(
        true_params,
        initial_values,
        params_to_est,
        snr=snr,
        bounds=bounds,
        normalize=True,
        plot=False,
        verbose=False,
    )

    for param in params_to_est:
        estimated_vals[param].append(est[param])

# %%
plt.figure(figsize=(10, 6))

for param in params_to_est:
    plt.plot(snr_range, estimated_vals[param], label=param)
    plt.axhline(true_params[param], c='k', ls='--')

    if '_' in param:
        label = f'${param}$'
    else:
        label = '{param}'
    plt.text(
        snr_range[-1] - 1,
        true_params[param] + 0.01,
        label,
        horizontalalignment='right',
        fontsize=10,
    )

plt.xlim(snr_range[0], snr_range[-1])
plt.xlabel('SNR')
plt.ylabel('Estimated Parameter')
plt.legend()
plt.show()

# %%
