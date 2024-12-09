# %%
# !%load_ext autoreload
# !%autoreload 2
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.linalg import inv
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess
from tqdm import tqdm

from dcsem.utils import stim_boxcar
from utils import (
    add_noise,
    add_underscore,
    filter_params,
    initialize_parameters,
    simulate_bold,
)

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'Times New Roman'


# %%
def objective(param_vals, param_names, bold_observed):
    # Map the parameter values to their names
    params = dict(zip(param_names, param_vals))

    # Simulate the estimated BOLD signal
    bold_simulated = simulate_bold(params, time=time, u=u, num_rois=num_rois)

    # Compute the mean squared error
    loss = np.mean((bold_simulated - bold_observed) ** 2)

    # Compute the sum of squared errors
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

    n = len(cov_mat)

    if n > 1:
        # Initialize lists to accumulate data
        pairs = []
        values = []

        # Extract upper half elements and their indices
        upper_half_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        upper_half_elements = cov_mat[upper_half_mask]
        row_indices, col_indices = np.where(upper_half_mask)

        # Store pairs and values
        for value, row, col in zip(upper_half_elements, row_indices, col_indices):
            pair_label = fr'{add_underscore(param_names[row])} $\leftrightarrow$ {add_underscore(param_names[col])}'
            pairs.append(pair_label)
            values.append(value)

    return estimated_params, hessian, cov_mat, std, pairs, values


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

    # Filter the relevant bounds if they are provided
    if bounds:
        bounds = [(bounds[param]) for param in params_to_est]

    # Estimate parameters
    est_params, hessian, covariance, std, pairs, values = estimate_parameters(
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

    return hessian, covariance, std, err, pairs, values


# %%
if __name__ == '__main__':
    # ==================================================================================
    # Specify the parameters for the simulation
    # ==================================================================================

    # Set up the time vector and stimulus function
    time = np.arange(100)
    u = stim_boxcar([[0, 30, 1]])

    # Model parameters
    num_rois = 2
    num_layers = 1

    # Set the colors for each parameter
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
    param_colors = dict(zip(['a01', 'a10', 'c0', 'c1'], color_cycle))

    # Parameters to use in the simulation and estimation
    params_to_set = ['a01', 'a10', 'c0', 'c1']
    params_to_est = ['a01', 'a10', 'c0', 'c1']

    # Generate all combinations of the parameters
    all_combinations = []
    for r in range(1, len(params_to_est) + 1):
        combinations_r = combinations(params_to_est, r)
        all_combinations.extend(combinations_r)

    # Convert each combination to a list (optional)
    all_combinations = [list(comb) for comb in all_combinations]

    # Remove single parameter combinations
    all_combinations = [comb for comb in all_combinations if len(comb) > 1]

    # Ground truth parameter values
    true_params = {
        'a01': 0.4,
        'a10': 0.4,
        'c0': 0.5,
        'c1': 0.5,
    }
    true_params = filter_params(true_params, params_to_set)

    # Bounds for the parameters
    bounds = {
        'a01': (0, 1),
        'a10': (0, 1),
        'c0': (0, 1),
        'c1': (0, 1),
    }
    bounds = filter_params(bounds, params_to_est)

    random = True

    # Signal-to-noise ratio
    n_sims = 3 if random else 1
    n_snrs = 20
    min_snr, max_snr = 0.1, 50
    snr_range = np.logspace(np.log10(min_snr), np.log10(max_snr), n_snrs)
    snr_range = np.linspace(min_snr, max_snr, n_snrs)

    # ==================================================================================
    # Run the simulation and estimation
    # ==================================================================================
    pair_list = []
    all_vals = []

    for comb in all_combinations:
        tmp_init = np.zeros((n_sims, len(comb)))
        tmp_stds = np.zeros((n_sims, len(comb)))
        tmp_errs = np.zeros((n_sims, len(comb)))
        tmp_vals = []
        init_list = []
        stds_list = []
        errs_list = []
        vals_list = []

        # Run the simulation and estimation
        for snr_db in tqdm(snr_range):
            non_nan_found = False

            while not non_nan_found:
                for sim_i in range(n_sims):
                    # Random initialization of the parameters
                    initial_values = initialize_parameters(bounds, comb, random=random)

                    hess, cov, std, err, pairs, vals = run_simulation(
                        true_params=true_params,
                        initial_values=initial_values,
                        params_to_est=comb,
                        snr=snr_db,
                        bounds=bounds,
                        normalize=True,
                        plot=False,
                        verbose=False,
                    )

                    # Collect initial guesses, standard deviations, and errors
                    tmp_init[sim_i, :] = initial_values
                    tmp_stds[sim_i, :] = std
                    tmp_errs[sim_i, :] = err
                    tmp_vals.append(vals)

                # Check if all simulations resulted in NaN
                if (np.isnan(tmp_stds).all()) or (
                    np.isnan(np.sum(tmp_stds, axis=1)).all()
                ):
                    print(f'All simulations failed at SNR {snr_db} dB, retrying...')
                    tmp_init = np.zeros((n_sims, len(comb)))
                    tmp_stds = np.zeros((n_sims, len(comb)))
                    tmp_errs = np.zeros((n_sims, len(comb)))
                else:
                    non_nan_found = True

            # Get the best estimation results
            best_run_idx = np.nanargmin(np.sum(tmp_stds, axis=1))

            init_list.append(tmp_init[best_run_idx])
            stds_list.append(tmp_stds[best_run_idx])
            errs_list.append(tmp_errs[best_run_idx])
            vals_list.append(tmp_vals[best_run_idx])

            tmp_init = np.zeros((n_sims, len(comb)))
            tmp_stds = np.zeros((n_sims, len(comb)))
            tmp_errs = np.zeros((n_sims, len(comb)))
            tmp_vals = []

        all_vals.append(vals_list)
        pair_list.append(pairs)

        # Plot the results
        stds_arr = np.array(stds_list)
        errs_arr = np.array(errs_list)
        vals_arr = np.array(vals_list)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].axhline(0, color='k', ls='--')
        axs[1].axhline(0, color='k', ls='--')

        for i, param in enumerate(comb):
            axs[0].plot(
                snr_range,
                stds_arr[:, i],
                '-x',
                color=param_colors[param],
                label=add_underscore(param),
            )
            axs[1].plot(
                snr_range,
                errs_arr[:, i],
                '-x',
                color=param_colors[param],
                label=add_underscore(param),
            )

        axs[0].set_xlabel('Signal-to-Noise Ratio (dB)')
        axs[0].set_ylabel('Standard Deviation')
        axs[0].legend()

        axs[1].set_xlabel('Signal-to-Noise Ratio (dB)')
        axs[1].set_ylabel('Estimation Error')
        axs[1].legend()

        tmp_names = comb.copy()
        tmp_names.sort()
        fig.suptitle(
            f'Parameter Estimation Results ({', '.join([add_underscore(name) for name in tmp_names])})'
        )
        plt.tight_layout()
        plt.savefig(
            f'img/presentation/estimation/random-{random}/on_diag-{'_'.join(tmp_names)}.png'
        )
        plt.show()

    for val, pairs, comb in zip(all_vals, pair_list, all_combinations):
        tmp_val = np.array(val)
        tmp_names = comb.copy()
        tmp_names.sort()
        plt.figure(figsize=(6, 4))
        plt.xlabel('Signal-to-Noise Ratio (dB)')
        plt.ylabel('Covariance')
        plt.title('Off Diagonal Covariance')
        for i, curr_off in enumerate(tmp_val.T):
            plt.plot(curr_off, label=pairs[i])
            plt.legend()
        plt.savefig(
            f'img/presentation/estimation/random-{random}/off_diag-{'_'.join(tmp_names)}.png'
        )
        plt.show()

# %%
