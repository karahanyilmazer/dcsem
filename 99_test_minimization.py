# %%
#!%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess

from utils import add_noise

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


# %%
def f(x, a, b):
    return a / (1 + np.exp(-b * x))


def loss(params):
    val = np.mean((signal - f(x, *params)) ** 2)
    return val


def callback(xk):
    print(f'Parameter val: {xk}\t-->\tLoss: {loss(xk)}')


def plot_estimation(x, signal, est_params):
    plt.figure()
    plt.plot(x, ground_truth, c='black', label='Ground Truth')
    plt.plot(x, signal, label='Observation')
    plt.plot(x, f(x, *est_params), ls='--', label='Estimated')
    plt.title('Parameter Estimation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def plot_loss(params_range, optimum):
    param_grid1, param_grid2 = np.meshgrid(params_range[0], params_range[1])
    loss_grid = np.zeros_like(param_grid1)

    # Compute loss values over the grid
    for i in range(param_grid1.shape[0]):
        for j in range(param_grid1.shape[1]):
            loss_grid[i, j] = loss([param_grid1[i, j], param_grid2[i, j]])

    fig = plt.figure(figsize=(12, 6))

    if len(params_range) == 2:
        # 3D Surface Plot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(param_grid1, param_grid2, loss_grid, cmap='viridis')
        ax.plot(
            optimum[0], optimum[1], loss(optimum), 'ro', label='Optimum', markersize=10
        )
        ax.axvline(optimum[0], color='red', ls='--', label='Optimum')
        ax.set_xlabel('Parameter a')
        ax.set_ylabel('Parameter b')
        ax.set_zlabel('MSE Loss')
        ax.set_title('Loss Surface')

        # 2D Contour Plot
        ax = fig.add_subplot(1, 2, 2)
        contour = ax.contourf(
            param_grid1, param_grid2, loss_grid, levels=50, cmap='viridis'
        )
        ax.plot(optimum[0], optimum[1], 'ro', label='Optimum', markersize=10)
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel('Parameter a')
        ax.set_ylabel('Parameter b')
        ax.set_title('Loss Contour')

        plt.tight_layout()
        plt.show()

    else:
        plt.plot(params_range)
        plt.axvline(optimum, color='red', ls='--', label='Optimum')
        plt.title('Loss Function Landscape')
        plt.xlabel('Parameter a')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.show()


def calc_std(est_params, loss, normalize=True, log=False):
    hess = approx_hess(est_params, loss)

    if normalize:
        # Calculate residuals at the estimated parameter
        residuals = signal - f(x, *est_params)
        p = len(est_params)
        s_sq = np.sum(residuals**2) / (n_samples - p)  # Residual variance

        cov = np.linalg.inv(hess) * s_sq

    else:
        cov = np.linalg.inv(hess)

    std = np.sqrt(np.diag(cov))

    if log:
        print('HESSIAN:\n', hess, '\n')
        print('COVARIANCE:\n', cov, '\n')
        print('STD:\n', std, '\n')

    return hess, cov, std


def plot_snr_to_std(snr_vals, std_vals, true_vals, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if normalize:
        ax.axhline(0, color='black', ls='--', label='Zero Line')
    else:
        # hess = approx_hess(true_vals, loss)
        # std_inf = np.sqrt(np.diag(hess)) / 2
        pass
    ax.plot(snr_vals, std_vals, '-o')
    # ax.set_title('Standard Deviation over SNR')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Standard Deviation')
    ax.legend()
    if ax is None:
        plt.show()


def plot_snr_to_err(snr_vals, est_params, std_vals, true_vals, shade=False, ax=None):
    est_params = np.array(est_params_history)
    true_vals = np.array(true_vals)
    std_vals = np.array(std_vals)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.axhline(0, color='black', ls='--', label='Zero Line')

    for i in range(est_params.shape[1]):
        est_err = true_vals[i] - est_params[:, i]
        ax.plot(snr_vals, est_err, '-o', label=f'Param {i}')
        if shade:
            ax.fill_between(
                snr_vals,
                est_err - std_vals[:, i],
                est_err + std_vals[:, i],
                alpha=0.2,
            )

    # ax.set_title('Estimation Error over SNR')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Estimation Error')
    ax.legend()

    if ax is None:
        plt.show()


def plot_snr_sweep(snr_vals, std_vals, true_vals, est_params_history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Parameter Estimation')
    plot_snr_to_std(snr_vals, std_vals, true_vals, ax=axs[0])
    plot_snr_to_err(snr_vals, est_params_history, std_vals, true_vals, ax=axs[1])
    plt.tight_layout()
    plt.savefig('img/presentation/min_test.png')
    plt.show()


# %%
plt.close('all')
n_samples = 100
n_simulations = 20
min_snr, max_snr = 0.01, 30
snr_vals = np.logspace(np.log10(min_snr), np.log10(max_snr), n_simulations)
snr_vals = np.linspace(min_snr, max_snr, n_simulations)
std_vals = []
est_params_history = []
normalize = True
plot = False
log = False

initial_guess = [1, 1]
a_true = 5
b_true = 10
true_vals = np.array([a_true, b_true])
x = np.linspace(-5, 5, n_samples)
ground_truth = f(x, a_true, b_true)

for snr in snr_vals:
    # Add noise to the ground truth to get the observed signal
    signal = add_noise(ground_truth, snr)

    # Esimate the parameters
    opt = minimize(loss, x0=initial_guess, callback=callback, method='L-BFGS-B')
    est_params = opt.x
    est_params_history.append(est_params)

    if plot:
        param_range = np.linspace(-100, 100, 100)
        param_ranges = [param_range, param_range]
        plot_estimation(x, signal, est_params)
        plot_loss(param_ranges, est_params)

    if log:
        print(f'True parameters: {a_true}, {b_true}')
        print(f'Estimated parameters: {est_params}\n')

    hess, cov, std = calc_std(est_params, loss, normalize=normalize, log=log)
    std_vals.append(std)

# plot_snr_to_std(snr_vals, std_vals, true_vals)
# plot_snr_to_err(snr_vals, est_params_history, std_vals, true_vals, shade=False)

plot_snr_sweep(snr_vals, std_vals, true_vals, est_params_history)

# %%
