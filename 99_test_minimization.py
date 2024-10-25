# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess


# %%
def f(x, c):
    return np.sin(x) * x**2 / c


def add_noise(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise = np.random.normal(0, 1, len(signal))
    noise = noise * np.std(signal) / snr_linear
    return signal + noise


def loss(c):
    val = np.mean((signal - f(x, c)) ** 2)
    return val


def callback(xk):
    print(f'Parameter val: {xk}\t-->\tLoss: {loss(xk)}')


def plot_estimation(x, signal, c_est):
    plt.plot(x, ground_truth, c='black', label='Ground Truth')
    plt.plot(x, signal, label='Observation')
    plt.plot(x, f(x, c_est), ls='--', label='Estimated')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Parameter Estimation')
    plt.legend()
    plt.show()


def plot_loss(c_values, loss_values):
    plt.plot(c_values, loss_values)
    plt.axvline(c_est, color='red', ls='--', label='Optimum')
    plt.xlabel('Parameter c')
    plt.ylabel('MSE Loss')
    plt.title('Loss Function Landscape')
    plt.legend()
    plt.show()


def calc_std(c_est, loss, normalize=True, log=False):
    hess = approx_hess(c_est, loss)

    if normalize:
        # Calculate residuals at the estimated parameter
        residuals = signal - f(x, c_est)
        p = len(c_est)
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


def plot_snr_to_std(snr_vals, std_vals):
    if normalize:
        plt.axhline(0, color='black', ls='--', label='Zero Line')
    plt.plot(snr_vals, std_vals, '-o')
    plt.title('Standard Deviation over SNR')
    plt.xlabel('SNR')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.show()


def plot_snr_to_est(snr_vals, est_params):
    plt.axhline(c_true, color='red', ls='--', label='True Value')
    plt.plot(snr_vals, est_params, '-o')
    plt.title('Estimated Parameter over SNR')
    plt.xlabel('SNR')
    plt.ylabel('Estimated Parameter')
    plt.legend()
    plt.show()


n_samples = 100
snr_vals = np.linspace(0.01, 30, 20)
snr_vals = np.logspace(np.log10(0.001), np.log10(100), 20)
std_vals = []
est_params = []
normalize = True
plot = False
log = False

for snr in snr_vals:
    x = np.linspace(-5, 5, n_samples)
    c_true = 5
    ground_truth = f(x, c_true)
    signal = add_noise(ground_truth, snr)

    initial_guess = 1

    opt = minimize(loss, x0=initial_guess, callback=callback)
    c_est = opt.x
    est_params.append(c_est)

    if plot:
        c_vals = np.linspace(1, 10, 100)
        loss_vals = [loss(c) for c in c_vals]
        plot_estimation(x, signal, c_est)
        plot_loss(c_vals, loss_vals)

    if log:
        print(f'True parameter: c = {c_true}')
        print(f'Estimated parameters: c = {c_est}\n')

    hess, cov, std = calc_std(c_est, loss, normalize=normalize, log=log)
    std_vals.append(std)

plot_snr_to_std(snr_vals, std_vals)
plot_snr_to_est(snr_vals, est_params)

# %%
