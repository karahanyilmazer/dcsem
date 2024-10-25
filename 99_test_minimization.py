# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess


# %%
def f(x, c):
    return np.sin(x) + c


def add_noise(signal, snr):
    noise = np.random.normal(0, 1, len(signal))
    noise = noise * np.std(signal) / snr
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


snr = 10
x = np.linspace(-5, 5, 100)
c_true = 5
ground_truth = f(x, c_true)
signal = add_noise(ground_truth, snr)

initial_guess = np.random.rand()

opt = minimize(loss, x0=initial_guess, callback=callback)
c_est = opt.x

plot_estimation(x, signal, c_est)

c_vals = np.linspace(0, 10, 100)
loss_vals = [loss(c) for c in c_vals]
plot_loss(c_vals, loss_vals)

print(f'True parameter: c = {c_true}')
print(f'Estimated parameters: c = {c_est}')
# %%
