# %%
# !%load_ext autoreload
# !%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
from bench.model_inversion import hessian, infer_change
from scipy.optimize import minimize

from dcsem.utils import stim_boxcar
from utils import set_style, simulate_bold

set_style()


# %%
loss_list = []


def loss(params):
    a01, a10, c0, c1 = params
    current_params = dict(true_params)
    current_params['a01'] = a01
    current_params['a10'] = a10
    current_params['c0'] = c0
    current_params['c1'] = c1
    obs_bold = simulate_bold(current_params, time=time, u=u, num_rois=NUM_ROIS)
    return np.mean((true_bold - obs_bold) ** 2)


time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])
NUM_ROIS = 2

true_params = {
    'a01': 0.4,
    'a10': 0.6,
    'c0': 0.9,
    'c1': 0.2,
}

# Generate "true" data using the known true parameters
true_bold = simulate_bold(true_params, time=time, u=u, num_rois=NUM_ROIS)

# Calculate the loss landscape
a01_range = np.linspace(0, 1, 100)
loss_scape = [loss((a, a, a, a)) for a in a01_range]

# Choose an initial guess for a01
initial_guess = [0.1, 0.2, 0.4, 0.7]
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
loss_list = []

# Call the minimization routine
result = minimize(
    loss,
    initial_guess,
    options={'gtol': 1e-12, 'ftol': 1e-12},
    method='L-BFGS-B',
)

# Extract the optimized parameter value
est_params = result.x

h = hessian(loss, est_params, bounds=bounds)

tmp_params = true_params.copy()
tmp_params['a01'] = est_params[0]
tmp_params['a10'] = est_params[1]
tmp_params['c0'] = est_params[2]
tmp_params['c1'] = est_params[3]

bold_pred = simulate_bold(tmp_params, time=time, u=u, num_rois=NUM_ROIS)

plt.figure()
plt.plot(time, true_bold, label='True BOLD')
plt.plot(time, bold_pred, '--', label='Fitted BOLD')
plt.legend()
plt.show()

# Print the results
print()
print("Optimization successful:", result.success)
print("Final loss:", result.fun)
print("Estimated parameters:")
print(f'a01:\t {initial_guess[0]} --> {est_params[0]:.4f}\t ({true_params['a01']})')
print(f'a10:\t {initial_guess[1]} --> {est_params[1]:.4f}\t ({true_params["a10"]})')
print(f'c0:\t {initial_guess[2]} --> {est_params[2]:.4f}\t ({true_params["c0"]})')
print(f'c1:\t {initial_guess[3]} --> {est_params[3]:.4f}\t ({true_params["c1"]})')

# %%
# plt.figure()
# plt.plot(a01_range, loss_scape, label='Loss Landscape')
# plt.axvline(est_params, color='r', linestyle='--', label='Optimized a01')
# plt.xlabel('a01')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.show()

# %%
