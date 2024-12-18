# %%
# !%load_ext autoreload
# !%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from dcsem.utils import stim_boxcar
from utils import set_style, simulate_bold

set_style()


# %%
loss_list = []


def loss(a01):
    params = dict(true_params)
    params['a01'] = a01
    obs_bold = simulate_bold(params, time=time, u=u, num_rois=NUM_ROIS)
    # obs_bold = obs_bold.reshape(-1, order='F')
    # return true_bold - obs_bold
    loss = np.mean((true_bold - obs_bold) ** 2)
    print(a01, loss)
    loss_list.append(loss)
    return loss


time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])
NUM_ROIS = 2

true_params = {
    'a01': 0.4,
    'a10': 0.4,
    'c0': 0.5,
    'c1': 0.5,
}

# Generate "true" data using the known true parameters
true_bold = simulate_bold(true_params, time=time, u=u, num_rois=NUM_ROIS)
# true_bold = true_bold.reshape(-1, order='F')

# Calculate the loss landscape
a01_range = np.linspace(0, 1, 100)
loss_scape = [loss(a) for a in a01_range]

# %%
# Choose an initial guess for a01
initial_guess = [0.7]
loss_list = []

# Call the minimization routine
result = minimize(
    loss,
    initial_guess,
    options={'gtol': 1e-12, 'xrtol': 1e-12, 'ftol': 1e-12},
    method='L-BFGS-B',
)

# Extract the optimized parameter value
optimized_a01 = result.x[0]

# Print the results
print()
print("Optimization successful:", result.success)
print("Optimized a01:", optimized_a01)
print("Final loss:", result.fun)

# %%
plt.figure()
plt.plot(a01_range, loss_scape, label='Loss Landscape')
plt.axvline(optimized_a01, color='r', linestyle='--', label='Optimized a01')
plt.xlabel('a01')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# %%
