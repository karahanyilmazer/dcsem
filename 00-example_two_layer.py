# %%
# !%load_ext autoreload
# !%autoreload 2
import matplotlib.pyplot as plt
import numpy as np

from dcsem import models, utils

# %%
TR = 1  # Repetition time
n_time = 100  # Number of time points
time = np.linspace(0, n_time * TR, n_time)  # Seconds

# Stimulus
u = utils.stim_boxcar([[0, 30, 1]])  # (onset, duration, amplitude)

# Connectivity parameters
num_rois = 1
num_layers = 2

# 1 ROI
A = utils.create_A_matrix(num_rois, num_layers, self_connections=-1)
print('A:\n', A)

# Input -> ROI0, Layer0 : c = 1
input_connections = ['R0, L0 = 1.0', 'R0, L1 = 1.0']
C = utils.create_C_matrix(num_rois, num_layers, input_connections)
print('C:\n', C)

# Iterate over different values of lambda
bold_tc = []
lambdas = [0.9, 0.8, 0.6, 0.4, 0.1, 0]

for l in lambdas:
    ldcm = models.TwoLayerDCM(num_rois, params={'A': A, 'C': C, 'l_d': l})
    bold_tc.append(ldcm.simulate(time, u)[0])

fig, axs = plt.subplots(1, 2)
axs[0].plot(bold_tc[0][:, 0], c=[0, 0, 0])

for bold, l in zip(bold_tc, lambdas):
    axs[1].plot(bold[:, 1], c=[l, l, l], label=fr'$\lambda_d$={l}')

# Adjust the plots
axs[0].set_title('Lower Layer')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('BOLD Signal')
axs[0].set_ylim([0, 0.1])
axs[0].grid()

axs[1].set_title('Upper Layer')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylim([0, 0.1])
axs[1].grid()
axs[1].legend()

plt.show()
# %%
