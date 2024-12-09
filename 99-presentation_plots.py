# %%
# !%load_ext autoreload
# !%autoreload 2
import matplotlib.pyplot as plt
import numpy as np

from dcsem import models, utils

# %%
time = np.arange(200)  # Time vector (seconds)
# Stimulus function (onset, duration, amplitude)
u = utils.stim_boxcar([[0, 10, 1]])
# u = utils.stim_boxcar([[0, 10, 1], [40, 10, 0.5], [50, 20, 1]])

# Connectivity parameters
num_rois = 2
num_layers = 1

# ROI0, Layer0 -> ROI1, Layer0 : Magnitude = 0.2
connections = ['R0, L0 -> R1, L0 = 0.2']
A = utils.create_A_matrix(num_rois, num_layers, connections, self_connections=-1)
print('A:\n', A)

# Input -> ROI0, Layer0 : c = 1
input_connections = ['R0, L0 = 1.0']
C = utils.create_C_matrix(num_rois, num_layers, input_connections)
print('C:\n', C)

# Instantiate the DCM object
dcm = models.DCM(num_rois, params={'A': A, 'C': C})
# Run simulation to get BOLD signal
bold, state_tc = dcm.simulate(time, u)

plt.figure(figsize=(4, 4))
plt.plot(bold[:, 0], lw=5, color='black')
plt.xlim(0, 200)
plt.xticks([])
plt.yticks([])

# Remove the frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('img/presentation/bold_control.png', dpi=300)
plt.show()
# %%
# Instantiate the DCM object
dcm = models.DCM(num_rois, params={'A': A, 'C': C, 'alpha': 0.9, 'tau': 30})
# Run simulation to get BOLD signal
bold, state_tc = dcm.simulate(time, u)

plt.figure(figsize=(4, 4))
plt.plot(bold[:, 0], lw=5, color='#EA5451')
plt.xlim(0, 200)
plt.xticks([])
plt.yticks([])

# Remove the frame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig('img/presentation/bold_affected.png', dpi=300)
plt.show()

# %%
plt.figure(figsize=(10, 4))
plt.plot(time, u(time), lw=2, c='black')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('Stimulus Function')
plt.tight_layout()
plt.savefig('img/presentation/stimulus.png', dpi=300)
plt.show()
