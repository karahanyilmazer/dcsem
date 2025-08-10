# %%
# !%load_ext autoreload
# !%autoreload 2
import matplotlib.pyplot as plt
import numpy as np

from dcsem.models import DCM
from dcsem.utils import create_A_matrix, create_C_matrix, stim_boxcar
from utils import set_style

set_style()

# %%
# Input
time = np.arange(200)  # Time vector (seconds)
# Stimulus function (onset, duration, amplitude)
# u = stim_boxcar([[0, 10, 1]])
u = stim_boxcar([[0, 10, 1], [40, 10, 0.5], [50, 20, 1]])

# Connectivity parameters
num_rois = 3
num_layers = 1

connections = []
# ROI0, Layer0 -> ROI1, Layer0 : Magnitude = 0.2
connections.append("R0, L0 -> R1, L0 = 0.2")
# ROI1, Layer0 -> ROI2, Layer0 : Magnitude = 0.4
connections.append("R1, L0 -> R2, L0 = 0.4")
# ROI2, Layer0 -> ROI0, Layer0 : Magnitude = -0.3
connections.append("R2, L0 -> R0, L0 = -0.3")
A = create_A_matrix(num_rois, num_layers, connections, self_connections=-1)
print("A:\n", A)

input_connections = []
# Input -> ROI0, Layer0 : c = 1
input_connections.append("R0, L0 = 1.0")
# Input -> ROI2, Layer0 : c = 0.5
input_connections.append("R2, L0 = 0.5")
C = create_C_matrix(num_rois, num_layers, input_connections)
print("C:\n", C)

# Instantiate the DCM object
dcm = DCM(num_rois, params={"A": A, "C": C})

# Run simulation to get BOLD signal
bold, state_tc = dcm.simulate(time, u)

# Normalize the BOLD signal
norm = False
if norm:
    bold = bold / np.max(bold, axis=0)

fig, axs = plt.subplots(2, 1)
axs[0].plot(time, u(time), label="Stimulus")
axs[1].plot(time, bold[:, 0], label="ROI 0")
axs[1].plot(time, bold[:, 1], label="ROI 1")
axs[1].plot(time, bold[:, 2], label="ROI 1")
axs[0].set_title("DCM Simulation")
axs[0].set_ylabel("Stimulus")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("BOLD Signal")
axs[1].legend()
plt.show()

# %%
