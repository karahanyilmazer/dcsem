# %%
# !%load_ext autoreload
# !%autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dcsem.models import DCM
from dcsem.utils import create_A_matrix, create_C_matrix, stim_boxcar
from utils import get_out_dir, set_style

set_style()
IMG_DIR = get_out_dir(type="img", subfolder="dcm")
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
IMG_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Input
time = np.arange(200)  # Time vector (seconds)
# Stimulus function (onset, duration, amplitude)
u = stim_boxcar(
    [
        [10, 10, 1],
        [40, 10, 0.5],
        [45, 20, 1],
        [100, 30, 0.3],
        [120, 30, 0.1],
    ]
)
# u = stim_boxcar([[20, 10, 1], [60, 10, 0.5], [70, 20, 1]])

# Connectivity parameters
num_rois = 2
num_layers = 1

# ROI0, Layer0 -> ROI1, Layer0 : Magnitude = 0.2
connections = ["R0, L0 -> R1, L0 = 0.2"]
A = create_A_matrix(num_rois, num_layers, connections, self_connections=-1)
print("A:\n", A)

# Input -> ROI0, Layer0 : c = 1
input_connections = ["R0, L0 = 1.0"]
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
axs[1].plot(time, bold[:, 0], label="ROI 1")
axs[1].plot(time, bold[:, 1], label="ROI 2")

axs[0].set_title(r"\textbf{Simulated BOLD Responses in a Two-Region DCM}")
axs[1].set_xlabel("Time (s)")
axs[0].set_ylabel("Stimulus (a.u.)")
axs[1].set_ylabel("BOLD Signal (a.u.)")
axs[1].legend()

plt.savefig(IMG_DIR / "2roi_input_output.png")
plt.savefig(LATEX_DIR / "2roi_input_output.pdf")
plt.show()

# %%
