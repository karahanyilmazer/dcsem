"""Forward-simulation demo — 1-layer DCM, 2 ROIs.

Cell-by-cell script. Open in VS Code's Interactive Window or Jupyter and
step through the ``# %%`` blocks to see how a small bidirectional 2-ROI
DCM responds to a brief boxcar stimulus delivered to ROI 0.
"""

# %% Imports + style + output dirs
import matplotlib.pyplot as plt
import numpy as np

from dcsem.models import DCM
from dcsem.utils import create_A_matrix, create_C_matrix, stim_boxcar
from utils import get_out_dir, set_style

set_style()
IMG_DIR = get_out_dir(type="img", subfolder="dcm")
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
FIG_NAME = "one_layer_two_roi"


# %% Time vector + stimulus
time = np.arange(100)  # seconds (1 s spacing)
u = stim_boxcar([[10, 2, 1]])  # onset=10s, duration=2s, amplitude=1


# %% Connectivity (A) and input (C) matrices
num_rois = 2
num_layers = 1

# Bidirectional connections; product a01·a10 must stay below
# self_connection^2 = 1 for the system to remain stable.
connections = [
    "R0, L0 -> R1, L0 = 0.8",
    "R1, L0 -> R0, L0 = 0.4",
]
A = create_A_matrix(num_rois, num_layers, connections, self_connections=-1)
print("A:\n", A)

input_connections = ["R0, L0 = 0.7", "R1, L0 = 0.0"]
C = create_C_matrix(num_rois, num_layers, input_connections)
print("C:\n", C)


# %% Forward simulate
dcm = DCM(num_rois, params={"A": A, "C": C})
bold, state_tc = dcm.simulate(time, u)


# %% Diagram of the DCM model (parameters labelled, saved as SVG/PNG/PDF)
from dcsem import plot_dcm_graph  # noqa: E402

fig, _ = plot_dcm_graph(dcm)
for ext in ("svg", "png", "pdf"):
    target = (LATEX_DIR if ext == "pdf" else IMG_DIR) / f"{FIG_NAME}_graph.{ext}"
    fig.savefig(target, bbox_inches="tight")
plt.show(block=False)


# %% Plot stimulus + per-ROI BOLD
fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

axs[0].plot(time, u(time), color="k")
axs[0].set_ylabel("Stimulus (a.u.)")
axs[0].set_title(r"\textbf{Simulated BOLD responses — 2-ROI DCM}")

for r in range(num_rois):
    axs[1].plot(time, bold[:, r], label=f"ROI {r}")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("BOLD signal (a.u.)")
axs[1].legend()

plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}.pdf")
plt.show(block=False)

# %%
