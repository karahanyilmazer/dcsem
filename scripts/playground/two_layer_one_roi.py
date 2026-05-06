"""Forward-simulation demo — 2-layer DCM, 1 ROI, sweep over blood-draining λ_d.

Cell-by-cell script. Open in VS Code's Interactive Window or Jupyter
and step through the ``# %%`` blocks. The blood-draining parameter
``λ_d`` controls how much of the deeper layer's hemodynamic response
leaks into the upper layer. The upper-layer panel shows that mixing
across a range of ``λ_d`` values; the lower-layer panel is unaffected
by ``λ_d`` (no draining flows into it from below).
"""

# %% Imports + style + output dirs
import matplotlib.pyplot as plt
import numpy as np

from dcsem.models import TwoLayerDCM
from dcsem.utils import create_A_matrix, create_C_matrix, stim_boxcar
from utils import get_out_dir, set_style

set_style()
IMG_DIR = get_out_dir(type="img", subfolder="dcm")
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
FIG_NAME = "two_layer_one_roi"


# %% Time vector + stimulus
TR = 1.0
n_time = 100
time = np.linspace(0, n_time * TR, n_time)
u = stim_boxcar([[0, 30, 1]])  # 30 s on, then off


# %% Connectivity (A) and input (C) matrices
num_rois = 1
num_layers = 2

A = create_A_matrix(num_rois, num_layers, self_connections=-1)
print("A:\n", A)

# Drive both layers equally; the difference between them comes from λ_d.
input_connections = ["R0, L0 = 1.0", "R0, L1 = 1.0"]
C = create_C_matrix(num_rois, num_layers, input_connections)
print("C:\n", C)


# %% Forward simulate across a sweep of λ_d
lambdas = [0.0, 0.1, 0.4, 0.6, 0.8, 0.9]
bold_tc = []
for l in lambdas:
    ldcm = TwoLayerDCM(num_rois, params={"A": A, "C": C, "l_d": l})
    bold_tc.append(ldcm.simulate(time, u)[0])


# %% Diagram of the layer-DCM (nodes = layers within the single ROI)
from dcsem import plot_dcm_graph  # noqa: E402

fig, _ = plot_dcm_graph(ldcm)
for ext in ("svg", "png", "pdf"):
    target = (LATEX_DIR if ext == "pdf" else IMG_DIR) / f"{FIG_NAME}_graph.{ext}"
    fig.savefig(target, bbox_inches="tight")
plt.show(block=False)


# %% Plot lower + upper layer BOLD (sweep colours = λ_d)
fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

# Lower layer is independent of λ_d — show one trace as ground truth.
axs[0].plot(time, bold_tc[0][:, 0], color="k")
axs[0].set_title("Lower layer")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("BOLD signal (a.u.)")
axs[0].grid(alpha=0.3)

# Upper layer absorbs draining from below — plot every λ_d.
for bold, l in zip(bold_tc, lambdas):
    axs[1].plot(time, bold[:, 1], color=str(l), label=rf"$\lambda_d$={l}")
axs[1].set_title("Upper layer")
axs[1].set_xlabel("Time (s)")
axs[1].grid(alpha=0.3)
axs[1].legend()

fig.suptitle(r"\textbf{Two-layer DCM — effect of blood-draining $\lambda_d$}")
plt.tight_layout()
plt.savefig(IMG_DIR / f"{FIG_NAME}.png")
plt.savefig(LATEX_DIR / f"{FIG_NAME}.pdf")
plt.show(block=False)

# %%
