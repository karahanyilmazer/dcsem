# %%
# !%load_ext autoreload
# !%autoreload 2
import pickle

import matplotlib.pyplot as plt
import numpy as np
from bench import change_model
from scipy.stats import uniform
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

from dcsem import NOISE_CONFIG, PARAM_BOUNDS, get_colormap, set_style
from dcsem.utils import stim_boxcar
from utils import (
    get_out_dir,
    get_summary_measures,
    get_width_height_latex,
    simulate_bold,
)

n_comps = 4
setting = f"no_noise_{n_comps}"
# setting = f"with_noise_{n_comps}"

set_style()
IMG_DIR = get_out_dir(type="img", subfolder=f"bench_final_{setting}")
LATEX_DIR = get_out_dir(type="latex", subfolder="figures")
MODEL_DIR = get_out_dir(type="model", subfolder=f"bench_{setting}")
cmap = get_colormap("YlGnBu")
width, height = get_width_height_latex()

SEED = 42
rng = np.random.default_rng(SEED)

# %%
# ======================================================================================
# Bilinear neural model parameters
NUM_LAYERS = 1
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[10, 20, 1]])  # Input stimulus
# u = stim_boxcar([[0, 10, 1], [40, 10, 0.5], [50, 20, 1]])

# Parameters to set and estimate
params_to_set = ["a01", "a10", "c0", "c1"]

# Parameter bounds from central config
bounds = PARAM_BOUNDS.get_bounds_dict()

METHOD = "PCA"
if METHOD == "PCA":
    with open(MODEL_DIR / f"pca_{setting}.pkl", "rb") as f:
        pca = pickle.load(f)
elif METHOD == "ICA":
    with open(MODEL_DIR / f"ica_{setting}.pkl", "rb") as f:
        ica = pickle.load(f)


# ======================================================================================
# %%
def calc_comps(method, **kwargs):
    """Thin wrapper around get_summary_measures for BENCH Trainer compatibility."""
    return get_summary_measures(method, time, u, NUM_ROIS, MODEL_DIR, setting, **kwargs)


# Check if the function works
comps = calc_comps("PCA", a01=[0.5, 1.0], a10=[1.0, 0.7])
print("PCA components:\n", comps)

# %%
priors = {
    "a01": uniform(loc=bounds["a01"][0], scale=bounds["a01"][1] - bounds["a01"][0]),
    "a10": uniform(loc=bounds["a10"][0], scale=bounds["a10"][1] - bounds["a10"][0]),
    "c0": uniform(loc=bounds["c0"][0], scale=bounds["c0"][1] - bounds["c0"][0]),
    "c1": uniform(loc=bounds["c1"][0], scale=bounds["c1"][1] - bounds["c1"][0]),
}

tr = change_model.Trainer(
    forward_model=calc_comps,
    priors=priors,
    kwargs={"method": "PCA"},
    measurement_names=["PC1", "PC2", "PC3", "PC4"],
)
mdl = tr.train(n_samples=5000, verbose=True)

# %%
if setting.startswith("no_noise"):
    # No observation noise on summary measures; use a tiny floor for BENCH numerics
    noise_level = 1e-4
else:
    noise_sigma_path = MODEL_DIR / f"noise_sigmas_{METHOD.lower()}_{setting}.pkl"
    with open(noise_sigma_path, "rb") as f:
        noise_sigmas = pickle.load(f)  # noqa: S301 — trusted local artifact
    noise_level = np.mean(noise_sigmas)
n_test_samples = 2000
effect_size = 0.3
n_repeats = 50

true_change, data, data2, sn = tr.generate_test_samples(
    n_samples=n_test_samples,
    n_repeats=n_repeats,
    effect_size=effect_size,
    noise_std=noise_level,
)

probs, infered_change_bench, amount, _ = mdl.infer(data, data2 - data, sn)
print("Accuracy:", np.mean(infered_change_bench == true_change))

# %%
conf_mat = confusion_matrix(true_change, infered_change_bench, normalize="true")

fig, ax = plt.subplots(1, 1, figsize=(width / 2, width / 2))
heatmap(
    conf_mat,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    cbar=False,
    square=True,
    xticklabels=[
        r"$\mathrm{no\ change}$",
        r"$a_{01}$",
        r"$a_{10}$",
        r"$c_0$",
        r"$c_1$",
    ],
    yticklabels=[
        r"$\mathrm{no\ change}$",
        r"$a_{01}$",
        r"$a_{10}$",
        r"$c_0$",
        r"$c_1$",
    ],
    ax=ax,
)
ax.set_xlabel("Inferred Change")
ax.set_ylabel("Actual Change")
plt.title("BENCH")
plt.tick_params(axis="x", which="minor", bottom=False, top=False)
plt.tick_params(axis="y", which="minor", left=False, right=False)
plt.savefig(IMG_DIR / f"confusion_matrix_bench_{setting}.png")
plt.savefig(LATEX_DIR / f"confusion_matrix_bench_{setting}.pdf")
plt.show()


# %%
with open(MODEL_DIR / f"conf_bench_{setting}.pkl", "wb") as f:
    pickle.dump(conf_mat, f)

with open(MODEL_DIR / f"mdl_{setting}.pkl", "wb") as f:
    pickle.dump(mdl, f)
# %%
