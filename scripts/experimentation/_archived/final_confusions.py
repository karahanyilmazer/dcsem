"""ARCHIVED - reads the unsuffixed ``results/models/bench/`` directory and the
old confusion-matrix file layout. Superseded by the workflow in
scripts/workflows/dcm_bench_comparison.py. Kept here for historical reference
only; do NOT run it.
"""

raise SystemExit(
    "scripts/experimentation/_archived/final_confusions.py is deprecated. "
    "Use scripts/workflows/dcm_bench_comparison.py instead."
)

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    get_colormap,
    get_out_dir,
    get_width_height_latex,
    initialize_parameters,
    set_style,
    simulate_bold,
    to_latex_label,
)

set_style()
IMG_DIR = get_out_dir(type="img", subfolder="bench_final")
LATEX_IMG_DIR = get_out_dir(type="latex", subfolder="figures")
LATEX_TEX_DIR = get_out_dir(type="latex", subfolder="tables")
MODEL_DIR = get_out_dir(type="model", subfolder="bench")
cmap = get_colormap("YlGnBu")
width, height = get_width_height_latex()

# %%

# Define Model Inversion data
model_inversion_data = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.36956522, 0.13043478, 0.13043478, 0.17391304, 0.19565217],
    [0.44761905, 0.07619048, 0.20952381, 0.11428571, 0.15238095],
    [0.35164835, 0.14285714, 0.16483516, 0.13186813, 0.20879121],
    [0.38888889, 0.14814815, 0.10185185, 0.14814815, 0.21296296],
]


labels = ["No Change", "a01", "a10", "c0", "c1"]
labels = ["No Change"] + [to_latex_label(label) for label in labels[1:]]
model_inversion_df = pd.DataFrame(model_inversion_data, index=labels, columns=labels)

# Define BENCH data
bench_data = [
    [0.87, 0.03, 0.08, 0.01, 0.00],
    [0.03, 0.84, 0.03, 0.03, 0.07],
    [0.03, 0.10, 0.62, 0.22, 0.03],
    [0.00, 0.00, 0.32, 0.68, 0.00],
    [0.00, 0.56, 0.03, 0.39, 0.02],
]
bench_df = pd.DataFrame(bench_data, index=labels, columns=labels)

# Plot side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(width, height))
sns.heatmap(
    model_inversion_df,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    cbar=False,
    square=True,
    ax=axes[0],
)
axes[0].set_title("Model Inversion")
axes[0].set_xlabel("Inferred Change")
axes[0].set_ylabel("Actual Change")

sns.heatmap(
    bench_df, annot=True, fmt=".2f", cmap=cmap, cbar=False, square=True, ax=axes[1]
)
axes[1].set_title("BENCH")
axes[1].set_xlabel("Inferred Change")
# axes[1].set_ylabel("Actual Change")

# Remove minor ticks from left and bottom
for ax in axes:
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.tick_params(which="both", left=False, bottom=False)

fig.suptitle(r"\textbf{Parameter Change Inference Confusion Matrices}", y=0.92)
plt.tight_layout()

plt.savefig(IMG_DIR / "change_inference_confusion_matrices.png")
plt.savefig(LATEX_IMG_DIR / "change_inference_confusion_matrices.pdf")
plt.show()


# %%
def confusion_metrics(conf_matrix, labels):
    # Assuming rows normalized
    recall = np.diag(conf_matrix)
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    f1 = 2 * (precision * recall) / (precision + recall)
    overall_acc = np.mean(recall)

    df = pd.DataFrame(
        {"Precision": precision, "Recall": recall, "F1": f1}, index=labels
    )

    return df, overall_acc


df_model, acc_model = confusion_metrics(model_inversion_data, labels)
df_bench, acc_bench = confusion_metrics(bench_data, labels)

print("Model Inversion accuracy:", acc_model)
print(df_model.round(2))
print("\nBENCH accuracy:", acc_bench)
print(df_bench.round(2))

with open(LATEX_TEX_DIR / "metrics_inversion.tex", "w") as f:
    f.write(df_model.to_latex(float_format="%.2f", caption="Model Inversion metrics"))

with open(LATEX_TEX_DIR / "metrics_bench.tex", "w") as f:
    f.write(df_bench.to_latex(float_format="%.2f", caption="BENCH metrics"))
# %%
