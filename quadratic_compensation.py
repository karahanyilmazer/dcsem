# %%
import matplotlib.pyplot as plt
import numpy as np

from utils import get_out_dir, set_style

set_style()
IMG_DIR = get_out_dir("img", "appendix")
LATEX_DIR = get_out_dir("latex", "figures")


# Model definition
def quadratic(theta, x):
    a, b, c = theta
    return a * x**2 + b * x + c


# Base parameters
theta_true = np.array([1.0, -12.0, 20.0])

# X-range similar to your inversion setup
x = np.linspace(-30, 30, 400)

# Several (a, c) combinations showing the negative trade-off
param_sets = [
    [1.0, -12.0, 20.0],  # true
    [1.2, -12.0, 18.0],  # ↑a, ↓c
    [0.8, -12.0, 22.0],  # ↓a, ↑c
    [1.0, -12.0, 20.0],  # baseline for comparison
]

plt.figure(figsize=(6, 4))
for a, b, c in param_sets[:-1]:
    y = quadratic([a, b, c], x)
    plt.plot(x, y, label=rf"$a={a:.1f},\, c={c:.1f}$")

# Highlight the true curve
plt.plot(x, quadratic(theta_true, x), "k--", lw=2, label="True (baseline)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trade-off Between Curvature (a) and Offset (c)")
plt.legend()

# Highlight regions where the curves are visually similar (centered x-values)
# These vertical lines indicate regions where the modeled curves overlap closely
for xv in [-10, 10]:
    plt.axvline(x=xv, color="gray", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(IMG_DIR / "quadratic_parameter_tradeoff.png")
plt.savefig(LATEX_DIR / "quadratic_parameter_tradeoff.pdf")
plt.show()

# %%
