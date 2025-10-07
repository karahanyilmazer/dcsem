# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# Function to minimize
def f(x):
    y = (x**2) - (12 * x) + 20
    return y


# Starting guess
x_start = 2.0

# Collect iterates via callback (includes initial guess)
x_hist = [x_start]


def _callback(xk):
    # xk can be scalar or array-like depending on method
    try:
        x_val = float(xk[0])
    except Exception:
        x_val = float(xk)
    x_hist.append(x_val)


# optimizing
result = minimize(f, x_start, callback=_callback, options={"disp": True})

optimal_x = result.x[0]

x_plot = np.linspace(-5, 10, 1000)
plt.plot(x_plot, f(x_plot), label="f(x)")

# Plot optimization path (initial guess + per-iteration x)
x_hist_arr = np.array(x_hist)
y_hist_arr = f(x_hist_arr)
plt.plot(x_hist_arr, y_hist_arr, "o-", color="black", label="iterates")

# Mark initial guess and optimum
plt.axvline(x_start, color="gray", linestyle="--", label="x0")
plt.axvline(optimal_x, color="tomato", label="optimum")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

# %%
