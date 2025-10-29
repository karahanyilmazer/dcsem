# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# Quadratic model: y = a x^2 + b x + c
def model(theta, x):
    a, b, c = theta
    return a * x**2 + b * x + c


# MSE loss over parameters theta = [a, b, c]
def mse(theta, x, y):
    y_pred = model(theta, x)
    return np.mean((y_pred - y) ** 2)


def _callback(xk):
    theta_hist.append(np.array(xk, dtype=float))


# Synthetic data (replace with your own x_data, y_data if available)
rng = np.random.default_rng(0)
x_data = np.linspace(-5, 15, 50)

# True parameters for data generation (match f(x) = x^2 - 12 x + 20)
a_true, b_true, c_true = 1.0, -12.0, 20.0
y_clean = model([a_true, b_true, c_true], x_data)
noise_sigma = 3.0  # set >0 to add noise, e.g., 1.0
y_data = y_clean + rng.normal(0, noise_sigma, size=x_data.shape)


# Initial guess for [a, b, c]
theta0 = np.array([0.5, 0.0, 0.0])

# Track parameter iterates
theta_hist = [theta0.copy()]


# Optimize MSE
result = minimize(
    mse,
    theta0,
    args=(x_data, y_data),
    callback=_callback,
    options={"disp": True},
)
theta_hat = result.x

print()
print("True parameters:\t", [a_true, b_true, c_true])
print("Estimated parameters:\t", theta_hat)


# Plot data and fitted curve
x_plot = np.linspace(x_data.min(), x_data.max(), 400)
y_fit = model(theta_hat, x_plot)

plt.scatter(x_data, y_data, s=20, alpha=0.7, label="data")
plt.plot(x_plot, y_fit, color="tomato", label="fitted")
plt.plot(
    x_plot,
    model([a_true, b_true, c_true], x_plot),
    color="gray",
    linestyle="--",
    label="true",
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# Loss landscapes: 1D for each param, 2D for each pair, 3D surface
# -------------------------------------------------------------

# Ranges around the estimated parameters
a_hat, b_hat, c_hat = theta_hat


def make_range(center, span, n):
    return np.linspace(center - span, center + span, n)


# Choose spans relative to magnitude for generality
span_a = 1.5 if abs(a_hat) < 1e-6 else 1.5 * max(1.0, abs(a_hat))
span_b = 8.0 if abs(b_hat) < 1e-6 else 0.7 * max(8.0, abs(b_hat))
span_c = 20.0 if abs(c_hat) < 1e-6 else 0.7 * max(20.0, abs(c_hat))

# 1D landscapes for each parameter
grids_1d = {
    "a": make_range(a_hat, span_a, 200),
    "b": make_range(b_hat, span_b, 200),
    "c": make_range(c_hat, span_c, 200),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (name, grid) in zip(axes, grids_1d.items()):
    if name == "a":
        losses = np.array([mse([v, b_hat, c_hat], x_data, y_data) for v in grid])
        ax.axvline(a_true, color="gray", linestyle="--", label="true")
        ax.axvline(a_hat, color="tomato", label="estimate")
    elif name == "b":
        losses = np.array([mse([a_hat, v, c_hat], x_data, y_data) for v in grid])
        ax.axvline(b_true, color="gray", linestyle="--", label="true")
        ax.axvline(b_hat, color="tomato", label="estimate")
    else:  # c
        losses = np.array([mse([a_hat, b_hat, v], x_data, y_data) for v in grid])
        ax.axvline(c_true, color="gray", linestyle="--", label="true")
        ax.axvline(c_hat, color="tomato", label="estimate")
    ax.plot(grid, losses)
    ax.set_xlabel(name)
    ax.set_title(f"MSE vs {name}")
axes[0].set_ylabel("MSE")
axes[0].legend()
plt.tight_layout()
plt.show()

# 2D landscapes for each parameter pair (contourf)
Na, Nb, Nc = 120, 120, 120
X = x_data[None, None, :]
YY = y_data[None, None, :]

# (a, b) with c fixed
a_grid = make_range(a_hat, span_a, Na)
b_grid = make_range(b_hat, span_b, Nb)
AA, BB = np.meshgrid(a_grid, b_grid, indexing="ij")
CC = c_hat
Zab = np.mean((AA[..., None] * (X**2) + BB[..., None] * X + CC - YY) ** 2, axis=-1)

# (a, c) with b fixed
c_grid = make_range(c_hat, span_c, Nc)
AA2, CC2 = np.meshgrid(a_grid, c_grid, indexing="ij")
BB2 = b_hat
Zac = np.mean((AA2[..., None] * (X**2) + BB2 * X + CC2[..., None] - YY) ** 2, axis=-1)

# (b, c) with a fixed
BB3, CC3 = np.meshgrid(b_grid, c_grid, indexing="ij")
AA3 = a_hat
Zbc = np.mean((AA3 * (X**2) + BB3[..., None] * X + CC3[..., None] - YY) ** 2, axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
cont = axes[0].contourf(AA, BB, Zab, levels=30, cmap="viridis")
axes[0].scatter([a_hat], [b_hat], color="tomato", s=50, label="estimate")
axes[0].scatter(
    [a_true], [b_true], color="white", edgecolors="black", s=40, label="true"
)
axes[0].set_xlabel("a")
axes[0].set_ylabel("b")
axes[0].set_title(f"(a,b) | c={c_hat:.3g}")

cont = axes[1].contourf(AA2, CC2, Zac, levels=30, cmap="viridis")
axes[1].scatter([a_hat], [c_hat], color="tomato", s=50)
axes[1].scatter([a_true], [c_true], color="white", edgecolors="black", s=40)
axes[1].set_xlabel("a")
axes[1].set_ylabel("c")
axes[1].set_title(f"(a,c) | b={b_hat:.3g}")

cont = axes[2].contourf(BB3, CC3, Zbc, levels=30, cmap="viridis")
axes[2].scatter([b_hat], [c_hat], color="tomato", s=50)
axes[2].scatter([b_true], [c_true], color="white", edgecolors="black", s=40)
axes[2].set_xlabel("b")
axes[2].set_ylabel("c")
axes[2].set_title(f"(b, c) | a={a_hat:.3g}")

plt.tight_layout()
cbar = plt.colorbar(cont, ax=axes.ravel().tolist(), shrink=0.9, label="MSE")
plt.show()

# 3D surface: z = MSE(a,b | c fixed)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(
    AA, BB, Zab, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95
)
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_zlabel("MSE")
ax.set_title(f"3D surface: MSE(a,b) | c={c_hat:.3g}")
# Mark estimate and true projection onto this surface (c fixed)
z_hat = mse([a_hat, b_hat, c_hat], x_data, y_data)
z_true_proj = mse([a_true, b_true, c_hat], x_data, y_data)
ax.scatter([a_hat], [b_hat], [z_hat], color="red", s=50, label="estimate")
ax.scatter(
    [a_true],
    [b_true],
    [z_true_proj],
    color="white",
    edgecolors="black",
    s=45,
    label="true (c fixed)",
)
fig.colorbar(surf, ax=ax, shrink=0.7, aspect=12, pad=0.1, label="MSE")
ax.legend(loc="best")
plt.tight_layout()
plt.show()

# %%
