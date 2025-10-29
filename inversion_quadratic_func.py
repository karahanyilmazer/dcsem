# %% Imports and config
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import seaborn as sns
from pypalettes import load_cmap
from scipy.optimize import minimize

# Plot settings
CMAP = load_cmap("blues9", cmap_type="continuous")

# Reproducibility and data settings
SEED = 0
N_POINTS = 50
X_MIN, X_MAX = -5.0, 15.0
NOISE_SIGMA = 3.0  # set 0 for noiseless

# Ground-truth parameters and initial guess
theta_true = np.array([1.0, -12.0, 20.0])  # [a, b, c]
theta_zero = np.array([0.5, 0.0, 0.0])

# Plot toggles
PLOT_1D = True
PLOT_2D = True
PLOT_3D = False

# Landscape resolution
N_1D = 200
N_2D = 120

# Loss plot span overrides (None = auto)
SPAN_A = None
SPAN_B = None
SPAN_C = None

# Diagnostics toggles
SHOW_DIAGNOSTICS = True  # Print Hessian diagnostics


# %% Helpers: model, loss, gradient, Hessian
def design_matrix(x):
    # Features: [x^2, x, 1]
    return np.stack([x**2, x, np.ones_like(x)], axis=1)


def model(theta, x):
    a, b, c = theta
    return a * x**2 + b * x + c


# MSE loss over parameters theta = [a, b, c]
def mse(theta, x, y):
    y_pred = model(theta, x)
    return np.mean((y_pred - y) ** 2)


def hessian_analytical(x):
    """Analytical Hessian of MSE for quadratic model."""
    H = np.zeros((3, 3))
    H[0, 0] = 2 * np.mean(x**4)
    H[0, 1] = H[1, 0] = 2 * np.mean(x**3)
    H[0, 2] = H[2, 0] = 2 * np.mean(x**2)
    H[1, 1] = 2 * np.mean(x**2)
    H[1, 2] = H[2, 1] = 2 * np.mean(x)
    H[2, 2] = 2.0

    # # For linear-in-parameters models: H = 2 * E[phi phi^T]
    # Features: [x^2, x, 1]
    # Phi = np.stack([x**2, x, np.ones_like(x)], axis=1)
    # n = Phi.shape[0]
    # H = 2.0 * (Phi.T @ Phi) / n

    return H


def make_range(center, span, n):
    return np.linspace(center - span, center + span, n)


# %% Data generation
rng = np.random.default_rng(SEED)
x_data = np.linspace(X_MIN, X_MAX, N_POINTS)
y_clean = model(theta_true, x_data)
y_data = y_clean + rng.normal(0.0, NOISE_SIGMA, size=x_data.shape)

# %% Fit
obj = lambda th: mse(th, x_data, y_data)

res = minimize(obj, theta_zero, method="BFGS")
theta_est = res.x
a_est, b_est, c_est = theta_est.tolist()

# Basic summary
mse_est = obj(theta_est)

# %% Plot: data and fitted curve
x_plot = np.linspace(x_data.min(), x_data.max(), 400)
y_fit = model(theta_est, x_plot)

plt.scatter(x_data, y_data, s=20, alpha=0.7, label="data")
plt.plot(x_plot, y_fit, color="tomato", label="fitted")
plt.plot(x_plot, model(theta_true, x_plot), color="gray", linestyle="--", label="true")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

print("Fit:")
print(f"  theta_true: {theta_true}")
print(f"  theta_est : {np.round(theta_est, 4)}")
print(f"  MSE: {mse_est:.4f}  (noise var ~ {NOISE_SIGMA**2:.2f})")

# %% Loss landscapes - 1D (optional)
if PLOT_1D:
    # Auto spans if not provided
    span_a = SPAN_A if SPAN_A is not None else 1.5 * max(1.0, abs(a_est))
    span_b = SPAN_B if SPAN_B is not None else 0.7 * max(8.0, abs(b_est))
    span_c = SPAN_C if SPAN_C is not None else 0.7 * max(20.0, abs(c_est))

    grids_1d = {
        "a": make_range(a_est, span_a, N_1D),
        "b": make_range(b_est, span_b, N_1D),
        "c": make_range(c_est, span_c, N_1D),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (name, grid) in zip(axes, grids_1d.items()):
        losses = []
        for v in grid:
            th = np.array([a_est, b_est, c_est], dtype=float)
            if name == "a":
                th[0] = v
                true_v = theta_true[0]
                est_v = a_est
            elif name == "b":
                th[1] = v
                true_v = theta_true[1]
                est_v = b_est
            else:
                th[2] = v
                true_v = theta_true[2]
                est_v = c_est
            losses.append(mse(th, x_data, y_data))
        ax.plot(grid, losses)
        ax.axvline(true_v, color="gray", linestyle="--", label="true")
        ax.axvline(est_v, color="tomato", label="estimate")
        ax.set_xlabel(name)
        ax.set_title(f"MSE vs {name}")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    plt.tight_layout()
    plt.show()

# %% Loss landscapes - 2D contours (optional)
if PLOT_2D:
    # Use same spans as above (compute if 1D cell was skipped)
    span_a = SPAN_A if SPAN_A is not None else 1.5 * max(1.0, abs(a_est))
    span_b = SPAN_B if SPAN_B is not None else 0.7 * max(8.0, abs(b_est))
    span_c = SPAN_C if SPAN_C is not None else 0.7 * max(20.0, abs(c_est))

    a_grid = make_range(a_est, span_a, N_2D)
    b_grid = make_range(b_est, span_b, N_2D)
    c_grid = make_range(c_est, span_c, N_2D)

    X = x_data[None, None, :]
    YY = y_data[None, None, :]

    # (a, b) with c fixed
    AA, BB = np.meshgrid(a_grid, b_grid, indexing="ij")
    CC = c_est
    Zab = np.mean((AA[..., None] * (X**2) + BB[..., None] * X + CC - YY) ** 2, axis=-1)

    # (a, c) with b fixed
    AA2, CC2 = np.meshgrid(a_grid, c_grid, indexing="ij")
    BB2 = b_est
    Zac = np.mean(
        (AA2[..., None] * (X**2) + BB2 * X + CC2[..., None] - YY) ** 2, axis=-1
    )

    # (b, c) with a fixed
    BB3, CC3 = np.meshgrid(b_grid, c_grid, indexing="ij")
    AA3 = a_est
    Zbc = np.mean(
        (AA3 * (X**2) + BB3[..., None] * X + CC3[..., None] - YY) ** 2, axis=-1
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    cont = axes[0].contourf(AA, BB, Zab, levels=30, cmap="viridis")
    axes[0].scatter([a_est], [b_est], color="tomato", s=50, label="estimate")
    axes[0].scatter(
        [theta_true[0]],
        [theta_true[1]],
        color="white",
        edgecolors="black",
        s=40,
        label="true",
    )
    axes[0].set_xlabel("a")
    axes[0].set_ylabel("b")
    axes[0].set_title(f"(a, b) | c={c_est:.3g}")

    cont = axes[1].contourf(AA2, CC2, Zac, levels=30, cmap="viridis")
    axes[1].scatter([a_est], [c_est], color="tomato", s=50)
    axes[1].scatter(
        [theta_true[0]], [theta_true[2]], color="white", edgecolors="black", s=40
    )
    axes[1].set_xlabel("a")
    axes[1].set_ylabel("c")
    axes[1].set_title(f"(a, c) | b={b_est:.3g}")

    cont = axes[2].contourf(BB3, CC3, Zbc, levels=30, cmap="viridis")
    axes[2].scatter([b_est], [c_est], color="tomato", s=50)
    axes[2].scatter(
        [theta_true[1]], [theta_true[2]], color="white", edgecolors="black", s=40
    )
    axes[2].set_xlabel("b")
    axes[2].set_ylabel("c")
    axes[2].set_title(f"(b, c) | a={a_est:.3g}")

    plt.tight_layout()
    plt.colorbar(cont, ax=axes.ravel().tolist(), shrink=0.9, label="MSE")
    plt.show()

# %% Loss landscape - 3D surface (optional)
if PLOT_3D:
    a_grid = make_range(a_est, 1.5 * max(1.0, abs(a_est)), N_2D)
    b_grid = make_range(b_est, 0.7 * max(8.0, abs(b_est)), N_2D)
    AA, BB = np.meshgrid(a_grid, b_grid, indexing="ij")
    X = x_data[None, None, :]
    YY = y_data[None, None, :]
    CC = c_est
    Zab = np.mean((AA[..., None] * (X**2) + BB[..., None] * X + CC - YY) ** 2, axis=-1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        AA, BB, Zab, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95
    )
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("MSE")
    ax.set_title(f"MSE(a, b) | c={c_est:.3g}")
    z_est = mse([a_est, b_est, c_est], x_data, y_data)
    z_true_proj = mse([theta_true[0], theta_true[1], c_est], x_data, y_data)
    ax.scatter([a_est], [b_est], [z_est], color="red", s=50, label="estimate")
    ax.scatter(
        [theta_true[0]],
        [theta_true[1]],
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

# %% Hessian-based diagnostics (concise)
if SHOW_DIAGNOSTICS:
    # Calculate Hessian at the optimum
    hess_func = nd.Hessian(lambda theta: mse(theta, x_data, y_data))
    H = hess_func(theta_est)
    H_analytical = hessian_analytical(x_data)

    # Condition number (max-min eigenvalue ratio)
    eigvals = np.linalg.eigvalsh(H)
    cond = float(np.max(eigvals) / max(np.min(eigvals), 1e-12))

    # Parameter covariance (approx): sigma^2 * H^{-1}
    residuals = y_data - model(theta_est, x_data)
    sigma_sq_est = np.var(residuals, ddof=3)

    try:
        H_inv = np.linalg.inv(H)  # Inverse Hessian
        cov = sigma_sq_est * H_inv  # Covariance estimate
        se = np.sqrt(np.diag(cov))  # Standard errors

        # 95% confidence intervals
        ci = np.vstack([theta_est - 1.96 * se, theta_est + 1.96 * se]).T

        # Correlation
        denom = np.outer(se, se)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / denom
        max_offdiag_corr = np.nanmax(np.abs(corr - np.eye(3)))

    except np.linalg.LinAlgError:
        cov = None
        se = np.array([np.nan, np.nan, np.nan])
        max_offdiag_corr = np.nan

    # Plot the inverse Hessian
    ax = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=CMAP,
        # vmin=-1,
        # vmax=1,
        # cbar_kws={"label": "Parameter correlation"},
    )
    ax.set_title("Correlation Matrix")
    ax.set_xticklabels(["a", "b", "c"])
    ax.set_yticklabels(["a", "b", "c"])
    plt.tight_layout()
    plt.show()

    print("\nHessian diagnostics:")
    print(f"  Difference to analytical Hessian: {np.linalg.norm(H - H_analytical):.2e}")
    print(f"  Eigenvalues: {np.round(eigvals, 4)}")
    print(f"  Condition number: {cond:.2e}")
    print(f"  Estimated noise variance: {sigma_sq_est:.4f}, True: {NOISE_SIGMA**2:.4f}")
    print(f"  Standard error[a,b,c]: {np.round(se, 4)}")
    if np.isfinite(max_offdiag_corr):
        print(f"  Max. off-diagonal correlation: {max_offdiag_corr:.3f}")

    print("  95% CI rows [low, high] per parameter (a,b,c):")
    for i, (lo, hi) in enumerate(ci):
        print(
            f"    {['a','b','c'][i]}: [{lo:.4f}, {hi:.4f}] (True {['a','b','c'][i]}: {theta_true[i]:.4f})"
        )

# %%
