# %%
import matplotlib.pyplot as plt
import numdifftools as nd
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

# %% HESSIAN-BASED DIAGNOSTICS
# Create Hessian function
hess_func = nd.Hessian(lambda theta: mse(theta, x_data, y_data))
H = hess_func(theta_hat)
print("\nHessian (numdifftools):")
print(H)

print("\n" + "=" * 60)
print("HESSIAN-BASED DIAGNOSTICS")
print("=" * 60)

# 1. Check for degeneracy via eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(H)
print("\n1. Eigenvalue Analysis (Degeneracy Check)")
print("-" * 40)
print(f"Eigenvalues: {eigenvalues}")
print(
    f"Condition number: {np.max(eigenvalues) / np.max([np.min(eigenvalues), 1e-10]):.2e}"
)

# Check for near-zero or negative eigenvalues
min_eig = np.min(eigenvalues)
if min_eig < 1e-6:
    print(
        f"⚠️  WARNING: Near-zero eigenvalue ({min_eig:.2e}) - model may be degenerate!"
    )
    # Find which parameter combination is problematic
    idx = np.argmin(eigenvalues)
    print(f"   Problematic direction: {eigenvectors[:, idx]}")
elif min_eig < 0:
    print(f"⚠️  WARNING: Negative eigenvalue ({min_eig:.2e}) - not at a minimum!")
else:
    print(f"✓  All eigenvalues positive - well-defined minimum")

# 2. Parameter uncertainty (standard errors)
print("\n2. Parameter Uncertainty")
print("-" * 40)
# For MSE, the Hessian relates to Fisher Information
# Covariance matrix ≈ σ² * H^(-1), where σ² is the noise variance
try:
    H_inv = np.linalg.inv(H)

    # Estimate noise variance from residuals
    residuals = y_data - model(theta_hat, x_data)
    sigma_sq_hat = np.var(residuals, ddof=3)  # ddof=3 for 3 parameters

    # Parameter covariance matrix
    cov_matrix = sigma_sq_hat * H_inv

    # Standard errors
    param_std = np.sqrt(np.diag(cov_matrix))
    param_names = ["a", "b", "c"]
    true_params = [a_true, b_true, c_true]

    print(f"Estimated noise variance: {sigma_sq_hat:.4f}")
    print(f"True noise variance: {noise_sigma**2:.4f}\n")

    print("Parameter estimates ± std error:")
    for i, name in enumerate(param_names):
        print(
            f"  {name}: {theta_hat[i]:8.4f} ± {param_std[i]:.4f}  "
            f"(true: {true_params[i]:8.4f})"
        )

    # Check if true parameters are within confidence intervals
    print("\n95% Confidence Intervals:")
    all_within = True
    for i, name in enumerate(param_names):
        ci_lower = theta_hat[i] - 1.96 * param_std[i]
        ci_upper = theta_hat[i] + 1.96 * param_std[i]
        within = ci_lower <= true_params[i] <= ci_upper
        all_within &= within
        status = "✓" if within else "✗"
        print(f"  {status} {name}: [{ci_lower:8.4f}, {ci_upper:8.4f}]")

    if all_within:
        print("\n✓  All true parameters within 95% CI")

except np.linalg.LinAlgError:
    print("⚠️  WARNING: Hessian is singular - model is degenerate!")
    print("   Cannot compute parameter uncertainties.")

# 3. Correlation between parameters
print("\n3. Parameter Correlations")
print("-" * 40)
try:
    # Correlation matrix from covariance
    corr_matrix = cov_matrix / np.outer(param_std, param_std)

    print("Correlation matrix:")
    print("       a       b       c")
    for i, name in enumerate(param_names):
        row_str = f"{name}  "
        for j in range(3):
            row_str += f"{corr_matrix[i,j]:7.3f} "
        print(row_str)

    # Check for high correlations (potential identifiability issues)
    high_corr_threshold = 0.95
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(corr_matrix[i, j]) > high_corr_threshold:
                print(
                    f"\n⚠️  High correlation between {param_names[i]} and {param_names[j]}: "
                    f"{corr_matrix[i,j]:.3f}"
                )
                print("   Parameters may be difficult to estimate independently.")

except:
    pass

# 4. Curvature analysis
print("\n4. Loss Landscape Curvature")
print("-" * 40)
print(f"Determinant of Hessian: {np.linalg.det(H):.2e}")
print(f"Trace of Hessian: {np.trace(H):.4f}")
print(f"Frobenius norm: {np.linalg.norm(H, 'fro'):.4f}")

# Effective dimensionality
eig_sum = np.sum(eigenvalues)
if eig_sum > 0:
    eff_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues**2)
    print(f"Effective dimensionality: {eff_dim:.2f} / 3")
    if eff_dim < 2.5:
        print("⚠️  Low effective dimensionality suggests redundant parameters")

# 5. Visualize Hessian structure
print("\n5. Hessian Visualization")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Heatmap of Hessian
im = axes[0].imshow(H, cmap="RdBu_r", aspect="auto")
axes[0].set_xticks(range(3))
axes[0].set_yticks(range(3))
axes[0].set_xticklabels(param_names)
axes[0].set_yticklabels(param_names)
axes[0].set_title("Hessian Matrix")
plt.colorbar(im, ax=axes[0])

# Eigenvalue spectrum
axes[1].bar(range(3), eigenvalues, color="steelblue", alpha=0.7)
axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
axes[1].set_xlabel("Eigenvalue index")
axes[1].set_ylabel("Eigenvalue")
axes[1].set_title("Eigenvalue Spectrum")
axes[1].set_xticks(range(3))

# Correlation matrix
try:
    im2 = axes[2].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_xticks(range(3))
    axes[2].set_yticks(range(3))
    axes[2].set_xticklabels(param_names)
    axes[2].set_yticklabels(param_names)
    axes[2].set_title("Parameter Correlation")
    plt.colorbar(im2, ax=axes[2])

    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            text = axes[2].text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
except:
    axes[2].text(
        0.5,
        0.5,
        "Could not compute\ncorrelation matrix",
        ha="center",
        va="center",
        transform=axes[2].transAxes,
    )
    axes[2].set_title("Parameter Correlation")

plt.tight_layout()
plt.show()

# 6. Compare numerical vs analytical Hessian (for validation)
print("\n6. Numerical Validation")
print("-" * 40)


def hessian_analytical(theta, x, y):
    """Analytical Hessian of MSE for quadratic model."""
    H = np.zeros((3, 3))
    H[0, 0] = 2 * np.mean(x**4)
    H[0, 1] = H[1, 0] = 2 * np.mean(x**3)
    H[0, 2] = H[2, 0] = 2 * np.mean(x**2)
    H[1, 1] = 2 * np.mean(x**2)
    H[1, 2] = H[2, 1] = 2 * np.mean(x)
    H[2, 2] = 2.0
    return H


H_analytical = hessian_analytical(theta_hat, x_data, y_data)
diff = np.abs(H - H_analytical)
print(f"Max difference (numerical vs analytical): {np.max(diff):.2e}")
if np.max(diff) < 1e-4:
    print("✓  Numerical Hessian agrees with analytical")
else:
    print("⚠️  Large discrepancy - check numerical precision")

# %%
