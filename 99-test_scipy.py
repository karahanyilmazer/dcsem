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

# %%
