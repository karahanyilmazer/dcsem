# %% Imports and config
import warnings

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Reproducibility and data settings
SEED = 0
N_POINTS = 50
X_MIN, X_MAX = -5.0, 15.0
NOISE_SIGMA = 5.0  # known observation noise std (set 0 for noiseless)

# Guard against zero/invalid noise (delta-like likelihood breaks MCMC)
if not np.isfinite(NOISE_SIGMA) or NOISE_SIGMA <= 0:
    warnings.warn(
        "NOISE_SIGMA <= 0 detected. Clamping to 1e-6 for numerical stability."
    )
    NOISE_SIGMA = 1e-6

# Ground-truth parameters and initial guess
THETA_TRUE = np.array([1.0, -12.0, 20.0])  # [a, b, c]
THETA_0 = np.array([0.5, 0.0, 0.0])

# MCMC settings
N_WALKERS = 24  # should be >= 2 * ndim
N_BURN = 2000
N_SAMPLES = 4000

# Plot toggles
PLOT_CORNER = True
PLOT_POSTERIOR_BANDS = True

# Landscape toggles (keep structure similar, but disabled by default to avoid overkill)
PLOT_1D = False
PLOT_2D = False
PLOT_3D = False

# Landscape resolution / spans (if you enable the above)
N_1D = 200
N_2D = 120
SPAN_A = None
SPAN_B = None
SPAN_C = None


# %% Helpers: model, loss, log-prob
def model(theta, x):
    a, b, c = theta
    return a * x**2 + b * x + c


def mse(theta, x, y):
    y_pred = model(theta, x)
    return np.mean((y_pred - y) ** 2)


def _lognorm(x, mu, sigma):
    """Log of univariate Normal density (up to numerical stability)."""
    z = (x - mu) / sigma
    return -0.5 * (z * z + np.log(2.0 * np.pi * sigma * sigma))


def log_prior(theta):
    """Independent Normal priors; broad and weakly-informative.

    a ~ N(0, 3^2), b ~ N(0, 20^2), c ~ N(0, 40^2)
    """
    a, b, c = theta
    return _lognorm(a, 0.0, 3.0) + _lognorm(b, 0.0, 20.0) + _lognorm(c, 0.0, 40.0)


def log_likelihood(theta, x, y, sigma):
    """Gaussian likelihood with known noise std sigma."""
    if sigma <= 0 or not np.isfinite(sigma):
        return -np.inf
    y_pred = model(theta, x)
    r = (y - y_pred) / sigma
    # sum over data points
    return -0.5 * (np.sum(r * r) + y.size * np.log(2.0 * np.pi * sigma * sigma))


def log_posterior(theta, x, y, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, x, y, sigma)
    return lp + ll


def make_range(center, span, n):
    return np.linspace(center - span, center + span, n)


def map_estimate(theta0, x, y, sigma):
    def neg_logpost(th):
        return -(log_posterior(th, x, y, sigma))

    res = optimize.minimize(neg_logpost, theta0, method="L-BFGS-B")
    return res.x


# %% Data generation
rng = np.random.default_rng(SEED)
x_data = np.linspace(X_MIN, X_MAX, N_POINTS)
y_clean = model(THETA_TRUE, x_data)
y_data = y_clean + rng.normal(0.0, NOISE_SIGMA, size=x_data.shape)


# %% MCMC: run sampler
ndim = THETA_TRUE.size
nwalkers = max(N_WALKERS, 2 * ndim)

theta_center = map_estimate(THETA_0, x_data, y_data, NOISE_SIGMA)

scale = np.maximum(0.05 * np.ones_like(theta_center), 0.05 * np.abs(theta_center))
rng = np.random.default_rng(SEED)
p0 = theta_center + rng.normal(0.0, scale, size=(nwalkers, ndim))

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior, args=(x_data, y_data, NOISE_SIGMA)
)

# Burn-in then production to allow progress reporting separation
state = sampler.run_mcmc(p0, N_BURN, progress=True)
sampler.reset()
sampler.run_mcmc(state, N_SAMPLES, progress=True)

chain = sampler.get_chain(flat=True)  # shape: (N_SAMPLES*nwalkers, ndim)
logp = sampler.get_log_prob(flat=True)

mask = np.isfinite(logp)
chain = chain[mask]
logp = logp[mask]

# Summaries
theta_mean = np.mean(chain, axis=0)
theta_median = np.median(chain, axis=0)
map_idx = int(np.argmax(logp))
theta_map = chain[map_idx]


# %% Basic summary and data/fit plot
x_plot = np.linspace(x_data.min(), x_data.max(), 400)
y_mean = model(theta_mean, x_plot)
y_true = model(THETA_TRUE, x_plot)

plt.scatter(x_data, y_data, s=20, alpha=0.7, label="data")
plt.plot(x_plot, y_mean, color="tomato", label="posterior mean fit")
plt.plot(x_plot, y_true, color="gray", linestyle="--", label="true")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

print("Posterior summary (quadratic y = ax^2 + bx + c):")
print(f"  theta_true : {np.round(THETA_TRUE, 4)}")
print(f"  mean       : {np.round(theta_mean, 4)}")
print(f"  median     : {np.round(theta_median, 4)}")
print(f"  MAP        : {np.round(theta_map, 4)}")
q025, q50, q975 = np.percentile(chain, [2.5, 50.0, 97.5], axis=0)
print(
    "  95% CI     : ["
    + ", ".join(f"{lo:.3f}, {hi:.3f}" for lo, hi in zip(q025.tolist(), q975.tolist()))
    + "]"
)


# %% Posterior predictive band (optional)
if PLOT_POSTERIOR_BANDS:
    nsamp = min(400, chain.shape[0])
    idx = rng.choice(chain.shape[0], size=nsamp, replace=False)
    thetas = chain[idx]
    # Vectorized: (nsamp, n_grid)
    A = thetas[:, 0][:, None]
    B = thetas[:, 1][:, None]
    C = thetas[:, 2][:, None]
    Xgrid = x_plot[None, :]
    Y = A * Xgrid**2 + B * Xgrid + C
    y_lo = np.percentile(Y, 2.5, axis=0)
    y_hi = np.percentile(Y, 97.5, axis=0)

    plt.scatter(x_data, y_data, s=18, alpha=0.6, label="data")
    plt.plot(x_plot, y_true, color="gray", linestyle="--", label="true")
    plt.plot(x_plot, y_mean, color="tomato", label="posterior mean")
    plt.fill_between(
        x_plot, y_lo, y_hi, color="tomato", alpha=0.2, label="95% posterior band"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% Minimal diagnostics
try:
    tau = sampler.get_autocorr_time(quiet=True)
    eff_per_walker = N_SAMPLES / tau
    eff_total = np.sum(eff_per_walker)
    tau_str = np.round(tau, 1)
except Exception:
    tau_str = "n/a"
    eff_total = np.nan

acc_frac = np.mean(sampler.acceptance_fraction)

print("Diagnostics:")
print(f"  acceptance fraction (mean): {acc_frac:.3f}")
print(f"  autocorr time (per dim)   : {tau_str}")
if np.isfinite(eff_total):
    print(f"  approx. effective samples : {int(eff_total)}")
if acc_frac < 0.05 or acc_frac > 0.8:
    print(
        "  ⚠️  Acceptance fraction outside [0.05, 0.8]. Consider adjusting initial spread or priors."
    )


# %% Corner plot (optional)
if PLOT_CORNER:
    fig = corner.corner(
        chain,
        labels=[r"$a$", r"$b$", r"$c$"],
        truths=THETA_TRUE,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        bins=50,
        smooth=0.8,
    )

    # # Get the axes array from the corner figure
    # axes = np.array(fig.axes).reshape((3, 3))  # ndim x ndim

    # for i, true_val in enumerate(THETA_TRUE):
    #     ax = axes[i, i]  # diagonal histogram
    #     ax.axvline(true_val, color="blue", lw=1.2, ls="--", alpha=0.8)
    #     ax.axvline(np.median(chain[:, i]), color="tomato", lw=1.2, ls="-", alpha=0.7)
    plt.show()


# %% (Optional) Loss landscapes — disabled by default to avoid overkill
if PLOT_1D or PLOT_2D or PLOT_3D:
    # This section mirrors the structure from 99-test_scipy.py and can be
    # enabled if you want comparable visualizations. Left minimal here.
    a_hat, b_hat, c_hat = theta_mean.tolist()

    def make_spans():
        span_a = SPAN_A if SPAN_A is not None else 1.5 * max(1.0, abs(a_hat))
        span_b = SPAN_B if SPAN_B is not None else 0.7 * max(8.0, abs(b_hat))
        span_c = SPAN_C if SPAN_C is not None else 0.7 * max(20.0, abs(c_hat))
        return span_a, span_b, span_c

    if PLOT_1D:
        span_a, span_b, span_c = make_spans()
        grids_1d = {
            "a": make_range(a_hat, span_a, N_1D),
            "b": make_range(b_hat, span_b, N_1D),
            "c": make_range(c_hat, span_c, N_1D),
        }
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        for ax, (name, grid) in zip(axes, grids_1d.items()):
            losses = []
            for v in grid:
                th = np.array([a_hat, b_hat, c_hat], dtype=float)
                if name == "a":
                    th[0] = v
                elif name == "b":
                    th[1] = v
                else:
                    th[2] = v
                losses.append(mse(th, x_data, y_data))
            ax.plot(grid, losses)
            ax.axvline(
                THETA_TRUE[0 if name == "a" else 1 if name == "b" else 2],
                color="gray",
                linestyle="--",
                label="true",
            )
            ax.axvline(
                [a_hat, b_hat, c_hat][0 if name == "a" else 1 if name == "b" else 2],
                color="tomato",
                label="estimate",
            )
            ax.set_xlabel(name)
            ax.set_title(f"MSE vs {name}")
        axes[0].set_ylabel("MSE")
        axes[0].legend()
        plt.tight_layout()
        plt.show()
