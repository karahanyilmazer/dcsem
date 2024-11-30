# %%
from typing import Callable, Tuple

import emcee
import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from IPython.display import Math, display


# %%
def mcmc(
    posterior: Callable,
    args: Tuple,
    p0: np.ndarray,
    cov=None,
    bounds=None,
    step_size=1e0,
    n_samples=1000,
    burnin=100,
    skips=10,
):
    """
    MCMC sampling of a distribution
    Args:

        posterior: The log pdf of distribution (Callable function like f(p0, *args)
        args: parameters of the distribution (tuple)
        p0: initial point (array like)
        cov: covariance of samples
        bounds: sample bounds (n_params, 2)

    Returns:
        Samples, pos
    """
    p0 = np.atleast_1d(p0)
    if cov is None:
        cov = np.eye(len(p0))

    cov = np.atleast_2d(cov)
    min_eig = np.linalg.eigvalsh(cov)[0]
    correction = np.maximum(1e-14 - min_eig, 0)
    L1 = np.linalg.cholesky(cov + correction * np.eye(cov.shape[0])) / np.sqrt(len(p0))

    if bounds is None:
        bounds = np.array([[-np.inf, np.inf]] * len(p0))

    assert (p0 >= bounds[:, 0]).all() and (
        p0 <= bounds[:, 1]
    ).all(), "P0 doesnt fit to the bounds"

    current = np.array(p0)
    prob_cur = posterior(current, *args)
    samples = []
    all_probs = []
    jumps = n_samples * skips + burnin
    for j in range(jumps):
        proposed = current + L1 @ np.random.randn(*current.shape) * step_size
        if (proposed >= bounds[:, 0]).all() and (proposed <= bounds[:, 1]).all():
            prob_next = posterior(proposed, *args)
            diff = np.clip(
                prob_next - prob_cur, -700, 700
            )  # to avoid overflow/underflow
            if np.exp(diff) > np.random.rand():
                current = proposed
                prob_cur = prob_next

        samples.append(current)
        all_probs.append(prob_cur)
    return (
        np.squeeze(np.stack(samples, axis=0))[burnin::skips],
        np.array(all_probs)[burnin::skips],
    )


def iterative_mcmc(
    posterior,
    args,
    p0,
    n_samples=1000,
    bounds=None,
    repeats=10,
    burnin=100,
    skips=10,
    starting_step_size=1e-3,
):
    """MCMC sampling with adjusting the covariance"""
    p0 = np.array(p0)
    current_cov = np.eye(len(p0)) * starting_step_size**2
    start = np.array(p0)
    for _ in range(repeats):
        results, _ = mcmc(
            posterior,
            args,
            start,
            current_cov,
            bounds=bounds,
            step_size=1,
            n_samples=n_samples // repeats + len(p0),
            burnin=0,
            skips=1,
        )
        start = results[-1]
        current_cov = np.cov(results.T)

    return mcmc(
        posterior,
        args,
        start,
        current_cov,
        bounds=bounds,
        n_samples=n_samples,
        burnin=burnin,
        skips=skips,
        step_size=1,
    )


def log_probability(slope, x, y, intercept):
    """
    Log-probability function for a linear model with fixed intercept.

    Args:
        slope (float): Slope parameter to estimate.
        x (array): Independent variable.
        y (array): Dependent variable.
        intercept (float): Fixed intercept.

    Returns:
        float: Log-probability of the data given the parameter.
    """
    if not (0 < slope < 5):  # Uniform prior on slope
        return -np.inf
    y_model = slope * x + intercept  # Predicted y
    likelihood = -0.5 * np.sum((y - y_model) ** 2)  # Gaussian likelihood
    return likelihood


# %%
# Generate synthetic data
m_true = 2.5
c_true = 1.0
x = np.linspace(0, 10, 50)  # Independent variable
noise = np.random.normal(0, 1, size=len(x))  # Add noise
y = m_true * x + c_true + noise  # Dependent variable

plt.plot(x, m_true * x + c_true, ls='--', c='black', label='Ground Truth')
plt.scatter(x, y, s=10, c='tomato', label='Observed')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
# MCMC setup
n_dim = 1  # Number of parameters to estimate (mu)
n_walkers = 10  # Number of walkers (chains)
n_burn = 500  # Burn-in steps
n_samples = 1000  # Sampling steps
n_iterations = 1000
bounds = np.array([[0.0, 10.0]])  # Bounds for mu
step_size = 0.1  # Step size for `mcmc`

# Initialize walkers around a guess
initial_guess = [m_true + 1.1 * np.random.randn(n_dim) for _ in range(n_walkers)]

# Run MCMC using emcee
sampler = emcee.EnsembleSampler(
    n_walkers,
    n_dim,
    log_probability,
    args=(x, y, c_true),
)
sampler.run_mcmc(initial_guess, n_burn + n_samples, progress=True)
samples_emcee = sampler.get_chain(discard=n_burn, flat=True)

initial_guess = [2.5]  # Slightly off from true value

# Run basic MCMC
samples_mcmc, probs_mcmc = mcmc(
    posterior=log_probability,
    args=(x, y, c_true),
    p0=initial_guess,
    bounds=bounds,
    step_size=step_size,
    n_samples=n_samples,
    burnin=n_burn,
    skips=1,
)


# Run iterative MCMC with adaptive covariance
samples_iterative_mcmc, probs_iterative_mcmc = iterative_mcmc(
    posterior=log_probability,
    args=(x, y, c_true),
    p0=initial_guess,
    n_samples=n_samples,
    bounds=bounds,
    repeats=10,  # Number of iterations to adjust covariance
    burnin=n_burn,
    skips=1,
    starting_step_size=0.01,  # Starting step size for adaptive covariance
)


mean_mcmc = np.mean(samples_mcmc, axis=0)
std_mcmc = np.std(samples_mcmc, axis=0)

mean_iterative_mcmc = np.mean(samples_iterative_mcmc, axis=0)
std_iterative_mcmc = np.std(samples_iterative_mcmc, axis=0)

mean_emcee = np.mean(samples_emcee, axis=0)
std_emcee = np.std(samples_emcee, axis=0)

print(f'MCMC Results:\tMean: {mean_mcmc:.2f}, Std: {std_mcmc:.2f}')
print(f'iMCMC Results:\tMean: {mean_iterative_mcmc:.2f}, Std: {std_iterative_mcmc:.2f}')
print(f'emcee Results:\tMean: {mean_emcee[0]:.2f}, Std: {std_emcee[0]:.2f}')

plt.figure(figsize=(12, 6))
plt.hist(samples_mcmc, bins=30, alpha=0.5, label='MCMC', density=True)
plt.hist(samples_iterative_mcmc, bins=30, alpha=0.5, label='iMCMC', density=True)
plt.hist(samples_emcee, bins=30, alpha=0.5, label='emcee', density=True)
plt.axvline(m_true, color='r', linestyle='--', label='True mu')
plt.xlabel('mu')
plt.ylabel('Density')
plt.title('Comparison of MCMC Results')
plt.legend()
plt.show()
# %%
fig = corner(samples_emcee, labels=['Slope'], truths=[m_true])
plt.show()

# %%
inds = np.random.randint(len(samples_emcee), size=100)
for ind in inds:
    sample = samples_emcee[ind]
    m_pred = sample
    plt.plot(x, x * sample + c_true, 'C4', alpha=0.1)
# plt.errorbar(x, y, yerr=yerr, fmt='.k', capsize=0)
plt.scatter(x, y, color='k', s=5, label='observed')
plt.plot(x, m_true * x + c_true, 'k', label='truth')
plt.legend()
plt.xlim(0, 10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
labels = ['m']
for i in range(n_dim):
    mcmc_est = np.percentile(samples_emcee[:, i], [16, 50, 84])
    q = np.diff(mcmc_est)
    txt = rf"\mathrm{{{labels[i]}}} = {mcmc_est[1]:.3f}_{{-{q[0]:.3f}}}^{{{q[1]:.3f}}}"
    display(Math(txt))

# %%
