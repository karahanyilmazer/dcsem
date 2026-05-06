"""Stochastic DCM trajectory statistics.

Tests the ``stochastic=True`` branch of ``DCM.integrate_x`` (Itô SDE via sdeint)
on the *neural* state alone — the haemodynamic ODE is deterministic given x(t).
We check three properties of the neural trajectory at stationarity:

1. Mean of x converges to 0 when there is no input (Wiener noise has zero mean).
2. Empirical variance matches the Lyapunov-equation prediction for ``dx = Ax dt
   + sigma dW``: ``A·Sigma + Sigma·A^T = -sigma^2 I`` ⇒ ``Sigma = sigma^2 / (-2λ)``
   for diagonal A = λ·I.
3. Autocorrelation decays with the expected time constant 1/|λ|.

These pin the SDE forward path. Tolerances are conservative because we use a
single realisation of moderate length.
"""

import numpy as np
import pytest

from dcsem import models, utils


def _neural_trajectory(seed: int, T: float = 1000.0, dt: float = 0.1,
                       lam: float = -1.0, sigma: float = 0.1):
    """Drive a 1-ROI stochastic DCM with no input, return the neural state.

    dt = 0.1 s is small enough that Euler-Maruyama discretisation bias on the
    stationary variance is well under 10 % for lam in [-2, -0.5].
    """
    A = np.array([[lam]])
    C = np.array([0.0])
    dcm = models.DCM(1, params={"A": A, "C": C}, stochastic=True)
    dcm.state_noise_std = sigma

    tvec = np.arange(0, T, dt)
    rng = np.random.default_rng(seed)
    x = dcm.integrate_x(tvec, u=None, generator=rng)  # shape (1, n_t)
    return tvec, x[0]


def test_stochastic_neural_mean_converges_to_zero():
    """No input + stable A ⇒ empirical mean of x near 0 at stationarity."""
    _, x = _neural_trajectory(seed=0)
    # Drop initial transient (first ~5 time-constants = 5 s)
    x_stat = x[500:]
    assert abs(np.mean(x_stat)) < 0.02


def test_stochastic_neural_variance_matches_lyapunov():
    """Stationary variance of x matches sigma^2 / (-2*lambda) within tolerance
    that accounts for Euler-Maruyama discretisation bias plus single-realisation
    sample noise."""
    lam, sigma = -1.0, 0.1
    _, x = _neural_trajectory(seed=1, lam=lam, sigma=sigma)
    x_stat = x[500:]
    var_emp = float(np.var(x_stat))
    var_theo = sigma**2 / (-2 * lam)  # = 0.005
    # 30 % relative tolerance: dt=0.1 ⇒ ~5 % discretisation high-bias for lam=-1,
    # plus ~10 % sample noise on var of var for ~1000 effective samples.
    assert abs(var_emp - var_theo) / var_theo < 0.3


def test_stochastic_neural_autocorr_decays_with_self_connection():
    """Autocorrelation of x(t) decays roughly exp(-|lambda| * tau)."""
    lam = -1.0
    dt = 0.1
    tvec, x = _neural_trajectory(seed=2, lam=lam, dt=dt)
    x_stat = x[500:] - np.mean(x[500:])

    # Lag of 1 s and 2 s in samples (dt = 0.1 s ⇒ 10 / 20 step lags).
    var = np.var(x_stat)
    ac_1s = np.mean(x_stat[:-10] * x_stat[10:]) / var
    ac_2s = np.mean(x_stat[:-20] * x_stat[20:]) / var
    # Theory: ac(tau) = exp(lam * tau) for stable A.
    expected_1s = np.exp(lam * 1.0)   # ≈ 0.368
    expected_2s = np.exp(lam * 2.0)   # ≈ 0.135
    assert abs(ac_1s - expected_1s) < 0.1
    assert abs(ac_2s - expected_2s) < 0.1


def test_stochastic_simulate_bold_finite_and_two_realisations_differ():
    """``simulate`` with stochastic=True returns finite BOLD; different rngs
    produce different realisations."""
    A = utils.create_A_matrix(num_rois=2, num_layers=1, self_connections=-2)
    C = utils.create_C_matrix(2, 1, input_connections=["R0,L0=1"])
    dcm = models.DCM(2, params={"A": A, "C": C}, stochastic=True)
    dcm.state_noise_std = 0.05

    tvec = np.linspace(0, 50, 200)
    u = utils.stim_boxcar([[0, 3, 1]])

    bold_a, _ = dcm.simulate(tvec, u=u, generator=np.random.default_rng(0))
    bold_b, _ = dcm.simulate(tvec, u=u, generator=np.random.default_rng(1))

    assert np.all(np.isfinite(bold_a))
    assert np.all(np.isfinite(bold_b))
    # Different seeds ⇒ different realisations, but means stay close
    assert not np.allclose(bold_a, bold_b)
    assert abs(np.mean(bold_a) - np.mean(bold_b)) < 0.1
