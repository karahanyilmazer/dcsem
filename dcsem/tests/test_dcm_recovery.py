"""End-to-end parameter-recovery tests for the time-domain DCM (audit H2).

These pin current behaviour: simulate with a known theta, fit, assert recovery.
If anyone changes the forward model in a way that biases recovery, these fail.
"""

import numpy as np
import pytest

from dcsem import models, utils


def _build_2roi_dcm():
    """A modest, stable 2-ROI DCM with one off-diagonal and one input."""
    dcm = models.DCM(2)
    A_true = np.array([[-1.5, 0.0], [0.4, -1.5]])
    C_true = np.array([0.8, 0.0])
    dcm.set_params({"A": A_true, "C": C_true})
    return dcm, A_true, C_true


def test_dcm_recovers_offdiag_and_input_from_clean_bold():
    """Clean (noise-free) BOLD: full-precision recovery of free parameters."""
    dcm, A_true, C_true = _build_2roi_dcm()
    tvec = np.linspace(0, 30, 120)
    u = utils.stim_boxcar([[0, 5, 1]])
    y, _ = dcm.simulate(tvec, u=u)

    names = dcm.get_p_names()
    p0 = dcm.get_p().copy()
    p0[names.index("a1_0")] = 5.0   # far from truth (0.4)
    p0[names.index("c0")] = 5.0     # far from truth (0.8)

    fixed = [n for n in names if n not in {"a1_0", "c0"}]
    res = dcm.fit(y, tvec, u=u, p0=p0, method="NL", fixed_vars=fixed)

    assert res.success
    assert np.isclose(res.x[names.index("a1_0")], A_true[1, 0], atol=5e-2)
    assert np.isclose(res.x[names.index("c0")], C_true[0], atol=5e-2)
    assert not res.at_bounds[names.index("a1_0")]
    assert not res.at_bounds[names.index("c0")]


def test_dcm_recovers_under_observation_noise():
    """With moderate BOLD noise (CNR=20), recovery should still be within ~10%."""
    rng = np.random.default_rng(0)
    dcm, A_true, C_true = _build_2roi_dcm()
    tvec = np.linspace(0, 30, 120)
    u = utils.stim_boxcar([[0, 5, 1]])
    y_clean, _ = dcm.simulate(tvec, u=u)

    sigma = np.std(y_clean) / 20.0
    y_noisy = y_clean + rng.normal(0, sigma, size=y_clean.shape)

    names = dcm.get_p_names()
    p0 = dcm.get_p().copy()
    p0[names.index("a1_0")] = 0.0
    p0[names.index("c0")] = 0.5

    fixed = [n for n in names if n not in {"a1_0", "c0"}]
    res = dcm.fit(y_noisy, tvec, u=u, p0=p0, method="NL", fixed_vars=fixed)

    assert res.success
    assert np.isclose(res.x[names.index("a1_0")], A_true[1, 0], atol=0.1)
    assert np.isclose(res.x[names.index("c0")], C_true[0], atol=0.15)


@pytest.mark.parametrize("a01_true", [-0.3, 0.0, 0.5])
def test_dcm_recovery_across_offdiag_signs(a01_true):
    """Sweep the recovered off-diagonal across negative, zero, positive values."""
    dcm = models.DCM(2)
    A_true = np.array([[-1.5, a01_true], [0.4, -1.5]])
    C_true = np.array([0.8, 0.0])
    dcm.set_params({"A": A_true, "C": C_true})

    tvec = np.linspace(0, 30, 120)
    u = utils.stim_boxcar([[0, 5, 1]])
    y, _ = dcm.simulate(tvec, u=u)

    names = dcm.get_p_names()
    p0 = dcm.get_p().copy()
    # Init off-diagonals at zero, input at 0.5
    if "a0_1" in names:
        p0[names.index("a0_1")] = 0.0
    if "a1_0" in names:
        p0[names.index("a1_0")] = 0.0
    if "c0" in names:
        p0[names.index("c0")] = 0.5

    free = {n for n in names if n.startswith("a") and "_" in n
            and not n.split("_")[0][1:] == n.split("_")[1]} | {"c0"}
    fixed = [n for n in names if n not in free]
    res = dcm.fit(y, tvec, u=u, p0=p0, method="NL", fixed_vars=fixed)

    assert res.success
    if "a0_1" in names:
        assert np.isclose(res.x[names.index("a0_1")], a01_true, atol=5e-2)
    assert np.isclose(res.x[names.index("a1_0")], A_true[1, 0], atol=5e-2)
    assert np.isclose(res.x[names.index("c0")], C_true[0], atol=5e-2)


def test_dcm_at_bounds_flag_set_when_optimum_pinned_to_lower_bound():
    """When the true value is far below the lower bound, the optimum sits at
    the bound and ``at_bounds`` should reflect that.

    Sets up a 2-ROI DCM, then pins ``c0`` to a free fit but with an artificial
    lower bound of 0.95 — well above the true value of 0.8 so the optimiser is
    forced to the lower bound.  Hooks into the same ``fit_NL`` path used in
    every recovery test.
    """
    dcm, _, C_true = _build_2roi_dcm()
    tvec = np.linspace(0, 30, 120)
    u = utils.stim_boxcar([[0, 5, 1]])
    y, _ = dcm.simulate(tvec, u=u)

    names = dcm.get_p_names()
    p0 = dcm.get_p().copy()
    fixed = [n for n in names if n != "c0"]

    # Monkey-patch ``get_bounds`` for this DCM instance so the optimiser must
    # park at the artificial LB.  True c0 = 0.8 < 0.95 = LB, so c0 is pinned.
    original_get_bounds = dcm.get_bounds

    def constrained_bounds():
        LB, UB = original_get_bounds()
        c0_idx = names.index("c0")
        LB = list(LB)
        UB = list(UB)
        LB[c0_idx] = 0.95
        UB[c0_idx] = 1.5
        return LB, UB

    dcm.get_bounds = constrained_bounds
    try:
        # Initialise above the bound so the line search drives down to LB.
        p0[names.index("c0")] = 1.2
        res = dcm.fit(y, tvec, u=u, p0=p0, method="NL", fixed_vars=fixed)
    finally:
        dcm.get_bounds = original_get_bounds

    assert res.success
    c0_idx = names.index("c0")
    assert np.isclose(res.x[c0_idx], 0.95, atol=1e-6)
    assert res.at_bounds[c0_idx]
    # Untouched (fixed) parameters should not be flagged at bounds (their
    # bounds are wide enough that the default values sit interior).
    other_at_bounds = np.delete(res.at_bounds, c0_idx)
    assert not other_at_bounds.any()
