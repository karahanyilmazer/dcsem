"""Tests for dcsem.diagnostics module."""

import logging

import numpy as np
import pytest

from dcsem.diagnostics import (
    compute_2d_loss_landscape,
    compute_hessian_diagnostics,
    profile_likelihood_1d,
)


class TestComputeHessianDiagnostics:
    """Tests for compute_hessian_diagnostics function."""

    def test_well_conditioned_matrix(self):
        H = np.array([[3.0, 0.0], [0.0, 1.0]])
        diag = compute_hessian_diagnostics(H)

        assert diag["eigvals_min"] == pytest.approx(1.0)
        assert diag["eigvals_max"] == pytest.approx(3.0)
        assert diag["n_negative_eigvals"] == 0
        assert diag["condition_number"] == pytest.approx(3.0)
        assert not diag["is_near_singular"]

    def test_indefinite_matrix(self):
        """Matrix with one negative eigenvalue should report it."""
        H = np.array([[1.0, 0.0], [0.0, -2.0]])
        diag = compute_hessian_diagnostics(H)

        assert diag["n_negative_eigvals"] == 1
        # Only one positive eigenvalue ⇒ condition_number is inf (no upper bound)
        assert np.isinf(diag["condition_number"])
        assert diag["is_near_singular"]

    def test_singular_matrix(self):
        """All-zero matrix has no positive eigenvalues."""
        H = np.zeros((2, 2))
        diag = compute_hessian_diagnostics(H)

        assert np.isnan(diag["eigvals_min_pos"])
        assert np.isinf(diag["condition_number"])


class TestProfileLikelihood1d:
    """Tests for profile_likelihood_1d function."""

    def test_simple_quadratic_profile(self):
        """A simple separable quadratic should produce a parabolic profile."""
        # obj(theta) = 0.5 * sum(theta**2) — minimum at zero, separable.
        def obj(theta):
            return 0.5 * float(np.sum(np.asarray(theta, dtype=float) ** 2))

        theta_est = np.zeros(2)
        param_grid = np.linspace(-1.0, 1.0, 5)
        free_bounds = [(-3.0, 3.0)]

        grid, profile = profile_likelihood_1d(
            obj, theta_est, fixed_idx=0, param_grid=param_grid, free_bounds=free_bounds
        )

        # Profile of theta[0] holding theta[1] free at its optimum (=0):
        # profile(v) = 0.5 * v**2.
        assert np.allclose(profile, 0.5 * param_grid**2, atol=1e-5)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in subtract:RuntimeWarning"
    )
    def test_records_nan_when_subproblem_fails(self, caplog):
        """When the inner ``minimize`` cannot return a finite result, the
        profile entry is NaN rather than a misleading numeric value.

        We construct an objective that returns ``np.inf`` for any attempted
        evaluation when the *fixed* parameter is at a sentinel value, so the
        L-BFGS-B optimiser has nothing finite to descend on. (scipy's
        finite-difference jacobian emits a benign RuntimeWarning we ignore.)
        """
        def obj(theta):
            theta = np.asarray(theta, dtype=float)
            if np.isclose(theta[0], 99.0):
                return float("inf")
            return 0.5 * float(np.sum(theta ** 2))

        theta_est = np.zeros(2)
        param_grid = np.array([-1.0, 0.0, 99.0])
        free_bounds = [(-3.0, 3.0)]

        with caplog.at_level(logging.DEBUG, logger="dcsem.diagnostics"):
            grid, profile = profile_likelihood_1d(
                obj, theta_est, fixed_idx=0, param_grid=param_grid, free_bounds=free_bounds
            )

        # First two entries are well-defined; third is NaN-flagged.
        assert np.isfinite(profile[0])
        assert np.isfinite(profile[1])
        assert np.isnan(profile[2])


class TestCompute2dLossLandscape:
    """Smoke test for compute_2d_loss_landscape."""

    def test_quadratic_landscape_shape(self):
        def obj(theta):
            return 0.5 * float(np.sum(np.asarray(theta, dtype=float) ** 2))

        theta_center = np.zeros(2)
        p1, p2, Z = compute_2d_loss_landscape(
            obj, theta_center, p1_idx=0, p2_idx=1, n_grid=5, half_range=1.0
        )

        assert p1.shape == (5,)
        assert p2.shape == (5,)
        assert Z.shape == (5, 5)
        # Center of grid is the minimum (0, 0) ⇒ Z[2, 2] is smallest.
        assert Z[2, 2] == pytest.approx(0.0, abs=1e-12)
