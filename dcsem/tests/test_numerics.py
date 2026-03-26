"""Tests for dcsem.numerics module."""

import numpy as np
import pytest

from dcsem.numerics import (
    compute_confidence_intervals,
    compute_correlation_matrix,
    compute_standard_errors,
    safe_hessian_inversion,
)


class TestSafeHessianInversion:
    """Tests for safe_hessian_inversion function."""

    def test_well_conditioned_matrix(self):
        """Test inversion of a well-conditioned positive definite matrix."""
        H = np.array([[2.0, 0.5], [0.5, 1.0]])
        sigma_sq = 0.1

        cov, diagnostics = safe_hessian_inversion(H, sigma_sq, method="tikhonov")

        # Check covariance is computed
        assert cov.shape == (2, 2)
        # Check it's symmetric
        assert np.allclose(cov, cov.T)
        # Check diagnostics
        assert "eigenvalues" in diagnostics
        assert "condition_number" in diagnostics
        assert diagnostics["method"] == "tikhonov"
        assert not diagnostics["rank_deficient"]

    def test_ill_conditioned_matrix_tikhonov(self):
        """Test Tikhonov regularization on ill-conditioned matrix."""
        # Create an ill-conditioned matrix
        H = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        sigma_sq = 0.1

        cov, diagnostics = safe_hessian_inversion(
            H, sigma_sq, regularization=1e-4, method="tikhonov"
        )

        # Should succeed without raising
        assert cov.shape == (2, 2)
        # Condition number should be high
        assert diagnostics["condition_number"] > 1e3

    def test_eigenvalue_method(self):
        """Test eigenvalue truncation method."""
        H = np.array([[2.0, 0.5], [0.5, 1.0]])
        sigma_sq = 0.1

        cov, diagnostics = safe_hessian_inversion(H, sigma_sq, method="eigenvalue")

        assert cov.shape == (2, 2)
        assert diagnostics["method"] == "eigenvalue"

    def test_raw_method(self):
        """Test raw inversion without regularization."""
        H = np.array([[2.0, 0.5], [0.5, 1.0]])
        sigma_sq = 0.1

        cov, diagnostics = safe_hessian_inversion(H, sigma_sq, method="raw")

        assert cov.shape == (2, 2)
        assert diagnostics["method"] == "raw"
        assert diagnostics["regularization_used"] == 0.0

    def test_singular_matrix_raw_fails(self):
        """Test that raw method fails on singular matrix."""
        H = np.array([[1.0, 1.0], [1.0, 1.0]])  # Singular
        sigma_sq = 0.1

        with pytest.raises(np.linalg.LinAlgError):
            safe_hessian_inversion(H, sigma_sq, method="raw")

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        H = np.array([[2.0, 0.5], [0.5, 1.0]])

        with pytest.raises(ValueError, match="Unknown method"):
            safe_hessian_inversion(H, 0.1, method="invalid")

    def test_diagnostics_eigenvalues(self):
        """Test that eigenvalues are correctly reported."""
        H = np.array([[3.0, 0.0], [0.0, 1.0]])
        sigma_sq = 0.1

        cov, diagnostics = safe_hessian_inversion(H, sigma_sq)

        # Eigenvalues should be 1 and 3
        eigvals = diagnostics["eigenvalues"]
        assert np.allclose(sorted(eigvals), [1.0, 3.0])

    def test_regularization_applied_when_needed(self):
        """Test that regularization is applied for rank-deficient matrices."""
        # Near-singular matrix
        H = np.array([[1.0, 1.0 - 1e-10], [1.0 - 1e-10, 1.0]])
        sigma_sq = 0.1

        cov, diagnostics = safe_hessian_inversion(
            H, sigma_sq, regularization=1e-6, method="tikhonov"
        )

        # Should detect rank deficiency and apply regularization
        assert diagnostics["rank_deficient"] or diagnostics["condition_number"] > 1e6


class TestComputeStandardErrors:
    """Tests for compute_standard_errors function."""

    def test_positive_variances(self):
        """Test SE computation with positive variances."""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])

        se = compute_standard_errors(cov)

        assert np.allclose(se, [0.2, 0.3])

    def test_negative_variance_handled(self, capsys):
        """Test that negative variances produce NaN SE with warning."""
        # Covariance with negative diagonal (invalid, but can happen numerically)
        cov = np.array([[0.04, 0.01], [0.01, -0.01]])

        se = compute_standard_errors(cov, warn_negative=True)

        # Positive variance → valid SE; negative variance → NaN
        assert se[0] == pytest.approx(0.2)
        assert np.isnan(se[1])
        # Check warning was printed
        captured = capsys.readouterr()
        assert "Negative variance" in captured.out

    def test_no_warning_when_disabled(self, capsys):
        """Test that warning can be suppressed."""
        cov = np.array([[0.04, 0.01], [0.01, -0.01]])

        se = compute_standard_errors(cov, warn_negative=False)

        captured = capsys.readouterr()
        assert "Negative variance" not in captured.out

    def test_zero_variance(self):
        """Test handling of zero variance."""
        cov = np.array([[0.04, 0.0], [0.0, 0.0]])

        se = compute_standard_errors(cov)

        assert se[0] == 0.2
        assert se[1] == 0.0


class TestComputeCorrelationMatrix:
    """Tests for compute_correlation_matrix function."""

    def test_valid_covariance(self):
        """Test correlation from valid covariance matrix."""
        # Covariance with known correlation
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])

        corr = compute_correlation_matrix(cov)

        # Diagonal should be 1
        assert np.allclose(np.diag(corr), [1.0, 1.0])
        # Off-diagonal should be correlation coefficient
        assert np.allclose(corr[0, 1], 0.5)
        assert np.allclose(corr[1, 0], 0.5)

    def test_different_variances(self):
        """Test correlation with different variances."""
        # var(X) = 4, var(Y) = 9, cov(X,Y) = 3
        # corr = 3 / (2 * 3) = 0.5
        cov = np.array([[4.0, 3.0], [3.0, 9.0]])

        corr = compute_correlation_matrix(cov)

        assert np.allclose(np.diag(corr), [1.0, 1.0])
        assert np.allclose(corr[0, 1], 0.5)

    def test_degenerate_variance_handled(self):
        """Test handling of zero variance (degenerate case)."""
        cov = np.array([[1.0, 0.0], [0.0, 0.0]])

        corr = compute_correlation_matrix(cov, handle_degenerate=True)

        # Diagonal should still be 1
        assert corr[0, 0] == 1.0
        assert corr[1, 1] == 1.0
        # Off-diagonal with degenerate should be 0
        assert corr[0, 1] == 0.0

    def test_negative_variance_handled(self):
        """Test handling of negative variance."""
        cov = np.array([[1.0, 0.1], [0.1, -0.01]])

        corr = compute_correlation_matrix(cov, handle_degenerate=True)

        # Should not raise and diagonal should be 1
        assert corr[0, 0] == 1.0
        assert corr[1, 1] == 1.0


class TestComputeConfidenceIntervals:
    """Tests for compute_confidence_intervals function."""

    def test_95_percent_ci(self):
        """Test 95% confidence interval computation."""
        theta_est = np.array([0.5, 1.0])
        se = np.array([0.1, 0.2])

        ci = compute_confidence_intervals(theta_est, se, alpha=0.05)

        # 95% CI uses z ~= 1.96
        assert ci.shape == (2, 2)
        # Check lower bounds are below estimates
        assert ci[0, 0] < 0.5
        assert ci[1, 0] < 1.0
        # Check upper bounds are above estimates
        assert ci[0, 1] > 0.5
        assert ci[1, 1] > 1.0
        # Check width is approximately 2 * 1.96 * se
        assert np.isclose(ci[0, 1] - ci[0, 0], 2 * 1.96 * 0.1, rtol=0.01)
        assert np.isclose(ci[1, 1] - ci[1, 0], 2 * 1.96 * 0.2, rtol=0.01)

    def test_90_percent_ci(self):
        """Test 90% confidence interval computation."""
        theta_est = np.array([0.5])
        se = np.array([0.1])

        ci = compute_confidence_intervals(theta_est, se, alpha=0.10)

        # 90% CI uses z ~= 1.645
        assert ci.shape == (1, 2)
        assert ci[0, 0] < 0.5
        assert ci[0, 1] > 0.5
        # 90% CI should be narrower than 95%
        width_90 = ci[0, 1] - ci[0, 0]
        ci_95 = compute_confidence_intervals(theta_est, se, alpha=0.05)
        width_95 = ci_95[0, 1] - ci_95[0, 0]
        assert width_90 < width_95

    def test_ci_centered_on_estimate(self):
        """Test that CI is centered on estimate."""
        theta_est = np.array([2.5, -1.0, 0.0])
        se = np.array([0.5, 0.3, 0.1])

        ci = compute_confidence_intervals(theta_est, se)

        # Midpoint should be the estimate
        midpoints = (ci[:, 0] + ci[:, 1]) / 2
        assert np.allclose(midpoints, theta_est)
