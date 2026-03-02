"""
Numerical stability utilities for DCM parameter estimation.

This module provides functions for safely inverting Hessian matrices and computing
standard errors, handling ill-conditioned matrices that commonly arise in DCM models.
"""

from typing import Literal

import numpy as np


def safe_hessian_inversion(
    hessian: np.ndarray,
    sigma_sq: float,
    regularization: float = 1e-6,
    method: Literal[
        "tikhonov", "eigenvalue", "raw", "pinvh", "adaptive_ridge"
    ] = "tikhonov",
) -> tuple[np.ndarray, dict]:
    """
    Safely invert Hessian with regularization for ill-conditioned matrices.

    DCM models can have near-degenerate parameters causing ill-conditioned Hessians.
    This function provides robust inversion with optional regularization.

    Args:
        hessian: Hessian matrix (n_params x n_params)
        sigma_sq: Estimated noise variance.
            For the canonical MLE formula Cov = H_NLL^{-1}, pass sigma_sq=1.0
            and compute H as the Hessian of the NLL directly (so Cov = H_NLL^{-1}).
            For MSE objectives, see the "canonical NLL" pattern in spdcm_noise_sweep.py.
        regularization: Regularization strength (lambda for Tikhonov / epsilon floor)
        method: Regularization method:
            - "tikhonov": (H + lambda*I)^-1 — adds regularization to diagonal
            - "eigenvalue": Truncates near-zero eigenvalues before inversion
            - "raw": Direct inversion with no regularization (may fail)
            - "pinvh": Moore-Penrose pseudoinverse via scipy.linalg.pinvh; zeros out
              flat directions. Diagnostic tool for degeneracy, not a calibrated CI.
            - "adaptive_ridge": Minimum ridge to restore PD, then inverts exactly.

    Returns:
        covariance: Parameter covariance matrix (sigma_sq * H^-1)
        diagnostics: Dict containing:
            - eigenvalues: Eigenvalues of original Hessian
            - condition_number: Condition number of Hessian
            - regularization_used: Regularization strength applied
            - method: Method used for inversion
            - rank_deficient: Whether matrix was detected as rank-deficient
            - regularized: Whether regularization was actually applied

    Raises:
        np.linalg.LinAlgError: If matrix is singular and cannot be inverted
    """
    n_params = hessian.shape[0]

    # Symmetrize before any eigenvalue work or inversion
    H_sym = 0.5 * (hessian + hessian.T)

    # Compute eigenvalue decomposition for diagnostics
    eigvals = np.linalg.eigvalsh(H_sym)
    eps = 1e-12
    eigvals_clipped = np.clip(eigvals, eps, None)
    condition_number = np.max(eigvals_clipped) / np.min(eigvals_clipped)

    # Check for rank deficiency
    rank_deficient = np.any(eigvals < 1e-8)

    diagnostics = {
        "eigenvalues": eigvals,
        "condition_number": condition_number,
        "regularization_used": 0.0,
        "method": method,
        "rank_deficient": rank_deficient,
        "regularized": False,
    }

    if method == "raw":
        # Direct inversion - may fail for ill-conditioned matrices
        H_inv = np.linalg.inv(H_sym)
        covariance = sigma_sq * H_inv
        return covariance, diagnostics

    elif method == "tikhonov":
        # Tikhonov regularization: (H + lambda*I)^-1
        # Apply regularization if needed (rank deficient or high condition number)
        if rank_deficient or condition_number > 1e6:
            H_reg = H_sym + regularization * np.eye(n_params)
            diagnostics["regularization_used"] = regularization
            diagnostics["regularized"] = True
        else:
            H_reg = H_sym

        H_inv = np.linalg.inv(H_reg)
        covariance = sigma_sq * H_inv
        return covariance, diagnostics

    elif method == "eigenvalue":
        # Eigenvalue truncation: reconstruct matrix with truncated eigenvalues
        eigvals_full, eigvecs = np.linalg.eigh(H_sym)

        # Truncate small eigenvalues
        min_eigval = regularization
        eigvals_truncated = np.maximum(eigvals_full, min_eigval)

        if np.any(eigvals_full < min_eigval):
            diagnostics["regularization_used"] = min_eigval
            diagnostics["regularized"] = True

        # Reconstruct inverse: V @ diag(1/lambda) @ V.T
        H_inv = eigvecs @ np.diag(1.0 / eigvals_truncated) @ eigvecs.T
        covariance = sigma_sq * H_inv
        return covariance, diagnostics

    elif method == "pinvh":
        # Moore-Penrose pseudoinverse via scipy.linalg.pinvh.
        # Zeros out flat directions (near-zero eigenvalues).
        # Diagnostic / stable-inverse tool — not a calibrated CI generator.
        # regularization is passed as atol to threshold near-zero eigenvalues.
        from scipy.linalg import pinvh as _pinvh

        H_inv = _pinvh(H_sym, lower=False, atol=regularization)
        covariance = sigma_sq * H_inv
        diagnostics["regularized"] = False
        return covariance, diagnostics

    elif method == "adaptive_ridge":
        # Minimum ridge to restore positive definiteness, then exact inversion.
        min_e = float(np.linalg.eigvalsh(H_sym)[0])
        ridge = (
            max(0.0, -min_e) + regularization
        )  # regularization acts as epsilon floor
        H_reg = H_sym + ridge * np.eye(n_params)
        H_inv = np.linalg.inv(H_reg)
        covariance = sigma_sq * H_inv
        diagnostics["regularization_used"] = ridge
        diagnostics["regularized"] = ridge > regularization
        return covariance, diagnostics

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Use 'tikhonov', 'eigenvalue', 'raw', 'pinvh', or 'adaptive_ridge'"
        )


def compute_standard_errors(
    covariance: np.ndarray,
    warn_negative: bool = True,
) -> np.ndarray:
    """
    Compute standard errors from covariance matrix, handling negative variances.

    Instead of simply clipping negative values (which hides the problem),
    this function uses absolute values and optionally warns the user.

    Args:
        covariance: Parameter covariance matrix (n_params x n_params)
        warn_negative: If True, prints warning when negative variances detected

    Returns:
        se: Standard errors for each parameter (sqrt of diagonal)
    """
    diag_cov = np.diag(covariance)

    if np.any(diag_cov < 0):
        if warn_negative:
            neg_indices = np.where(diag_cov < 0)[0]
            print(
                f"⚠️  Negative variance detected at parameter indices {neg_indices.tolist()}. "
                "Hessian may not be positive definite. Using absolute values."
            )
        # Use absolute value instead of clipping to zero
        diag_cov = np.abs(diag_cov)

    se = np.sqrt(diag_cov)
    return se


def compute_correlation_matrix(
    covariance: np.ndarray,
    handle_degenerate: bool = True,
) -> np.ndarray:
    """
    Compute correlation matrix from covariance with proper handling of degenerate cases.

    Args:
        covariance: Parameter covariance matrix (n_params x n_params)
        handle_degenerate: If True, handles zero/negative variances gracefully

    Returns:
        corr: Correlation matrix (n_params x n_params)
    """
    se = np.sqrt(np.abs(np.diag(covariance)))
    denom = np.outer(se, se)

    if handle_degenerate:
        # Avoid division by zero
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(denom > 0, covariance / denom, 0)
        # Set diagonal to 1 (by definition)
        np.fill_diagonal(corr, 1.0)
    else:
        corr = covariance / denom

    return corr


def compute_confidence_intervals(
    theta_est: np.ndarray,
    se: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Compute confidence intervals for parameter estimates.

    Args:
        theta_est: Estimated parameter values (n_params,)
        se: Standard errors for each parameter (n_params,)
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        ci: Confidence intervals, shape (n_params, 2) with [lower, upper] bounds
    """
    from scipy import stats

    z = stats.norm.ppf(1 - alpha / 2)  # 1.96 for alpha=0.05
    ci = np.column_stack([theta_est - z * se, theta_est + z * se])
    return ci
