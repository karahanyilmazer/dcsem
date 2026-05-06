"""
Diagnostic utilities for Spectral DCM parameter estimation.

Provides Hessian diagnostics, profile likelihood, 2D loss landscapes,
and parametric bootstrap uncertainty quantification.
"""

import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def compute_hessian_diagnostics(H, near_singular_threshold=1e8):
    """Compute diagnostic statistics for a Hessian matrix.

    Uses the positive eigenvalue spectrum for condition number to avoid
    negative curvature from numerical issues skewing the metric.

    Args:
        H: Hessian matrix (n_params x n_params)
        near_singular_threshold: Condition number threshold for near-singularity flag

    Returns:
        dict with keys:
            eigvals_min: Raw minimum eigenvalue (may be negative)
            eigvals_med: Median eigenvalue
            eigvals_max: Maximum eigenvalue
            eigvals_min_pos: Smallest positive eigenvalue (nan if none)
            n_negative_eigvals: Number of negative eigenvalues
            condition_number: Based on positive spectrum (inf if fewer than 2 positive)
            is_near_singular: True if cond > threshold or is inf
    """
    H_sym = 0.5 * (H + H.T)
    eigvals = np.linalg.eigvalsh(H_sym)
    n_negative = int(np.sum(eigvals < 0))
    pos_eigvals = eigvals[eigvals > 0]

    if len(pos_eigvals) >= 2:
        cond = pos_eigvals[-1] / pos_eigvals[0]  # max_pos / min_pos
    elif len(pos_eigvals) == 1:
        cond = np.inf  # only one positive → not bounded
    else:
        cond = np.inf  # all non-positive

    eig_min_pos = pos_eigvals[0] if len(pos_eigvals) > 0 else np.nan
    is_near_singular = np.isinf(cond) or (cond > near_singular_threshold)

    return dict(
        eigvals_min=float(eigvals[0]),  # raw min (may be negative)
        eigvals_med=float(np.median(eigvals)),
        eigvals_max=float(eigvals[-1]),
        eigvals_min_pos=float(eig_min_pos),  # smallest positive eigenvalue
        n_negative_eigvals=n_negative,
        condition_number=float(cond),  # based on positive spectrum
        is_near_singular=is_near_singular,
    )


def profile_likelihood_1d(obj, theta_est, fixed_idx, param_grid, free_bounds):
    """Compute profile likelihood by fixing one parameter and optimizing over others.

    For each value in param_grid, fixes theta[fixed_idx] and minimizes over
    remaining parameters.

    Args:
        obj: Objective function theta -> scalar
        theta_est: Starting parameter estimate (n_params,)
        fixed_idx: Index of the parameter to profile
        param_grid: Array of values for the fixed parameter
        free_bounds: List of (lo, hi) tuples for the non-fixed parameters

    Returns:
        param_grid: Same as input (np.ndarray)
        profile: Objective values at each grid point (np.ndarray)
    """
    n = len(theta_est)
    free_idx = [k for k in range(n) if k != fixed_idx]
    profile = []

    for val in param_grid:

        def f(theta_free, v=val):
            theta_full = theta_est.copy()
            for k, idx in enumerate(free_idx):
                theta_full[idx] = theta_free[k]
            theta_full[fixed_idx] = v
            return obj(theta_full)

        res = minimize(f, theta_est[free_idx], method="L-BFGS-B", bounds=free_bounds)
        # Only trust the result if the optimiser converged AND the value is
        # finite. Otherwise the profile entry is NaN — better than recording
        # a spike from a failed sub-problem and pretending it is a real
        # likelihood drop.
        if res.success and np.isfinite(res.fun):
            profile.append(float(res.fun))
        else:
            logger.debug(
                "profile_likelihood_1d: sub-optimisation failed at fixed=%g "
                "(success=%s, fun=%s, message=%s)",
                val,
                res.success,
                res.fun,
                getattr(res, "message", ""),
            )
            profile.append(np.nan)

    return np.asarray(param_grid), np.asarray(profile)


def compute_2d_loss_landscape(
    obj, theta_center, p1_idx, p2_idx, n_grid=40, half_range=None
):
    """Grid scan of the loss landscape for two parameters; all others fixed at center.

    Args:
        obj: Objective function theta -> scalar
        theta_center: Center point for the grid (n_params,)
        p1_idx: Index of first parameter (row axis)
        p2_idx: Index of second parameter (column axis)
        n_grid: Number of grid points per axis
        half_range: Half-range for each axis. If None, uses
            0.4 * max(|theta_center[p_i]|, 0.1) per axis independently.

    Returns:
        p1_vals: Grid values for parameter 1 (n_grid,)
        p2_vals: Grid values for parameter 2 (n_grid,)
        Z: Loss values, shape (n_grid, n_grid); Z[i, j] = obj at p1=p1_vals[i], p2=p2_vals[j]
    """
    theta_center = np.asarray(theta_center, dtype=float)

    if half_range is None:
        hr1 = 0.4 * max(abs(theta_center[p1_idx]), 0.1)
        hr2 = 0.4 * max(abs(theta_center[p2_idx]), 0.1)
    else:
        hr1 = hr2 = half_range

    p1_vals = np.linspace(
        theta_center[p1_idx] - hr1, theta_center[p1_idx] + hr1, n_grid
    )
    p2_vals = np.linspace(
        theta_center[p2_idx] - hr2, theta_center[p2_idx] + hr2, n_grid
    )

    Z = np.full((n_grid, n_grid), np.nan)
    theta = theta_center.copy()

    for i, v1 in enumerate(p1_vals):
        for j, v2 in enumerate(p2_vals):
            theta[p1_idx] = v1
            theta[p2_idx] = v2
            Z[i, j] = obj(theta)

    return p1_vals, p2_vals, Z


def parametric_bootstrap_uncertainty(
    spdcm, theta_est, snr, make_obj_fn, bounds, n_bootstrap, rng
):
    """Estimate uncertainty via parametric bootstrap.

    Generates n_bootstrap synthetic CSD datasets from theta_est at given snr,
    refits each, and returns bootstrap SE and percentile CIs.

    Args:
        spdcm: SpectralDCM instance
        theta_est: MLE estimate to use as generating parameters (n_params,)
        snr: Signal-to-noise ratio for synthetic data generation
        make_obj_fn: Callable y_obs -> (obj, y_mean, y_std, y_obs_norm)
        bounds: List of (lo, hi) tuples for optimization bounds
        n_bootstrap: Number of bootstrap replicates
        rng: np.random.Generator instance

    Returns:
        boot_se: Bootstrap standard errors (n_params,); all-NaN if < 3 replicates succeed
        boot_ci_dict: Dict mapping CI level (50, 80, 95) to (n_params, 2) arrays [lo, hi];
            empty dict if < 3 replicates succeed
    """
    from scipy.optimize import minimize as _minimize

    boot_thetas = []
    for _ in range(n_bootstrap):
        rng_b = np.random.default_rng(rng.integers(2**31))
        y_boot = spdcm.generate_noisy_csd(theta_est, snr=snr, rng=rng_b)
        obj_b, *_ = make_obj_fn(y_boot)
        res_b = _minimize(obj_b, theta_est, method="L-BFGS-B", bounds=bounds)
        if res_b.success and res_b.fun < 1e9:
            boot_thetas.append(res_b.x)

    if len(boot_thetas) < 3:
        return np.full(len(theta_est), np.nan), {}

    boot_arr = np.array(boot_thetas)
    boot_se = np.std(boot_arr, axis=0)
    boot_ci = {
        level: np.column_stack(
            [
                np.percentile(boot_arr, (100 - level) / 2, axis=0),
                np.percentile(boot_arr, 50 + level / 2, axis=0),
            ]
        )
        for level in [50, 80, 95]
    }
    return boot_se, boot_ci
