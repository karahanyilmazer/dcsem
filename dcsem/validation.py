"""
Input validation utilities for DCM simulations.

This module provides validation functions to catch shape mismatches and invalid
inputs early, providing clear error messages for common mistakes.
"""

from typing import Optional

import numpy as np


class ShapeError(ValueError):
    """Raised when array shape doesn't match expected format."""

    pass


def validate_bold_shape(
    bold: np.ndarray,
    expected_timepoints: Optional[int] = None,
    expected_rois: Optional[int] = None,
    name: str = "BOLD",
) -> None:
    """
    Validate BOLD signal has shape (T, R) for timepoints and ROIs.

    Args:
        bold: BOLD signal array to validate
        expected_timepoints: If provided, check T dimension matches
        expected_rois: If provided, check R dimension matches
        name: Name for error messages (default "BOLD")

    Raises:
        ShapeError: If shape doesn't match (T, R) format or dimensions don't match

    Examples:
        >>> bold = np.random.randn(100, 2)
        >>> validate_bold_shape(bold, expected_timepoints=100, expected_rois=2)
        >>> validate_bold_shape(bold)  # Just check 2D
    """
    if bold.ndim != 2:
        raise ShapeError(
            f"{name} must be 2D with shape (T, R) where T=timepoints and R=ROIs. "
            f"Got {bold.ndim}D array with shape {bold.shape}."
        )

    T, R = bold.shape

    if expected_timepoints is not None and T != expected_timepoints:
        raise ShapeError(
            f"{name} has {T} timepoints but expected {expected_timepoints}. "
            f"Shape is {bold.shape}, expected ({expected_timepoints}, {R})."
        )

    if expected_rois is not None and R != expected_rois:
        raise ShapeError(
            f"{name} has {R} ROIs but expected {expected_rois}. "
            f"Shape is {bold.shape}, expected ({T}, {expected_rois})."
        )


def validate_connectivity_matrix(
    A: np.ndarray,
    num_rois: int,
    name: str = "A",
) -> None:
    """
    Validate connectivity matrix is square with correct dimensions.

    Args:
        A: Connectivity matrix to validate
        num_rois: Expected number of ROIs (matrix should be num_rois x num_rois)
        name: Name for error messages (default "A")

    Raises:
        ShapeError: If matrix is not square or dimensions don't match
    """
    if A.ndim != 2:
        raise ShapeError(
            f"{name} matrix must be 2D. Got {A.ndim}D array with shape {A.shape}."
        )

    if A.shape[0] != A.shape[1]:
        raise ShapeError(
            f"{name} matrix must be square. Got shape {A.shape}."
        )

    if A.shape[0] != num_rois:
        raise ShapeError(
            f"{name} matrix has size {A.shape[0]}x{A.shape[0]} but expected "
            f"{num_rois}x{num_rois} for {num_rois} ROIs."
        )


def validate_input_matrix(
    C: np.ndarray,
    num_rois: int,
    num_inputs: int = 1,
    name: str = "C",
) -> None:
    """
    Validate input matrix has correct dimensions.

    Args:
        C: Input matrix to validate (num_rois x num_inputs)
        num_rois: Expected number of ROIs
        num_inputs: Expected number of inputs (default 1)
        name: Name for error messages (default "C")

    Raises:
        ShapeError: If dimensions don't match expected
    """
    if C.ndim != 2:
        raise ShapeError(
            f"{name} matrix must be 2D. Got {C.ndim}D array with shape {C.shape}."
        )

    if C.shape[0] != num_rois:
        raise ShapeError(
            f"{name} matrix has {C.shape[0]} rows but expected {num_rois} ROIs. "
            f"Shape is {C.shape}, expected ({num_rois}, {num_inputs})."
        )

    if C.shape[1] != num_inputs:
        raise ShapeError(
            f"{name} matrix has {C.shape[1]} columns but expected {num_inputs} inputs. "
            f"Shape is {C.shape}, expected ({num_rois}, {num_inputs})."
        )


def validate_stimulus(
    u: np.ndarray,
    expected_timepoints: Optional[int] = None,
    name: str = "stimulus",
) -> None:
    """
    Validate stimulus array has correct shape.

    Args:
        u: Stimulus array to validate (should be 1D with length T)
        expected_timepoints: If provided, check length matches
        name: Name for error messages (default "stimulus")

    Raises:
        ShapeError: If stimulus is not 1D or length doesn't match
    """
    if u.ndim != 1:
        raise ShapeError(
            f"{name} must be 1D with shape (T,). Got {u.ndim}D array with shape {u.shape}."
        )

    if expected_timepoints is not None and len(u) != expected_timepoints:
        raise ShapeError(
            f"{name} has length {len(u)} but expected {expected_timepoints} timepoints."
        )


def validate_parameters_in_bounds(
    params: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    strict: bool = False,
) -> list[str]:
    """
    Check if parameters are within specified bounds.

    Args:
        params: Dictionary of parameter values
        bounds: Dictionary of (min, max) bounds for each parameter
        strict: If True, raise error on violation. If False, return list of violations.

    Returns:
        List of parameter names that are out of bounds (empty if all valid)

    Raises:
        ValueError: If strict=True and any parameter is out of bounds
    """
    violations = []

    for name, value in params.items():
        if name in bounds:
            low, high = bounds[name]
            if value < low or value > high:
                violations.append(name)

    if strict and violations:
        msg_parts = []
        for name in violations:
            value = params[name]
            low, high = bounds[name]
            msg_parts.append(f"{name}={value:.4f} not in [{low}, {high}]")
        raise ValueError(f"Parameters out of bounds: {', '.join(msg_parts)}")

    return violations
