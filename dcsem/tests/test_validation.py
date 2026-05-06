"""Tests for dcsem.validation module."""

import numpy as np
import pytest

from dcsem.validation import (
    ShapeError,
    validate_bold_shape,
    validate_connectivity_matrix,
    validate_input_matrix,
    validate_parameters_in_bounds,
    validate_stimulus,
)


class TestValidateBoldShape:
    """Tests for validate_bold_shape function."""

    def test_valid_2d_shape(self):
        """Test validation passes for correct 2D shape."""
        bold = np.random.randn(100, 2)
        # Should not raise
        validate_bold_shape(bold)

    def test_valid_shape_with_expected_dims(self):
        """Test validation with expected dimensions."""
        bold = np.random.randn(100, 4)
        # Should not raise
        validate_bold_shape(bold, expected_timepoints=100, expected_rois=4)

    def test_1d_array_raises(self):
        """Test that 1D array raises ShapeError."""
        bold = np.random.randn(100)

        with pytest.raises(ShapeError, match="must be 2D"):
            validate_bold_shape(bold)

    def test_3d_array_raises(self):
        """Test that 3D array raises ShapeError."""
        bold = np.random.randn(10, 100, 2)

        with pytest.raises(ShapeError, match="must be 2D"):
            validate_bold_shape(bold)

    def test_wrong_timepoints_raises(self):
        """Test that wrong timepoints raises ShapeError."""
        bold = np.random.randn(100, 2)

        with pytest.raises(ShapeError, match="100 timepoints but expected 50"):
            validate_bold_shape(bold, expected_timepoints=50)

    def test_wrong_rois_raises(self):
        """Test that wrong ROIs raises ShapeError."""
        bold = np.random.randn(100, 2)

        with pytest.raises(ShapeError, match="2 ROIs but expected 4"):
            validate_bold_shape(bold, expected_rois=4)

    def test_custom_name_in_error(self):
        """Test that custom name appears in error message."""
        bold = np.random.randn(100)

        with pytest.raises(ShapeError, match="MySignal must be 2D"):
            validate_bold_shape(bold, name="MySignal")


class TestValidateConnectivityMatrix:
    """Tests for validate_connectivity_matrix function."""

    def test_valid_square_matrix(self):
        """Test validation passes for correct square matrix."""
        A = np.random.randn(4, 4)
        # Should not raise
        validate_connectivity_matrix(A, num_rois=4)

    def test_1d_array_raises(self):
        """Test that 1D array raises ShapeError."""
        A = np.random.randn(4)

        with pytest.raises(ShapeError, match="must be 2D"):
            validate_connectivity_matrix(A, num_rois=4)

    def test_non_square_raises(self):
        """Test that non-square matrix raises ShapeError."""
        A = np.random.randn(4, 3)

        with pytest.raises(ShapeError, match="must be square"):
            validate_connectivity_matrix(A, num_rois=4)

    def test_wrong_dimensions_raises(self):
        """Test that wrong dimensions raises ShapeError."""
        A = np.random.randn(3, 3)

        with pytest.raises(ShapeError, match="3x3 but expected 4x4"):
            validate_connectivity_matrix(A, num_rois=4)

    def test_custom_name_in_error(self):
        """Test that custom name appears in error message."""
        A = np.random.randn(4)

        with pytest.raises(ShapeError, match="Connectivity matrix must be 2D"):
            validate_connectivity_matrix(A, num_rois=4, name="Connectivity")


class TestValidateInputMatrix:
    """Tests for validate_input_matrix function."""

    def test_valid_matrix(self):
        """Test validation passes for correct dimensions."""
        C = np.random.randn(4, 1)
        # Should not raise
        validate_input_matrix(C, num_rois=4, num_inputs=1)

    def test_valid_multi_input(self):
        """Test validation with multiple inputs."""
        C = np.random.randn(3, 2)
        validate_input_matrix(C, num_rois=3, num_inputs=2)

    def test_1d_array_raises(self):
        """Test that 1D array raises ShapeError."""
        C = np.random.randn(4)

        with pytest.raises(ShapeError, match="must be 2D"):
            validate_input_matrix(C, num_rois=4)

    def test_wrong_rois_raises(self):
        """Test that wrong number of ROIs raises ShapeError."""
        C = np.random.randn(3, 1)

        with pytest.raises(ShapeError, match="3 rows but expected 4"):
            validate_input_matrix(C, num_rois=4)

    def test_wrong_inputs_raises(self):
        """Test that wrong number of inputs raises ShapeError."""
        C = np.random.randn(4, 2)

        with pytest.raises(ShapeError, match="2 columns but expected 1"):
            validate_input_matrix(C, num_rois=4, num_inputs=1)


class TestValidateStimulus:
    """Tests for validate_stimulus function."""

    def test_valid_1d_stimulus(self):
        """Test validation passes for 1D stimulus."""
        u = np.random.randn(100)
        # Should not raise
        validate_stimulus(u)

    def test_valid_with_expected_timepoints(self):
        """Test validation with expected timepoints."""
        u = np.random.randn(100)
        validate_stimulus(u, expected_timepoints=100)

    def test_2d_array_raises(self):
        """Test that 2D array raises ShapeError."""
        u = np.random.randn(100, 1)

        with pytest.raises(ShapeError, match="must be 1D"):
            validate_stimulus(u)

    def test_wrong_length_raises(self):
        """Test that wrong length raises ShapeError."""
        u = np.random.randn(100)

        with pytest.raises(ShapeError, match="length 100 but expected 50"):
            validate_stimulus(u, expected_timepoints=50)

    def test_custom_name_in_error(self):
        """Test that custom name appears in error message."""
        u = np.random.randn(100, 1)

        with pytest.raises(ShapeError, match="input_signal must be 1D"):
            validate_stimulus(u, name="input_signal")


class TestValidateParametersInBounds:
    """Tests for validate_parameters_in_bounds function."""

    def test_params_in_bounds(self):
        """Test that params in bounds returns empty list."""
        params = {"a01": 0.5, "a10": 0.3}
        bounds = {"a01": (0.0, 1.0), "a10": (0.0, 1.0)}

        violations = validate_parameters_in_bounds(params, bounds)

        assert violations == []

    def test_param_below_bound(self):
        """Test detection of parameter below lower bound."""
        params = {"a01": -0.1, "a10": 0.5}
        bounds = {"a01": (0.0, 1.0), "a10": (0.0, 1.0)}

        violations = validate_parameters_in_bounds(params, bounds)

        assert "a01" in violations
        assert "a10" not in violations

    def test_param_above_bound(self):
        """Test detection of parameter above upper bound."""
        params = {"a01": 0.5, "a10": 1.5}
        bounds = {"a01": (0.0, 1.0), "a10": (0.0, 1.0)}

        violations = validate_parameters_in_bounds(params, bounds)

        assert "a10" in violations
        assert "a01" not in violations

    def test_multiple_violations(self):
        """Test detection of multiple violations."""
        params = {"a01": -0.1, "a10": 1.5, "c0": 0.5}
        bounds = {"a01": (0.0, 1.0), "a10": (0.0, 1.0), "c0": (0.0, 1.0)}

        violations = validate_parameters_in_bounds(params, bounds)

        assert set(violations) == {"a01", "a10"}

    def test_strict_mode_raises(self):
        """Test that strict mode raises ValueError."""
        params = {"a01": -0.1}
        bounds = {"a01": (0.0, 1.0)}

        with pytest.raises(ValueError, match="Parameters out of bounds"):
            validate_parameters_in_bounds(params, bounds, strict=True)

    def test_strict_mode_passes_when_valid(self):
        """Test that strict mode doesn't raise when valid."""
        params = {"a01": 0.5}
        bounds = {"a01": (0.0, 1.0)}

        # Should not raise
        violations = validate_parameters_in_bounds(params, bounds, strict=True)
        assert violations == []

    def test_param_not_in_bounds_ignored(self):
        """Test that params without bounds are ignored."""
        params = {"a01": 0.5, "unknown": 999.0}
        bounds = {"a01": (0.0, 1.0)}

        violations = validate_parameters_in_bounds(params, bounds)

        assert violations == []

    def test_boundary_values_valid(self):
        """Test that boundary values are considered valid."""
        params = {"a01": 0.0, "a10": 1.0}
        bounds = {"a01": (0.0, 1.0), "a10": (0.0, 1.0)}

        violations = validate_parameters_in_bounds(params, bounds)

        assert violations == []


class TestShapeError:
    """Tests for ShapeError exception."""

    def test_is_value_error(self):
        """Test that ShapeError is a ValueError subclass."""
        assert issubclass(ShapeError, ValueError)

    def test_can_be_raised_and_caught(self):
        """Test that ShapeError can be raised and caught."""
        with pytest.raises(ShapeError):
            raise ShapeError("Test error message")

    def test_message_preserved(self):
        """Test that error message is preserved."""
        try:
            raise ShapeError("Custom message")
        except ShapeError as e:
            assert "Custom message" in str(e)
