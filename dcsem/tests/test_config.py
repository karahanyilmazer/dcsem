"""Tests for dcsem.config module."""

import numpy as np
import pytest

from dcsem.config import NOISE_CONFIG, PARAM_BOUNDS, NoiseConfig, ParameterBounds


class TestParameterBounds:
    """Tests for ParameterBounds configuration."""

    def test_default_bounds(self):
        """Test default bounds are (0, 1)."""
        bounds = ParameterBounds()

        assert bounds.a_min == 0.0
        assert bounds.a_max == 1.0
        assert bounds.c_min == 0.0
        assert bounds.c_max == 1.0

    def test_custom_bounds(self):
        """Test custom bounds can be set."""
        bounds = ParameterBounds(a_min=-1.0, a_max=2.0, c_min=0.5, c_max=1.5)

        assert bounds.a_min == -1.0
        assert bounds.a_max == 2.0
        assert bounds.c_min == 0.5
        assert bounds.c_max == 1.5

    def test_get_bounds_dict(self):
        """Test get_bounds_dict returns correct dictionary."""
        bounds = ParameterBounds()

        bounds_dict = bounds.get_bounds_dict()

        assert bounds_dict["a01"] == (0.0, 1.0)
        assert bounds_dict["a10"] == (0.0, 1.0)
        assert bounds_dict["c0"] == (0.0, 1.0)
        assert bounds_dict["c1"] == (0.0, 1.0)

    def test_get_bounds_list(self):
        """Test get_bounds_list returns correct list for param names."""
        bounds = ParameterBounds()
        param_names = ["a01", "a10", "c0", "c1"]

        bounds_list = bounds.get_bounds_list(param_names)

        assert len(bounds_list) == 4
        assert all(b == (0.0, 1.0) for b in bounds_list)

    def test_get_bounds_list_unknown_param(self):
        """Test get_bounds_list falls back to default for unknown params."""
        bounds = ParameterBounds()
        param_names = ["unknown_param"]

        bounds_list = bounds.get_bounds_list(param_names)

        # Should use a_min, a_max as fallback
        assert bounds_list[0] == (0.0, 1.0)

    def test_frozen_dataclass(self):
        """Test that ParameterBounds is immutable."""
        bounds = ParameterBounds()

        with pytest.raises(AttributeError):
            bounds.a_min = 0.5

    def test_global_param_bounds_instance(self):
        """Test global PARAM_BOUNDS instance exists and has correct values."""
        assert PARAM_BOUNDS.a_min == 0.0
        assert PARAM_BOUNDS.a_max == 1.0


class TestNoiseConfig:
    """Tests for NoiseConfig configuration."""

    def test_default_noise_fraction(self):
        """Test default noise fraction is 0.10."""
        config = NoiseConfig()

        assert config.noise_fraction == 0.10

    def test_custom_noise_fraction(self):
        """Test custom noise fraction can be set."""
        config = NoiseConfig(noise_fraction=0.05)

        assert config.noise_fraction == 0.05

    def test_get_noise_std(self):
        """Test get_noise_std computes correct value."""
        config = NoiseConfig(noise_fraction=0.10)
        signal_std = 2.0

        noise_std = config.get_noise_std(signal_std)

        assert noise_std == 0.2  # 0.10 * 2.0

    def test_get_noise_std_zero_signal(self):
        """Test get_noise_std with zero signal std."""
        config = NoiseConfig(noise_fraction=0.10)

        noise_std = config.get_noise_std(0.0)

        assert noise_std == 0.0

    def test_frozen_dataclass(self):
        """Test that NoiseConfig is immutable."""
        config = NoiseConfig()

        with pytest.raises(AttributeError):
            config.noise_fraction = 0.5

    def test_global_noise_config_instance(self):
        """Test global NOISE_CONFIG instance exists and has correct value."""
        assert NOISE_CONFIG.noise_fraction == 0.10


class TestNoiseConfigIntegration:
    """Integration tests for NoiseConfig with typical usage patterns."""

    def test_noise_addition_pattern(self):
        """Test typical noise addition pattern."""
        # Simulate BOLD signal with larger sample for better statistics
        rng = np.random.default_rng(42)
        bold_true = rng.standard_normal((1000, 2))

        # Add noise using NOISE_CONFIG
        noise_sigma = NOISE_CONFIG.get_noise_std(np.std(bold_true))
        noise = rng.normal(0.0, noise_sigma, size=bold_true.shape)
        bold_obs = bold_true + noise

        # Verify noise level is approximately correct (with wider tolerance for sampling variance)
        actual_noise_std = np.std(bold_obs - bold_true)
        assert np.isclose(actual_noise_std, noise_sigma, rtol=0.15)

    def test_consistency_across_calls(self):
        """Test that same signal std gives same noise std."""
        signal_std = 1.5

        noise_std_1 = NOISE_CONFIG.get_noise_std(signal_std)
        noise_std_2 = NOISE_CONFIG.get_noise_std(signal_std)

        assert noise_std_1 == noise_std_2


class TestParameterBoundsIntegration:
    """Integration tests for ParameterBounds with typical usage patterns."""

    def test_scipy_optimize_bounds_format(self):
        """Test bounds work with scipy.optimize format."""
        param_names = ["a01", "a10", "c0", "c1"]
        bounds_list = PARAM_BOUNDS.get_bounds_list(param_names)

        # scipy.optimize expects list of (min, max) tuples
        for low, high in bounds_list:
            assert isinstance(low, float)
            assert isinstance(high, float)
            assert low < high

    def test_parameter_generation_within_bounds(self):
        """Test that random parameters can be generated within bounds."""
        rng = np.random.default_rng(42)
        param_names = ["a01", "a10", "c0", "c1"]
        bounds_list = PARAM_BOUNDS.get_bounds_list(param_names)

        # Generate random parameters within bounds
        params = np.array([rng.uniform(low, high) for (low, high) in bounds_list])

        # Verify all within bounds
        for i, (low, high) in enumerate(bounds_list):
            assert low <= params[i] <= high
