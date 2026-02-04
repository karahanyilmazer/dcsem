"""
Central configuration for DCSEM package.

This module provides a single source of truth for parameter bounds, noise settings,
and path configuration across all scripts in the repository.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    # Look for .env in the project root (parent of dcsem package)
    _project_root = Path(__file__).parent.parent
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass  # python-dotenv not installed, skip .env loading


@dataclass(frozen=True)
class ParameterBounds:
    """
    Canonical parameter bounds for DCM estimation.

    These bounds are used consistently across all estimation scripts to ensure
    comparable results. The default (0, 1) range is appropriate for normalized
    connectivity and input strength parameters.
    """

    # A matrix bounds (connectivity strengths)
    a_min: float = 0.0
    a_max: float = 1.0

    # C matrix bounds (input strengths)
    c_min: float = 0.0
    c_max: float = 1.0

    def get_bounds_dict(self) -> dict:
        """Return bounds as a dictionary for common parameter names."""
        return {
            "a01": (self.a_min, self.a_max),
            "a10": (self.a_min, self.a_max),
            "c0": (self.c_min, self.c_max),
            "c1": (self.c_min, self.c_max),
        }

    def get_bounds_list(self, param_names: list[str]) -> list[tuple[float, float]]:
        """Return bounds as a list of tuples for scipy.optimize.minimize."""
        bounds_dict = self.get_bounds_dict()
        return [bounds_dict.get(name, (self.a_min, self.a_max)) for name in param_names]


@dataclass(frozen=True)
class NoiseConfig:
    """
    Configuration for noise addition to BOLD signals.

    The default noise_fraction of 0.10 (10% of signal std) is commonly used
    across the codebase for simulation studies.
    """

    noise_fraction: float = 0.10  # Fraction of signal std to use as noise std

    def get_noise_std(self, signal_std: float) -> float:
        """Calculate noise standard deviation from signal std."""
        return self.noise_fraction * signal_std


@dataclass
class PathConfig:
    """
    Configuration for output paths.

    The latex_dir is loaded from the DCSEM_LATEX_DIR environment variable,
    which can be set in a .env file in the project root.
    """

    latex_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("DCSEM_LATEX_DIR")
    )

    def get_latex_path(self) -> Optional[Path]:
        """Return latex directory as a Path object, or None if not configured."""
        if self.latex_dir is not None:
            return Path(self.latex_dir)
        return None


# Default instances for use throughout the codebase
PARAM_BOUNDS = ParameterBounds()
NOISE_CONFIG = NoiseConfig()
PATH_CONFIG = PathConfig()
