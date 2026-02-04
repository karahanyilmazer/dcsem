"""
DCSEM: Dynamic Causal Modeling for fMRI BOLD Signal Simulation and Parameter Estimation.

This package provides tools for simulating fMRI BOLD signals using Dynamic Causal
Modeling (DCM) and for estimating model parameters through various methods including
L-BFGS-B optimization, MCMC, and BENCH.
"""

# Configuration - single source of truth for parameters
from .config import NOISE_CONFIG, PARAM_BOUNDS, PATH_CONFIG

# Core models
from .models import DCM, MultiLayerDCM, MultiLayerSEM, SEM, TwoLayerDCM

# Plotting utilities
from .plotting import (
    add_underscore,
    get_colormap,
    get_param_colors,
    get_width_height_latex,
    list_available_colormaps,
    set_style,
    to_latex_label,
)

# Core utilities
from .utils import (
    A_to_text,
    C_to_text,
    MH,
    create_A_matrix,
    create_C_matrix,
    plot_posterior,
    plot_signals,
    stim_boxcar,
)

__all__ = [
    # Configuration
    "PARAM_BOUNDS",
    "NOISE_CONFIG",
    "PATH_CONFIG",
    # Models
    "DCM",
    "TwoLayerDCM",
    "MultiLayerDCM",
    "SEM",
    "MultiLayerSEM",
    # Plotting
    "set_style",
    "get_param_colors",
    "to_latex_label",
    "add_underscore",
    "get_colormap",
    "list_available_colormaps",
    "get_width_height_latex",
    # Utilities
    "create_A_matrix",
    "create_C_matrix",
    "A_to_text",
    "C_to_text",
    "stim_boxcar",
    "plot_signals",
    "plot_posterior",
    "MH",
]
