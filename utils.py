import json
import pickle
import re
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from pypalettes import load_cmap

from dcsem.config import PATH_CONFIG
from dcsem.models import DCM

# Import plotting functions from dcsem.plotting for backwards compatibility
# These are re-exported here with deprecation warnings
from dcsem.plotting import (
    add_underscore as _add_underscore,
)
from dcsem.plotting import (
    get_colormap as _get_colormap,
)
from dcsem.plotting import (
    get_param_colors as _get_param_colors,
)
from dcsem.plotting import (
    get_width_height_latex as _get_width_height_latex,
)
from dcsem.plotting import (
    list_available_colormaps as _list_available_colormaps,
)
from dcsem.plotting import (
    set_style as _set_style,
)
from dcsem.plotting import (
    to_latex_label as _to_latex_label,
)
from dcsem.utils import create_A_matrix, create_C_matrix


def filter_params(params, keys, exclude=False):
    if exclude:
        return {k: params[k] for k in params.keys() if k not in keys}
    return {k: params[k] for k in keys}


def initialize_parameters(bounds, params_to_sim, random=False):
    initial_values = []
    for param in params_to_sim:
        if random:
            initial_values.append(np.random.uniform(*bounds[param]))
        else:
            initial_values.append(np.mean(bounds[param]))

    return initial_values


def get_one_layer_A(a01=0.4, a10=0.4, self_connections=-1):
    connections = []
    connections.append(f"R0, L0 -> R1, L0 = {a01}")  # ROI0 -> ROI1 connection
    connections.append(f"R1, L0 -> R0, L0 = {a10}")  # ROI1 -> ROI0 connection
    return create_A_matrix(
        num_rois=2,
        num_layers=1,
        paired_connections=connections,
        self_connections=self_connections,
    )


def get_one_layer_C(c0=0.5, c1=0.5):
    connections = []
    connections.append(f"R0, L0 = {c0}")  # Input --> ROI0 connection
    connections.append(f"R1, L0 = {c1}")  # Input --> ROI1 connection
    return create_C_matrix(num_rois=2, num_layers=1, input_connections=connections)


def simulate_bold(
    params,
    num_rois,
    time,
    u,
    squeeze=True,
    ode_method=None,
    ode_rtol=None,
    ode_atol=None,
    ode_max_step=None,
):
    """
    Simulate BOLD signals for the given parameters.

    Args:
        params: Dictionary of parameters, where each key can be a single value or an array of values.
        num_rois: Number of regions of interest (ROIs).
        time: Time vector for simulation.
        u: Input signal.

    Returns:
        A numpy array of shape (N, T, R) where N is number of parameter sets,
        T is time points, and R is number of ROIs.
    """
    # Define input arguments for defining A and C matrices
    A_param_names = ["a01", "a10", "self_connections"]
    C_param_names = ["c0", "c1"]

    # Determine if any parameter is an array
    param_keys = list(params.keys())

    array_keys = [
        key
        for key in param_keys
        if isinstance(params[key], np.ndarray)
        or isinstance(params[key], list)
        and len(params[key]) > 1
    ]

    if not array_keys:  # If no parameter is an array, run single simulation
        A_kwargs = {k: float(params[k]) for k in A_param_names if k in params}
        C_kwargs = {k: float(params[k]) for k in C_param_names if k in params}
        A = get_one_layer_A(**A_kwargs)
        C = get_one_layer_C(**C_kwargs)
        dcm = DCM(
            num_rois,
            params={
                "A": A,
                "C": C,
                **params,
            },
        )
        # Set optional solver controls
        if ode_method is not None:
            dcm.ode_method = ode_method
        if ode_rtol is not None:
            dcm.ode_rtol = ode_rtol
        if ode_atol is not None:
            dcm.ode_atol = ode_atol
        if ode_max_step is not None:
            dcm.ode_max_step = ode_max_step
        bold, _ = dcm.simulate(time, u)
        if squeeze:
            return bold  # Shape (T, R)
        else:
            return bold[np.newaxis, :, :]  # Shape (1, T, R)

    # If there are arrays, run multiple simulations
    results = []
    max_length = len(params[array_keys[0]])  # Assume all arrays have the same length
    for i in range(max_length):
        # Extract single values for array parameters
        single_params = {
            k: params[k][i] if k in array_keys else params[k] for k in param_keys
        }

        # Generate A and C matrices for the current parameter set
        A_kwargs = {
            k: float(single_params[k]) for k in A_param_names if k in single_params
        }
        C_kwargs = {
            k: float(single_params[k]) for k in C_param_names if k in single_params
        }
        A = get_one_layer_A(**A_kwargs)
        C = get_one_layer_C(**C_kwargs)

        # Simulate BOLD signal
        dcm = DCM(
            num_rois,
            params={
                "A": A,
                "C": C,
                **single_params,
            },
        )
        # Set optional solver controls
        if ode_method is not None:
            dcm.ode_method = ode_method
        if ode_rtol is not None:
            dcm.ode_rtol = ode_rtol
        if ode_atol is not None:
            dcm.ode_atol = ode_atol
        if ode_max_step is not None:
            dcm.ode_max_step = ode_max_step
        bold, _ = dcm.simulate(time, u)
        results.append(bold)

    return np.array(results)  # Shape (N, T, R)


def add_noise(signal, snr_db=None, tsnr=None, noise_std=None, rng=None):
    """
    Add Gaussian noise to a BOLD signal using one of three methods.

    Args:
        signal: numpy array of BOLD signal to add noise to (shape: T or T×R).
        snr_db: (optional) signal-to-noise ratio in decibels (power-based).
        tsnr: (optional) temporal SNR = mean(signal) / std(noise) (amplitude-based, fMRI standard).
        noise_std: (optional) direct specification of noise standard deviation.
        rng: numpy random Generator instance or None.

    Note: Provide exactly ONE of snr_db, tsnr, or noise_std.

    Temporal SNR (tSNR) is the standard metric in fMRI:
        - tSNR = mean(signal) / std(noise)
        - Typical values: 7-50 for raw voxels, up to 400 for regional averages
        - Lower tSNR = more noise relative to baseline signal

    Returns:
        noisy_signal: signal plus Gaussian noise.
        noise_std_used: the standard deviation of noise that was added.

    Examples:
        # Add noise with tSNR=50 (typical for fMRI regional average)
        noisy, noise_std = add_noise(bold_signal, tsnr=50)

        # Add noise with specific noise level
        noisy, noise_std = add_noise(bold_signal, noise_std=0.1)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Check that exactly one noise specification is provided
    specs_provided = sum(x is not None for x in [snr_db, tsnr, noise_std])
    if specs_provided != 1:
        raise ValueError("Provide exactly ONE of: snr_db, tsnr, or noise_std")

    # Calculate noise standard deviation based on the method
    if noise_std is not None:
        # Direct specification
        sigma = noise_std

    elif tsnr is not None:
        # Temporal SNR (fMRI standard): tSNR = mean(signal) / std(noise)
        # Therefore: std(noise) = mean(signal) / tSNR
        signal_mean = np.mean(signal)
        sigma = signal_mean / tsnr

    else:  # snr_db is not None
        # Power-based SNR (convert dB to linear scale)
        signal_power = np.mean(signal**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        sigma = np.sqrt(noise_power)

    # Generate and add noise
    noise = rng.normal(0, sigma, signal.shape)
    noisy_signal = signal + noise

    return noisy_signal, sigma


# Wrapper functions that delegate to dcsem.plotting
# These maintain backwards compatibility for existing code importing from utils


def add_underscore(param, bold=False):
    """Add LaTeX subscript formatting. Delegated to dcsem.plotting."""
    return _add_underscore(param, bold)


def to_latex_label(param):
    """Convert parameter name to LaTeX label. Delegated to dcsem.plotting."""
    return _to_latex_label(param)


def get_width_height_latex(column_width=483.6969):
    """Calculate figure dimensions for LaTeX. Delegated to dcsem.plotting."""
    return _get_width_height_latex(column_width)


def set_style(dpi=300, cmap="science"):
    """Set matplotlib style. Delegated to dcsem.plotting."""
    return _set_style(dpi, cmap)


def get_param_colors():
    """Get consistent colors for parameters. Delegated to dcsem.plotting."""
    return _get_param_colors()


def get_summary_measures(method, time, u, num_rois, model_dir, setting, **kwargs):
    # Define the allowed parameters
    allowed_keys = ["a01", "a10", "c0", "c1"]

    # Find invalid keys
    invalid_keys = [key for key in kwargs.keys() if key not in allowed_keys]

    # Assert that all keys are allowed
    assert not invalid_keys, (
        f"Invalid parameter keys: {invalid_keys}. Allowed keys are: {allowed_keys}."
    )
    # Filter all arguments that are not None
    params = {}
    for key, val in kwargs.items():
        if key == "method":
            continue
        if val is not None:
            # Convert the values to a numpy array
            if not isinstance(val, (list, np.ndarray)):
                val = [val]
            if not isinstance(val, np.ndarray):
                val = np.array(val)

            params[key] = val

    # Assert that all values have the same length
    lengths = [len(v) for v in params.values()]
    assert all(length == lengths[0] for length in lengths), (
        "All values must have the same length!"
    )

    # Initialize the BOLD signals
    bold_true = simulate_bold(
        params,
        time=time,
        u=u,
        num_rois=num_rois,
    )
    bold_obsv = bold_true

    # Concatenate along the last axis (ROIs) for PCA/ICA input
    # bold_obsv shape: (N, T, R) --> (N, T*R)
    tmp_bold = bold_obsv.reshape(bold_obsv.shape[0], -1)

    # Center the data
    tmp_bold_c = tmp_bold - np.mean(tmp_bold, axis=1, keepdims=True)

    if method == "PCA":
        with open(model_dir / f"pca_{setting}.pkl", "rb") as f:
            pca = pickle.load(f)
        components = pca.transform(tmp_bold_c)
    elif method == "ICA":
        with open(model_dir / f"ica_{setting}.pkl", "rb") as f:
            ica = pickle.load(f)
        components = ica.transform(tmp_bold_c)
    else:
        raise ValueError(f"Method '{method}' not supported. Use 'PCA' or 'ICA'.")

    return components


def get_out_dir(type="img", subfolder=None, extra_subfolders=None):
    """
    Get output directory with flexible subdirectory creation.

    Args:
        type: Type of output directory ('img' or 'model')
        subfolder: Main subfolder (e.g., 'wip', 'final')
        extra_subfolders: Additional nested subfolders as string or list
                              (e.g., 'estimation' or ['estimation', 'plots'])

    Returns:
        Path object pointing to the created directory

    Examples:
        get_out_dir("img", "wip", "estimation")  # results/images/wip/estimation/
        get_out_dir("img", "final", ["plots", "snr"])  # results/images/final/plots/snr/
    """
    if type == "img":
        out_dir = Path("results/images")
    elif type == "model":
        out_dir = Path("results/models")
    elif type == "latex":
        latex_path = PATH_CONFIG.get_latex_path()
        if latex_path is None:
            raise ValueError(
                "DCSEM_LATEX_DIR environment variable not set. "
                "Please set it in your .env file or environment."
            )
        out_dir = latex_path
    else:
        raise ValueError(f"Unknown output type: {type}. Use 'img', 'model' or 'latex'.")

    # Get the absolute path to the output directory
    out_dir = Path(__file__).parent / out_dir

    # Add main subfolder if provided
    if subfolder:
        out_dir = out_dir / subfolder

    # Add additional subfolders if provided
    if extra_subfolders:
        if isinstance(extra_subfolders, str):
            out_dir = out_dir / extra_subfolders
        elif isinstance(extra_subfolders, (list, tuple)):
            for sub in extra_subfolders:
                out_dir = out_dir / sub
        else:
            raise ValueError("extra_subfolders must be string, list, or tuple")

    # Create the output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def log_run(
    model_name,
    method,
    seed,
    settings,
    params,
    hessian,
    performance,
    log_dir="results/logs",
    diagnostics=None,
    correlation=None,
    overwrite=False,
):
    """
    Log a run (standard optimization or MCMC).
    If diagnostics is provided, treat as MCMC-style run and include diagnostics in metadata.
    Hessian is stored as {} if None.
    Correlation matrix can be provided separately.
    Files are saved with timestamps to avoid overwriting existing logs.
    """
    # Handle hessian as empty dict if None
    hessian_to_store = hessian if hessian is not None else {}

    # Add correlation to hessian dict if provided
    if correlation is not None:
        hessian_to_store["correlation"] = correlation

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "method": method,
        "seed": seed,
        "settings": settings,
        "params": params,
        "hessian": hessian_to_store,
        "performance": performance,
        "diagnostics": diagnostics,
    }
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp to avoid overwriting
    if overwrite:
        fname = Path(log_dir) / f"{model_name}_{method}.json"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = Path(log_dir) / f"{model_name}_{method}_{timestamp}.json"

    with open(fname, "w") as f:
        json.dump(record, f, indent=2)
    if diagnostics is not None:
        print(f"Logged MCMC results to {fname}")
    else:
        print(f"Logged standard run results to {fname}")


def get_colormap(name="parula", as_colors=False):
    """Get a colormap by name. Delegated to dcsem.plotting."""
    return _get_colormap(name, as_colors)


def list_available_colormaps():
    """List all available colormaps. Delegated to dcsem.plotting."""
    return _list_available_colormaps()
