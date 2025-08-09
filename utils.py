import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from dcsem.models import DCM
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


def simulate_bold(params, num_rois, time, u):
    """
    Simulate BOLD signals for the given parameters.

    Args:
        params: Dictionary of parameters, where each key can be a single value or an array of values.
        num_rois: Number of regions of interest (ROIs).
        time: Time vector for simulation.
        u: Input signal.

    Returns:
        A list of simulated BOLD signals, one for each value in the parameter arrays.
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
        return dcm.simulate(time, u)[0]

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
        bold_signal = dcm.simulate(time, u)[0]
        results.append(bold_signal)

    return np.array(results)


def add_noise(signal, snr_db):
    signal_power = np.mean(signal**2)
    snr = 10 ** (snr_db / 10)  # Convert dB to linear scale
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise
    return noisy_signal


def add_underscore(param):
    # Use regex to insert an underscore before a digit sequence and group digits for LaTeX
    latex_param = re.sub(r"(\D)(\d+)", r"\1_{\2}", param)
    return r"${" + latex_param + r"}$"


def set_style():
    plt.style.use(["science", "no-latex"])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 300


def get_param_colors():
    set_style()
    # Set the colors for each parameter
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]
    param_colors = dict(zip(["a01", "a10", "c0", "c1"], color_cycle))
    return param_colors


def get_summary_measures(method, time, u, num_rois, **kwargs):
    # Define the allowed parameters
    allowed_keys = ["a01", "a10", "c0", "c1"]

    # Find invalid keys
    invalid_keys = [key for key in kwargs.keys() if key not in allowed_keys]

    # Assert that all keys are allowed
    assert (
        not invalid_keys
    ), f"Invalid parameter keys: {invalid_keys}. Allowed keys are: {allowed_keys}."
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
    assert all(
        length == lengths[0] for length in lengths
    ), "All values must have the same length!"

    # Initialize the BOLD signals
    bold_true = simulate_bold(
        params,
        time=time,
        u=u,
        num_rois=num_rois,
    )
    bold_obsv = bold_true

    tmp_bold = np.concatenate([bold_obsv[:, :, 0], bold_obsv[:, :, 1]], axis=1)
    tmp_bold_c = tmp_bold - np.mean(tmp_bold, axis=1, keepdims=True)

    if method == "PCA":
        pca = pickle.load(open("models/pca.pkl", "rb"))
        components = pca.transform(tmp_bold_c)
    elif method == "ICA":
        ica = pickle.load(open("models/ica.pkl", "rb"))
        components = ica.transform(tmp_bold_c)

    return components
