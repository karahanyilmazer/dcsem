import numpy as np

from dcsem.models import DCM
from dcsem.utils import create_A_matrix, create_C_matrix


def get_one_layer_A(w01=0.4, w10=0.4, self_connections=-1):
    connections = []
    connections.append(f'R0, L0 -> R1, L0 = {w01}')  # ROI0 -> ROI1 connection
    connections.append(f'R1, L0 -> R0, L0 = {w10}')  # ROI1 -> ROI0 connection
    return create_A_matrix(
        num_rois=2,
        num_layers=1,
        paired_connections=connections,
        self_connections=self_connections,
    )


def get_one_layer_C(i0=1, i1=0):
    connections = []
    connections.append(f'R0, L0 = {i0}')  # Input --> ROI0 connection
    connections.append(f'R1, L0 = {i1}')  # Input --> ROI1 connection
    return create_C_matrix(num_rois=2, num_layers=1, input_connections=connections)


def simulate_bold(params, num_rois, time, u):
    # Define input arguments for defining A and C matrices
    A_param_names = ['w01', 'w10', 'self_connections']
    C_param_names = ['i0', 'i1']

    # Extract relevant parameters for A and C matrices from params
    A_kwargs = {k: params[k] for k in A_param_names if k in params}
    C_kwargs = {k: params[k] for k in C_param_names if k in params}

    # Create A and C matrices using extracted parameters
    A = get_one_layer_A(**A_kwargs)
    C = get_one_layer_C(**C_kwargs)

    dcm = DCM(
        num_rois,
        params={
            'A': A,
            'C': C,
            **params,
        },
    )
    return dcm.simulate(time, u)[0]


def simulate_bold_multi(params, num_rois, time, u):
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
    A_param_names = ['w01', 'w10', 'self_connections']
    C_param_names = ['i0', 'i1']

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
                'A': A,
                'C': C,
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
                'A': A,
                'C': C,
                **single_params,
            },
        )
        bold_signal = dcm.simulate(time, u)[0]
        results.append(bold_signal)

    return np.array(results)


def add_noise(bold_true, snr_db):
    snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear scale
    sigma = np.max(bold_true) / snr_linear
    return bold_true + np.random.normal(0, sigma, bold_true.shape)


def filter_params(params, keys):
    return {k: params[k] for k in keys}


def initialize_parameters(bounds, params_to_sim, random=False):
    initial_values = []
    for param in params_to_sim:
        if random:
            initial_values.append(np.random.uniform(*bounds[param]))
        else:
            initial_values.append(np.mean(bounds[param]))

    return initial_values
