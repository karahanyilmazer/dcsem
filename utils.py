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
