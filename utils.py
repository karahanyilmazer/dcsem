import json
import pickle
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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


def add_noise(signal, snr_db, rng=None):
    """
    Add Gaussian noise to a signal given a target SNR in dB.

    Args:
        signal: numpy array of signal to add noise to.
        snr_db: desired signal-to-noise ratio in decibels.
        rng: numpy random Generator instance or None.

    Returns:
        noisy_signal: signal plus Gaussian noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    signal_power = np.mean(signal**2)
    snr = 10 ** (snr_db / 10)  # Convert dB to linear scale
    noise_power = signal_power / snr
    noise = rng.normal(0, np.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise
    return noisy_signal


def add_underscore(param):
    # Use regex to insert an underscore before a digit sequence and group digits for LaTeX
    latex_param = re.sub(r"(\D)(\d+)", r"\1_{\2}", param)
    return r"${" + latex_param + r"}$"


def set_style(font_family=None, use_science=True, use_latex=False, dpi=300):
    styles = []
    if use_science:
        styles.append("science")
    if use_latex:
        styles.append("latex")
    else:
        styles.append("no-latex")
    plt.style.use(styles)
    if font_family is not None:
        plt.rcParams["font.family"] = font_family
    plt.rcParams["figure.dpi"] = dpi


def get_param_colors():
    # Set the colors for each parameter
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]
    param_colors = dict(zip(["a01", "a10", "c0", "c1"], color_cycle))
    return param_colors


def get_summary_measures(method, time, u, num_rois, model_dir, **kwargs):
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

    # Concatenate along the last axis (ROIs) for PCA/ICA input
    # bold_obsv shape: (N, T, R) --> (N, T*R)
    tmp_bold = bold_obsv.reshape(bold_obsv.shape[0], -1)

    # Center the data
    tmp_bold_c = tmp_bold - np.mean(tmp_bold, axis=1, keepdims=True)

    if method == "PCA":
        pca = pickle.load(open(model_dir / "pca.pkl", "rb"))
        components = pca.transform(tmp_bold_c)
    elif method == "ICA":
        ica = pickle.load(open(model_dir / "ica.pkl", "rb"))
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
    else:
        raise ValueError(f"Unknown output type: {type}. Use 'img' or 'model'.")

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
    model_name, method, seed, settings, params, hessian, performance, log_dir="logs"
):
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "method": method,
        "seed": seed,
        "settings": settings,
        "params": params,
        "hessian": hessian,
        "performance": performance,
    }
    Path(log_dir).mkdir(exist_ok=True)
    fname = Path(log_dir) / f"{model_name}_{method}.json"
    with open(fname, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Logged results to {fname}")


def get_colormap(name="parula", as_colors=False):
    """
    Get a colormap by name. First checks palettable library, then falls back to custom colormaps.

    Parameters:
    -----------
    name : str
        Name of the colormap to retrieve

    Returns:
    --------
    matplotlib colormap object (mpl.cm.Colormap or mpl.colors.Colormap if as_colors=True)
    """
    import inspect

    import matplotlib.colors as mcolors
    import numpy as np

    # First, try to find the colormap in palettable
    try:
        import palettable

        # Search through all palettable modules and submodules
        for module_name in dir(palettable):
            if module_name.startswith("_"):
                continue

            module = getattr(palettable, module_name)
            if not inspect.ismodule(module):
                continue

            # Check submodules (e.g., sequential, diverging, qualitative)
            for submodule_name in dir(module):
                if submodule_name.startswith("_"):
                    continue

                submodule = getattr(module, submodule_name)
                if not inspect.ismodule(submodule):
                    continue

                # Look for the colormap in this submodule
                for attr_name in dir(submodule):
                    if attr_name.lower() == name.lower() and hasattr(
                        getattr(submodule, attr_name), "mpl_colormap"
                    ):
                        colormap_obj = getattr(submodule, attr_name)
                        print(
                            f"Found '{name}' in palettable.{module_name}.{submodule_name}"
                        )
                        if as_colors and hasattr(colormap_obj, "mpl_colors"):
                            cmap = colormap_obj.mpl_colors
                        else:
                            cmap = colormap_obj.mpl_colormap
                        return cmap

    except ImportError:
        print("Palettable not available, using custom colormaps only")
    except Exception as e:
        print(f"Error searching palettable: {e}")

    # Fall back to custom colormaps if not found in palettable
    if name.lower() == "parula":
        cm_data = [
            [0.2422, 0.1504, 0.6603],
            [0.2444, 0.1534, 0.6728],
            [0.2464, 0.1569, 0.6847],
            [0.2484, 0.1607, 0.6961],
            [0.2503, 0.1648, 0.7071],
            [0.2522, 0.1689, 0.7179],
            [0.254, 0.1732, 0.7286],
            [0.2558, 0.1773, 0.7393],
            [0.2576, 0.1814, 0.7501],
            [0.2594, 0.1854, 0.761],
            [0.2611, 0.1893, 0.7719],
            [0.2628, 0.1932, 0.7828],
            [0.2645, 0.1972, 0.7937],
            [0.2661, 0.2011, 0.8043],
            [0.2676, 0.2052, 0.8148],
            [0.2691, 0.2094, 0.8249],
            [0.2704, 0.2138, 0.8346],
            [0.2717, 0.2184, 0.8439],
            [0.2729, 0.2231, 0.8528],
            [0.274, 0.228, 0.8612],
            [0.2749, 0.233, 0.8692],
            [0.2758, 0.2382, 0.8767],
            [0.2766, 0.2435, 0.884],
            [0.2774, 0.2489, 0.8908],
            [0.2781, 0.2543, 0.8973],
            [0.2788, 0.2598, 0.9035],
            [0.2794, 0.2653, 0.9094],
            [0.2798, 0.2708, 0.915],
            [0.2802, 0.2764, 0.9204],
            [0.2806, 0.2819, 0.9255],
            [0.2809, 0.2875, 0.9305],
            [0.2811, 0.293, 0.9352],
            [0.2813, 0.2985, 0.9397],
            [0.2814, 0.304, 0.9441],
            [0.2814, 0.3095, 0.9483],
            [0.2813, 0.315, 0.9524],
            [0.2811, 0.3204, 0.9563],
            [0.2809, 0.3259, 0.96],
            [0.2807, 0.3313, 0.9636],
            [0.2803, 0.3367, 0.967],
            [0.2798, 0.3421, 0.9702],
            [0.2791, 0.3475, 0.9733],
            [0.2784, 0.3529, 0.9763],
            [0.2776, 0.3583, 0.9791],
            [0.2766, 0.3638, 0.9817],
            [0.2754, 0.3693, 0.984],
            [0.2741, 0.3748, 0.9862],
            [0.2726, 0.3804, 0.9881],
            [0.271, 0.386, 0.9898],
            [0.2691, 0.3916, 0.9912],
            [0.267, 0.3973, 0.9924],
            [0.2647, 0.403, 0.9935],
            [0.2621, 0.4088, 0.9946],
            [0.2591, 0.4145, 0.9955],
            [0.2556, 0.4203, 0.9965],
            [0.2517, 0.4261, 0.9974],
            [0.2473, 0.4319, 0.9983],
            [0.2424, 0.4378, 0.9991],
            [0.2369, 0.4437, 0.9996],
            [0.2311, 0.4497, 0.9995],
            [0.225, 0.4559, 0.9985],
            [0.2189, 0.462, 0.9968],
            [0.2128, 0.4682, 0.9948],
            [0.2066, 0.4743, 0.9926],
            [0.2006, 0.4803, 0.9906],
            [0.195, 0.4861, 0.9887],
            [0.1903, 0.4919, 0.9867],
            [0.1869, 0.4975, 0.9844],
            [0.1847, 0.503, 0.9819],
            [0.1831, 0.5084, 0.9793],
            [0.1818, 0.5138, 0.9766],
            [0.1806, 0.5191, 0.9738],
            [0.1795, 0.5244, 0.9709],
            [0.1785, 0.5296, 0.9677],
            [0.1778, 0.5349, 0.9641],
            [0.1773, 0.5401, 0.9602],
            [0.1768, 0.5452, 0.956],
            [0.1764, 0.5504, 0.9516],
            [0.1755, 0.5554, 0.9473],
            [0.174, 0.5605, 0.9432],
            [0.1716, 0.5655, 0.9393],
            [0.1686, 0.5705, 0.9357],
            [0.1649, 0.5755, 0.9323],
            [0.161, 0.5805, 0.9289],
            [0.1573, 0.5854, 0.9254],
            [0.154, 0.5902, 0.9218],
            [0.1513, 0.595, 0.9182],
            [0.1492, 0.5997, 0.9147],
            [0.1475, 0.6043, 0.9113],
            [0.1461, 0.6089, 0.908],
            [0.1446, 0.6135, 0.905],
            [0.1429, 0.618, 0.9022],
            [0.1408, 0.6226, 0.8998],
            [0.1383, 0.6272, 0.8975],
            [0.1354, 0.6317, 0.8953],
            [0.1321, 0.6363, 0.8932],
            [0.1288, 0.6408, 0.891],
            [0.1253, 0.6453, 0.8887],
            [0.1219, 0.6497, 0.8862],
            [0.1185, 0.6541, 0.8834],
            [0.1152, 0.6584, 0.8804],
            [0.1119, 0.6627, 0.877],
            [0.1085, 0.6669, 0.8734],
            [0.1048, 0.671, 0.8695],
            [0.1009, 0.675, 0.8653],
            [0.0964, 0.6789, 0.8609],
            [0.0914, 0.6828, 0.8562],
            [0.0855, 0.6865, 0.8513],
            [0.0789, 0.6902, 0.8462],
            [0.0713, 0.6938, 0.8409],
            [0.0628, 0.6972, 0.8355],
            [0.0535, 0.7006, 0.8299],
            [0.0433, 0.7039, 0.8242],
            [0.0328, 0.7071, 0.8183],
            [0.0234, 0.7103, 0.8124],
            [0.0155, 0.7133, 0.8064],
            [0.0091, 0.7163, 0.8003],
            [0.0046, 0.7192, 0.7941],
            [0.0019, 0.722, 0.7878],
            [0.0009, 0.7248, 0.7815],
            [0.0018, 0.7275, 0.7752],
            [0.0046, 0.7301, 0.7688],
            [0.0094, 0.7327, 0.7623],
            [0.0162, 0.7352, 0.7558],
            [0.0253, 0.7376, 0.7492],
            [0.0369, 0.74, 0.7426],
            [0.0504, 0.7423, 0.7359],
            [0.0638, 0.7446, 0.7292],
            [0.077, 0.7468, 0.7224],
            [0.0899, 0.7489, 0.7156],
            [0.1023, 0.751, 0.7088],
            [0.1141, 0.7531, 0.7019],
            [0.1252, 0.7552, 0.695],
            [0.1354, 0.7572, 0.6881],
            [0.1448, 0.7593, 0.6812],
            [0.1532, 0.7614, 0.6741],
            [0.1609, 0.7635, 0.6671],
            [0.1678, 0.7656, 0.6599],
            [0.1741, 0.7678, 0.6527],
            [0.1799, 0.7699, 0.6454],
            [0.1853, 0.7721, 0.6379],
            [0.1905, 0.7743, 0.6303],
            [0.1954, 0.7765, 0.6225],
            [0.2003, 0.7787, 0.6146],
            [0.2061, 0.7808, 0.6065],
            [0.2118, 0.7828, 0.5983],
            [0.2178, 0.7849, 0.5899],
            [0.2244, 0.7869, 0.5813],
            [0.2318, 0.7887, 0.5725],
            [0.2401, 0.7905, 0.5636],
            [0.2491, 0.7922, 0.5546],
            [0.2589, 0.7937, 0.5454],
            [0.2695, 0.7951, 0.536],
            [0.2809, 0.7964, 0.5266],
            [0.2929, 0.7975, 0.517],
            [0.3052, 0.7985, 0.5074],
            [0.3176, 0.7994, 0.4975],
            [0.3301, 0.8002, 0.4876],
            [0.3424, 0.8009, 0.4774],
            [0.3548, 0.8016, 0.4669],
            [0.3671, 0.8021, 0.4563],
            [0.3795, 0.8026, 0.4454],
            [0.3921, 0.8029, 0.4344],
            [0.405, 0.8031, 0.4233],
            [0.4184, 0.803, 0.4122],
            [0.4322, 0.8028, 0.4013],
            [0.4463, 0.8024, 0.3904],
            [0.4608, 0.8018, 0.3797],
            [0.4753, 0.8011, 0.3691],
            [0.4899, 0.8002, 0.3586],
            [0.5044, 0.7993, 0.348],
            [0.5187, 0.7982, 0.3374],
            [0.5329, 0.797, 0.3267],
            [0.547, 0.7957, 0.3159],
            [0.5609, 0.7943, 0.305],
            [0.5748, 0.7929, 0.2941],
            [0.5886, 0.7913, 0.2833],
            [0.6024, 0.7896, 0.2726],
            [0.6161, 0.7878, 0.2622],
            [0.6297, 0.7859, 0.2521],
            [0.6433, 0.7839, 0.2423],
            [0.6567, 0.7818, 0.2329],
            [0.6701, 0.7796, 0.2239],
            [0.6833, 0.7773, 0.2155],
            [0.6963, 0.775, 0.2075],
            [0.7091, 0.7727, 0.1998],
            [0.7218, 0.7703, 0.1924],
            [0.7344, 0.7679, 0.1852],
            [0.7468, 0.7654, 0.1782],
            [0.759, 0.7629, 0.1717],
            [0.771, 0.7604, 0.1658],
            [0.7829, 0.7579, 0.1608],
            [0.7945, 0.7554, 0.157],
            [0.806, 0.7529, 0.1546],
            [0.8172, 0.7505, 0.1535],
            [0.8281, 0.7481, 0.1536],
            [0.8389, 0.7457, 0.1546],
            [0.8495, 0.7435, 0.1564],
            [0.86, 0.7413, 0.1587],
            [0.8703, 0.7392, 0.1615],
            [0.8804, 0.7372, 0.165],
            [0.8903, 0.7353, 0.1695],
            [0.9, 0.7336, 0.1749],
            [0.9093, 0.7321, 0.1815],
            [0.9184, 0.7308, 0.189],
            [0.9272, 0.7298, 0.1973],
            [0.9357, 0.729, 0.2061],
            [0.944, 0.7285, 0.2151],
            [0.9523, 0.7284, 0.2237],
            [0.9606, 0.7285, 0.2312],
            [0.9689, 0.7292, 0.2373],
            [0.977, 0.7304, 0.2418],
            [0.9842, 0.733, 0.2446],
            [0.99, 0.7365, 0.2429],
            [0.9946, 0.7407, 0.2394],
            [0.9966, 0.7458, 0.2351],
            [0.9971, 0.7513, 0.2309],
            [0.9972, 0.7569, 0.2267],
            [0.9971, 0.7626, 0.2224],
            [0.9969, 0.7683, 0.2181],
            [0.9966, 0.774, 0.2138],
            [0.9962, 0.7798, 0.2095],
            [0.9957, 0.7856, 0.2053],
            [0.9949, 0.7915, 0.2012],
            [0.9938, 0.7974, 0.1974],
            [0.9923, 0.8034, 0.1939],
            [0.9906, 0.8095, 0.1906],
            [0.9885, 0.8156, 0.1875],
            [0.9861, 0.8218, 0.1846],
            [0.9835, 0.828, 0.1817],
            [0.9807, 0.8342, 0.1787],
            [0.9778, 0.8404, 0.1757],
            [0.9748, 0.8467, 0.1726],
            [0.972, 0.8529, 0.1695],
            [0.9694, 0.8591, 0.1665],
            [0.9671, 0.8654, 0.1636],
            [0.9651, 0.8716, 0.1608],
            [0.9634, 0.8778, 0.1582],
            [0.9619, 0.884, 0.1557],
            [0.9608, 0.8902, 0.1532],
            [0.9601, 0.8963, 0.1507],
            [0.9596, 0.9023, 0.148],
            [0.9595, 0.9084, 0.145],
            [0.9597, 0.9143, 0.1418],
            [0.9601, 0.9203, 0.1382],
            [0.9608, 0.9262, 0.1344],
            [0.9618, 0.932, 0.1304],
            [0.9629, 0.9379, 0.1261],
            [0.9642, 0.9437, 0.1216],
            [0.9657, 0.9494, 0.1168],
            [0.9674, 0.9552, 0.1116],
            [0.9692, 0.9609, 0.1061],
            [0.9711, 0.9667, 0.1001],
            [0.973, 0.9724, 0.0938],
            [0.9749, 0.9782, 0.0872],
            [0.9769, 0.9839, 0.0805],
        ]

        cmap = LinearSegmentedColormap.from_list("parula", cm_data)
        if as_colors:
            N = getattr(cmap, "N", 256)
            colors = cmap(np.linspace(0, 1, N))
            cmap = mcolors.ListedColormap(colors)
        return cmap

    # If colormap not found in palettable or custom colormaps, try matplotlib
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(name)
        print(f"Found '{name}' in matplotlib")
        if as_colors:
            N = getattr(cmap, "N", 256)
            colors = cmap(np.linspace(0, 1, N))
            cmap = mcolors.ListedColormap(colors)
        return cmap
    except ValueError:
        print(
            f"Colormap '{name}' not found in palettable, custom colormaps, or matplotlib"
        )
        print("Available custom colormaps: parula")
        print("Falling back to 'viridis'")
        cmap = plt.get_cmap("viridis")
        if as_colors:
            N = getattr(cmap, "N", 256)
            colors = cmap(np.linspace(0, 1, N))
            cmap = mcolors.ListedColormap(colors)
        return cmap


def list_available_colormaps():
    """
    List all available colormaps from palettable and custom colormaps.

    Returns:
    --------
    dict : Dictionary with categories as keys and colormap names as values
    """
    import inspect

    available_colormaps = {
        "custom": ["parula"],
        "matplotlib": [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "turbo",
            "coolwarm",
            "seismic",
        ],
    }

    # Get palettable colormaps
    try:
        import palettable

        for module_name in dir(palettable):
            if module_name.startswith("_"):
                continue

            module = getattr(palettable, module_name)
            if not inspect.ismodule(module):
                continue

            # Check submodules
            for submodule_name in dir(module):
                if submodule_name.startswith("_"):
                    continue

                submodule = getattr(module, submodule_name)
                if not inspect.ismodule(submodule):
                    continue

                category_name = f"{module_name}_{submodule_name}"
                available_colormaps[category_name] = []

                # Get all colormap objects
                for attr_name in dir(submodule):
                    if attr_name.startswith("_"):
                        continue

                    attr = getattr(submodule, attr_name)
                    if hasattr(attr, "mpl_colormap"):
                        available_colormaps[category_name].append(attr_name)

                # Remove empty categories
                if not available_colormaps[category_name]:
                    del available_colormaps[category_name]

    except ImportError:
        print("Palettable not available")

    return available_colormaps
