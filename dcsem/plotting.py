"""
Unified plotting utilities for DCSEM package.

This module provides consistent plotting styles and helper functions
for visualization across all scripts in the repository.
"""

import re
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch


def get_width_height_latex(column_width: float = 483.6969) -> tuple[float, float]:
    """
    Calculate figure dimensions for LaTeX documents.

    Args:
        column_width: Column width in points (from LaTeX \\columnwidth).
                      Default is for a typical two-column article.

    Returns:
        Tuple of (width, height) in inches, using golden ratio.
    """
    pt = 1.0 / 72.27
    width = column_width * pt
    golden = (1 + 5**0.5) / 2
    height = width / golden
    return width, height


def set_style(dpi: int = 300, cmap: str = "science") -> None:
    """
    Set consistent matplotlib style for publication-quality figures.

    Args:
        dpi: Resolution for saved figures.
        cmap: Colormap name. 'science' uses a custom scientific color palette,
              otherwise loads from pypalettes.
    """
    width, height = get_width_height_latex()

    if cmap == "science":
        colors = [
            "#0C5DA5",
            "#00B945",
            "#FF9500",
            "#FF2C00",
            "#845B97",
            "#474747",
            "#9e9e9e",
        ]
    else:
        try:
            from pypalettes import load_cmap as pypal_load_cmap

            colors = pypal_load_cmap(cmap).colors
        except ImportError:
            # Fall back to science colors if pypalettes not available
            colors = [
                "#0C5DA5",
                "#00B945",
                "#FF9500",
                "#FF2C00",
                "#845B97",
                "#474747",
                "#9e9e9e",
            ]

    plt.rcParams.update(
        {
            "figure.figsize": (width, height),
            "axes.linewidth": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.top": False,
            "ytick.right": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "legend.edgecolor": "black",
            "legend.borderaxespad": 0.7,
            "text.usetex": True,
            "savefig.bbox": "tight",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.family": "serif",
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.prop_cycle": cycler("color", colors),
        }
    )


def get_param_colors(param_names: list[str] = None) -> dict[str, str]:
    """
    Get consistent colors for DCM parameters.

    Args:
        param_names: List of parameter names. Defaults to ['a01', 'a10', 'c0', 'c1'].

    Returns:
        Dictionary mapping parameter names to colors.
    """
    if param_names is None:
        param_names = ["a01", "a10", "c0", "c1"]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return dict(zip(param_names, color_cycle[: len(param_names)]))


def to_latex_label(param: str) -> str:
    """
    Convert parameter name to LaTeX label.

    Examples:
        'a' -> r'$a$'
        'alpha' -> r'$\\alpha$'
        'Vmax' -> r'$V_{\\mathrm{max}}$'
        'KM' -> r'$K_M$'
        'k1' -> r'$k_1$'
        'A1' -> r'$A_1$'
        'x0' -> r'$x_0$'

    Args:
        param: Parameter name as string.

    Returns:
        LaTeX-formatted string.
    """
    # Greek letters mapping
    greek = {
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "delta": r"\delta",
        "epsilon": r"\epsilon",
        "theta": r"\theta",
        "lambda": r"\lambda",
        "mu": r"\mu",
        "sigma": r"\sigma",
        "tau": r"\tau",
        "phi": r"\phi",
        "omega": r"\omega",
    }

    # Check if it's a Greek letter
    if param.lower() in greek:
        return f"${greek[param.lower()]}$"

    # Special cases
    if param == "Vmax":
        return r"$V_{\mathrm{max}}$"
    if param == "KM":
        return r"$K_M$"

    # Handle mixed case with subscript (e.g., 'k1', 'A2', 'x0')
    # Pattern: letter(s) followed by digit(s)
    match = re.match(r"^([A-Za-z]+)(\d+)$", param)
    if match:
        base, subscript = match.groups()
        return f"${base}_{{{subscript}}}$"

    # Default: wrap in $...$
    return f"${param}$"


def add_underscore(param: str, bold: bool = False) -> str:
    """
    Add LaTeX subscript formatting to parameter name.

    Args:
        param: Parameter name (e.g., 'a01').
        bold: If True, use bold math font.

    Returns:
        LaTeX-formatted string with subscript.
    """
    # Use regex to insert an underscore before a digit sequence and group digits for LaTeX
    latex_param = re.sub(r"(\D)(\d+)", r"\1_{\2}", param)
    if bold:
        return r"$\mathbf{" + latex_param + r"}$"
    else:
        return r"${" + latex_param + r"}$"


def get_colormap(name: str = "parula", as_colors: bool = False):
    """
    Get a colormap by name.

    First checks palettable library, then falls back to custom colormaps,
    then matplotlib built-ins.

    Args:
        name: Name of the colormap to retrieve.
        as_colors: If True, return as ListedColormap instead of LinearSegmentedColormap.

    Returns:
        matplotlib colormap object.
    """
    import inspect
    import matplotlib.colors as mcolors

    # First, try to find the colormap in palettable
    try:
        import palettable

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
                        if as_colors and hasattr(colormap_obj, "mpl_colors"):
                            return colormap_obj.mpl_colors
                        return colormap_obj.mpl_colormap

    except ImportError:
        pass  # palettable not available
    except Exception:
        pass  # Error searching palettable

    # Fall back to custom colormaps if not found in palettable
    if name.lower() == "parula":
        return _get_parula_colormap(as_colors)

    # If colormap not found in palettable or custom colormaps, try matplotlib
    try:
        cmap = plt.get_cmap(name)
        if as_colors:
            N = getattr(cmap, "N", 256)
            colors = cmap(np.linspace(0, 1, N))
            return mcolors.ListedColormap(colors)
        return cmap
    except ValueError:
        # Fall back to viridis
        cmap = plt.get_cmap("viridis")
        if as_colors:
            N = getattr(cmap, "N", 256)
            colors = cmap(np.linspace(0, 1, N))
            return mcolors.ListedColormap(colors)
        return cmap


def _get_parula_colormap(as_colors: bool = False):
    """Return the MATLAB parula colormap."""
    import matplotlib.colors as mcolors

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
        import matplotlib.colors as mcolors

        return mcolors.ListedColormap(colors)
    return cmap


def list_available_colormaps() -> dict[str, list[str]]:
    """
    List all available colormaps from palettable and custom colormaps.

    Returns:
        Dictionary with categories as keys and colormap names as values.
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
        pass  # Palettable not available

    return available_colormaps


# ---------------------------------------------------------------------------
# DCM connectivity diagram
# ---------------------------------------------------------------------------


def _node_positions(n: int) -> list[tuple[float, float]]:
    """Place ``n`` nodes for a clean DCM diagram.

    1 → centred. 2 → horizontal pair. 3+ → regular polygon, top-most node
    sitting at 90 degrees so the diagram reads naturally top-to-bottom.
    """
    if n == 1:
        return [(0.0, 0.0)]
    if n == 2:
        return [(-1.0, 0.0), (1.0, 0.0)]
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    return [(float(np.cos(a)), float(np.sin(a))) for a in angles]


def _curved_arrow(
    ax,
    p_start: tuple[float, float],
    p_end: tuple[float, float],
    *,
    rad: float = 0.0,
    color: str = "black",
    lw: float = 1.5,
    mutation_scale: float = 14,
):
    """Draw an arrow from ``p_start`` to ``p_end`` with bezier curvature ``rad``."""
    arrow = FancyArrowPatch(
        p_start,
        p_end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        color=color,
        lw=lw,
        mutation_scale=mutation_scale,
        shrinkA=0,
        shrinkB=0,
        zorder=2,
    )
    ax.add_patch(arrow)
    return arrow


def plot_dcm_graph(
    dcm,
    *,
    show_self_connections: bool = True,
    show_inputs: bool = True,
    threshold: float = 1e-12,
    figsize: Optional[tuple[float, float]] = None,
    node_color: Optional[Union[str, Sequence]] = None,
    node_radius: float = 0.18,
    fontsize: int = 14,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """Render a DCM connectivity diagram with all parameters labelled.

    Nodes are labelled ``x_i`` and connection labels follow the diagram
    convention ``a_ij`` = strength of the connection FROM ``x_i`` TO ``x_j``
    (which is matrix entry ``A[j, i]`` since DCM uses ``dx/dt = A x + C u``
    with rows indexing destination). Self-connections render as outward
    loops labelled ``a_ii``; external inputs render as upward arrows from
    below each node, labelled ``c_i`` with ``u(t)`` underneath.

    Args:
        dcm: any object exposing ``.p.A`` (square matrix) and ``.p.C``
            (1D vector of length n_rois). Works with ``DCM``, ``TwoLayerDCM``,
            ``MultiLayerDCM``, or any duck-typed equivalent.
        show_self_connections: include ``a_ii`` self-loops (default True).
        show_inputs: include the ``c_i / u(t)`` input arrows (default True).
        threshold: connections with absolute weight below this are skipped.
        figsize: passed to ``plt.subplots``. Auto-scaled by ROI count if None.
        node_color: a single colour, a list of per-node colours, or None
            to default to a viridis-style gradient.
        node_radius: node circle radius in layout units.
        fontsize: base font size (math labels are scaled up slightly).
        title: optional figure title.
        ax: bring-your-own axis. If None, a new figure is created.
        save_path: if given, ``fig.savefig`` is called on it. The format is
            inferred from the extension — pass ``.svg`` for vector output
            suitable for slides and thesis documents.

    Returns:
        ``(fig, ax)`` so callers can further customise or save in multiple formats.

    Example:
        >>> from dcsem.models import DCM
        >>> from dcsem.utils import create_A_matrix, create_C_matrix
        >>> A = create_A_matrix(2, 1, ["R0,L0->R1,L0=0.5"], self_connections=-1)
        >>> C = create_C_matrix(2, 1, ["R0,L0=1.0"])
        >>> dcm = DCM(2, params={"A": A, "C": C})
        >>> fig, ax = plot_dcm_graph(dcm, save_path="dcm.svg")
    """
    A = np.asarray(dcm.p.A, dtype=float)
    C = np.asarray(dcm.p.C, dtype=float).flatten()
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got {A.shape}")
    if C.shape != (n,):
        raise ValueError(f"C must be 1D of length {n}, got {C.shape}")

    positions = _node_positions(n)
    layout_centre = np.mean(positions, axis=0) if n > 1 else np.array([0.0, 0.0])

    if ax is None:
        if figsize is None:
            figsize = (max(5.0, 2.5 + 1.2 * n), max(4.0, 2.0 + 1.0 * n))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_aspect("equal")
    ax.axis("off")

    # Per-node colours
    if node_color is None:
        cmap = plt.get_cmap("viridis")
        denom = max(n - 1, 1)
        colors = [cmap(0.2 + 0.6 * (i / denom)) for i in range(n)]
    elif isinstance(node_color, str):
        colors = [node_color] * n
    else:
        colors = list(node_color)
        if len(colors) < n:
            colors = (colors * ((n // len(colors)) + 1))[:n]

    # 1) Off-diagonal arrows: a_ij rendered on i → j
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weight = A[j, i]  # DCM: A[dest, src] = strength of src → dest
            if abs(weight) < threshold:
                continue
            xi, yi = positions[i]
            xj, yj = positions[j]
            dx, dy = xj - xi, yj - yi
            d = float(np.hypot(dx, dy))
            ux, uy = dx / d, dy / d
            start = (xi + node_radius * ux, yi + node_radius * uy)
            end = (xj - node_radius * ux, yj - node_radius * uy)

            # If the reverse connection also exists, curve the two arrows
            # in opposite directions so they don't overlap.
            has_reverse = abs(A[i, j]) >= threshold
            rad = 0.25 if has_reverse else 0.0

            _curved_arrow(ax, start, end, rad=rad)

            # Label at midpoint, offset perpendicular toward the curve
            # (positive rad bows to the LEFT looking from start→end).
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2
            perp = np.array([-uy, ux])  # left-perpendicular
            offset_mag = 0.18 + 0.5 * abs(rad)
            sign = 1.0 if rad >= 0 else -1.0
            lx = mx + sign * offset_mag * perp[0]
            ly = my + sign * offset_mag * perp[1]
            ax.text(
                lx,
                ly,
                rf"$a_{{{i}{j}}}$",
                ha="center",
                va="center",
                fontsize=fontsize + 1,
                zorder=4,
            )

    # 2) Nodes (drawn after arrows so they cover arrow tails neatly)
    for i, (x, y) in enumerate(positions):
        circle = Circle(
            (x, y),
            node_radius,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            rf"$x_{{{i}}}$",
            ha="center",
            va="center",
            fontsize=fontsize + 3,
            zorder=4,
        )

    # 3) Self-loops (a_ii)
    if show_self_connections:
        for i in range(n):
            if abs(A[i, i]) < threshold:
                continue
            x, y = positions[i]
            # Outward direction (from layout centre to node)
            if n == 1:
                outward = np.array([0.0, 1.0])
            else:
                outward = np.array([x - layout_centre[0], y - layout_centre[1]])
                outward = outward / max(np.linalg.norm(outward), 1e-9)

            # Two attachment points on the node boundary, separated by an
            # angular opening so the bezier produces a visible loop.
            base_angle = float(np.arctan2(outward[1], outward[0]))
            half_opening = np.deg2rad(28)
            p_left = (
                x + node_radius * np.cos(base_angle - half_opening),
                y + node_radius * np.sin(base_angle - half_opening),
            )
            p_right = (
                x + node_radius * np.cos(base_angle + half_opening),
                y + node_radius * np.sin(base_angle + half_opening),
            )
            _curved_arrow(
                ax, p_left, p_right, rad=2.5, lw=1.2, mutation_scale=10
            )

            label_pos = (
                x + 2.6 * node_radius * outward[0],
                y + 2.6 * node_radius * outward[1],
            )
            ax.text(
                label_pos[0],
                label_pos[1],
                rf"$a_{{{i}{i}}}$",
                ha="center",
                va="center",
                fontsize=fontsize + 1,
                zorder=4,
            )

    # 4) External inputs (u(t) → c_i → x_i), arrows coming from below
    if show_inputs:
        for i in range(n):
            if abs(C[i]) < threshold:
                continue
            x, y = positions[i]
            tip = (x, y - node_radius - 0.04)
            tail = (x, y - node_radius - 0.45)
            ax.annotate(
                "",
                xy=tip,
                xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
                zorder=2,
            )
            # c_i label sits beside the arrow shaft
            ax.text(
                x + 0.08,
                y - node_radius - 0.25,
                rf"$c_{{{i}}}$",
                ha="left",
                va="center",
                fontsize=fontsize + 1,
            )
            # u(t) label below the arrow tail
            ax.text(
                x,
                y - node_radius - 0.58,
                r"$u(t)$",
                ha="center",
                va="center",
                fontsize=fontsize + 1,
            )

    if title is not None:
        ax.set_title(title)

    # Set view limits with padding
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    pad_x = 0.7 + node_radius
    pad_y_top = 0.9 + node_radius  # room for a_ii labels above top node
    pad_y_bot = 0.95 + node_radius  # room for u(t) below bottom node
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y_bot, max(ys) + pad_y_top)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax
