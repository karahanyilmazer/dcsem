# %%
# Parameter estimation via L-BFGS-B with Hessian-based uncertainty quantification.
#
# Sweeps over SNR levels and parameter combinations to measure how estimation
# quality degrades with noise.  Uses z-score normalisation and safe Hessian
# inversion (matching inversion_generic.py).
from itertools import combinations

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from dcsem import NOISE_CONFIG, PARAM_BOUNDS
from dcsem.numerics import (
    compute_confidence_intervals,
    compute_standard_errors,
    safe_hessian_inversion,
)
from dcsem.plotting import add_underscore, get_param_colors, set_style
from dcsem.utils import stim_boxcar
from utils import add_noise, get_out_dir, simulate_bold

set_style()
IMG_DIR = get_out_dir(type="img", subfolder="wip", extra_subfolders="estimation")


# %%
# =============================================================================
# CORE FUNCTIONS  (aligned with inversion_generic.py)
# =============================================================================


def make_objective(model_func, y_obs, param_names, remaining_params, normalize=True):
    """Build a scalar objective that matches inversion_generic's normalised MSE."""
    if normalize:
        y_mean = y_obs.mean(axis=0, keepdims=True)
        y_std = y_obs.std(axis=0, keepdims=True) + 1e-12
        y_obs_norm = (y_obs - y_mean) / y_std

        def objective(theta):
            params = dict(zip(param_names, theta))
            params.update(remaining_params)
            y_pred = model_func(params)
            y_pred_norm = (y_pred - y_mean) / y_std
            return mean_squared_error(y_obs_norm, y_pred_norm)

        return objective, y_mean, y_std
    else:

        def objective(theta):
            params = dict(zip(param_names, theta))
            params.update(remaining_params)
            y_pred = model_func(params)
            return mean_squared_error(y_obs, y_pred)

        return objective, None, None


def estimate_parameters(
    objective,
    initial_values,
    bounds,
    n_params,
):
    """Run L-BFGS-B then compute Hessian-based SE via safe_hessian_inversion."""
    from scipy.optimize import minimize

    res = minimize(
        objective,
        x0=initial_values,
        bounds=bounds,
        method="L-BFGS-B",
    )
    theta_est = res.x

    # --- Hessian in scaled parameter space (matching inversion_generic) ------
    if bounds is not None:
        lowers = np.array([b[0] for b in bounds])
        scales = np.array([b[1] - b[0] for b in bounds])
    else:
        lowers = np.zeros(n_params)
        scales = np.ones(n_params)

    def to_scaled(theta):
        return (theta - lowers) / scales

    def from_scaled(s):
        return s * scales + lowers

    def nll_scaled(s):
        return objective(from_scaled(s))

    theta_s = to_scaled(theta_est)

    try:
        H_s = nd.Hessian(nll_scaled, step=1e-3)(theta_s)
        H_s = 0.5 * (H_s + H_s.T)  # symmetrise
        H = H_s / np.outer(scales, scales)  # unscale

        cov, _ = safe_hessian_inversion(H, 1.0, regularization=1e-6, method="pinvh")
        se = compute_standard_errors(cov, warn_negative=False)
        ci = compute_confidence_intervals(theta_est, se, alpha=0.05)
    except Exception:
        se = np.full(n_params, np.nan)
        ci = np.full((n_params, 2), np.nan)
        cov = np.full((n_params, n_params), np.nan)

    return theta_est, se, ci, cov


def run_simulation(
    true_params,
    initial_values,
    params_to_est,
    snr,
    all_bounds,
    time_vec,
    u,
    num_rois,
    normalize=True,
):
    """Generate noisy data, estimate params, return (se, error)."""
    # Simulate ground truth
    bold_true = simulate_bold(true_params, time=time_vec, u=u, num_rois=num_rois)

    # Add noise (correctly unpack the tuple)
    bold_noisy, _noise_std = add_noise(bold_true, snr_db=snr)

    # Separate free vs fixed params
    remaining_params = {k: v for k, v in true_params.items() if k not in params_to_est}
    est_bounds = [all_bounds[p] for p in params_to_est]

    def model_func(params):
        return simulate_bold(params, time=time_vec, u=u, num_rois=num_rois)

    obj, _, _ = make_objective(
        model_func, bold_noisy, params_to_est, remaining_params, normalize=normalize
    )

    theta_est, se, ci, cov = estimate_parameters(
        obj, initial_values, est_bounds, len(params_to_est)
    )

    # Estimation error
    true_vals = np.array([true_params[p] for p in params_to_est])
    err = true_vals - theta_est

    return se, err


# %%
if __name__ == "__main__":
    # =========================================================================
    # Simulation settings
    # =========================================================================
    time = np.arange(100)
    u = stim_boxcar([[10, 20, 1]])

    NUM_ROIS = 2
    RANDOM = False

    param_colors = get_param_colors()

    # Use central config for bounds
    param_names_all = ["a01", "a10", "c0", "c1"]
    bounds_dict = PARAM_BOUNDS.get_bounds_dict()

    # Ground truth
    true_params = {"a01": 0.6, "a10": 0.4, "c0": 0.5, "c1": 0.5}

    # Which params to estimate (set to a subset to see single-param behaviour)
    params_to_est = ["a01"]

    # All combinations of those params
    all_combinations = []
    for r in range(1, len(params_to_est) + 1):
        all_combinations.extend([list(c) for c in combinations(params_to_est, r)])

    # SNR sweep
    n_sims = 3 if RANDOM else 1
    n_snrs = 20
    snr_range = np.linspace(0.1, 50, n_snrs)

    # =========================================================================
    # Run estimation over SNR sweep
    # =========================================================================
    for comb in all_combinations:
        n_p = len(comb)
        stds_list = []
        errs_list = []

        for snr_db in tqdm(snr_range, desc=f"Estimating {comb}"):
            best_se = None
            best_total_se = np.inf

            for _sim_i in range(n_sims):
                if RANDOM:
                    init = np.array([np.random.uniform(*bounds_dict[p]) for p in comb])
                else:
                    init = np.array([np.mean(bounds_dict[p]) for p in comb])

                se, err = run_simulation(
                    true_params=true_params,
                    initial_values=init,
                    params_to_est=comb,
                    snr=snr_db,
                    all_bounds=bounds_dict,
                    time_vec=time,
                    u=u,
                    num_rois=NUM_ROIS,
                    normalize=True,
                )

                total_se = np.nansum(se)
                if total_se < best_total_se:
                    best_total_se = total_se
                    best_se = se
                    best_err = err

            stds_list.append(best_se)
            errs_list.append(best_err)

        # =====================================================================
        # Plot: std and error vs SNR
        # =====================================================================
        stds_arr = np.array(stds_list)
        errs_arr = np.array(errs_list)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].axhline(0, color="k", ls="--")
        axs[1].axhline(0, color="k", ls="--")

        for i, param in enumerate(comb):
            axs[0].plot(
                snr_range,
                stds_arr[:, i],
                "-x",
                color=param_colors[param],
                label=add_underscore(param),
            )
            axs[1].plot(
                snr_range,
                errs_arr[:, i],
                "-x",
                color=param_colors[param],
                label=add_underscore(param),
            )

        axs[0].set_xlabel("Signal-to-Noise Ratio (dB)")
        axs[0].set_ylabel("Standard Deviation")
        axs[0].legend()

        axs[1].set_xlabel("Signal-to-Noise Ratio (dB)")
        axs[1].set_ylabel("Estimation Error")
        axs[1].legend()

        sorted_names = sorted(comb)
        fig.suptitle(
            f"Parameter Estimation Results "
            f"({', '.join([add_underscore(n) for n in sorted_names])})"
        )
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"{'_'.join(sorted_names)}_estimation.png")
        # plt.close("all")
        plt.show()

# %%
