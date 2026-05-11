# %%
# Off-diagonal covariance analysis across SNR levels.
#
# For every multi-parameter combination (2-param, 3-param, 4-param),
# estimates parameters at each SNR, then extracts and plots the
# off-diagonal covariance elements.  Uses z-score normalisation and
# safe Hessian inversion (matching inversion_generic.py).
from itertools import combinations

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from dcsem import NOISE_CONFIG, PARAM_BOUNDS
from dcsem.numerics import (
    compute_correlation_matrix,
    compute_standard_errors,
    safe_hessian_inversion,
)
from dcsem.plotting import add_underscore, get_param_colors, set_style
from dcsem.utils import stim_boxcar
from utils import add_noise, get_out_dir, simulate_bold

set_style()


# %%
# =============================================================================
# CORE FUNCTIONS  (aligned with inversion_generic.py)
# =============================================================================


def make_objective(model_func, y_obs, param_names):
    """Build a raw MSE objective (no normalisation) matching inversion_generic."""

    def objective(theta):
        params = dict(zip(param_names, theta))
        y_pred = model_func(params)
        return mean_squared_error(y_obs, y_pred)

    return objective


def estimate_with_covariance(
    objective, model_func, y_obs, param_names, initial_values, bounds, n_params
):
    """L-BFGS-B + Hessian-based covariance via proper NLL Hessian.

    MSE is used for optimisation; a separate NLL is built for the Hessian
    so that Cov = H_NLL^{-1} gives correctly scaled uncertainties.
    """
    from scipy.optimize import minimize

    # DCM forward models integrate stiff ODEs (BDF); scipy's default FD step
    # (~1.5e-8) falls below the solver's relative tolerance, so the gradient
    # comes back as integration noise. Mirror the fix in inversion_generic.py.
    res = minimize(
        objective,
        x0=initial_values,
        bounds=bounds,
        method="L-BFGS-B",
        options={"eps": 1e-3},
    )
    theta_est = res.x

    # --- Build canonical NLL (matching inversion_generic.py) -----------------
    params_est = dict(zip(param_names, theta_est))
    y_pred_est = model_func(params_est)
    r_est = (y_obs - y_pred_est).ravel()
    sigma2_est = float(np.dot(r_est, r_est)) / max(r_est.size - n_params, 1)

    def nll_obj(theta):
        params = dict(zip(param_names, theta))
        y_pred = model_func(params)
        r = (y_obs - y_pred).ravel()
        if not np.all(np.isfinite(r)):
            return 1e10
        return 0.5 * float(np.dot(r, r)) / sigma2_est

    # --- Hessian in scaled parameter space -----------------------------------
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
        return nll_obj(from_scaled(s))

    theta_s = to_scaled(theta_est)

    try:
        H_s = nd.Hessian(nll_scaled, step=1e-3)(theta_s)
        H_s = 0.5 * (H_s + H_s.T)
        H = H_s / np.outer(scales, scales)

        cov, _ = safe_hessian_inversion(H, 1.0, regularization=1e-6, method="adaptive_ridge")
        se = compute_standard_errors(cov, warn_negative=False)
    except Exception:
        cov = np.full((n_params, n_params), np.nan)
        se = np.full(n_params, np.nan)

    return theta_est, se, cov


def extract_off_diagonal(cov, param_names):
    """Extract upper-triangle off-diagonal elements with pair labels."""
    n = len(param_names)
    pairs = []
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            label = (
                rf"{add_underscore(param_names[i])} "
                rf"$\leftrightarrow$ "
                rf"{add_underscore(param_names[j])}"
            )
            pairs.append(label)
            values.append(cov[i, j])
    return pairs, values


def run_simulation(
    true_params,
    initial_values,
    params_to_est,
    snr,
    bounds_dict,
    time_vec,
    u,
    num_rois,
):
    """Generate noisy data, estimate, return (se, error, off_diag_pairs, off_diag_values)."""
    bold_true = simulate_bold(true_params, time=time_vec, u=u, num_rois=num_rois)
    bold_noisy, _noise_std = add_noise(bold_true, snr_db=snr)

    est_bounds = [bounds_dict[p] for p in params_to_est]

    def model_func(params):
        return simulate_bold(params, time=time_vec, u=u, num_rois=num_rois)

    obj = make_objective(model_func, bold_noisy, params_to_est)

    theta_est, se, cov = estimate_with_covariance(
        obj, model_func, bold_noisy, params_to_est,
        initial_values, est_bounds, len(params_to_est)
    )

    true_vals = np.array([true_params[p] for p in params_to_est])
    err = true_vals - theta_est

    pairs, off_vals = extract_off_diagonal(cov, params_to_est)

    return se, err, pairs, off_vals


# %%
if __name__ == "__main__":
    # =========================================================================
    # Settings
    # =========================================================================
    time = np.arange(100)
    u = stim_boxcar([[10, 20, 1]])
    num_rois = 2

    param_colors = get_param_colors()
    bounds_dict = PARAM_BOUNDS.get_bounds_dict()

    # All four params estimated
    params_to_est = ["a01", "a10", "c0", "c1"]

    # Ground truth
    true_params = {"a01": 0.4, "a10": 0.4, "c0": 0.5, "c1": 0.5}

    # Generate multi-param combinations (skip single-param — no off-diagonal)
    all_combinations = []
    for r in range(2, len(params_to_est) + 1):
        all_combinations.extend([list(c) for c in combinations(params_to_est, r)])

    random = True
    n_sims = 3 if random else 1
    n_snrs = 20
    snr_range = np.linspace(0.1, 50, n_snrs)

    IMG_DIR = get_out_dir(
        type="img",
        subfolder="wip",
        extra_subfolders=["estimation", f"random-{random}"],
    )

    # =========================================================================
    # Run
    # =========================================================================
    all_pair_labels = []
    all_off_vals = []

    for comb in all_combinations:
        n_p = len(comb)
        stds_list = []
        errs_list = []
        vals_list = []
        pair_labels = None  # same for every SNR

        for snr_db in tqdm(snr_range, desc=f"Estimating {comb}"):
            best_se = None
            best_total_se = np.inf

            for _sim_i in range(n_sims):
                if random:
                    init = np.array(
                        [np.random.uniform(*bounds_dict[p]) for p in comb]
                    )
                else:
                    init = np.array([np.mean(bounds_dict[p]) for p in comb])

                se, err, pairs, off_vals = run_simulation(
                    true_params=true_params,
                    initial_values=init,
                    params_to_est=comb,
                    snr=snr_db,
                    bounds_dict=bounds_dict,
                    time_vec=time,
                    u=u,
                    num_rois=num_rois,
                )

                total_se = np.nansum(se)
                if total_se < best_total_se:
                    best_total_se = total_se
                    best_se = se
                    best_err = err
                    best_off = off_vals
                    pair_labels = pairs

            stds_list.append(best_se)
            errs_list.append(best_err)
            vals_list.append(best_off)

        all_off_vals.append(vals_list)
        all_pair_labels.append(pair_labels)

        # =================================================================
        # Plot: on-diagonal (std and error vs SNR)
        # =================================================================
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
        plt.savefig(IMG_DIR / f"on_diag-{'_'.join(sorted_names)}.png")
        plt.close("all")

    # =====================================================================
    # Plot: off-diagonal covariance vs SNR
    # =====================================================================
    for vals_list, pair_labels, comb in zip(
        all_off_vals, all_pair_labels, all_combinations
    ):
        vals_arr = np.array(vals_list)
        sorted_names = sorted(comb)

        plt.figure(figsize=(6, 4))
        plt.xlabel("Signal-to-Noise Ratio (dB)")
        plt.ylabel("Covariance")
        plt.title("Off-Diagonal Covariance")
        for i, label in enumerate(pair_labels):
            plt.plot(snr_range, vals_arr[:, i], label=label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"off_diag-{'_'.join(sorted_names)}.png")
        plt.close("all")

# %%
