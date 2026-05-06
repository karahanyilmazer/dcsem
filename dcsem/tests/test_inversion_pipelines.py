"""Regression tests for the time-domain inversion scripts.

Stage 2.5 hardening: pins the new NPZ artifact schema and convergence
behaviour in ``inversion_generic.py`` (L-BFGS-B) and ``mcmc_generic.py``
(emcee).  Mirrors the spectral coverage in ``test_spdcm_pipeline.py`` so
stage-3 sweep harnesses can rely on a uniform schema across pipelines.
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# T1 — unit tests for the shared MCMC convergence helper
# ---------------------------------------------------------------------------


def test_is_chain_converged_passes_when_acc_and_ess_in_band():
    """Healthy chain: acceptance in [0.15, 0.80] AND ESS > 50 * n_params."""
    from dcsem.utils import is_chain_converged

    assert is_chain_converged(acc_frac=0.30, eff_total=600, n_params=8) is True


def test_is_chain_converged_rejects_low_acceptance():
    from dcsem.utils import is_chain_converged

    assert is_chain_converged(acc_frac=0.10, eff_total=600, n_params=8) is False


def test_is_chain_converged_rejects_high_acceptance():
    from dcsem.utils import is_chain_converged

    assert is_chain_converged(acc_frac=0.85, eff_total=600, n_params=8) is False


def test_is_chain_converged_rejects_low_ess():
    """ESS = 50 should not pass the strict-greater threshold (50 * 8 = 400)."""
    from dcsem.utils import is_chain_converged

    assert is_chain_converged(acc_frac=0.30, eff_total=50, n_params=8) is False


def test_is_chain_converged_passes_just_above_ess_threshold():
    from dcsem.utils import is_chain_converged

    # n_params=8, ess_factor=50 -> threshold is 400
    assert is_chain_converged(acc_frac=0.30, eff_total=401, n_params=8) is True


def test_is_chain_converged_rejects_nan_ess():
    """``np.isfinite(np.nan) is False`` so the chain is not converged when
    autocorr time could not be estimated."""
    from dcsem.utils import is_chain_converged

    assert is_chain_converged(acc_frac=0.30, eff_total=np.nan, n_params=8) is False


def test_is_chain_converged_honours_keyword_overrides():
    """Custom ``ess_factor`` lowers the threshold so a smaller chain passes."""
    from dcsem.utils import is_chain_converged

    # default: 50 * 8 = 400 → 100 fails
    assert is_chain_converged(acc_frac=0.30, eff_total=100, n_params=8) is False
    # override: 10 * 8 = 80 → 100 passes
    assert (
        is_chain_converged(acc_frac=0.30, eff_total=100, n_params=8, ess_factor=10)
        is True
    )


# ---------------------------------------------------------------------------
# T2 / T3 — runpy-based artifact schema regression tests
#
# These run the real scripts end-to-end with a fast analytical model
# ("quadratic") so the NPZ artifact contains the new fields. Slow but pins
# the public schema.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepare_runpy_env(monkeypatch, tmp_path, active_model="quadratic"):
    """Common monkeypatch setup for the runpy integration tests."""
    import matplotlib

    matplotlib.use("Agg")

    # Even though get_out_dir resolves images relative to the project (not
    # cwd), chdir keeps any cwd-sensitive side effects (matplotlib temp
    # files, etc.) inside tmp_path.
    monkeypatch.chdir(tmp_path)
    latex_dir = tmp_path / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DCSEM_LATEX_DIR", str(latex_dir))
    monkeypatch.setenv("DCSEM_ACTIVE_MODEL", active_model)


def test_inversion_generic_artifact_schema(tmp_path, monkeypatch):
    """``inversion_generic.py`` must save ``cov_is_calibrated``,
    ``hess_is_near_singular``, ``converged`` and the rest of the stage-2.5
    schema to the NPZ artifact.  Uses the fast quadratic model.
    """
    import runpy

    _prepare_runpy_env(monkeypatch, tmp_path, active_model="quadratic")

    runpy.run_path(
        str(PROJECT_ROOT / "inversion_generic.py"),
        run_name="__main__",
    )

    from utils import get_out_dir

    img_dir = get_out_dir(
        type="img",
        subfolder="inversion",
        extra_subfolders=["L-BFGS-B", "quadratic"],
    )
    artifact_path = img_dir / "run_results.npz"
    assert artifact_path.exists(), f"NPZ not written at {artifact_path}"

    with np.load(artifact_path) as artifact:
        required = {
            "y_obs",
            "y_pred",
            "theta_est",
            "theta_true",
            "theta_zero",
            "se",
            "ci",
            "cov",
            "hess_cond",
            "cov_is_calibrated",
            "hess_is_near_singular",
            "converged",
        }
        missing = required - set(artifact.files)
        assert not missing, f"NPZ missing required keys: {missing}"

        assert artifact["cov_is_calibrated"].dtype == bool
        assert artifact["hess_is_near_singular"].dtype == bool
        assert artifact["converged"].dtype == bool
        assert artifact["cov_is_calibrated"].shape == (1,)
        assert artifact["hess_is_near_singular"].shape == (1,)
        assert artifact["converged"].shape == (1,)
        assert artifact["hess_cond"].shape == (1,)

        # theta_est / theta_true / theta_zero share shape (n_params,)
        n_params = artifact["theta_est"].shape[0]
        assert artifact["theta_true"].shape == (n_params,)
        assert artifact["theta_zero"].shape == (n_params,)
        assert artifact["se"].shape == (n_params,)
        assert artifact["ci"].shape == (n_params, 2)
        assert artifact["cov"].shape == (n_params, n_params)


def test_mcmc_generic_artifact_schema(tmp_path, monkeypatch):
    """``mcmc_generic.py`` must save the MCMC artifact schema (mean / median
    / map / ci / acceptance / ess / converged / cov_is_calibrated) so the
    stage-3 sweep harness can read it uniformly.
    """
    import runpy

    _prepare_runpy_env(monkeypatch, tmp_path, active_model="quadratic")
    # Reduce sample counts so the test runs in seconds, not minutes.
    monkeypatch.setenv("DCSEM_N_WALKERS", "20")
    monkeypatch.setenv("DCSEM_N_BURN", "50")
    monkeypatch.setenv("DCSEM_N_SAMPLES_MCMC", "300")

    runpy.run_path(
        str(PROJECT_ROOT / "mcmc_generic.py"),
        run_name="__main__",
    )

    from utils import get_out_dir

    img_dir = get_out_dir(
        type="img",
        subfolder="inversion",
        extra_subfolders=["MCMC", "quadratic"],
    )
    artifact_path = img_dir / "run_results.npz"
    assert artifact_path.exists(), f"NPZ not written at {artifact_path}"

    with np.load(artifact_path) as artifact:
        required = {
            "y_obs",
            "y_pred",
            "theta_mean",
            "theta_median",
            "theta_map",
            "theta_true",
            "theta_zero",
            "se",
            "ci",
            "cov",
            "converged",
            "cov_is_calibrated",
            "acceptance_fraction",
            "ess_total",
        }
        missing = required - set(artifact.files)
        assert not missing, f"NPZ missing required keys: {missing}"

        assert artifact["converged"].dtype == bool
        assert artifact["cov_is_calibrated"].dtype == bool
        assert artifact["converged"].shape == (1,)
        assert artifact["cov_is_calibrated"].shape == (1,)
        assert artifact["acceptance_fraction"].shape == (1,)
        assert artifact["ess_total"].shape == (1,)

        n_params = artifact["theta_mean"].shape[0]
        assert artifact["theta_median"].shape == (n_params,)
        assert artifact["theta_map"].shape == (n_params,)
        assert artifact["theta_true"].shape == (n_params,)
        assert artifact["theta_zero"].shape == (n_params,)
        assert artifact["se"].shape == (n_params,)
        assert artifact["ci"].shape == (n_params, 2)
        assert artifact["cov"].shape == (n_params, n_params)

        # CI is stored as np.stack([q025, q975], axis=1); each lower bound
        # must be ≤ the matching upper bound.
        ci = artifact["ci"]
        assert np.all(ci[:, 0] <= ci[:, 1]), "CI lower bound exceeds upper"

        # Cov diagonal must be finite + non-negative — implies the chain is
        # non-empty and the new ``_in_bounds`` guard is not silently rejecting
        # every walker for the unbounded ``quadratic`` model.
        diag = np.diag(artifact["cov"])
        assert np.all(np.isfinite(diag)), f"cov diagonal not finite: {diag}"
        assert np.all(diag >= 0.0), f"cov diagonal negative: {diag}"
