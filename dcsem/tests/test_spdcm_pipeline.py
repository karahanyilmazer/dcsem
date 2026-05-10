"""Regression tests for the top-level spectral DCM inversion scripts.

Stage 2 audit: catches pipeline-level bugs in ``spdcm_generic.py`` and
``spdcm_mcmc_generic.py`` (TR resolution, covariance calibration flag,
artifact schema). Forward-model correctness is covered by
``test_dcm_spectral_regressions.py``.
"""

import numpy as np
import pytest


def test_resolve_effective_tr_uses_npz_tr_over_cfg(tmp_path):
    """Empirical NPZ with explicit TR must override ``cfg.TR``.

    Previously ``SpectralDCM`` was constructed with ``cfg.TR`` *before* the
    empirical NPZ was loaded, so ``observed_csd`` ran ``scipy.signal.csd``
    at the wrong sampling rate when the file's TR differed.  The helper
    now reads the NPZ's TR up-front so the model sees the right sampling rate.
    """
    from scripts.pipelines.spdcm_generic import RunConfig, _resolve_effective_tr

    npz_path = tmp_path / "bold.npz"
    bold = np.random.default_rng(0).standard_normal((200, 2)).astype(np.float32)
    np.savez(npz_path, bold=bold, TR=np.float64(2.0))

    cfg = RunConfig(data_mode="empirical", bold_path=str(npz_path), TR=1.0)
    tr_eff, bold_loaded = _resolve_effective_tr(cfg)

    assert tr_eff == pytest.approx(2.0)
    assert bold_loaded is not None
    assert bold_loaded.shape == (200, 2)


def test_resolve_effective_tr_falls_back_to_cfg_when_npz_lacks_tr(tmp_path):
    """If the NPZ has no ``TR`` key, ``cfg.TR`` is used."""
    from scripts.pipelines.spdcm_generic import RunConfig, _resolve_effective_tr

    npz_path = tmp_path / "bold_no_tr.npz"
    bold = np.random.default_rng(0).standard_normal((100, 2)).astype(np.float32)
    np.savez(npz_path, bold=bold)

    cfg = RunConfig(data_mode="empirical", bold_path=str(npz_path), TR=0.8)
    tr_eff, bold_loaded = _resolve_effective_tr(cfg)

    assert tr_eff == pytest.approx(0.8)
    assert bold_loaded is not None
    assert bold_loaded.shape == (100, 2)


def test_resolve_effective_tr_passthrough_for_synthetic_modes():
    """Non-empirical modes return ``cfg.TR`` and ``None`` for bold."""
    from scripts.pipelines.spdcm_generic import RunConfig, _resolve_effective_tr

    for mode in ("synthetic_csd", "synthetic_bold"):
        cfg = RunConfig(data_mode=mode, TR=1.5)
        tr_eff, bold_loaded = _resolve_effective_tr(cfg)
        assert tr_eff == pytest.approx(1.5)
        assert bold_loaded is None


def test_resolve_effective_tr_warns_on_mismatch(tmp_path, capsys):
    """Mismatch between ``cfg.TR`` and NPZ TR must surface a warning so the
    silent-override behaviour is impossible."""
    from scripts.pipelines.spdcm_generic import RunConfig, _resolve_effective_tr

    npz_path = tmp_path / "bold_tr_mismatch.npz"
    np.savez(
        npz_path,
        bold=np.zeros((50, 2), dtype=np.float32),
        TR=np.float64(2.0),
    )

    cfg = RunConfig(data_mode="empirical", bold_path=str(npz_path), TR=1.0)
    _resolve_effective_tr(cfg)

    captured = capsys.readouterr().out
    assert "TR" in captured and ("override" in captured.lower() or "mismatch" in captured.lower())


def test_mcmc_runconfig_routes_through_resolve_effective_tr(tmp_path):
    """``spdcm_mcmc_generic.RunConfig`` is duck-typed compatible with
    ``_resolve_effective_tr`` — same data_mode/bold_path/TR fields. Pinned so
    that future field renames cannot silently bypass the TR resolution path
    in the MCMC script.
    """
    from scripts.pipelines.spdcm_generic import _resolve_effective_tr
    from scripts.pipelines.spdcm_mcmc_generic import RunConfig as MCMCRunConfig

    npz_path = tmp_path / "bold_mcmc.npz"
    np.savez(
        npz_path,
        bold=np.zeros((50, 2), dtype=np.float32),
        TR=np.float64(2.5),
    )

    cfg = MCMCRunConfig(data_mode="empirical", bold_path=str(npz_path), TR=1.0)
    tr_eff, bold_loaded = _resolve_effective_tr(cfg)

    assert tr_eff == pytest.approx(2.5)
    assert bold_loaded is not None


def test_resolve_effective_tr_csv_uses_cfg_tr(tmp_path):
    """CSV files carry no TR metadata; must fall back to ``cfg.TR`` and load
    the BOLD without raising."""
    from scripts.pipelines.spdcm_generic import RunConfig, _resolve_effective_tr

    csv_path = tmp_path / "bold.csv"
    bold = np.random.default_rng(0).standard_normal((50, 2))
    import pandas as pd

    pd.DataFrame(
        {"time_s": np.arange(50) * 1.5, "r0": bold[:, 0], "r1": bold[:, 1]}
    ).to_csv(csv_path, index=False)

    cfg = RunConfig(data_mode="empirical", bold_path=str(csv_path), TR=1.5)
    tr_eff, bold_loaded = _resolve_effective_tr(cfg)

    assert tr_eff == pytest.approx(1.5)
    assert bold_loaded is not None
    assert bold_loaded.shape == (50, 2)


def test_mcmc_runconfig_supports_csv_empirical_mode(tmp_path):
    """``spdcm_mcmc_generic`` empirical mode now accepts CSV input via the
    shared ``_resolve_effective_tr`` helper. Pinned so the symmetry with the
    L-BFGS script is not lost."""
    import pandas as pd

    from scripts.pipelines.spdcm_generic import _resolve_effective_tr
    from scripts.pipelines.spdcm_mcmc_generic import RunConfig as MCMCRunConfig

    csv_path = tmp_path / "bold_mcmc.csv"
    bold = np.random.default_rng(0).standard_normal((40, 2))
    pd.DataFrame(
        {"time_s": np.arange(40) * 1.2, "r0": bold[:, 0], "r1": bold[:, 1]}
    ).to_csv(csv_path, index=False)

    cfg = MCMCRunConfig(data_mode="empirical", bold_path=str(csv_path), TR=1.2)
    tr_eff, bold_loaded = _resolve_effective_tr(cfg)

    assert tr_eff == pytest.approx(1.2)
    assert bold_loaded is not None
    assert bold_loaded.shape == (40, 2)


def test_resolve_effective_tr_raises_on_missing_bold_key(tmp_path):
    """An NPZ that lacks the ``"bold"`` key must produce a clear error
    naming the available keys, not an opaque ``KeyError`` from numpy.
    """
    from scripts.pipelines.spdcm_generic import RunConfig, _resolve_effective_tr

    npz_path = tmp_path / "wrong_key.npz"
    np.savez(npz_path, BOLD=np.zeros((50, 2)), TR=np.float64(1.0))

    cfg = RunConfig(data_mode="empirical", bold_path=str(npz_path), TR=1.0)
    with pytest.raises(KeyError, match="bold"):
        _resolve_effective_tr(cfg)


def test_resolve_effective_tr_requires_bold_path_in_empirical_mode():
    """Empirical mode without ``cfg.bold_path`` raises before silently using
    cfg.TR on no-data."""
    from scripts.pipelines.spdcm_generic import RunConfig, _resolve_effective_tr

    cfg = RunConfig(data_mode="empirical", bold_path=None, TR=1.0)
    with pytest.raises(ValueError, match="bold_path"):
        _resolve_effective_tr(cfg)


def test_spdcm_artifact_schema_includes_calibration_flag(tmp_path, monkeypatch):
    """``run_single`` must save ``cov_is_calibrated``, ``hess_is_near_singular``
    and ``tr`` to the NPZ artifact so downstream BENCH-coupling work can
    filter on calibration without re-running the fit.

    Smoke test: synthetic_csd mode, profile sweep disabled. Output is
    redirected to tmp via ``chdir`` because ``get_out_dir`` resolves
    ``results/...`` relative to the cwd.
    """
    import matplotlib

    matplotlib.use("Agg")

    # get_out_dir writes under <cwd>/results/...; chdir to tmp to keep the
    # test self-contained (does not pollute the project's actual results dir).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DCSEM_LATEX_DIR", str(tmp_path / "latex"))
    (tmp_path / "latex").mkdir(parents=True, exist_ok=True)

    from scripts.pipelines.spdcm_generic import RunConfig, run_single

    cfg = RunConfig(
        data_mode="synthetic_csd",
        snr=20.0,
        seed=123,
        profile_params=[],  # skip profile sweep
    )
    run_single(cfg)

    # Locate the artifact via the same path logic as run_single
    from utils import get_out_dir

    img_dir = get_out_dir(
        type="img",
        subfolder="inversion",
        extra_subfolders=["L-BFGS-B", "spdcm_2roi"],
    )
    artifact = np.load(img_dir / "run_results.npz")

    required = {
        "cov_is_calibrated",
        "hess_is_near_singular",
        "tr",
        "theta_est",
        "theta_true",
        "theta_zero",
        "se",
        "ci",
        "cov",
        "hess_cond",
        "converged",
    }
    missing = required - set(artifact.files)
    assert not missing, f"artifact NPZ is missing required keys: {missing}"

    assert artifact["cov_is_calibrated"].dtype == bool
    assert artifact["hess_is_near_singular"].dtype == bool
    assert artifact["tr"].shape == (1,)

    assert bool(artifact["cov_is_calibrated"]), (
        "Clean synthetic_csd run at SNR=20 should yield a calibrated covariance "
        "under adaptive_ridge. cov_is_calibrated=False here would mean the "
        "Hessian inversion needed a non-trivial ridge — regression in the "
        "spdcm_generic.py inversion path."
    )
    # theta_zero must match theta_est dimensionality so downstream tooling
    # can reproduce the run from the artifact alone.
    assert artifact["theta_zero"].shape == artifact["theta_est"].shape
