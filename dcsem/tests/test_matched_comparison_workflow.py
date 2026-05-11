import os
import subprocess
from pathlib import Path

import numpy as np
import scipy.stats as st

from bench.change_model import Trainer
from scripts.workflows.dcm_bench_comparison import (
    CLASS_LABELS,
    ComparisonConfig,
    ComparisonPaths,
    SweepResult,
    generate_change_designs,
    load_sweep_artifact,
    run_bench_effect_size_sweep,
    save_sweep_artifact,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_run_overnight_dry_run_succeeds(tmp_path):
    script = _repo_root() / "scripts" / "run_overnight.sh"
    completed = subprocess.run(
        ["bash", str(script), "--foreground", "--dry-run", "--skip-caffeinate"],
        cwd=_repo_root(),
        env={**os.environ, "LOG_DIR": str(tmp_path / "logs")},
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ALL DONE exit=0" in completed.stdout
    assert "scripts/workflows/05-apply_bench.py" in completed.stdout


def test_sweep_artifact_contains_configured_effect_sizes(tmp_path):
    cfg = ComparisonConfig(effect_size_grid=(0.05, 0.1, 0.2), n_test_samples=5)
    result = SweepResult(
        effect_size=np.array(cfg.effect_size_grid),
        accuracy=np.array([0.2, 0.4, 0.6]),
        confusion_matrices=np.zeros((3, len(CLASS_LABELS), len(CLASS_LABELS))),
        true_change=np.zeros((3, cfg.n_test_samples), dtype=int),
        inferred_change=np.zeros((3, cfg.n_test_samples), dtype=int),
    )

    path = tmp_path / "accuracy_vs_effect_size_bench.npz"
    save_sweep_artifact(path, result, cfg, "bench")
    loaded = np.load(path, allow_pickle=False)

    np.testing.assert_allclose(loaded["effect_size"], cfg.effect_size_grid)
    assert loaded["accuracy"].shape == (3,)
    assert loaded["confusion_matrices"].shape == (3, len(CLASS_LABELS), len(CLASS_LABELS))


def test_bench_and_model_inversion_artifacts_share_effect_grid(tmp_path):
    cfg = ComparisonConfig(effect_size_grid=(0.05, 0.1, 0.2, 0.3, 0.5), n_test_samples=5)
    result = SweepResult(
        effect_size=np.array(cfg.effect_size_grid),
        accuracy=np.zeros(len(cfg.effect_size_grid)),
        confusion_matrices=np.zeros(
            (len(cfg.effect_size_grid), len(CLASS_LABELS), len(CLASS_LABELS))
        ),
        true_change=np.zeros((len(cfg.effect_size_grid), cfg.n_test_samples), dtype=int),
        inferred_change=np.zeros(
            (len(cfg.effect_size_grid), cfg.n_test_samples), dtype=int
        ),
    )

    bench_path = tmp_path / "accuracy_vs_effect_size_bench.npz"
    inversion_path = tmp_path / "accuracy_vs_effect_size_model_inversion.npz"
    save_sweep_artifact(bench_path, result, cfg, "bench")
    save_sweep_artifact(inversion_path, result, cfg, "model_inversion")

    bench = load_sweep_artifact(bench_path)
    inversion = load_sweep_artifact(inversion_path)
    np.testing.assert_allclose(bench.effect_size, inversion.effect_size)


def test_fast_bench_sweep_smoke_uses_current_api(tmp_path):
    def summary_model(a01, a10, c0, c1):
        return np.column_stack([a01, a10, c0, c1])

    priors = {name: st.uniform(0, 1) for name in ("a01", "a10", "c0", "c1")}
    trainer = Trainer(
        forward_model=summary_model,
        priors=priors,
        measurement_names=["a01", "a10", "c0", "c1"],
    )
    bench_model = trainer.train(n_samples=10, parallel=False)

    cfg = ComparisonConfig(
        effect_size_grid=(0.1,),
        n_test_samples=5,
        n_train_samples=10,
        bench_parallel=False,
        reuse_bench_model=False,
    )
    paths = ComparisonPaths(model_dir=tmp_path)
    result = run_bench_effect_size_sweep(
        cfg,
        paths,
        bench_model,
        designs=generate_change_designs(cfg),
        summary_model=summary_model,
    )

    np.testing.assert_allclose(result.effect_size, [0.1])
    assert result.accuracy.shape == (1,)
    assert result.confusion_matrices.shape == (1, len(CLASS_LABELS), len(CLASS_LABELS))
