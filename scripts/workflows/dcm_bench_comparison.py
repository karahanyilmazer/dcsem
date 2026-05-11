"""Matched 2-ROI DCM model-inversion vs BENCH effect-size study.

This module keeps the BENCH and model-inversion workflows importable and
testable.  The numbered scripts in this directory are thin entry points around
these functions.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from bench import change_model
from scipy.optimize import minimize
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix, mean_squared_error

from dcsem import NOISE_CONFIG, PARAM_BOUNDS, get_colormap, set_style
from dcsem.utils import stim_boxcar
from scripts._artifact_metadata import (
    compute_sha256,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)
from utils import get_out_dir, get_summary_measures, get_width_height_latex, simulate_bold

PARAM_NAMES: tuple[str, ...] = ("a01", "a10", "c0", "c1")
CLASS_LABELS: tuple[str, ...] = ("no change", *PARAM_NAMES)
DEFAULT_EFFECT_SIZE_GRID: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True)
class ComparisonConfig:
    """Shared configuration for matched BENCH and inversion sweeps."""

    seed: int = 42
    n_components: int = 4
    noise_mode: str = "no_noise"
    effect_size_grid: tuple[float, ...] = DEFAULT_EFFECT_SIZE_GRID
    n_test_samples: int = 200
    n_train_samples: int = 5000
    # n_repeats was historically declared as 50 but never read; with the
    # wiring planned for PR 3 it would multiply the inversion sweep cost by 50x.
    # Default to 1 (no-op) until that PR explicitly opts in.
    n_repeats: int = 1
    method: str = "PCA"
    ode_method: str = "BDF"
    summary_noise_floor: float = 1e-4
    change_threshold_fraction: float = 0.5
    inversion_maxiter: int = 100
    bench_parallel: bool = True
    inversion_verbose: bool = True
    reuse_bench_model: bool = True
    # BENCH Trainer.train hyperparameters. Surfaced here so they participate
    # in the cache fingerprint (mdl_{setting}.pkl.meta.json) and can be tuned
    # without monkey-patching.
    bench_dv0: float = 1e-6
    bench_mu_poly_degree: int = 2
    bench_sigma_poly_degree: int = 1
    bench_alpha: float = 0.1

    @property
    def setting(self) -> str:
        return f"{self.noise_mode}_{self.n_components}"


@dataclass(frozen=True)
class ComparisonPaths:
    """Output and model-artifact paths for the comparison workflow."""

    model_dir: Path
    img_dir: Path | None = None
    latex_dir: Path | None = None

    @classmethod
    def from_config(cls, cfg: ComparisonConfig) -> "ComparisonPaths":
        latex_dir: Path | None
        try:
            latex_dir = get_out_dir(type="latex", subfolder="figures")
        except ValueError:
            latex_dir = None

        return cls(
            model_dir=get_out_dir(type="model", subfolder=f"bench_{cfg.setting}"),
            img_dir=get_out_dir(type="img", subfolder=f"bench_final_{cfg.setting}"),
            latex_dir=latex_dir,
        )


@dataclass(frozen=True)
class ChangeDesign:
    effect_size: float
    baseline_theta: np.ndarray
    perturbed_theta: np.ndarray
    true_change: np.ndarray


@dataclass(frozen=True)
class SweepResult:
    effect_size: np.ndarray
    accuracy: np.ndarray
    confusion_matrices: np.ndarray
    true_change: np.ndarray
    inferred_change: np.ndarray


def _default_time_and_stimulus() -> tuple[np.ndarray, Callable[[float], float]]:
    return np.arange(100), stim_boxcar([[10, 20, 1]])


def _bounds_dict() -> dict[str, tuple[float, float]]:
    return PARAM_BOUNDS.get_bounds_dict()


def _bounds_list() -> list[tuple[float, float]]:
    return PARAM_BOUNDS.get_bounds_list(list(PARAM_NAMES))


def generate_change_design(
    cfg: ComparisonConfig,
    effect_size: float,
    effect_index: int = 0,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> ChangeDesign:
    """Generate one matched test design for both BENCH and inversion.

    Changes are positive shifts of exactly ``effect_size``.  Baseline values
    for changed parameters are sampled from the admissible subrange so the
    perturbation does not need clipping.
    """

    bounds = _bounds_dict() if bounds is None else bounds
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, effect_index]))
    n_classes = len(CLASS_LABELS)
    repeats = int(np.ceil(cfg.n_test_samples / n_classes))
    true_change = np.tile(np.arange(n_classes, dtype=int), repeats)[: cfg.n_test_samples]
    rng.shuffle(true_change)

    baseline = np.zeros((cfg.n_test_samples, len(PARAM_NAMES)), dtype=float)
    for param_idx, name in enumerate(PARAM_NAMES):
        low, high = bounds[name]
        baseline[:, param_idx] = rng.uniform(low, high, size=cfg.n_test_samples)

    for sample_idx, change_idx in enumerate(true_change):
        if change_idx == 0:
            continue
        param_idx = change_idx - 1
        low, high = bounds[PARAM_NAMES[param_idx]]
        upper = high - effect_size
        if upper < low:
            raise ValueError(
                f"effect_size={effect_size} exceeds admissible range for "
                f"{PARAM_NAMES[param_idx]}: {bounds[PARAM_NAMES[param_idx]]}"
            )
        baseline[sample_idx, param_idx] = rng.uniform(low, upper)

    perturbed = baseline.copy()
    changed = true_change > 0
    perturbed[changed, true_change[changed] - 1] += effect_size

    return ChangeDesign(
        effect_size=float(effect_size),
        baseline_theta=baseline,
        perturbed_theta=perturbed,
        true_change=true_change,
    )


def generate_change_designs(cfg: ComparisonConfig) -> list[ChangeDesign]:
    return [
        generate_change_design(cfg, effect_size, effect_index)
        for effect_index, effect_size in enumerate(cfg.effect_size_grid)
    ]


def theta_matrix_to_params(theta: np.ndarray) -> dict[str, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"theta must have shape (n, {len(PARAM_NAMES)}), got {theta.shape}."
        )
    return {name: theta[:, idx] for idx, name in enumerate(PARAM_NAMES)}


def build_summary_forward_model(
    cfg: ComparisonConfig,
    paths: ComparisonPaths,
) -> Callable[..., np.ndarray]:
    time, u = _default_time_and_stimulus()
    method = cfg.method

    def forward_model(**params: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Trying to unpickle estimator",
                category=UserWarning,
            )
            return get_summary_measures(
                method,
                time,
                u,
                num_rois=2,
                model_dir=paths.model_dir,
                setting=cfg.setting,
                **params,
            )

    forward_model.__name__ = f"dcm_{cfg.setting}_{method.lower()}_summary"
    return forward_model


def _summary_noise_std(cfg: ComparisonConfig, paths: ComparisonPaths) -> float:
    if cfg.noise_mode == "no_noise":
        return cfg.summary_noise_floor

    noise_sigma_path = (
        paths.model_dir / f"noise_sigmas_{cfg.method.lower()}_{cfg.setting}.pkl"
    )
    with open(noise_sigma_path, "rb") as f:
        noise_sigmas = pickle.load(f)
    return float(np.mean(noise_sigmas))


def _pca_artifact_path(cfg: ComparisonConfig, paths: ComparisonPaths) -> Path:
    return paths.model_dir / f"{cfg.method.lower()}_{cfg.setting}.pkl"


def _trainer_config_dict(cfg: ComparisonConfig) -> dict[str, object]:
    """Trainer hyperparams that should invalidate the cache when changed."""
    bounds = _bounds_dict()
    return {
        "n_train_samples": cfg.n_train_samples,
        "dv0": cfg.bench_dv0,
        "mu_poly_degree": cfg.bench_mu_poly_degree,
        "sigma_poly_degree": cfg.bench_sigma_poly_degree,
        "alpha": cfg.bench_alpha,
        "seed": cfg.seed,
        "parallel": cfg.bench_parallel,
        "change_vecs": [{p: 1} for p in PARAM_NAMES],
        "lims": ["twosided"] * len(PARAM_NAMES),
        "param_names": list(PARAM_NAMES),
        "param_bounds": bounds,
        "priors_kind": "uniform_per_param",
        "method": cfg.method,
        "n_components": cfg.n_components,
        "noise_mode": cfg.noise_mode,
        "normaliser": "default_normaliser",
        "forward_model_name": f"dcm_{cfg.setting}_{cfg.method.lower()}_summary",
    }


def _can_reuse_cached_bench(
    cfg: ComparisonConfig,
    paths: ComparisonPaths,
    model_path: Path,
) -> bool:
    """Multi-step staleness check; logs the reason when retraining is needed."""
    if not model_path.exists():
        print(f"[cache] {model_path.name}: artifact missing; retraining")
        return False

    mdl_meta = read_sidecar(model_path)
    if mdl_meta is None:
        print(f"[cache] {model_path.name}: sidecar missing; retraining")
        return False

    pca_path = _pca_artifact_path(cfg, paths)
    if not pca_path.exists():
        print(f"[cache] {pca_path.name}: upstream PCA missing; retraining")
        return False
    current_pca_sha = compute_sha256(pca_path)
    recorded_pca_sha = mdl_meta.get("pca_sha256")
    if recorded_pca_sha != current_pca_sha:
        print(
            f"[cache] {model_path.name}: upstream PCA hash changed "
            f"({(recorded_pca_sha or '<missing>')[:12]}... -> "
            f"{current_pca_sha[:12]}...); retraining"
        )
        return False

    # JSON-roundtrip the desired config so tuples (e.g. param_bounds values)
    # compare equal to the lists that come back from the sidecar JSON.
    import json as _json

    desired = _json.loads(_json.dumps(_trainer_config_dict(cfg), default=str))
    drift = [
        key
        for key, want in desired.items()
        if mdl_meta.get(key) != want
    ]
    if drift:
        print(
            f"[cache] {model_path.name}: trainer config drift in fields "
            f"{drift}; retraining"
        )
        return False

    if mdl_meta.get("git_dirty") and mdl_meta.get("git_sha") != _short_sha():
        print(
            f"[cache] {model_path.name}: artifact was built from a dirty "
            f"worktree at {mdl_meta.get('git_sha')}; retraining"
        )
        return False

    return True


def _short_sha() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        return None


def _write_bench_sidecar(
    model_path: Path,
    cfg: ComparisonConfig,
    paths: ComparisonPaths,
) -> None:
    sidecar_payload: dict[str, object] = {**_trainer_config_dict(cfg)}
    pca_path = _pca_artifact_path(cfg, paths)
    if pca_path.exists():
        sidecar_payload["pca_sha256"] = compute_sha256(pca_path)
    write_sidecar(model_path, sidecar_payload, sha_key="mdl_sha256")


def train_or_load_bench_model(
    cfg: ComparisonConfig,
    paths: ComparisonPaths,
    summary_model: Callable[..., np.ndarray] | None = None,
):
    """Train BENCH on the configured PCA summary, reusing a saved model if asked."""

    model_path = paths.model_dir / f"mdl_{cfg.setting}.pkl"
    if cfg.reuse_bench_model and _can_reuse_cached_bench(cfg, paths, model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    summary_model = summary_model or build_summary_forward_model(cfg, paths)
    bounds = _bounds_dict()
    priors = {
        name: uniform(loc=bounds[name][0], scale=bounds[name][1] - bounds[name][0])
        for name in PARAM_NAMES
    }
    trainer = change_model.Trainer(
        forward_model=summary_model,
        priors=priors,
        measurement_names=[f"PC{i + 1}" for i in range(cfg.n_components)],
    )
    # BENCH samples prior draws via scipy.stats.rvs(), which routes through
    # the global numpy RNG. Seed it so the training set is reproducible and
    # the cache fingerprint actually corresponds to a deterministic fit.
    np.random.seed(cfg.seed)
    model = trainer.train(
        n_samples=cfg.n_train_samples,
        dv0=cfg.bench_dv0,
        mu_poly_degree=cfg.bench_mu_poly_degree,
        sigma_poly_degree=cfg.bench_sigma_poly_degree,
        alpha=cfg.bench_alpha,
        verbose=True,
        parallel=cfg.bench_parallel,
    )
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    _write_bench_sidecar(model_path, cfg, paths)
    return model


def _confusion_and_accuracy(
    true_change: np.ndarray,
    inferred_change: np.ndarray,
) -> tuple[np.ndarray, float]:
    labels = np.arange(len(CLASS_LABELS))
    conf = confusion_matrix(
        true_change,
        inferred_change,
        labels=labels,
        normalize="true",
    )
    accuracy = float(np.mean(inferred_change == true_change))
    return conf, accuracy


def run_bench_effect_size_sweep(
    cfg: ComparisonConfig,
    paths: ComparisonPaths,
    bench_model,
    designs: Iterable[ChangeDesign] | None = None,
    summary_model: Callable[..., np.ndarray] | None = None,
) -> SweepResult:
    """Evaluate a trained BENCH model on the shared effect-size grid."""

    summary_model = summary_model or build_summary_forward_model(cfg, paths)
    noise_std = _summary_noise_std(cfg, paths)
    designs = list(generate_change_designs(cfg) if designs is None else designs)

    accuracies: list[float] = []
    confusions: list[np.ndarray] = []
    true_by_effect: list[np.ndarray] = []
    inferred_by_effect: list[np.ndarray] = []

    for effect_index, design in enumerate(designs):
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 10_000, effect_index]))
        baseline = summary_model(**theta_matrix_to_params(design.baseline_theta))
        perturbed = summary_model(**theta_matrix_to_params(design.perturbed_theta))

        if noise_std > 0:
            baseline = baseline + rng.normal(0.0, noise_std, size=baseline.shape)
            perturbed = perturbed + rng.normal(0.0, noise_std, size=perturbed.shape)

        sigma_n = (noise_std**2) * np.eye(baseline.shape[1])
        sigma_n = np.broadcast_to(sigma_n, (baseline.shape[0], *sigma_n.shape)).copy()
        # NB: a DeprecationWarning ("Conversion of an array with ndim > 0 to
        # a scalar") used to be silenced here. It comes from BENCH's
        # log-likelihood code path and may indicate sigma_n collapse; left
        # visible so we can diagnose during the linearity / wald-test work.
        _, inferred, _, _ = bench_model.infer(
            baseline,
            perturbed - baseline,
            sigma_n,
            parallel=cfg.bench_parallel,
        )

        conf, accuracy = _confusion_and_accuracy(design.true_change, inferred)
        accuracies.append(accuracy)
        confusions.append(conf)
        true_by_effect.append(design.true_change)
        inferred_by_effect.append(np.asarray(inferred, dtype=int))

    return SweepResult(
        effect_size=np.array([d.effect_size for d in designs], dtype=np.float64),
        accuracy=np.array(accuracies, dtype=np.float64),
        confusion_matrices=np.stack(confusions, axis=0),
        true_change=np.stack(true_by_effect, axis=0),
        inferred_change=np.stack(inferred_by_effect, axis=0),
    )


def _simulate_observed_bold(
    theta: np.ndarray,
    cfg: ComparisonConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    time, u = _default_time_and_stimulus()
    params = dict(zip(PARAM_NAMES, theta))
    bold = simulate_bold(
        params,
        time=time,
        u=u,
        num_rois=2,
        ode_method=cfg.ode_method,
    )
    if cfg.noise_mode == "no_noise":
        return bold

    noise_sigma = NOISE_CONFIG.get_noise_std(float(np.std(bold)))
    return bold + rng.normal(0.0, noise_sigma, size=bold.shape)


def invert_bold_observation(y_obs: np.ndarray, cfg: ComparisonConfig) -> np.ndarray:
    time, u = _default_time_and_stimulus()
    bounds = _bounds_list()
    initial_guess = np.array([np.mean(bound) for bound in bounds], dtype=float)
    scale = float(np.std(y_obs)) + 1e-12

    def objective(theta: np.ndarray) -> float:
        params = dict(zip(PARAM_NAMES, theta))
        try:
            y_pred = simulate_bold(
                params,
                time=time,
                u=u,
                num_rois=2,
                ode_method=cfg.ode_method,
            )
        except Exception:
            return 1e12
        if not np.all(np.isfinite(y_pred)):
            return 1e12
        return float(mean_squared_error(y_obs / scale, y_pred / scale))

    # DCM forward models integrate stiff ODEs (BDF); scipy's default FD step
    # (~1.5e-8) falls below the solver's relative tolerance, so the gradient
    # comes back as integration noise. Mirror the fix in inversion_generic.py.
    res = minimize(
        objective,
        initial_guess,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": cfg.inversion_maxiter, "eps": 1e-3},
    )
    return np.asarray(res.x, dtype=float)


def run_model_inversion_effect_size_sweep(
    cfg: ComparisonConfig,
    designs: Iterable[ChangeDesign] | None = None,
) -> SweepResult:
    """Run matched model-inversion confusion sweeps over effect size."""

    designs = list(generate_change_designs(cfg) if designs is None else designs)
    accuracies: list[float] = []
    confusions: list[np.ndarray] = []
    true_by_effect: list[np.ndarray] = []
    inferred_by_effect: list[np.ndarray] = []

    for effect_index, design in enumerate(designs):
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 20_000, effect_index]))
        inferred = np.zeros_like(design.true_change)
        threshold = cfg.change_threshold_fraction * design.effect_size

        iterator = range(len(design.true_change))
        for sample_idx in iterator:
            if cfg.inversion_verbose and sample_idx % max(1, len(design.true_change) // 10) == 0:
                print(
                    f"model inversion effect={design.effect_size:g}: "
                    f"{sample_idx}/{len(design.true_change)}"
                )

            y_baseline = _simulate_observed_bold(
                design.baseline_theta[sample_idx],
                cfg,
                rng,
            )
            y_perturbed = _simulate_observed_bold(
                design.perturbed_theta[sample_idx],
                cfg,
                rng,
            )
            theta_first = invert_bold_observation(y_baseline, cfg)
            theta_second = invert_bold_observation(y_perturbed, cfg)
            diff = np.abs(theta_second - theta_first)
            max_diff = float(np.max(diff))
            inferred[sample_idx] = int(np.argmax(diff) + 1) if max_diff > threshold else 0

        conf, accuracy = _confusion_and_accuracy(design.true_change, inferred)
        accuracies.append(accuracy)
        confusions.append(conf)
        true_by_effect.append(design.true_change)
        inferred_by_effect.append(inferred)

    return SweepResult(
        effect_size=np.array([d.effect_size for d in designs], dtype=np.float64),
        accuracy=np.array(accuracies, dtype=np.float64),
        confusion_matrices=np.stack(confusions, axis=0),
        true_change=np.stack(true_by_effect, axis=0),
        inferred_change=np.stack(inferred_by_effect, axis=0),
    )


def save_sweep_artifact(
    path: Path,
    result: SweepResult,
    cfg: ComparisonConfig,
    method_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        effect_size=result.effect_size,
        accuracy=result.accuracy,
        confusion_matrices=result.confusion_matrices,
        true_change=result.true_change,
        inferred_change=result.inferred_change,
        labels=np.array(CLASS_LABELS),
        param_names=np.array(PARAM_NAMES),
        method=np.array(method_name),
        setting=np.array(cfg.setting),
        seed=np.array(cfg.seed, dtype=np.int64),
        n_test_samples=np.array(cfg.n_test_samples, dtype=np.int64),
        n_repeats=np.array(cfg.n_repeats, dtype=np.int64),
        noise_mode=np.array(cfg.noise_mode),
    )


def load_sweep_artifact(path: Path) -> SweepResult:
    artifact = np.load(path, allow_pickle=False)
    return SweepResult(
        effect_size=artifact["effect_size"],
        accuracy=artifact["accuracy"],
        confusion_matrices=artifact["confusion_matrices"],
        true_change=artifact["true_change"],
        inferred_change=artifact["inferred_change"],
    )


def _canonical_effect_index(result: SweepResult, canonical: float = 0.3) -> int:
    matches = np.where(np.isclose(result.effect_size, canonical))[0]
    if matches.size:
        return int(matches[0])
    return int(np.argmin(np.abs(result.effect_size - canonical)))


def save_confusion_pickle(
    path: Path,
    result: SweepResult,
    canonical_effect: float = 0.3,
) -> None:
    idx = _canonical_effect_index(result, canonical_effect)
    with open(path, "wb") as f:
        pickle.dump(result.confusion_matrices[idx], f)


def plot_accuracy_comparison(
    bench_result: SweepResult,
    inversion_result: SweepResult,
    cfg: ComparisonConfig,
    paths: ComparisonPaths,
) -> None:
    if paths.img_dir is None and paths.latex_dir is None:
        return
    if not np.allclose(bench_result.effect_size, inversion_result.effect_size):
        raise ValueError("BENCH and model-inversion sweeps use different effect grids.")

    set_style()
    width, height = get_width_height_latex()
    fig, ax = plt.subplots(figsize=(width / 1.5, height * 0.75))
    ax.plot(
        bench_result.effect_size,
        bench_result.accuracy,
        marker="o",
        label="BENCH (PCA-4)",
    )
    ax.plot(
        inversion_result.effect_size,
        inversion_result.accuracy,
        marker="s",
        label="model inversion",
    )
    ax.axhline(1 / len(CLASS_LABELS), color="0.5", ls="--", lw=1, label="chance")
    ax.set_xlabel("test effect size")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("2-ROI DCM change detection")
    ax.legend(fontsize="small")
    fig.tight_layout()

    if paths.img_dir is not None:
        paths.img_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.img_dir / f"accuracy_vs_effect_size_comparison_{cfg.setting}.png")
    if paths.latex_dir is not None:
        paths.latex_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.latex_dir / f"accuracy_vs_effect_size_comparison_{cfg.setting}.pdf")
    plt.close(fig)


def plot_confusion_matrix(
    result: SweepResult,
    title: str,
    output_stem: str,
    paths: ComparisonPaths,
    canonical_effect: float = 0.3,
) -> None:
    if paths.img_dir is None and paths.latex_dir is None:
        return
    idx = _canonical_effect_index(result, canonical_effect)
    set_style()
    cmap = get_colormap("YlGnBu")
    width, _ = get_width_height_latex()
    fig, ax = plt.subplots(1, 1, figsize=(width / 2, width / 2))
    import seaborn as sns

    sns.heatmap(
        result.confusion_matrices[idx],
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar=False,
        square=True,
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Inferred Change")
    ax.set_ylabel("Actual Change")
    ax.set_title(title)
    fig.tight_layout()

    if paths.img_dir is not None:
        paths.img_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.img_dir / f"{output_stem}.png")
    if paths.latex_dir is not None:
        paths.latex_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.latex_dir / f"{output_stem}.pdf")
    plt.close(fig)


def run_bench_workflow(cfg: ComparisonConfig, paths: ComparisonPaths) -> SweepResult:
    designs = generate_change_designs(cfg)
    summary_model = build_summary_forward_model(cfg, paths)
    bench_model = train_or_load_bench_model(cfg, paths, summary_model)
    result = run_bench_effect_size_sweep(cfg, paths, bench_model, designs, summary_model)

    artifact_path = paths.model_dir / "accuracy_vs_effect_size_bench.npz"
    save_sweep_artifact(artifact_path, result, cfg, "bench")

    legacy_path = paths.model_dir / f"accuracy_vs_effect_size_{cfg.setting}.npz"
    if legacy_path != artifact_path:
        shutil.copyfile(artifact_path, legacy_path)

    save_confusion_pickle(paths.model_dir / f"conf_bench_{cfg.setting}.pkl", result)
    plot_confusion_matrix(
        result,
        title="BENCH",
        output_stem=f"confusion_matrix_bench_{cfg.setting}",
        paths=paths,
    )
    return result


def run_model_inversion_workflow(
    cfg: ComparisonConfig,
    paths: ComparisonPaths,
) -> SweepResult:
    designs = generate_change_designs(cfg)
    result = run_model_inversion_effect_size_sweep(cfg, designs)
    artifact_path = paths.model_dir / "accuracy_vs_effect_size_model_inversion.npz"
    save_sweep_artifact(artifact_path, result, cfg, "model_inversion")
    save_confusion_pickle(paths.model_dir / "conf_inversion.pkl", result)
    plot_confusion_matrix(
        result,
        title="Model inversion",
        output_stem=f"confusion_matrix_model_inversion_{cfg.setting}",
        paths=paths,
    )
    return result


def maybe_plot_existing_comparison(
    cfg: ComparisonConfig,
    paths: ComparisonPaths,
) -> bool:
    bench_path = paths.model_dir / "accuracy_vs_effect_size_bench.npz"
    inversion_path = paths.model_dir / "accuracy_vs_effect_size_model_inversion.npz"
    if not bench_path.exists() or not inversion_path.exists():
        return False
    bench_result = load_sweep_artifact(bench_path)
    inversion_result = load_sweep_artifact(inversion_path)
    plot_accuracy_comparison(bench_result, inversion_result, cfg, paths)
    return True


def run_matched_workflow(cfg: ComparisonConfig, paths: ComparisonPaths) -> dict[str, SweepResult]:
    bench_result = run_bench_workflow(cfg, paths)
    inversion_result = run_model_inversion_workflow(cfg, paths)
    plot_accuracy_comparison(bench_result, inversion_result, cfg, paths)
    return {"bench": bench_result, "model_inversion": inversion_result}


def _parse_effect_sizes(values: str | None) -> tuple[float, ...]:
    if values is None:
        return DEFAULT_EFFECT_SIZE_GRID
    return tuple(float(value.strip()) for value in values.split(",") if value.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("bench", "inversion", "matched", "plot"),
        nargs="?",
        default="matched",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--noise-mode", choices=("no_noise", "with_noise"), default="no_noise")
    parser.add_argument("--effect-sizes", default=None, help="Comma-separated effect sizes.")
    parser.add_argument("--n-test-samples", type=int, default=200)
    parser.add_argument("--n-train-samples", type=int, default=5000)
    parser.add_argument(
        "--bench-dv0",
        type=float,
        default=1e-6,
        help=(
            "Perturbation magnitude for BENCH Jacobian training. "
            "1e-6 is BENCH's default; raise (e.g. 0.2) if the linearity "
            "diagnostic flags cells with cosine < 0.95 or magnitude ratio "
            "outside [0.8, 1.25]."
        ),
    )
    parser.add_argument("--inversion-maxiter", type=int, default=100)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--serial-bench", action="store_true")
    parser.add_argument("--quiet-inversion", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = ComparisonConfig(
        seed=args.seed,
        n_components=args.n_components,
        noise_mode=args.noise_mode,
        effect_size_grid=_parse_effect_sizes(args.effect_sizes),
        n_test_samples=args.n_test_samples,
        n_train_samples=args.n_train_samples,
        bench_dv0=args.bench_dv0,
        inversion_maxiter=args.inversion_maxiter,
        reuse_bench_model=not args.force_train,
        bench_parallel=not args.serial_bench,
        inversion_verbose=not args.quiet_inversion,
    )
    paths = ComparisonPaths.from_config(cfg)

    if args.command == "bench":
        run_bench_workflow(cfg, paths)
        maybe_plot_existing_comparison(cfg, paths)
    elif args.command == "inversion":
        run_model_inversion_workflow(cfg, paths)
        maybe_plot_existing_comparison(cfg, paths)
    elif args.command == "plot":
        if not maybe_plot_existing_comparison(cfg, paths):
            raise SystemExit("Both BENCH and model-inversion sweep artifacts are required.")
    else:
        run_matched_workflow(cfg, paths)


if __name__ == "__main__":
    main()
