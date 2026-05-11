"""Quantify how well BENCH's local Jacobian fit predicts finite-difference deltas.

BENCH trains on perturbations of size ``dv0 = 1e-6`` but is queried at test
effect sizes ``e in [0.05, 0.5]`` - six orders of magnitude larger. The
underlying change model is a polynomial regression of the Jacobian; its
predicted change for effect ``e`` is just ``mu(y_baseline) * e``.

This script measures, per parameter and per effect size, how that linear
extrapolation compares against the true forward-model delta::

    delta_true(theta_b, k, e) = m(theta_b + e * e_k) - m(theta_b)
    delta_pred(theta_b, k, e) = mu_k(y_norm) * e

where ``m`` is the trained summary-measure forward model, ``mu_k`` is the
Jacobian regressor for parameter k (BENCH's MLChangeVector), and ``y_norm``
is the normalised baseline summary.

Metrics aggregated over ``n_baseline`` random baselines per cell:
    cosine        direction agreement (drives BENCH's argmax classification)
    mag_ratio     ||delta_pred|| / ||delta_true||
    rel_err       per-PC componentwise relative error
    mahal         Mahalanobis residual under BENCH's reported sigma_p
                  (calibration check for the uncertainty model)

Decision threshold: any (param, effect_size) cell with median cosine < 0.95
or median magnitude ratio outside [0.8, 1.25] is FLAGGED and printed to
stdout. If any cell is flagged, single-magnitude training at ``dv0 = 1e-6``
is inadequate and PR 2.2 (dv0 tuning) is warranted.

Side experiment (open question A): every metric is computed twice, once with
the standard joint demean of the flattened BOLD (the production code path)
and once with per-ROI demeaning prior to flattening. Both use the SAME PCA
object so the comparison answers "what if inference used per-ROI demean
despite the PCA having been fit on joint-demeaned data".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dcsem import PARAM_BOUNDS, set_style
from dcsem.utils import stim_boxcar
from scripts._artifact_metadata import load_artifact
from scripts.workflows.dcm_bench_comparison import (
    ComparisonConfig,
    ComparisonPaths,
    PARAM_NAMES,
    build_summary_forward_model,
    train_or_load_bench_model,
)
from utils import get_width_height_latex, simulate_bold


DEFAULT_EFFECT_SIZE_GRID: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True)
class LinearityConfig:
    """Configuration knobs that don't already live in ComparisonConfig."""

    n_baseline: int = 200
    effect_size_grid: tuple[float, ...] = DEFAULT_EFFECT_SIZE_GRID
    cosine_threshold: float = 0.95
    mag_ratio_low: float = 0.8
    mag_ratio_high: float = 1.25
    eps: float = 1e-12
    include_per_roi_demean: bool = True


def _sample_baselines(
    rng: np.random.Generator,
    n_baseline: int,
    bounds: dict[str, tuple[float, float]],
    effect_size: float,
) -> np.ndarray:
    """Uniform baseline theta within bounds restricted so baseline + effect_size
    stays admissible.
    """
    baseline = np.zeros((n_baseline, len(PARAM_NAMES)), dtype=float)
    for k, name in enumerate(PARAM_NAMES):
        low, high = bounds[name]
        upper = high - effect_size
        if upper < low:
            raise ValueError(
                f"effect_size={effect_size} exceeds admissible range for {name}"
            )
        baseline[:, k] = rng.uniform(low, upper, size=n_baseline)
    return baseline


def _params_dict(theta_matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {name: theta_matrix[:, k] for k, name in enumerate(PARAM_NAMES)}


def _bold_from_thetas(theta_matrix: np.ndarray, time: np.ndarray, u) -> np.ndarray:
    """Vectorised simulate_bold -> (N, T, R)."""
    return simulate_bold(_params_dict(theta_matrix), time=time, u=u, num_rois=2)


def _summary_joint(bold: np.ndarray, pca) -> np.ndarray:
    """Production path: flatten then demean across the joint axis."""
    flat = bold.reshape(bold.shape[0], -1)
    flat_c = flat - flat.mean(axis=1, keepdims=True)
    return pca.transform(flat_c)


def _summary_per_roi(bold: np.ndarray, pca) -> np.ndarray:
    """Alternative: per-ROI demean along time, THEN flatten."""
    bold_c = bold - bold.mean(axis=1, keepdims=True)
    flat_c = bold_c.reshape(bold.shape[0], -1)
    return pca.transform(flat_c)


def _identify_change_models(bench_model) -> dict[str, int]:
    """Map parameter name to its index in ``bench_model.models``.

    ``models[0]`` is NoChangeModel; per-param MLChangeVectors follow in the
    order of ``Trainer.priors`` keys (PARAM_NAMES). Identified by the model's
    ``name`` attribute, which carries the parameter name.
    """
    name_to_idx: dict[str, int] = {}
    for idx, m in enumerate(bench_model.models):
        name = getattr(m, "name", "")
        for pname in PARAM_NAMES:
            if name == pname or name.startswith(f"{pname}_"):
                name_to_idx[pname] = idx
                break
    missing = set(PARAM_NAMES) - set(name_to_idx)
    if missing:
        raise RuntimeError(
            f"Could not match BENCH change models to params; missing {missing}, "
            f"got {name_to_idx}"
        )
    return name_to_idx


def _bench_predict_delta(
    bench_model,
    model_index: int,
    y_baseline: np.ndarray,
    effect_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-baseline (mu, sigma_p) -> (delta_pred, sigma_pred) at this effect.

    Runs y_baseline through the same normaliser BENCH uses internally
    (default_normaliser is identity, but call it anyway in case it changes).
    """
    y_norm, _, _ = bench_model.normaliser(
        baseline=y_baseline,
        change=None,
        noise_cov=None,
        names=bench_model.measurement_names,
    )
    mdl = bench_model.models[model_index]
    deltas = np.zeros_like(y_norm)
    sigmas = np.zeros((y_norm.shape[0], y_norm.shape[1], y_norm.shape[1]))
    for i, y in enumerate(y_norm):
        mu, sigma_p = mdl.distribution(y)
        deltas[i] = np.squeeze(mu)
        sigmas[i] = np.squeeze(sigma_p)
    return deltas * effect_size, (effect_size**2) * sigmas


def _cell_metrics(
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    sigma_pred: np.ndarray,
    eps: float,
) -> dict[str, np.ndarray]:
    """Per-sample metrics for one (param, effect_size, demean) cell."""
    norm_true = np.linalg.norm(delta_true, axis=1)
    norm_pred = np.linalg.norm(delta_pred, axis=1)
    inner = (delta_true * delta_pred).sum(axis=1)
    cosine = inner / np.maximum(norm_true * norm_pred, eps)
    mag_ratio = norm_pred / np.maximum(norm_true, eps)
    denom = np.maximum(np.abs(delta_true), eps)
    rel_err = (np.abs(delta_true - delta_pred) / denom).mean(axis=1)

    resid = delta_pred - delta_true
    n_dim = sigma_pred.shape[-1]
    mahal = np.zeros(delta_true.shape[0])
    for i in range(delta_true.shape[0]):
        cov = sigma_pred[i] + eps * np.eye(n_dim)
        try:
            x = np.linalg.solve(cov, resid[i])
            mahal[i] = float(resid[i] @ x)
        except np.linalg.LinAlgError:
            mahal[i] = np.nan
    return {
        "cosine": cosine,
        "mag_ratio": mag_ratio,
        "rel_err": rel_err,
        "mahal": mahal,
    }


def _load_pca(paths: ComparisonPaths, cfg: ComparisonConfig):
    pca_path = paths.model_dir / f"{cfg.method.lower()}_{cfg.setting}.pkl"
    return load_artifact(pca_path)


@dataclass
class CellResult:
    """Aggregated metrics for one (param, effect_size, demean_kind) cell."""

    param: str
    effect_size: float
    demean: str  # "joint" or "per_roi"
    samples: dict[str, np.ndarray]  # per-baseline metrics + raw deltas

    @property
    def median_cosine(self) -> float:
        return float(np.median(self.samples["cosine"]))

    @property
    def median_mag_ratio(self) -> float:
        return float(np.median(self.samples["mag_ratio"]))

    @property
    def median_rel_err(self) -> float:
        return float(np.median(self.samples["rel_err"]))

    @property
    def median_mahal(self) -> float:
        return float(np.nanmedian(self.samples["mahal"]))


def run_diagnostic(
    cfg: ComparisonConfig,
    lin_cfg: LinearityConfig,
    paths: ComparisonPaths,
) -> list[CellResult]:
    """Compute predicted-vs-true deltas across the (param, effect_size) grid."""
    set_style()
    print(f"\n=== Linearity diagnostic for setting={cfg.setting} ===")
    print(
        f"n_baseline={lin_cfg.n_baseline}, "
        f"effect_sizes={lin_cfg.effect_size_grid}, "
        f"per_roi_demean={lin_cfg.include_per_roi_demean}"
    )

    summary_model = build_summary_forward_model(cfg, paths)
    bench_model = train_or_load_bench_model(cfg, paths, summary_model)
    pca = _load_pca(paths, cfg)
    pname_to_idx = _identify_change_models(bench_model)
    print(f"BENCH change-model index map: {pname_to_idx}")

    time = np.arange(100)
    u = stim_boxcar([[10, 20, 1]])
    bounds = PARAM_BOUNDS.get_bounds_dict()
    rng = np.random.default_rng(cfg.seed)

    cells: list[CellResult] = []
    demean_kinds: tuple[str, ...] = (
        ("joint", "per_roi") if lin_cfg.include_per_roi_demean else ("joint",)
    )

    for effect_size in lin_cfg.effect_size_grid:
        baseline_theta = _sample_baselines(rng, lin_cfg.n_baseline, bounds, effect_size)
        baseline_bold = _bold_from_thetas(baseline_theta, time, u)
        summaries_baseline = {
            "joint": _summary_joint(baseline_bold, pca),
            "per_roi": _summary_per_roi(baseline_bold, pca),
        }

        for k, pname in enumerate(PARAM_NAMES):
            perturbed_theta = baseline_theta.copy()
            perturbed_theta[:, k] += effect_size
            perturbed_bold = _bold_from_thetas(perturbed_theta, time, u)
            summaries_perturbed = {
                "joint": _summary_joint(perturbed_bold, pca),
                "per_roi": _summary_per_roi(perturbed_bold, pca),
            }

            for demean in demean_kinds:
                y_b = summaries_baseline[demean]
                y_p = summaries_perturbed[demean]
                delta_true = y_p - y_b

                delta_pred, sigma_pred = _bench_predict_delta(
                    bench_model, pname_to_idx[pname], y_b, effect_size
                )

                samples = _cell_metrics(delta_true, delta_pred, sigma_pred, lin_cfg.eps)
                cells.append(
                    CellResult(
                        param=pname,
                        effect_size=float(effect_size),
                        demean=demean,
                        samples={
                            **samples,
                            "delta_true": delta_true,
                            "delta_pred": delta_pred,
                            "sigma_pred": sigma_pred,
                            "baseline_theta": baseline_theta,
                        },
                    )
                )
                print(
                    f"  {pname:>4s} e={effect_size:.2f} demean={demean:<7s}  "
                    f"cos={cells[-1].median_cosine:+.3f}  "
                    f"mag={cells[-1].median_mag_ratio:+.3f}  "
                    f"rel={cells[-1].median_rel_err:.3f}  "
                    f"mahal={cells[-1].median_mahal:.3f}"
                )

    return cells


def _flagged_cells(
    cells: list[CellResult], lin_cfg: LinearityConfig
) -> list[CellResult]:
    out = []
    for c in cells:
        if c.demean != "joint":
            continue  # decision is on the production code path only
        cos_bad = c.median_cosine < lin_cfg.cosine_threshold
        mag_bad = not (
            lin_cfg.mag_ratio_low <= c.median_mag_ratio <= lin_cfg.mag_ratio_high
        )
        if cos_bad or mag_bad:
            out.append(c)
    return out


def save_artifact(cells: list[CellResult], paths: ComparisonPaths) -> Path:
    out_path = paths.model_dir / "linearity_diagnostic.npz"
    npz: dict[str, np.ndarray] = {}
    npz["param"] = np.array([c.param for c in cells])
    npz["effect_size"] = np.array([c.effect_size for c in cells], dtype=float)
    npz["demean"] = np.array([c.demean for c in cells])
    for metric in ("cosine", "mag_ratio", "rel_err", "mahal"):
        npz[metric] = np.stack([c.samples[metric] for c in cells], axis=0)
    np.savez(out_path, **npz)
    print(f"  wrote {out_path}")
    return out_path


def plot_metrics(
    cells: list[CellResult],
    paths: ComparisonPaths,
    lin_cfg: LinearityConfig,
) -> None:
    set_style()
    width, height = get_width_height_latex()
    fig, axes = plt.subplots(
        len(PARAM_NAMES), 4,
        figsize=(width * 1.5, height * 2.4),
        sharex=True,
    )
    metric_specs = [
        ("cosine", "cosine"),
        ("mag_ratio", "||pred|| / ||true||"),
        ("rel_err", "per-PC rel err"),
        ("mahal", "Mahalanobis"),
    ]
    demeans = sorted({c.demean for c in cells})
    colors = {"joint": "C0", "per_roi": "C3"}

    for row, pname in enumerate(PARAM_NAMES):
        for col, (metric_key, metric_label) in enumerate(metric_specs):
            ax = axes[row, col]
            for demean in demeans:
                xs, med, q25, q75 = [], [], [], []
                for c in cells:
                    if c.param != pname or c.demean != demean:
                        continue
                    arr = c.samples[metric_key]
                    if metric_key == "mahal":
                        arr = arr[~np.isnan(arr)]
                    if arr.size == 0:
                        continue
                    xs.append(c.effect_size)
                    med.append(np.median(arr))
                    q25.append(np.quantile(arr, 0.25))
                    q75.append(np.quantile(arr, 0.75))
                xs_a = np.asarray(xs)
                med_a = np.asarray(med)
                q25_a = np.asarray(q25)
                q75_a = np.asarray(q75)
                ax.plot(xs_a, med_a, "o-", color=colors.get(demean, "k"), label=demean)
                ax.fill_between(xs_a, q25_a, q75_a, alpha=0.2, color=colors.get(demean, "k"))

            if metric_key == "cosine":
                ax.axhline(lin_cfg.cosine_threshold, color="0.5", ls=":", lw=1)
                ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1.05)
            if metric_key == "mag_ratio":
                ax.axhline(lin_cfg.mag_ratio_low, color="0.5", ls=":", lw=1)
                ax.axhline(lin_cfg.mag_ratio_high, color="0.5", ls=":", lw=1)
            if col == 0:
                ax.set_ylabel(pname)
            if row == 0:
                ax.set_title(metric_label)
            if row == len(PARAM_NAMES) - 1:
                ax.set_xlabel("effect_size")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if paths.img_dir is not None:
        paths.img_dir.mkdir(parents=True, exist_ok=True)
        path_png = paths.img_dir / "linearity_diagnostic.png"
        fig.savefig(path_png)
        print(f"  wrote {path_png}")
    if paths.latex_dir is not None:
        paths.latex_dir.mkdir(parents=True, exist_ok=True)
        path_pdf = paths.latex_dir / "linearity_diagnostic.pdf"
        fig.savefig(path_pdf)
        print(f"  wrote {path_pdf}")
    plt.close(fig)


def print_decision(cells: list[CellResult], lin_cfg: LinearityConfig) -> int:
    flagged = _flagged_cells(cells, lin_cfg)
    print()
    if not flagged:
        print(
            "[linearity] PASS - all (param, effect_size) cells under joint demean "
            f"meet median cosine >= {lin_cfg.cosine_threshold} and "
            f"mag ratio in [{lin_cfg.mag_ratio_low}, {lin_cfg.mag_ratio_high}]."
        )
        print("[linearity] single-magnitude dv0 training is adequate.")
        return 0

    print(f"[linearity] FLAGGED - {len(flagged)} cells fall outside thresholds:")
    for c in flagged:
        print(
            f"  {c.param} e={c.effect_size:.2f}  "
            f"cosine={c.median_cosine:+.3f}  "
            f"mag_ratio={c.median_mag_ratio:+.3f}"
        )
    print(
        "[linearity] consider PR 2.2 (--bench-dv0 retraining) before relying on "
        "the affected cells."
    )
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--setting", default="no_noise_4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-baseline", type=int, default=200)
    parser.add_argument(
        "--effect-sizes",
        default=None,
        help="Comma-separated effect sizes; default mirrors dcm_bench_comparison.",
    )
    parser.add_argument("--n-train-samples", type=int, default=5000)
    parser.add_argument("--bench-dv0", type=float, default=1e-6)
    parser.add_argument(
        "--skip-per-roi",
        action="store_true",
        help="Skip per-ROI demean side experiment.",
    )
    return parser.parse_args(argv)


def _parse_setting(setting: str) -> tuple[str, int]:
    head, _, tail = setting.rpartition("_")
    if head not in ("no_noise", "with_noise") or not tail.isdigit():
        raise ValueError(
            f"Invalid --setting='{setting}'; expected '<noise_mode>_<n_components>'."
        )
    return head, int(tail)


def _parse_effect_sizes(text: str | None) -> tuple[float, ...]:
    if text is None:
        return DEFAULT_EFFECT_SIZE_GRID
    return tuple(float(v.strip()) for v in text.split(",") if v.strip())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    noise_mode, n_components = _parse_setting(args.setting)
    effect_sizes = _parse_effect_sizes(args.effect_sizes)

    cfg = ComparisonConfig(
        seed=args.seed,
        n_components=n_components,
        noise_mode=noise_mode,
        n_train_samples=args.n_train_samples,
        bench_dv0=args.bench_dv0,
    )
    lin_cfg = LinearityConfig(
        n_baseline=args.n_baseline,
        effect_size_grid=effect_sizes,
        include_per_roi_demean=not args.skip_per_roi,
    )
    paths = ComparisonPaths.from_config(cfg)

    cells = run_diagnostic(cfg, lin_cfg, paths)
    save_artifact(cells, paths)
    plot_metrics(cells, paths, lin_cfg)
    return print_decision(cells, lin_cfg)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
