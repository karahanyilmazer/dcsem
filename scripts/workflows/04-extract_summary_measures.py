"""Fit PCA/ICA summary-measure projections for the BENCH-vs-inversion comparison.

Produces, per ``(noise_mode, n_components)`` setting, the on-disk artifacts that
``scripts/workflows/dcm_bench_comparison.py`` and friends read at run time:

    results/models/bench_{setting}/
        pca_{setting}.pkl                          fitted sklearn PCA
        pca_{setting}.pkl.meta.json                content hash + config
        ica_{setting}.pkl                          fitted sklearn FastICA
        ica_{setting}.pkl.meta.json
        noise_sigmas_pca_{setting}.pkl             per-sample noise std list
        noise_sigmas_pca_{setting}.pkl.meta.json
        noise_sigmas_ica_{setting}.pkl             (same data, ICA-side name)
        noise_sigmas_ica_{setting}.pkl.meta.json

Use ``--all-settings`` to iterate the four combinations
``{no_noise, with_noise} x {3, 4}``; the bare CLI form runs the default single
setting (``no_noise_4``).
"""

from __future__ import annotations

import argparse
import pickle as _pkl
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
from tqdm import tqdm

from dcsem import PARAM_BOUNDS
from dcsem.utils import stim_boxcar
from scripts._artifact_metadata import write_sidecar
from utils import (
    get_out_dir,
    get_width_height_latex,
    initialize_parameters,
    set_style,
    simulate_bold,
)

# Bilinear neural model parameters - fixed for the BENCH 2-ROI workflow.
NUM_LAYERS = 1
NUM_ROIS = 2
TIME = np.arange(100)
STIM_SPEC = [[10, 20, 1]]
PARAMS_TO_SET = ["a01", "a10", "c0", "c1"]

# The four legitimate (noise_mode, n_components) combinations that the
# downstream comparison code currently reads.
ALL_SETTINGS: tuple[tuple[str, int], ...] = (
    ("no_noise", 3),
    ("no_noise", 4),
    ("with_noise", 3),
    ("with_noise", 4),
)


@dataclass(frozen=True)
class ExtractConfig:
    noise_mode: Literal["no_noise", "with_noise"] = "no_noise"
    n_components: int = 4
    n_samples: int = 10000
    seed: int = 42

    @property
    def setting(self) -> str:
        return f"{self.noise_mode}_{self.n_components}"


def _bounds_dict() -> dict[str, tuple[float, float]]:
    return PARAM_BOUNDS.get_bounds_dict()


def _stim_spec_metadata() -> dict[str, object]:
    return {
        "boxcar": STIM_SPEC,
        "time_start": int(TIME[0]),
        "time_stop": int(TIME[-1]) + 1,
        "time_step": int(TIME[1] - TIME[0]) if len(TIME) > 1 else 1,
        "n_timepoints": int(len(TIME)),
    }


def generate_training_bold(cfg: ExtractConfig) -> tuple[np.ndarray, list[float]]:
    """Simulate ``n_samples`` BOLD draws under ``cfg``.

    Returns the centred concatenated BOLD matrix ``(n_samples, 2*T)`` and the
    per-sample noise sigma used (zero throughout when
    ``noise_mode == 'no_noise'``).
    """
    rng = np.random.default_rng(cfg.seed)
    bounds = _bounds_dict()
    u = stim_boxcar(STIM_SPEC)

    bolds_roi0: list[np.ndarray] = []
    bolds_roi1: list[np.ndarray] = []
    noise_sigmas: list[float] = []

    for _ in tqdm(range(cfg.n_samples), desc=f"BOLD sim ({cfg.setting})"):
        initial_values = initialize_parameters(
            bounds, PARAMS_TO_SET, random=True, rng=rng
        )
        bold_true = simulate_bold(
            dict(zip(PARAMS_TO_SET, initial_values)),
            time=TIME,
            u=u,
            num_rois=NUM_ROIS,
        )

        if cfg.noise_mode == "no_noise":
            bold_obsv = bold_true
            noise_sigma = 0.0
        else:
            noise_sigma = float(0.10 * np.std(bold_true))
            bold_obsv = bold_true + rng.normal(
                0.0, noise_sigma, size=bold_true.shape
            )

        noise_sigmas.append(noise_sigma)
        bolds_roi0.append(bold_obsv[:, 0])
        bolds_roi1.append(bold_obsv[:, 1])

    bolds_roi0_arr = np.asarray(bolds_roi0)
    bolds_roi1_arr = np.asarray(bolds_roi1)
    bold_concat = np.concatenate([bolds_roi0_arr, bolds_roi1_arr], axis=1)
    bold_concat_c = bold_concat - np.mean(bold_concat, axis=1, keepdims=True)
    return bold_concat_c, noise_sigmas


def _plot_elbow(
    bold_concat_c: np.ndarray,
    fitter_kind: Literal["PCA", "ICA"],
    elbow: int,
    img_dir: Path,
    latex_dir: Path | None,
    setting: str,
    seed: int,
) -> None:
    """Reconstruction-error sweep over n_components for visual diagnostics."""
    n_vals = np.arange(1, 21)
    errors: list[float] = []
    for n in n_vals:
        if fitter_kind == "ICA":
            model: PCA | FastICA = FastICA(n_components=int(n), random_state=seed)
        else:
            model = PCA(n_components=int(n))
        comps = model.fit_transform(bold_concat_c)
        recon = model.inverse_transform(comps)
        errors.append(float(np.mean((bold_concat_c - recon) ** 2)))

    fig, ax = plt.subplots()
    ax.plot(n_vals, errors)
    ax.axvline(elbow, color="C1", linestyle="--", label="Elbow")
    ax.set_xlabel(f"Number of {fitter_kind} Components")
    ax.set_ylabel("Reconstruction Error")
    ax.legend()
    ax.grid(True)
    img_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_dir / f"{fitter_kind.lower()}_elbow_{setting}.png")
    if latex_dir is not None:
        latex_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(latex_dir / f"{fitter_kind.lower()}_elbow_{setting}.pdf")
    plt.close(fig)


def _plot_components(
    model: PCA | FastICA,
    bold_concat_c: np.ndarray,
    img_dir: Path,
    latex_dir: Path | None,
    setting: str,
    name: Literal["PCA", "ICA"],
) -> None:
    width, height = get_width_height_latex()
    components = model.transform(bold_concat_c)
    fig, axs = plt.subplots(2, 1, figsize=(width, height * 1.2))
    axs[0].plot(model.components_.T)
    axs[0].set_title(f"{name} Components")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend(
        [f"Component {i + 1}" for i in range(model.components_.shape[0])]
    )

    axs[1].plot(components)
    axs[1].set_title("Transformed Data")
    axs[1].set_xlabel("Sample")
    axs[1].set_ylabel(f"{name} Value")
    fig.tight_layout()

    img_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_dir / f"{name.lower()}_components_{setting}.png")
    if latex_dir is not None:
        latex_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(latex_dir / f"{name.lower()}_components_{setting}.pdf")
    plt.close(fig)


def _dump_artifact(
    obj: object,
    artifact_path: Path,
    sidecar_config: dict[str, object],
    sha_key: str,
) -> None:
    """Serialise ``obj`` to ``artifact_path`` and write its sidecar JSON."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "wb") as f:
        _pkl.dump(obj, f)
    write_sidecar(artifact_path, sidecar_config, sha_key=sha_key)


def run_extract(cfg: ExtractConfig) -> None:
    """Fit + save PCA, ICA, and noise-sigma artifacts for one setting."""
    set_style()
    setting = cfg.setting
    print(f"\n=== Extracting summary measures for setting={setting} ===")

    img_dir = get_out_dir(type="img", subfolder=f"bench_{setting}")
    model_dir = get_out_dir(type="model", subfolder=f"bench_{setting}")
    try:
        latex_dir: Path | None = get_out_dir(type="latex", subfolder="figures")
    except ValueError:
        latex_dir = None

    bold_concat_c, noise_sigmas = generate_training_bold(cfg)

    pca = PCA(n_components=cfg.n_components)
    pca.fit(bold_concat_c)
    ica = FastICA(n_components=cfg.n_components, random_state=cfg.seed)
    ica.fit(bold_concat_c)

    _plot_elbow(
        bold_concat_c, "PCA", cfg.n_components, img_dir, latex_dir, setting, cfg.seed
    )
    _plot_elbow(
        bold_concat_c, "ICA", cfg.n_components, img_dir, latex_dir, setting, cfg.seed
    )
    _plot_components(pca, bold_concat_c, img_dir, latex_dir, setting, "PCA")
    _plot_components(ica, bold_concat_c, img_dir, latex_dir, setting, "ICA")

    base_config: dict[str, object] = {
        "n_components": cfg.n_components,
        "n_samples": cfg.n_samples,
        "noise_mode": cfg.noise_mode,
        "seed": cfg.seed,
        "param_bounds": _bounds_dict(),
        "params_to_set": list(PARAMS_TO_SET),
        "stim_spec": _stim_spec_metadata(),
        "num_rois": NUM_ROIS,
        "num_layers": NUM_LAYERS,
    }

    pca_path = model_dir / f"pca_{setting}.pkl"
    _dump_artifact(
        pca, pca_path, {**base_config, "method": "PCA"}, sha_key="pca_sha256"
    )

    ica_path = model_dir / f"ica_{setting}.pkl"
    _dump_artifact(
        ica, ica_path, {**base_config, "method": "ICA"}, sha_key="ica_sha256"
    )

    noise_pca_path = model_dir / f"noise_sigmas_pca_{setting}.pkl"
    _dump_artifact(
        noise_sigmas,
        noise_pca_path,
        {**base_config, "method": "PCA", "kind": "noise_sigmas"},
        sha_key="noise_sigmas_sha256",
    )

    noise_ica_path = model_dir / f"noise_sigmas_ica_{setting}.pkl"
    _dump_artifact(
        noise_sigmas,
        noise_ica_path,
        {**base_config, "method": "ICA", "kind": "noise_sigmas"},
        sha_key="noise_sigmas_sha256",
    )

    print(f"  wrote {pca_path.name} (+ sidecar)")
    print(f"  wrote {ica_path.name} (+ sidecar)")
    print(f"  wrote {noise_pca_path.name} (+ sidecar)")
    print(f"  wrote {noise_ica_path.name} (+ sidecar)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else ""
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all-settings",
        action="store_true",
        help="Iterate {no_noise, with_noise} x {3, 4}",
    )
    parser.add_argument(
        "--noise-mode", choices=("no_noise", "with_noise"), default="no_noise"
    )
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings: Iterable[tuple[str, int]]
    if args.all_settings:
        settings = ALL_SETTINGS
    else:
        settings = ((args.noise_mode, args.n_components),)
    for noise_mode, n_components in settings:
        run_extract(
            ExtractConfig(
                noise_mode=noise_mode,  # type: ignore[arg-type]
                n_components=n_components,
                n_samples=args.n_samples,
                seed=args.seed,
            )
        )


if __name__ == "__main__":
    main(sys.argv[1:])
