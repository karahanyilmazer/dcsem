"""Sanity-check that BENCH's ``infer`` actually honors the sigma_n argument.

Open question E from the plan: ``summary_noise_floor = 1e-4`` may be
effectively zero in PCA-transformed space, and the silenced DeprecationWarning
"Conversion of an array with ndim > 0 to a scalar" hints that sigma_n could
be collapsed somewhere inside ``bench/change_model.py``. Before we tune the
floor value, we need to confirm that varying sigma_n_const actually changes
posteriors / inferred classes.

Usage:
    python scripts/_inspect_bench_sigma.py [--setting no_noise_4]

Output: a table to stdout, one row per sigma_n_const, plus a heatmap saved
to results/images/bench_final_{setting}/sigma_n_sweep.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dcsem import PARAM_BOUNDS, set_style
from dcsem.utils import stim_boxcar
from scripts.workflows.dcm_bench_comparison import (
    ComparisonConfig,
    ComparisonPaths,
    PARAM_NAMES,
    build_summary_forward_model,
    train_or_load_bench_model,
)
from utils import simulate_bold


def _parse_setting(setting: str) -> tuple[str, int]:
    head, _, tail = setting.rpartition("_")
    if head not in ("no_noise", "with_noise") or not tail.isdigit():
        raise ValueError(f"Invalid setting='{setting}'")
    return head, int(tail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--setting", default="no_noise_4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-samples", type=int, default=5000)
    parser.add_argument("--effect-size", type=float, default=0.3)
    parser.add_argument(
        "--sigma-grid",
        default="1e-8,1e-6,1e-4,1e-2,1e-1",
        help="Comma-separated isotropic sigma_n_const values to test.",
    )
    args = parser.parse_args(argv)

    set_style()
    noise_mode, n_components = _parse_setting(args.setting)
    cfg = ComparisonConfig(
        seed=args.seed,
        n_components=n_components,
        noise_mode=noise_mode,
        n_train_samples=args.n_train_samples,
    )
    paths = ComparisonPaths.from_config(cfg)

    summary_model = build_summary_forward_model(cfg, paths)
    bench_model = train_or_load_bench_model(cfg, paths, summary_model)

    # Fixed baseline + perturbed pair: a01 shifted by effect_size, others at 0.4.
    e = float(args.effect_size)
    bounds = PARAM_BOUNDS.get_bounds_dict()
    base = np.array([0.4, 0.4, 0.4, 0.4])[None, :]
    pert = base.copy()
    pert[0, 0] += e  # change a01
    if pert[0, 0] > bounds["a01"][1]:
        raise ValueError(f"effect_size={e} pushes a01 out of bounds")

    base_summary = summary_model(**{n: base[:, k] for k, n in enumerate(PARAM_NAMES)})
    pert_summary = summary_model(**{n: pert[:, k] for k, n in enumerate(PARAM_NAMES)})
    delta = pert_summary - base_summary
    d = base_summary.shape[1]

    sigmas = [float(v.strip()) for v in args.sigma_grid.split(",") if v.strip()]
    print(f"\nsigma_n sweep on setting={cfg.setting}, a01 shift={e}")
    print(f"baseline summary shape={base_summary.shape}, delta L2={np.linalg.norm(delta):.4e}")
    print(f"{'sigma_n':>12s}  {'inferred':>12s}  {'posterior':>50s}")
    posteriors = np.zeros((len(sigmas), 1 + len(PARAM_NAMES)))
    inferred_classes = []
    for i, s in enumerate(sigmas):
        sigma_n = (s**2) * np.eye(d)
        sigma_n = np.broadcast_to(sigma_n, (1, d, d)).copy()
        post, inferred, _, _ = bench_model.infer(
            base_summary, delta, sigma_n, parallel=False
        )
        post_row = np.asarray(post).reshape(-1)
        posteriors[i] = post_row[: posteriors.shape[1]]
        inferred_classes.append(int(np.asarray(inferred).reshape(-1)[0]))
        post_str = ", ".join(f"{p:.3f}" for p in post_row[: posteriors.shape[1]])
        print(f"  {s:.0e}  {inferred_classes[-1]:>12d}  [{post_str}]")

    spread = float(posteriors.max(axis=0) - posteriors.min(axis=0)).real if False else None
    spread = float(np.max(posteriors.max(axis=0) - posteriors.min(axis=0)))
    print(f"\nposterior range across the sweep: max - min = {spread:.4e}")
    if spread < 1e-3:
        print(
            "[sigma_n] FLAGGED - posteriors essentially constant across "
            f"sigma_n in [{sigmas[0]:.0e}, {sigmas[-1]:.0e}]; "
            "sigma_n may be being collapsed inside bench/change_model.py "
            "(see lines 515/519 of MLChangeVector.log_lh)."
        )
        rc = 1
    else:
        print(
            "[sigma_n] OK - posteriors vary with sigma_n; the noise covariance "
            "is being honoured."
        )
        rc = 0

    # Plot posteriors vs sigma_n
    fig, ax = plt.subplots()
    labels = ["nochange", *PARAM_NAMES]
    for j, label in enumerate(labels):
        ax.plot(sigmas, posteriors[:, j], "o-", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("sigma_n_const")
    ax.set_ylabel("posterior probability")
    ax.set_title(f"BENCH posterior vs noise floor (a01 shift={e})")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    if paths.img_dir is not None:
        paths.img_dir.mkdir(parents=True, exist_ok=True)
        out = paths.img_dir / "sigma_n_sweep.png"
        fig.savefig(out)
        print(f"  wrote {out}")
    plt.close(fig)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
