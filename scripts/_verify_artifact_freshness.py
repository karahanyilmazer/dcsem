"""Verify on-disk BENCH workflow artifacts against their sidecar metadata.

Walks ``results/models/bench_{setting}/`` for the requested setting(s) and
checks each artifact's recorded SHA-256 (in its ``*.meta.json`` sidecar)
against a freshly computed hash of the bytes on disk. For BENCH model
artifacts, also verifies that the recorded upstream ``pca_sha256`` matches the
current PCA artifact in the same directory.

Two modes, picked by ``--mode``:

  pre-train (default before BENCH training runs)
    PCA, ICA, noise-sigma artifacts must exist and match their sidecars.
    Missing BENCH model artifact (``mdl_{setting}.pkl``) is a *warning*
    (consumer hasn't been built yet).

  post-train (after BENCH training runs)
    Everything must match. Missing BENCH model or stale sidecar is fatal.

Exit codes:
  0  all clean
  1  fatal mismatch (something downstream would be using a stale artifact)
  2  warnings only (acceptable in pre-train mode)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts._artifact_metadata import (
    compute_sha256,
    read_sidecar,
    sidecar_path,
)


# Match the supported (noise_mode, n_components) set in script 04.
ALL_SETTINGS = ("no_noise_3", "no_noise_4", "with_noise_3", "with_noise_4")


@dataclass
class CheckResult:
    artifact: Path
    ok: bool
    severity: str  # "ok", "warn", "fatal"
    reason: str


def _check_self_hash(artifact: Path, sha_key: str) -> CheckResult:
    if not artifact.exists():
        return CheckResult(artifact, False, "fatal", "artifact missing")
    side = read_sidecar(artifact)
    if side is None:
        return CheckResult(
            artifact, False, "fatal", f"sidecar missing: {sidecar_path(artifact).name}"
        )
    recorded = side.get(sha_key)
    if recorded is None:
        return CheckResult(
            artifact, False, "fatal", f"sidecar has no '{sha_key}' field"
        )
    current = compute_sha256(artifact)
    if recorded != current:
        return CheckResult(
            artifact,
            False,
            "fatal",
            f"{sha_key} mismatch: recorded={recorded[:12]}... vs current={current[:12]}...",
        )
    return CheckResult(artifact, True, "ok", "")


def _check_pca_upstream(mdl_path: Path, pca_path: Path) -> CheckResult:
    side = read_sidecar(mdl_path)
    if side is None:
        return CheckResult(mdl_path, False, "fatal", "BENCH model sidecar missing")
    recorded = side.get("pca_sha256")
    if recorded is None:
        return CheckResult(
            mdl_path, False, "fatal", "BENCH model sidecar has no pca_sha256"
        )
    if not pca_path.exists():
        return CheckResult(
            mdl_path,
            False,
            "fatal",
            f"upstream PCA artifact missing: {pca_path.name}",
        )
    current_pca_sha = compute_sha256(pca_path)
    if recorded != current_pca_sha:
        return CheckResult(
            mdl_path,
            False,
            "fatal",
            f"BENCH model trained against PCA {recorded[:12]}... but current PCA is {current_pca_sha[:12]}...",
        )
    return CheckResult(mdl_path, True, "ok", "")


def _project_root() -> Path:
    # this file lives at scripts/_verify_artifact_freshness.py
    return Path(__file__).resolve().parent.parent


def _bench_dir(setting: str) -> Path:
    return _project_root() / "results" / "models" / f"bench_{setting}"


def check_setting(setting: str, mode: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    base = _bench_dir(setting)
    if not base.exists():
        return [
            CheckResult(base, False, "fatal", f"bench_{setting}/ directory missing")
        ]

    pca = base / f"pca_{setting}.pkl"
    ica = base / f"ica_{setting}.pkl"
    nsp = base / f"noise_sigmas_pca_{setting}.pkl"
    nsi = base / f"noise_sigmas_ica_{setting}.pkl"
    mdl = base / f"mdl_{setting}.pkl"

    results.append(_check_self_hash(pca, "pca_sha256"))
    results.append(_check_self_hash(ica, "ica_sha256"))
    results.append(_check_self_hash(nsp, "noise_sigmas_sha256"))
    results.append(_check_self_hash(nsi, "noise_sigmas_sha256"))

    if not mdl.exists():
        sev = "warn" if mode == "pre-train" else "fatal"
        results.append(
            CheckResult(mdl, sev == "ok", sev, "BENCH model artifact not produced yet")
        )
    else:
        mdl_self = _check_self_hash(mdl, "mdl_sha256")
        if mode == "pre-train" and not mdl_self.ok:
            # Pre-train: BENCH model exists but has no/invalid sidecar.
            # Probably legacy. Cache invalidation in train_or_load_bench_model
            # will retrain it; downgrade to a warning here.
            mdl_self = CheckResult(
                mdl_self.artifact, False, "warn", mdl_self.reason + " (will retrain)"
            )
        results.append(mdl_self)
        if mdl_self.ok:
            results.append(_check_pca_upstream(mdl, pca))

    return results


def _format_result(r: CheckResult) -> str:
    if r.severity == "ok":
        return f"  [OK]    {r.artifact.name}"
    if r.severity == "warn":
        return f"  [WARN]  {r.artifact.name}: {r.reason}"
    return f"  [FATAL] {r.artifact.name}: {r.reason}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--setting",
        help=f"Setting like 'no_noise_4'. Defaults to no_noise_4. Use --all to check {list(ALL_SETTINGS)}.",
        default=None,
    )
    parser.add_argument("--all", action="store_true", help="Check every supported setting.")
    parser.add_argument(
        "--mode",
        choices=("pre-train", "post-train"),
        default="pre-train",
        help="pre-train: missing BENCH model is a warning. post-train: everything must match.",
    )
    args = parser.parse_args(argv)

    if args.all:
        settings = list(ALL_SETTINGS)
    else:
        settings = [args.setting or "no_noise_4"]

    overall_fatal = False
    overall_warn = False
    for setting in settings:
        print(f"\n[freshness] setting={setting} mode={args.mode}")
        results = check_setting(setting, args.mode)
        for r in results:
            print(_format_result(r))
            if r.severity == "fatal":
                overall_fatal = True
            elif r.severity == "warn":
                overall_warn = True

    if overall_fatal:
        print("\n[freshness] FATAL — at least one artifact is stale or missing.")
        return 1
    if overall_warn:
        print("\n[freshness] WARN — pre-train state acceptable (BENCH model not yet built).")
        return 2
    print("\n[freshness] all artifacts verified clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
