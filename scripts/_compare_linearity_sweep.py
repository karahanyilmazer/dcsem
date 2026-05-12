"""Multi-NPZ linearity comparator: dv0 sweep summary.

Takes a list of ``linearity_diagnostic_dv*.npz`` files (one per dv0 value)
and prints three tables:

1. Cosine medians per (param, effect_size) cell.
2. Magnitude-ratio medians per cell.
3. PASS / FLAG per cell, plus a headline count of cells passing both
   thresholds (cos >= 0.95 AND mag ratio in [0.8, 1.25]).

The dv0 value is parsed back from each filename; an unparseable name falls
back to the file stem.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


COSINE_THRESHOLD = 0.95
MAG_RATIO_LOW = 0.8
MAG_RATIO_HIGH = 1.25


def parse_dv0_from_path(p: Path) -> str:
    m = re.search(r"linearity_diagnostic_dv(?P<dv>.+?)\.npz", p.name)
    if not m:
        return p.stem
    return m.group("dv")


def _sort_key(dv: str) -> tuple[float, str]:
    try:
        return (float(dv), dv)
    except ValueError:
        return (float("inf"), dv)


def summarise(npz_path: Path) -> dict[tuple[str, float], dict[str, float]]:
    d = np.load(npz_path)
    out: dict[tuple[str, float], dict[str, float]] = {}
    for i in range(len(d["param"])):
        if str(d["demean"][i]) != "joint":
            continue
        key = (str(d["param"][i]), float(d["effect_size"][i]))
        out[key] = {
            "cos": float(np.median(d["cosine"][i])),
            "mag": float(np.median(d["mag_ratio"][i])),
        }
    return out


def cell_passes(cos: float, mag: float) -> bool:
    return (
        cos >= COSINE_THRESHOLD
        and MAG_RATIO_LOW <= mag <= MAG_RATIO_HIGH
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "npzs",
        nargs="+",
        help="linearity_diagnostic_dv*.npz files; one per dv0 value.",
    )
    args = parser.parse_args(argv)

    pairs = [(parse_dv0_from_path(Path(p)), Path(p)) for p in args.npzs]
    pairs.sort(key=lambda x: _sort_key(x[0]))
    dv0s = [dv for dv, _ in pairs]
    summaries = [summarise(p) for _, p in pairs]

    # Union of keys across all NPZs (some sweeps may use different effect grids).
    keys = sorted(
        set().union(*(s.keys() for s in summaries)),
        key=lambda k: (k[0], k[1]),
    )

    header = f"{'param':>5s} {'e':>5s} | " + " ".join(
        f"{('dv0=' + dv):>11s}" for dv in dv0s
    )

    print(
        f"\nCosine medians (joint demean), '*' marks cos >= {COSINE_THRESHOLD}"
    )
    print(header)
    print("-" * len(header))
    for k in keys:
        line = f"{k[0]:>5s} {k[1]:.2f} |"
        for s in summaries:
            c = s.get(k)
            if c is None:
                line += f" {'-':>11s}"
            else:
                star = "*" if c["cos"] >= COSINE_THRESHOLD else " "
                line += f" {c['cos']:+9.3f}{star} "
        print(line)

    print(
        f"\nMag-ratio medians (joint demean), '*' marks "
        f"mag in [{MAG_RATIO_LOW}, {MAG_RATIO_HIGH}]"
    )
    print(header)
    print("-" * len(header))
    for k in keys:
        line = f"{k[0]:>5s} {k[1]:.2f} |"
        for s in summaries:
            c = s.get(k)
            if c is None:
                line += f" {'-':>11s}"
            else:
                star = (
                    "*"
                    if MAG_RATIO_LOW <= c["mag"] <= MAG_RATIO_HIGH
                    else " "
                )
                line += f" {c['mag']:+9.3f}{star} "
        print(line)

    print(
        f"\nPer-cell decision (both cos>={COSINE_THRESHOLD} AND "
        f"mag in [{MAG_RATIO_LOW}, {MAG_RATIO_HIGH}]):"
    )
    print(header)
    print("-" * len(header))
    for k in keys:
        line = f"{k[0]:>5s} {k[1]:.2f} |"
        for s in summaries:
            c = s.get(k)
            if c is None:
                line += f" {'-':>11s}"
            else:
                mark = "OK" if cell_passes(c["cos"], c["mag"]) else "FLAG"
                line += f" {mark:>11s}"
        print(line)

    print()
    print(f"Headline pass-count per dv0 (out of {len(keys)} cells):")
    best_dv0 = None
    best_pass = -1
    for dv, s in zip(dv0s, summaries):
        n_pass = sum(
            1 for k in keys if k in s and cell_passes(s[k]["cos"], s[k]["mag"])
        )
        marker = ""
        if n_pass > best_pass:
            best_pass = n_pass
            best_dv0 = dv
        print(f"  dv0={dv:<10s}  {n_pass:>2d}/{len(keys)}")
    if best_dv0 is not None:
        print(f"\nBest dv0 in this sweep: {best_dv0}  ({best_pass}/{len(keys)} cells pass)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
