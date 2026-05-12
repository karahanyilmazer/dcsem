"""Side-by-side comparison of two linearity_diagnostic.npz files.

Used after a retrain at a different ``n_train_samples`` (or ``dv0``) to see
whether the BENCH-vs-true delta agreement actually moved. Prints a median
table per (param, effect_size, demean) and flags cells whose median cosine
or magnitude ratio crossed the decision thresholds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

COSINE_THRESHOLD = 0.95
MAG_RATIO_LOW = 0.8
MAG_RATIO_HIGH = 1.25


def _summarise(npz_path: Path) -> dict[tuple[str, float, str], dict[str, float]]:
    d = np.load(npz_path)
    params = d["param"]
    effects = d["effect_size"]
    demeans = d["demean"]
    cos = d["cosine"]
    mag = d["mag_ratio"]
    rel = d["rel_err"]
    mahal = d["mahal"]

    out: dict[tuple[str, float, str], dict[str, float]] = {}
    for i in range(len(params)):
        key = (str(params[i]), float(effects[i]), str(demeans[i]))
        m = np.isfinite(mahal[i])
        out[key] = {
            "cosine": float(np.median(cos[i])),
            "mag_ratio": float(np.median(mag[i])),
            "rel_err": float(np.median(rel[i])),
            "mahal": float(np.median(mahal[i][m])) if m.any() else float("nan"),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", help="Path to baseline NPZ (e.g. smoke run).")
    parser.add_argument("candidate", help="Path to candidate NPZ (e.g. fresh retrain).")
    parser.add_argument(
        "--demean",
        default="joint",
        choices=("joint", "per_roi"),
        help="Which demean variant to compare (default: joint).",
    )
    args = parser.parse_args(argv)

    base = _summarise(Path(args.baseline))
    cand = _summarise(Path(args.candidate))

    keys = sorted(
        (k for k in base if k[2] == args.demean),
        key=lambda k: (k[0], k[1]),
    )

    print(
        f"\n{'param':>5s} {'e':>5s} | "
        f"{'cos_base':>9s} {'cos_cand':>9s} {'d_cos':>8s} | "
        f"{'mag_base':>9s} {'mag_cand':>9s} {'d_mag':>8s} | "
        f"baseline candidate"
    )
    print("-" * 102)
    flipped = []
    for k in keys:
        b = base[k]
        c = cand[k]
        if k not in cand:
            print(f"{k[0]:>5s} {k[1]:.2f}  (missing in candidate)")
            continue
        d_cos = c["cosine"] - b["cosine"]
        d_mag = c["mag_ratio"] - b["mag_ratio"]
        base_ok = (
            b["cosine"] >= COSINE_THRESHOLD
            and MAG_RATIO_LOW <= b["mag_ratio"] <= MAG_RATIO_HIGH
        )
        cand_ok = (
            c["cosine"] >= COSINE_THRESHOLD
            and MAG_RATIO_LOW <= c["mag_ratio"] <= MAG_RATIO_HIGH
        )
        mark = ""
        if base_ok != cand_ok:
            mark = " <- crossed threshold"
            flipped.append((k, base_ok, cand_ok))
        print(
            f"{k[0]:>5s} {k[1]:.2f} | "
            f"{b['cosine']:+9.3f} {c['cosine']:+9.3f} {d_cos:+8.3f} | "
            f"{b['mag_ratio']:+9.3f} {c['mag_ratio']:+9.3f} {d_mag:+8.3f} | "
            f"{'OK ' if base_ok else 'FLAG'} -> {'OK ' if cand_ok else 'FLAG'}{mark}"
        )

    print()
    if not flipped:
        print(
            "[compare] No cells crossed the (cosine>=0.95, mag in [0.8,1.25]) "
            "thresholds between baseline and candidate."
        )
    else:
        print(f"[compare] {len(flipped)} cells crossed thresholds:")
        for (k, b_ok, c_ok) in flipped:
            direction = "OK->FLAG" if b_ok else "FLAG->OK"
            print(f"  {k[0]:>5s} e={k[1]:.2f}  {direction}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
