#!/usr/bin/env python
"""Print BENCH effect-size sweep metrics from accuracy_vs_effect_size_*.npz.

Used by scripts/verify_dcm_pipeline.sh when generating SUMMARY.md.
"""
from __future__ import annotations

import sys

import numpy as np


def main(path: str) -> int:
    d = np.load(path)
    files = sorted(d.files)
    print(f"keys: {files}")

    for key in ("effect_size", "accuracy", "n_test_samples", "n_repeats"):
        if key in d.files:
            arr = np.asarray(d[key])
            if arr.ndim == 0:
                print(f"{key:<16}: {arr.item()}")
            else:
                print(f"{key:<16}: {np.array2string(arr, precision=4, suppress_small=True)}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
