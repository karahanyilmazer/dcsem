#!/usr/bin/env python
"""Print key calibration / convergence metrics from an L-BFGS-B run_results.npz.

Used by scripts/verify_dcm_pipeline.sh when generating SUMMARY.md.
"""
from __future__ import annotations

import sys

import numpy as np


def main(path: str) -> int:
    d = np.load(path)
    files = sorted(d.files)
    print(f"keys: {files}")

    def scalar_bool(key: str) -> str:
        if key not in d.files:
            return "MISSING"
        v = d[key]
        return str(bool(np.asarray(v).ravel()[0]))

    def scalar_float(key: str, fmt: str = "{:.4g}") -> str:
        if key not in d.files:
            return "MISSING"
        v = d[key]
        return fmt.format(float(np.asarray(v).ravel()[0]))

    print(f"cov_is_calibrated    : {scalar_bool('cov_is_calibrated')}")
    print(f"converged            : {scalar_bool('converged')}")
    print(f"hess_is_near_singular: {scalar_bool('hess_is_near_singular')}")
    print(f"hess_cond            : {scalar_float('hess_cond', '{:.3e}')}")

    for key in ("theta_true", "theta_est", "theta_zero", "se"):
        if key in d.files:
            arr = np.asarray(d[key])
            print(f"{key:<13}: {np.array2string(arr, precision=4, suppress_small=True)}")

    if "ci" in d.files:
        ci = np.asarray(d["ci"])
        print(f"ci shape={ci.shape}:")
        print(np.array2string(ci, precision=4, suppress_small=True))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
