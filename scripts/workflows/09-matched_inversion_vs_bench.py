"""Run the full matched 2-ROI DCM inversion-vs-BENCH comparison."""

import sys

from scripts.workflows.dcm_bench_comparison import main


if __name__ == "__main__":
    main(["matched", *sys.argv[1:]])
