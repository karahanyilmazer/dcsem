"""Run the BENCH side of the matched 2-ROI DCM effect-size sweep."""

import sys

from scripts.workflows.dcm_bench_comparison import main


if __name__ == "__main__":
    main(["bench", *sys.argv[1:]])
