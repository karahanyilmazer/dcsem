"""Run the model-inversion side of the matched 2-ROI DCM effect-size sweep."""

import sys

from scripts.workflows.dcm_bench_comparison import main


if __name__ == "__main__":
    main(["inversion", *sys.argv[1:]])
