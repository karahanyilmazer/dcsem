#!/usr/bin/env bash
# _test_bench_dv0.sh - Re-run the BENCH linearity diagnostic at a chosen dv0.
#
# Usage:
#   bash scripts/_test_bench_dv0.sh                       # defaults: dv0=0.2
#   bash scripts/_test_bench_dv0.sh 0.2                   # explicit dv0
#   bash scripts/_test_bench_dv0.sh 0.2 200 1000          # dv0, n_baseline, n_train
#   DV0=0.1 N_BASELINE=200 N_TRAIN=1000 bash scripts/_test_bench_dv0.sh
#
# What it does:
#   1. Snapshots the current results/models/bench_no_noise_4/linearity_diagnostic.npz
#      to linearity_diagnostic_dv<dv0_old>.npz so the next diagnostic does
#      not clobber it. The "_old" tag comes from the upstream mdl sidecar's
#      recorded dv0, so subsequent runs at different dv0 produce a chain of
#      comparable baselines.
#   2. Runs scripts/workflows/10-bench_linearity_diagnostic.py at the new dv0.
#      The BENCH cache auto-invalidates because bench_dv0 is part of the
#      mdl sidecar fingerprint, so the model is retrained on the fly.
#   3. Runs scripts/_compare_linearity.py against the snapshot.
#
# Output locations (printed at the end of the run):
#   stdout (tee'd to /tmp/linearity_dv<dv0>.log)
#   results/models/bench_no_noise_4/linearity_diagnostic.npz       (new, dv0)
#   results/models/bench_no_noise_4/linearity_diagnostic_dv<old>.npz (frozen)
#   results/images/bench_final_no_noise_4/linearity_diagnostic.png
#   results/models/bench_no_noise_4/mdl_no_noise_4.pkl.meta.json

set -euo pipefail

# ---------- args / defaults -------------------------------------------------
DV0="${1:-${DV0:-0.2}}"
N_BASELINE="${2:-${N_BASELINE:-200}}"
N_TRAIN="${3:-${N_TRAIN:-1000}}"
SETTING="${SETTING:-no_noise_4}"

# ---------- locate project root --------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="results/models/bench_${SETTING}"
NPZ="$MODEL_DIR/linearity_diagnostic.npz"
MDL_META="$MODEL_DIR/mdl_${SETTING}.pkl.meta.json"

echo "============================================================"
echo "BENCH dv0 diagnostic test"
echo "  dv0          = $DV0"
echo "  n_baseline   = $N_BASELINE"
echo "  n_train      = $N_TRAIN"
echo "  setting      = $SETTING"
echo "  project root = $ROOT_DIR"
echo "============================================================"

# ---------- step 1: snapshot the existing diagnostic NPZ -------------------
if [[ -f "$NPZ" ]]; then
  # Tag the snapshot with the dv0 that produced it, taken from the
  # current BENCH mdl sidecar so we never mislabel the snapshot.
  if [[ -f "$MDL_META" ]]; then
    OLD_DV0="$(uv run python -c "import json; print(json.load(open('$MDL_META'))['dv0'])")"
  else
    OLD_DV0="unknown"
  fi
  OLD_TAG="$(echo "$OLD_DV0" | sed 's/[^0-9a-zA-Z._-]/_/g')"
  SNAPSHOT="$MODEL_DIR/linearity_diagnostic_dv${OLD_TAG}.npz"
  if [[ ! -f "$SNAPSHOT" ]]; then
    cp "$NPZ" "$SNAPSHOT"
    echo "[snapshot] saved current diagnostic as $SNAPSHOT"
  else
    echo "[snapshot] $SNAPSHOT already exists; leaving it alone"
  fi
else
  SNAPSHOT=""
  echo "[snapshot] no existing $NPZ to snapshot; will only produce the new run"
fi

# ---------- step 2: re-run the diagnostic at the new dv0 --------------------
DV0_TAG="$(echo "$DV0" | sed 's/[^0-9a-zA-Z._-]/_/g')"
LOG="/tmp/linearity_dv${DV0_TAG}.log"
echo
echo "[diagnostic] running with --bench-dv0 $DV0 (logs tee'd to $LOG)"
echo
uv run python scripts/workflows/10-bench_linearity_diagnostic.py \
    --setting "$SETTING" \
    --n-baseline "$N_BASELINE" \
    --bench-dv0 "$DV0" \
    --n-train-samples "$N_TRAIN" \
    2>&1 | tee "$LOG"

# ---------- step 3: compare against the snapshot ---------------------------
COMPARE_LOG="/tmp/linearity_compare_dv${DV0_TAG}.log"
if [[ -n "$SNAPSHOT" ]]; then
  echo
  echo "[compare] dv0=${OLD_DV0} vs dv0=${DV0}"
  echo
  uv run python scripts/_compare_linearity.py "$SNAPSHOT" "$NPZ" 2>&1 | tee "$COMPARE_LOG"
else
  COMPARE_LOG=""
  echo
  echo "[compare] skipped (no snapshot to compare against)"
fi

# ---------- output summary --------------------------------------------------
echo
echo "============================================================"
echo "DONE. Paste any of these back when reviewing:"
echo "  diagnostic stdout : $LOG"
[[ -n "$COMPARE_LOG" ]] && echo "  compare table     : $COMPARE_LOG"
echo "  new diagnostic NPZ: $NPZ"
[[ -n "$SNAPSHOT" ]] && echo "  baseline snapshot : $SNAPSHOT"
echo "  figure (PNG)      : results/images/bench_final_${SETTING}/linearity_diagnostic.png"
echo "  BENCH mdl sidecar : $MDL_META"
echo "============================================================"
