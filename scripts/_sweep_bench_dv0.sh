#!/usr/bin/env bash
# _sweep_bench_dv0.sh - Run the BENCH linearity diagnostic across multiple dv0 values.
#
# Usage:
#   bash scripts/_sweep_bench_dv0.sh                                 # defaults
#   bash scripts/_sweep_bench_dv0.sh 0.05,0.1,0.2,0.3                # custom dv0 grid
#   bash scripts/_sweep_bench_dv0.sh 0.05,0.1,0.2,0.3 200 1000       # + n_baseline, n_train
#   FORCE_RERUN=1 bash scripts/_sweep_bench_dv0.sh                   # ignore cached snapshots
#   SETTING=with_noise_4 bash scripts/_sweep_bench_dv0.sh            # other setting
#
# What it does:
#   1. Snapshots whatever linearity_diagnostic.npz currently lives on disk,
#      tagging it with the dv0 recorded in the BENCH mdl sidecar (so no
#      pre-sweep state is lost).
#   2. For each dv0 in DVALUES, runs the linearity diagnostic. BENCH cache
#      auto-invalidates because bench_dv0 is in the sidecar fingerprint, so
#      the model retrains on the fly. Each run snapshots the new NPZ as
#      results/models/bench_<setting>/linearity_diagnostic_dv<dv0>.npz.
#      If a snapshot already exists for a given dv0 and FORCE_RERUN is not
#      set, that dv0 is skipped (re-running is a no-op).
#   3. Runs scripts/_compare_linearity_sweep.py across ALL dv0 snapshots in
#      the model directory (including any pre-existing ones like
#      dv1e-06 from earlier pre-flights).
#
# Expected wall time: ~3-5 min per dv0 value (BENCH retrain dominates).
# Cached dv0 values are reused instantly.
#
# Output:
#   /tmp/linearity_dv<dv0>.log                per-dv0 diagnostic stdout
#   /tmp/linearity_sweep_summary.log          combined comparison table
#   results/models/bench_<setting>/linearity_diagnostic_dv<dv0>.npz   per-dv0 NPZ
#   results/models/bench_<setting>/mdl_<setting>.pkl.meta.json        sidecar of last-trained model

set -euo pipefail

# ---------- args / defaults -------------------------------------------------
DVALUES="${1:-0.05,0.1,0.2,0.3}"
N_BASELINE="${2:-${N_BASELINE:-200}}"
N_TRAIN="${3:-${N_TRAIN:-1000}}"
SETTING="${SETTING:-no_noise_4}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# ---------- locate project root --------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="results/models/bench_${SETTING}"
NPZ="$MODEL_DIR/linearity_diagnostic.npz"
MDL_META="$MODEL_DIR/mdl_${SETTING}.pkl.meta.json"

echo "============================================================"
echo "BENCH dv0 sweep"
echo "  dv0 grid     = $DVALUES"
echo "  n_baseline   = $N_BASELINE"
echo "  n_train      = $N_TRAIN"
echo "  setting      = $SETTING"
echo "  force_rerun  = $FORCE_RERUN"
echo "  project root = $ROOT_DIR"
echo "============================================================"

# ---------- pre-sweep snapshot of whatever is currently on disk -------------
if [[ -f "$NPZ" && -f "$MDL_META" ]]; then
  CURRENT_DV0="$(uv run python -c "import json; print(json.load(open('$MDL_META'))['dv0'])")"
  CURRENT_TAG="$(echo "$CURRENT_DV0" | sed 's/[^0-9a-zA-Z._-]/_/g')"
  CURRENT_SNAP="$MODEL_DIR/linearity_diagnostic_dv${CURRENT_TAG}.npz"
  if [[ ! -f "$CURRENT_SNAP" ]]; then
    cp "$NPZ" "$CURRENT_SNAP"
    echo "[pre-sweep] snapshotted current diagnostic as $CURRENT_SNAP"
  else
    echo "[pre-sweep] $CURRENT_SNAP already exists; leaving it"
  fi
fi

# ---------- iterate over dv0 values ----------------------------------------
IFS=',' read -ra DV_ARRAY <<< "$DVALUES"

for raw_dv0 in "${DV_ARRAY[@]}"; do
  dv0="$(echo "$raw_dv0" | xargs)"
  [[ -z "$dv0" ]] && continue
  TAG="$(echo "$dv0" | sed 's/[^0-9a-zA-Z._-]/_/g')"
  SNAP="$MODEL_DIR/linearity_diagnostic_dv${TAG}.npz"
  LOG="/tmp/linearity_dv${TAG}.log"

  echo
  echo "------------------------------------------------------------"
  echo "[sweep] dv0=$dv0"
  echo "------------------------------------------------------------"

  if [[ -f "$SNAP" && "$FORCE_RERUN" -ne 1 ]]; then
    echo "[sweep] $SNAP already exists; reusing (set FORCE_RERUN=1 to redo)"
    continue
  fi

  uv run python scripts/workflows/10-bench_linearity_diagnostic.py \
      --setting "$SETTING" \
      --n-baseline "$N_BASELINE" \
      --bench-dv0 "$dv0" \
      --n-train-samples "$N_TRAIN" \
      2>&1 | tee "$LOG"

  cp "$NPZ" "$SNAP"
  echo "[sweep] saved $SNAP"
done

# ---------- combined comparison across ALL dv0 snapshots -------------------
echo
echo "============================================================"
echo "[sweep] all runs complete; building combined comparison"
echo "============================================================"
SUMMARY_LOG="/tmp/linearity_sweep_summary.log"
# Collect every dv0 snapshot in the model dir, including ones from prior runs
# (e.g. dv1e-06 from the pre-flight) so the table shows the full history.
shopt -s nullglob
SNAPSHOTS=("$MODEL_DIR"/linearity_diagnostic_dv*.npz)
shopt -u nullglob
if [[ ${#SNAPSHOTS[@]} -eq 0 ]]; then
  echo "[sweep] no snapshots found in $MODEL_DIR; nothing to compare"
  exit 0
fi
uv run python scripts/_compare_linearity_sweep.py "${SNAPSHOTS[@]}" 2>&1 | tee "$SUMMARY_LOG"

echo
echo "============================================================"
echo "DONE. Paste back when ready:"
echo "  summary table : $SUMMARY_LOG"
for snap in "${SNAPSHOTS[@]}"; do
  echo "  snapshot      : $snap"
done
echo "============================================================"
