#!/usr/bin/env bash
# verify_dcm_pipeline.sh — End-to-end deterministic-DCM pipeline verification.
#
# Runs the test suite, both forward-model playgrounds, both single-fit
# pipelines, and workflows 02–08 in sequence. Writes one log per stage plus
# a SUMMARY.md with NPZ key metrics and links to the produced artifacts.
#
# Skips spectral-DCM and laminar-DCM scripts entirely (out of scope here).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/results/logs/verify_$TIMESTAMP}"
SKIP_TESTS=0
SKIP_HEADLINE=0
STOP_ON_ERROR=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

End-to-end verification of the deterministic-DCM pipeline.
Runs from $ROOT_DIR/.

Stages (in order):
  01-tests                      uv run python -m pytest dcsem/tests
  02a-playground_2roi           scripts/playground/dcm_1layer_2roi.py
  02b-playground_3roi           scripts/playground/dcm_1layer_3roi.py
  03a-inversion_dcm             scripts/pipelines/inversion_generic.py
                                  (env: DCSEM_ACTIVE_MODEL=dcm_2roi DCSEM_SEED=42)
  03b-mcmc_dcm                  scripts/pipelines/mcmc_generic.py
                                  (env: DCSEM_ACTIVE_MODEL=dcm_2roi DCSEM_SEED=42
                                        DCSEM_N_SAMPLES_MCMC=3500 DCSEM_N_BURN=300
                                        DCSEM_N_WALKERS=16; ~15-20 min vs ~110 at full)
  04a-wf02_solver_snr           scripts/workflows/02-estimate_params_solver.py
  04b-wf03_off_diag             scripts/workflows/03-off_diag_errors.py
  04c-wf04_summary_measures     scripts/workflows/04-extract_summary_measures.py
  04d-pre_train_freshness       scripts/_verify_artifact_freshness.py --mode pre-train
                                  (HARD-GATE: aborts pipeline on stale upstream PCA)
  04e-wf08_identifiability      scripts/workflows/08-identifiability_analysis.py
  05a-wf05_bench_sweep          scripts/workflows/05-apply_bench.py
  05b-post_train_freshness      scripts/_verify_artifact_freshness.py --mode post-train
                                  (HARD-GATE: aborts pipeline on stale BENCH model)
  05c-wf05_bench_headline2000   scripts/workflows/05-apply_bench.py --n-test-samples 2000
                                  (skip with --skip-headline)
  05d-wf06_investigate          scripts/workflows/06-investigate_bench.py
  05e-wf10_linearity            scripts/workflows/10-bench_linearity_diagnostic.py
                                  (soft gate: FLAGGED cells do not abort)
  06-wf07_inversion_confusion   scripts/workflows/07-model_inversion_confusion.py

Total wall clock: ~50–90 min sequential.

Options:
  --skip-tests        Skip Stage 01 (pytest already ran).
  --skip-headline     Skip Stage 05b (2000-sample BENCH headline; ~15–25 min savings).
  --stop-on-error     Exit at the first failing stage (default: continue + summarize).
  --log-dir PATH      Override log dir (default: results/logs/verify_TIMESTAMP/).
  --dry-run           Print stages without running them.
  --help              Show this message.

Output:
  $LOG_DIR/
    master.log              Orchestration log (start/end/status per stage)
    SUMMARY.md              Per-stage table, NPZ key metrics, error tails
    <stage>.log             stdout+stderr from each stage
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tests) SKIP_TESTS=1 ;;
    --skip-headline) SKIP_HEADLINE=1 ;;
    --stop-on-error) STOP_ON_ERROR=1 ;;
    --log-dir) shift; LOG_DIR="${1:?Missing value for --log-dir}" ;;
    --dry-run) DRY_RUN=1 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"
SUMMARY="$LOG_DIR/SUMMARY.md"

timestamp() { date "+%F %T"; }

log() {
  local msg="[$(timestamp)] $*"
  echo "$msg"
  printf '%s\n' "$msg" >> "$MASTER_LOG"
}

# ----- Pre-flight ------------------------------------------------------------

cd "$ROOT_DIR"

SCRIPT_PATHS=(
  "scripts/playground/dcm_1layer_2roi.py"
  "scripts/playground/dcm_1layer_3roi.py"
  "scripts/pipelines/inversion_generic.py"
  "scripts/pipelines/mcmc_generic.py"
  "scripts/workflows/02-estimate_params_solver.py"
  "scripts/workflows/03-off_diag_errors.py"
  "scripts/workflows/04-extract_summary_measures.py"
  "scripts/workflows/05-apply_bench.py"
  "scripts/workflows/06-investigate_bench.py"
  "scripts/workflows/07-model_inversion_confusion.py"
  "scripts/workflows/08-identifiability_analysis.py"
)
for p in "${SCRIPT_PATHS[@]}"; do
  if [[ ! -f "$ROOT_DIR/$p" ]]; then
    echo "Missing script: $ROOT_DIR/$p" >&2
    exit 1
  fi
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

GIT_BRANCH="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

log "================================================================"
log "Deterministic-DCM verification"
log "ROOT_DIR=$ROOT_DIR"
log "LOG_DIR=$LOG_DIR"
log "GIT_BRANCH=$GIT_BRANCH GIT_SHA=$GIT_SHA"
log "SKIP_TESTS=$SKIP_TESTS SKIP_HEADLINE=$SKIP_HEADLINE STOP_ON_ERROR=$STOP_ON_ERROR DRY_RUN=$DRY_RUN"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
log "MPLBACKEND=$MPLBACKEND PYTHONUNBUFFERED=$PYTHONUNBUFFERED"

LATEX_OK="$(uv run python -c 'from dcsem.config import PATH_CONFIG; print("ok" if PATH_CONFIG.get_latex_path() else "fail")' 2>/dev/null || echo fail)"
if [[ "$LATEX_OK" != "ok" ]]; then
  export DCSEM_LATEX_DIR="$LOG_DIR/latex"
  mkdir -p "$DCSEM_LATEX_DIR"
  log "DCSEM_LATEX_DIR not configured; falling back to $DCSEM_LATEX_DIR"
else
  log "DCSEM_LATEX_DIR resolves OK"
fi
log "================================================================"

# ----- Stage runner ----------------------------------------------------------

# STAGES_RUN entries: "name|status" (status=0 PASS, otherwise FAIL exit code)
STAGES_RUN=()

generate_summary() {
  local rc=$?
  set +e
  log "Generating SUMMARY.md"

  {
    echo "# Deterministic-DCM Verification Summary"
    echo
    echo "- **Run:** $TIMESTAMP"
    echo "- **Branch:** $GIT_BRANCH"
    echo "- **Commit:** $GIT_SHA"
    echo "- **Log dir:** \`$LOG_DIR\`"
    echo "- **Master log:** \`master.log\`"
    echo "- **Final exit code:** $rc"
    echo
    echo "## Stages"
    echo
    echo "| Stage | Result | Exit | Log |"
    echo "|---|---|---|---|"
    for entry in "${STAGES_RUN[@]}"; do
      stage_name="${entry%%|*}"
      stage_status="${entry##*|}"
      if [[ "$stage_status" == "0" ]]; then
        mark="PASS"
      else
        mark="FAIL"
      fi
      echo "| \`$stage_name\` | $mark | $stage_status | \`$stage_name.log\` |"
    done
    echo
    echo "## Pipeline NPZ inspection"
    echo
  } > "$SUMMARY"

  inspect_npz "L-BFGS-B inversion (\`dcm_2roi\`)" \
    "$ROOT_DIR/results/images/inversion/L-BFGS-B/dcm_2roi/run_results.npz" \
    inversion

  inspect_npz "MCMC inversion (\`dcm_2roi\`)" \
    "$ROOT_DIR/results/images/inversion/MCMC/dcm_2roi/run_results.npz" \
    mcmc

  inspect_npz "BENCH sweep accuracy (wf05)" \
    "$ROOT_DIR/results/models/bench_no_noise_4/accuracy_vs_effect_size_bench.npz" \
    bench

  inspect_npz "BENCH inversion-confusion accuracy (wf07)" \
    "$ROOT_DIR/results/models/bench_no_noise_4/accuracy_vs_effect_size_model_inversion.npz" \
    bench

  {
    echo "## Output directories to inspect"
    echo
    for d in \
      "results/images/inversion/L-BFGS-B/dcm_2roi" \
      "results/images/inversion/MCMC/dcm_2roi" \
      "results/images/wip/estimation" \
      "results/images/bench_final_no_noise_4" \
      "results/images/bench_with_noise_4" \
      "results/images/identifiability" \
      "results/images/dcm" \
      "results/models/bench" \
      "results/models/bench_no_noise_4" \
      "results/models/bench_with_noise_4"; do
      if [[ -d "$ROOT_DIR/$d" ]]; then
        echo "- \`$d/\`"
      else
        echo "- \`$d/\` _(missing)_"
      fi
    done
    echo
    echo "## Errors / tracebacks in logs"
    echo
  } >> "$SUMMARY"

  err_logs="$(grep -l -iE "Traceback|FAILED|^Error:" "$LOG_DIR"/*.log 2>/dev/null || true)"
  if [[ -z "$err_logs" ]]; then
    echo "_No tracebacks or test failures detected in any log._" >> "$SUMMARY"
  else
    while IFS= read -r f; do
      [[ -z "$f" ]] && continue
      echo "### \`$(basename "$f")\`" >> "$SUMMARY"
      echo >> "$SUMMARY"
      echo "Last 30 lines:" >> "$SUMMARY"
      echo '```' >> "$SUMMARY"
      tail -30 "$f" >> "$SUMMARY"
      echo '```' >> "$SUMMARY"
      echo >> "$SUMMARY"
    done <<< "$err_logs"
  fi

  log "Summary written to $SUMMARY"
  log "================================================================"
  log "DONE exit=$rc"
  exit "$rc"
}

# Inspect a single NPZ artifact and append a markdown block to SUMMARY.md.
# Args: label, path, kind (inversion|mcmc|bench)
inspect_npz() {
  local label="$1"
  local path="$2"
  local kind="$3"

  echo "### $label" >> "$SUMMARY"
  echo >> "$SUMMARY"
  echo "Path: \`$path\`" >> "$SUMMARY"
  echo >> "$SUMMARY"

  if [[ ! -f "$path" ]]; then
    echo "_NPZ not found — pipeline likely failed or skipped._" >> "$SUMMARY"
    echo >> "$SUMMARY"
    return
  fi

  echo '```' >> "$SUMMARY"
  case "$kind" in
    inversion) uv run python "$SCRIPT_DIR/_inspect_inversion_npz.py" "$path" 2>&1 >> "$SUMMARY" || echo "(inspection failed)" >> "$SUMMARY" ;;
    mcmc)      uv run python "$SCRIPT_DIR/_inspect_mcmc_npz.py" "$path"      2>&1 >> "$SUMMARY" || echo "(inspection failed)" >> "$SUMMARY" ;;
    bench)     uv run python "$SCRIPT_DIR/_inspect_bench_npz.py" "$path"     2>&1 >> "$SUMMARY" || echo "(inspection failed)" >> "$SUMMARY" ;;
    *)         echo "(unknown kind: $kind)" >> "$SUMMARY" ;;
  esac
  echo '```' >> "$SUMMARY"
  echo >> "$SUMMARY"
}

trap 'generate_summary' EXIT

run_stage() {
  local name="$1"; shift
  local env_str="$1"; shift
  local log_file="$LOG_DIR/${name}.log"

  log "START $name${env_str:+ (env: $env_str)}"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRYRUN ${env_str:+env $env_str }$* > $log_file"
    log "END   $name exit=0 (dry-run)"
    STAGES_RUN+=("$name|0")
    return 0
  fi

  local status
  set +e
  if [[ -n "$env_str" ]]; then
    # shellcheck disable=SC2086  # we want word-splitting on env_str
    env $env_str "$@" >"$log_file" 2>&1
  else
    "$@" >"$log_file" 2>&1
  fi
  status=$?
  set -e

  log "END   $name exit=$status"
  STAGES_RUN+=("$name|$status")

  if [[ $status -ne 0 && $STOP_ON_ERROR -eq 1 ]]; then
    log "STOPPING after failure in $name (--stop-on-error set)"
    exit "$status"
  fi
  return 0
}

# hard_run_stage runs like run_stage but ALWAYS aborts the whole pipeline on
# non-zero exit, regardless of --stop-on-error. Use for freshness gates whose
# failure means downstream stages would otherwise consume stale state.
hard_run_stage() {
  local name="$1"; shift
  local env_str="$1"; shift
  local log_file="$LOG_DIR/${name}.log"

  log "START $name [HARD-GATE]${env_str:+ (env: $env_str)}"

  if [[ $DRY_RUN -eq 1 ]]; then
    log "DRYRUN ${env_str:+env $env_str }$* > $log_file"
    log "END   $name exit=0 (dry-run)"
    STAGES_RUN+=("$name|0")
    return 0
  fi

  local status
  set +e
  if [[ -n "$env_str" ]]; then
    # shellcheck disable=SC2086
    env $env_str "$@" >"$log_file" 2>&1
  else
    "$@" >"$log_file" 2>&1
  fi
  status=$?
  set -e

  log "END   $name exit=$status"
  STAGES_RUN+=("$name|$status")

  if [[ $status -ne 0 ]]; then
    log "HARD-GATE FAIL in $name (exit=$status); aborting pipeline"
    exit "$status"
  fi
  return 0
}

# ----- Stages ----------------------------------------------------------------

if [[ $SKIP_TESTS -eq 0 ]]; then
  run_stage "01-tests" "" \
    uv run python -m pytest dcsem/tests -v --tb=short
else
  log "SKIP 01-tests (--skip-tests set)"
fi

run_stage "02a-playground_2roi" "" \
  uv run python "scripts/playground/dcm_1layer_2roi.py"
run_stage "02b-playground_3roi" "" \
  uv run python "scripts/playground/dcm_1layer_3roi.py"

run_stage "03a-inversion_dcm" "DCSEM_ACTIVE_MODEL=dcm_2roi DCSEM_SEED=42" \
  uv run python "scripts/pipelines/inversion_generic.py"
# Scaled-down MCMC (3500 samples / 300 burn-in / 16 walkers) clears the
# ESS > 200 gate with margin while keeping verification budget ~15-20 min.
# Full-fidelity (10000 samples) takes ~110 min and is exercised separately
# by test_inversion_pipelines.py (quadratic model, fast).
run_stage "03b-mcmc_dcm" \
  "DCSEM_ACTIVE_MODEL=dcm_2roi DCSEM_SEED=42 DCSEM_N_SAMPLES_MCMC=3500 DCSEM_N_BURN=300 DCSEM_N_WALKERS=16" \
  uv run python "scripts/pipelines/mcmc_generic.py"

run_stage "04a-wf02_solver_snr" "" \
  uv run python "scripts/workflows/02-estimate_params_solver.py"
run_stage "04b-wf03_off_diag" "" \
  uv run python "scripts/workflows/03-off_diag_errors.py"
run_stage "04c-wf04_summary_measures" "" \
  uv run python "scripts/workflows/04-extract_summary_measures.py"
# Pre-train freshness gate: the PCA, ICA, and noise-sigma artifacts that
# 05-apply_bench.py is about to consume must hash-match their sidecars.
# A missing or mismatched PCA aborts the pipeline. Missing BENCH model is OK
# here (we're about to build it).
hard_run_stage "04d-pre_train_freshness" "" \
  uv run python "scripts/_verify_artifact_freshness.py" --setting no_noise_4 --mode pre-train
run_stage "04e-wf08_identifiability" "" \
  uv run python "scripts/workflows/08-identifiability_analysis.py"

run_stage "05a-wf05_bench_sweep" "" \
  uv run python "scripts/workflows/05-apply_bench.py"
# Post-train freshness gate: now everything must match - in particular the
# BENCH model sidecar must declare a pca_sha256 that equals the current PCA.
hard_run_stage "05b-post_train_freshness" "" \
  uv run python "scripts/_verify_artifact_freshness.py" --setting no_noise_4 --mode post-train
if [[ $SKIP_HEADLINE -eq 0 ]]; then
  run_stage "05c-wf05_bench_headline2000" "" \
    uv run python "scripts/workflows/05-apply_bench.py" --n-test-samples 2000
else
  log "SKIP 05c-wf05_bench_headline2000 (--skip-headline set)"
fi
run_stage "05d-wf06_investigate" "" \
  uv run python "scripts/workflows/06-investigate_bench.py"
# Linearity diagnostic: characterises BENCH's Jacobian extrapolation across
# the test effect-size grid. Soft-gated (does not abort the pipeline) so a
# FLAGGED outcome still produces the full SUMMARY.md.
run_stage "05e-wf10_linearity" "" \
  uv run python "scripts/workflows/10-bench_linearity_diagnostic.py" --setting no_noise_4

run_stage "06-wf07_inversion_confusion" "" \
  uv run python "scripts/workflows/07-model_inversion_confusion.py"

# Compute final exit code from stage statuses (trap will produce SUMMARY).
overall=0
for entry in "${STAGES_RUN[@]}"; do
  if [[ "${entry##*|}" != "0" ]]; then overall=1; fi
done
exit "$overall"
