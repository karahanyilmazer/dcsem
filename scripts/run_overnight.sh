#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_PY="/Users/karahanyilmazer/Coding/python/venvs/dcsem_13/bin/python"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PY}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/overnight_$TIMESTAMP}"
USE_CAFFEINATE=1
STOP_ON_ERROR=0
MODE="detach"
DRY_RUN=0

SCRIPTS=(
  "02-estimate_params_solver.py"
  "02-estimate_params_mcmc.py"
  "03-off_diag_errors.py"
  "04-extract_summary_measures.py"
  "05-apply_bench.py"
  "06-investigate_bench.py"
  "07-model_inversion_confusion.py"
  "08-identifiability_analysis.py"
  "inversion_generic.py"
  "mcmc_generic.py"
  "spdcm_generic.py"
  "spdcm_mcmc_generic.py"
  "spdcm_noise_sweep.py"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Default behavior:
  Starts the overnight batch in the background with nohup, keeps the machine
  awake via caffeinate when available, and writes logs under:
    $ROOT_DIR/logs/overnight_TIMESTAMP/

Options:
  --foreground       Run in the current shell instead of detaching.
  --worker           Internal mode used by the detached launcher.
  --python PATH      Python interpreter to use.
  --log-dir PATH     Directory for master and per-script logs.
  --skip-caffeinate  Do not use caffeinate even if available.
  --stop-on-error    Stop at the first failing script.
  --dry-run          Print what would run without executing anything.
  --help             Show this message.

Examples:
  $(basename "$0")
  $(basename "$0") --foreground
  $(basename "$0") --python "$DEFAULT_PY" --log-dir "$ROOT_DIR/logs/my_run"
EOF
}

timestamp() {
  date "+%F %T"
}

validate_scripts() {
  local script
  for script in "${SCRIPTS[@]}"; do
    if [[ ! -f "$ROOT_DIR/$script" ]]; then
      echo "Missing script: $ROOT_DIR/$script" >&2
      exit 1
    fi
  done
}

run_worker() {
  mkdir -p "$LOG_DIR"
  validate_scripts

  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 1
  fi

  cd "$ROOT_DIR"
  export MPLBACKEND="${MPLBACKEND:-Agg}"
  export PYTHONUNBUFFERED=1

  echo "[$(timestamp)] ROOT_DIR=$ROOT_DIR"
  echo "[$(timestamp)] PYTHON_BIN=$PYTHON_BIN"
  echo "[$(timestamp)] LOG_DIR=$LOG_DIR"
  echo "[$(timestamp)] MPLBACKEND=$MPLBACKEND"

  local overall=0
  local script
  for script in "${SCRIPTS[@]}"; do
    local log_file="$LOG_DIR/${script%.py}.log"
    echo "[$(timestamp)] START $script"

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[$(timestamp)] DRYRUN $PYTHON_BIN -u $script > $log_file 2>&1"
      echo "[$(timestamp)] END   $script exit=0"
      continue
    fi

    set +e
    "$PYTHON_BIN" -u "$ROOT_DIR/$script" >"$log_file" 2>&1
    local status=$?
    set -e

    echo "[$(timestamp)] END   $script exit=$status"
    if [[ "$status" -ne 0 ]]; then
      overall=1
      if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
        echo "[$(timestamp)] STOPPING after failure in $script"
        break
      fi
    fi
  done

  echo "[$(timestamp)] ALL DONE exit=$overall"
  return "$overall"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground)
      MODE="foreground"
      ;;
    --worker)
      MODE="worker"
      ;;
    --python)
      shift
      PYTHON_BIN="${1:?Missing value for --python}"
      ;;
    --log-dir)
      shift
      LOG_DIR="${1:?Missing value for --log-dir}"
      ;;
    --skip-caffeinate)
      USE_CAFFEINATE=0
      ;;
    --stop-on-error)
      STOP_ON_ERROR=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ "$MODE" == "worker" || "$MODE" == "foreground" ]]; then
  run_worker
  exit $?
fi

mkdir -p "$LOG_DIR"
validate_scripts

launcher=(nohup)
if [[ "$USE_CAFFEINATE" -eq 1 ]] && command -v caffeinate >/dev/null 2>&1; then
  launcher+=(caffeinate -dimsu)
fi

worker_cmd=(
  env
  "PYTHON_BIN=$PYTHON_BIN"
  "LOG_DIR=$LOG_DIR"
  "STOP_ON_ERROR=$STOP_ON_ERROR"
  "DRY_RUN=$DRY_RUN"
  "$SCRIPT_PATH"
  --worker
)

master_log="$LOG_DIR/master.log"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Would launch:"
  printf '  %q' "${launcher[@]}" "${worker_cmd[@]}"
  printf ' > %q 2>&1 < /dev/null &\n' "$master_log"
  exit 0
fi

"${launcher[@]}" "${worker_cmd[@]}" >"$master_log" 2>&1 < /dev/null &
pid=$!
disown "$pid" || true

echo "PID=$pid"
echo "LOG_DIR=$LOG_DIR"
echo "MASTER_LOG=$master_log"
echo "Watch progress with:"
echo "  tail -f \"$master_log\""
