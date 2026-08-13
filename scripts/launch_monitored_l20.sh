#!/usr/bin/env bash
# Launch one l20 command only after a fresh GPU check, then detach its monitor.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: launch_monitored_l20.sh --tag TAG --gpu INDEX --experiment-dir DIR \
  [--interval SECONDS] [--min-free-gb N] [--ready-pattern REGEX] -- COMMAND [ARG ...]

Set CYTOBRIDGE_LAUNCH_CONFIRMED=1 only after the user confirms the resource estimate.
EOF
}

TAG=""
GPU=""
EXPERIMENT_DIR=""
INTERVAL=300
MIN_FREE_GB=80
READY_PATTERN='Epoch 0.*loss|train/loss'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --gpu)
      GPU="${2:-}"
      shift 2
      ;;
    --experiment-dir)
      EXPERIMENT_DIR="${2:-}"
      shift 2
      ;;
    --interval)
      INTERVAL="${2:-}"
      shift 2
      ;;
    --min-free-gb)
      MIN_FREE_GB="${2:-}"
      shift 2
      ;;
    --ready-pattern)
      READY_PATTERN="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$TAG" || -z "$GPU" || -z "$EXPERIMENT_DIR" || $# -eq 0 ]]; then
  usage
  exit 2
fi
if [[ "${CYTOBRIDGE_LAUNCH_CONFIRMED:-0}" != "1" ]]; then
  echo "launch refused: set CYTOBRIDGE_LAUNCH_CONFIRMED=1 only after explicit confirmation" >&2
  exit 3
fi
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
  echo "GPU index must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ && "$MIN_FREE_GB" =~ ^[0-9]+$ ]]; then
  echo "interval must be positive and min-free-gb must be non-negative" >&2
  exit 2
fi
if [[ ! -d "$EXPERIMENT_DIR/logs" ]]; then
  echo "missing experiment logs directory: $EXPERIMENT_DIR/logs" >&2
  echo "create the experiment with ~/scripts/new_experiment.sh first" >&2
  exit 4
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENT_DIR="$(cd "$EXPERIMENT_DIR" && pwd)"
PID_FILE="$EXPERIMENT_DIR/logs/run.pid"
STATUS_FILE="$EXPERIMENT_DIR/logs/run.status"
LOG_FILE="$EXPERIMENT_DIR/logs/run.log"
MONITOR_LOG="$EXPERIMENT_DIR/logs/monitor.log"
READINESS_REPORT="$EXPERIMENT_DIR/logs/readiness.json"

for path in "$PID_FILE" "$STATUS_FILE" "$LOG_FILE" "$MONITOR_LOG"; do
  if [[ -e "$path" ]]; then
    echo "launch refused: existing run artifact would be overwritten: $path" >&2
    exit 5
  fi
done

READINESS_CHECK="$ROOT/../handoff/check_readiness.py"
if [[ -f "$READINESS_CHECK" ]]; then
  PYTHON_BIN="${CYTOBRIDGE_PYTHON:-python}"
  READINESS_TMP="${READINESS_REPORT}.tmp.$$"
  if ! "$PYTHON_BIN" "$READINESS_CHECK" --code-dir "$ROOT" >"$READINESS_TMP"; then
    mv "$READINESS_TMP" "$READINESS_REPORT"
    echo "launch refused: handoff readiness checks failed; see $READINESS_REPORT" >&2
    exit 8
  fi
  mv "$READINESS_TMP" "$READINESS_REPORT"
fi

GPU_ROW="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits -i "$GPU")"
IFS=',' read -r OBSERVED_GPU UTIL MEM_USED MEM_TOTAL <<<"$GPU_ROW"
OBSERVED_GPU="${OBSERVED_GPU//[[:space:]]/}"
UTIL="${UTIL//[[:space:]]/}"
MEM_USED="${MEM_USED//[[:space:]]/}"
MEM_TOTAL="${MEM_TOTAL//[[:space:]]/}"
if [[ "$OBSERVED_GPU" != "$GPU" || ! "$UTIL" =~ ^[0-9]+$ || ! "$MEM_USED" =~ ^[0-9]+$ ]]; then
  echo "could not parse nvidia-smi preflight row: $GPU_ROW" >&2
  exit 6
fi
if ((UTIL >= 5 || MEM_USED >= 1024)); then
  echo "launch refused: GPU $GPU is not eligible (util=${UTIL}%, mem=${MEM_USED}/${MEM_TOTAL} MiB)" >&2
  exit 7
fi

cd "$ROOT"
nohup env CUDA_VISIBLE_DEVICES="$GPU" bash scripts/run_tracked.sh \
  --pid-file "$PID_FILE" \
  --status-file "$STATUS_FILE" \
  --log-file "$LOG_FILE" \
  -- "$@" >/dev/null 2>&1 &
RUN_LAUNCH_PID=$!

nohup bash scripts/monitor_experiment.sh \
  --tag "$TAG" \
  --pid-file "$PID_FILE" \
  --status-file "$STATUS_FILE" \
  --log-file "$LOG_FILE" \
  --interval "$INTERVAL" \
  --min-free-gb "$MIN_FREE_GB" \
  --ready-pattern "$READY_PATTERN" >"$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!

printf '%s\n' "$RUN_LAUNCH_PID" >"$EXPERIMENT_DIR/logs/launcher.pid"
printf '%s\n' "$MONITOR_PID" >"$EXPERIMENT_DIR/logs/monitor.pid"
echo "detached run launcher pid=$RUN_LAUNCH_PID monitor pid=$MONITOR_PID gpu=$GPU"
echo "status=$STATUS_FILE log=$LOG_FILE monitor_log=$MONITOR_LOG"
