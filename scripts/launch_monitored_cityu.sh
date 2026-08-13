#!/usr/bin/env bash
# Launch one confirmed cityu job in tmux with a detached monitor window.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: launch_monitored_cityu.sh --tag TAG --gpu INDEX --experiment-dir DIR \
  [--interval SECONDS] [--min-free-gb N] [--ready-pattern REGEX] -- COMMAND [ARG ...]

Set CYTOBRIDGE_LAUNCH_CONFIRMED=1 only after the PI confirms the fresh resource estimate.
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
    --tag) TAG="${2:-}"; shift 2 ;;
    --gpu) GPU="${2:-}"; shift 2 ;;
    --experiment-dir) EXPERIMENT_DIR="${2:-}"; shift 2 ;;
    --interval) INTERVAL="${2:-}"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="${2:-}"; shift 2 ;;
    --ready-pattern) READY_PATTERN="${2:-}"; shift 2 ;;
    --) shift; break ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "$TAG" || -z "$GPU" || -z "$EXPERIMENT_DIR" || $# -eq 0 ]]; then
  usage
  exit 2
fi
if [[ "${CYTOBRIDGE_LAUNCH_CONFIRMED:-0}" != "1" ]]; then
  echo "launch refused: explicit confirmation is not recorded" >&2
  exit 3
fi
if [[ ! "$TAG" =~ ^[A-Za-z0-9_.-]+$ || ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "tag or GPU index has an invalid format" >&2
  exit 2
fi
if [[ ! -d "$EXPERIMENT_DIR/logs" ]]; then
  echo "missing experiment logs directory; use ~/scripts/new_experiment.sh first" >&2
  exit 4
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENT_DIR="$(cd "$EXPERIMENT_DIR" && pwd)"
PID_FILE="$EXPERIMENT_DIR/logs/run.pid"
STATUS_FILE="$EXPERIMENT_DIR/logs/run.status"
LOG_FILE="$EXPERIMENT_DIR/logs/run.log"
MONITOR_LOG="$EXPERIMENT_DIR/logs/monitor.log"
READINESS_REPORT="$EXPERIMENT_DIR/logs/readiness.json"
SESSION="${TAG}-cityu"
for path in "$PID_FILE" "$STATUS_FILE" "$LOG_FILE" "$MONITOR_LOG"; do
  if [[ -e "$path" ]]; then
    echo "launch refused: existing run artifact would be overwritten: $path" >&2
    exit 5
  fi
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "launch refused: tmux session already exists: $SESSION" >&2
  exit 5
fi

PYTHON_BIN="${CYTOBRIDGE_PYTHON:-$ROOT/env/core_python.sh}"
READINESS=("$PYTHON_BIN" "$ROOT/../handoff/check_readiness.py" --code-dir "$ROOT")
if [[ -n "${SCGPT_CKPT_DIR:-}" ]]; then
  READINESS+=(--scgpt-ckpt-dir "$SCGPT_CKPT_DIR")
fi
if ! "${READINESS[@]}" >"${READINESS_REPORT}.tmp"; then
  mv "${READINESS_REPORT}.tmp" "$READINESS_REPORT"
  echo "launch refused: readiness checks failed; see $READINESS_REPORT" >&2
  exit 8
fi
mv "${READINESS_REPORT}.tmp" "$READINESS_REPORT"

if [[ -n "${CYTOBRIDGE_SOURCE_MANIFEST:-}" ]]; then
  "$PYTHON_BIN" -m scripts.source_tree_manifest verify \
    --project-root "$ROOT/.." --manifest "$CYTOBRIDGE_SOURCE_MANIFEST"
fi

GPU_ROW="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits -i "$GPU")"
IFS=',' read -r OBSERVED_GPU UTIL MEM_USED MEM_TOTAL <<<"$GPU_ROW"
OBSERVED_GPU="${OBSERVED_GPU//[[:space:]]/}"
UTIL="${UTIL//[[:space:]]/}"
MEM_USED="${MEM_USED//[[:space:]]/}"
if [[ "$OBSERVED_GPU" != "$GPU" || ! "$UTIL" =~ ^[0-9]+$ || ! "$MEM_USED" =~ ^[0-9]+$ ]]; then
  echo "could not parse nvidia-smi preflight row: $GPU_ROW" >&2
  exit 6
fi
if (( UTIL >= 5 || MEM_USED >= 1024 )); then
  echo "launch refused: GPU $GPU is not eligible (${UTIL}%, ${MEM_USED}/${MEM_TOTAL} MiB)" >&2
  exit 7
fi

printf -v ROOT_Q '%q' "$ROOT"
printf -v TRACKED_Q '%q ' bash scripts/run_tracked.sh \
  --pid-file "$PID_FILE" --status-file "$STATUS_FILE" --log-file "$LOG_FILE" -- "$@"
printf -v MONITOR_Q '%q ' bash scripts/monitor_experiment.sh \
  --tag "$TAG" --pid-file "$PID_FILE" --status-file "$STATUS_FILE" \
  --log-file "$LOG_FILE" --interval "$INTERVAL" --min-free-gb "$MIN_FREE_GB" \
  --ready-pattern "$READY_PATTERN" --terminate-on-anomaly

tmux new-session -d -s "$SESSION" -n run \
  "cd $ROOT_Q && CUDA_VISIBLE_DEVICES=$GPU $TRACKED_Q"
tmux new-window -d -t "$SESSION" -n monitor \
  "cd $ROOT_Q && $MONITOR_Q >$(printf '%q' "$MONITOR_LOG") 2>&1"
printf '%s\n' "$SESSION" >"$EXPERIMENT_DIR/logs/tmux.session"
echo "launched session=$SESSION gpu=$GPU"
echo "status=$STATUS_FILE log=$LOG_FILE monitor=$MONITOR_LOG"
