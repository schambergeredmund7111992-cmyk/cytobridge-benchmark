#!/usr/bin/env bash
# Poll a detached run and notify only on start, anomaly, low disk, or process exit.
set -uo pipefail

usage() {
  cat >&2 <<'EOF'
usage: monitor_experiment.sh --tag TAG --pid-file PATH --status-file PATH \
  --log-file PATH [--interval SECONDS] [--min-free-gb N] \
  [--ready-pattern REGEX] [--terminate-on-anomaly] [--once]
EOF
}

TAG=""
PID_FILE=""
STATUS_FILE=""
LOG_FILE=""
INTERVAL=300
MIN_FREE_GB=80
ONCE=0
READY_PATTERN=""
TERMINATE_ON_ANOMALY=0
NOTIFY_BIN="${CYTOBRIDGE_NOTIFY_BIN:-$HOME/scripts/notify.py}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --pid-file)
      PID_FILE="${2:-}"
      shift 2
      ;;
    --status-file)
      STATUS_FILE="${2:-}"
      shift 2
      ;;
    --log-file)
      LOG_FILE="${2:-}"
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
    --once)
      ONCE=1
      shift
      ;;
    --terminate-on-anomaly)
      TERMINATE_ON_ANOMALY=1
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$TAG" || -z "$PID_FILE" || -z "$STATUS_FILE" || -z "$LOG_FILE" ]]; then
  usage
  exit 2
fi
if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ && "$MIN_FREE_GB" =~ ^[0-9]+$ ]]; then
  echo "interval must be positive and min-free-gb must be non-negative" >&2
  exit 2
fi

STATE_DIR="$(dirname "$STATUS_FILE")"
MONITOR_STATE="${STATE_DIR}/monitor.state"
mkdir -p "$STATE_DIR"

notify() {
  local level="$1"
  local text="$2"
  if [[ ! -x "$NOTIFY_BIN" ]]; then
    echo "[monitor] notification unavailable: $NOTIFY_BIN" >&2
    return 1
  fi
  if ! "$NOTIFY_BIN" --title "CytoBridge: ${TAG}" --text "$text" --level "$level"; then
    echo "[monitor] notification rejected or failed for event: $level" >&2
    return 1
  fi
}

read_status_value() {
  local key="$1"
  if [[ -f "$STATUS_FILE" ]]; then
    awk -F= -v key="$key" '$1 == key {print substr($0, index($0, "=") + 1); exit}' \
      "$STATUS_FILE"
  fi
}

last_log_line() {
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 20 "$LOG_FILE" | awk 'NF {line=$0} END {print substr(line, 1, 500)}'
  fi
}

has_event() {
  [[ -f "$MONITOR_STATE" ]] && grep -Fxq "$1" "$MONITOR_STATE"
}

mark_event() {
  if ! has_event "$1"; then
    printf '%s\n' "$1" >>"$MONITOR_STATE"
  fi
}

check_once() {
  local state pid exit_code free_kb free_gb last_line
  state="$(read_status_value state)"
  pid="$(read_status_value pid)"
  exit_code="$(read_status_value exit_code)"

  if [[ -f "$LOG_FILE" ]] && tail -n 500 "$LOG_FILE" | grep -Eiq \
    '(^|[^[:alpha:]])nan([^[:alpha:]]|$)|(^|[^[:alpha:]])inf([^[:alpha:]]|$)|out of memory|cuda oom|traceback|data leakage|baseline([^[:alpha:]]+).?mismatch|gradient explosion'; then
    if has_event anomaly; then
      return 20
    fi
    last_line="$(last_log_line)"
    if [[ $TERMINATE_ON_ANOMALY -eq 1 && "$pid" =~ ^[1-9][0-9]*$ ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
    notify error "检测到质量/运行异常，monitor 已停止且不会自动重试。日志末行：${last_line}" || true
    mark_event anomaly
    return 20
  fi

  if [[ "$state" == "running" && -n "$READY_PATTERN" && -f "$LOG_FILE" ]]; then
    if grep -Eiq "$READY_PATTERN" "$LOG_FILE" && ! has_event normal_start; then
      last_line="$(last_log_line)"
      notify info "已观察到首个正常进度；monitor 将继续后台检查。日志末行：${last_line}" || true
      mark_event normal_start
    fi
  elif [[ "$state" == "running" ]] && ! has_event process_start; then
    notify info "后台进程已启动；monitor 每 ${INTERVAL}s 检查日志、退出状态和磁盘。" || true
    mark_event process_start
  fi

  free_kb="$(df -Pk "$(dirname "$LOG_FILE")" | awk 'NR == 2 {print $4}')"
  if [[ "$free_kb" =~ ^[0-9]+$ ]]; then
    free_gb=$((free_kb / 1024 / 1024))
    if ((free_gb < MIN_FREE_GB)); then
      if has_event low_disk; then
        return 21
      fi
      notify error "可用磁盘仅 ${free_gb} GiB，低于 ${MIN_FREE_GB} GiB 阈值；monitor 已停止。" || true
      if [[ $TERMINATE_ON_ANOMALY -eq 1 && "$pid" =~ ^[1-9][0-9]*$ ]]; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
      mark_event low_disk
      return 21
    fi
  fi

  if [[ "$state" == "complete" ]]; then
    if has_event complete; then
      return 10
    fi
    last_line="$(last_log_line)"
    notify info "后台进程已正常退出（exit 0）。这只表示进程完成，论文结果仍须通过 result gate。日志末行：${last_line}" || true
    mark_event complete
    return 10
  fi
  if [[ "$state" == "failed" ]]; then
    if has_event failed; then
      return 22
    fi
    last_line="$(last_log_line)"
    notify error "后台进程失败（exit ${exit_code:-unknown}），不会自动重试。日志末行：${last_line}" || true
    mark_event failed
    return 22
  fi

  if [[ -n "$pid" && "$state" == "running" ]] && ! kill -0 "$pid" 2>/dev/null; then
    if has_event missing_process; then
      return 23
    fi
    last_line="$(last_log_line)"
    notify error "进程已消失但未写入终态；按异常处理且不会自动重试。日志末行：${last_line}" || true
    mark_event missing_process
    return 23
  fi

  return 0
}

while true; do
  check_once
  RESULT=$?
  case "$RESULT" in
    10)
      exit 0
      ;;
    20|21|22|23)
      exit "$RESULT"
      ;;
  esac
  if [[ $ONCE -eq 1 ]]; then
    exit 0
  fi
  sleep "$INTERVAL"
done
