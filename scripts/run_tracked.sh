#!/usr/bin/env bash
# Run one foreground command while writing machine-readable PID and exit status files.
set -uo pipefail

usage() {
  echo "usage: $0 --pid-file PATH --status-file PATH --log-file PATH -- COMMAND [ARG ...]" >&2
}

PID_FILE=""
STATUS_FILE=""
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
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

if [[ -z "$PID_FILE" || -z "$STATUS_FILE" || -z "$LOG_FILE" || $# -eq 0 ]]; then
  usage
  exit 2
fi

mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$STATUS_FILE")" "$(dirname "$LOG_FILE")"

printf '[tracked] started_at=%s pid=%s command=' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" >>"$LOG_FILE"
printf '%q ' "$@" >>"$LOG_FILE"
printf '\n' >>"$LOG_FILE"

set +e
"$@" >>"$LOG_FILE" 2>&1 &
CHILD_PID=$!
printf '%s\n' "$CHILD_PID" >"$PID_FILE"
printf 'state=running\nstarted_at=%s\npid=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CHILD_PID" >"$STATUS_FILE"
trap 'kill -TERM "$CHILD_PID" 2>/dev/null || true' TERM INT
wait "$CHILD_PID"
EXIT_CODE=$?
set -e

if [[ $EXIT_CODE -eq 0 ]]; then
  STATE="complete"
else
  STATE="failed"
fi

STATUS_TMP="${STATUS_FILE}.tmp.$$"
printf 'state=%s\nfinished_at=%s\npid=%s\nexit_code=%s\n' \
  "$STATE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CHILD_PID" "$EXIT_CODE" >"$STATUS_TMP"
mv "$STATUS_TMP" "$STATUS_FILE"
printf '[tracked] CYTOBRIDGE_RUN_STATUS=%s exit_code=%s finished_at=%s\n' \
  "$STATE" "$EXIT_CODE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG_FILE"

exit "$EXIT_CODE"
