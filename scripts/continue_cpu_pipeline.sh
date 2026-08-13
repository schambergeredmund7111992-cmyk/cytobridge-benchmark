#!/usr/bin/env bash
# Continue external exports and Tahoe preparation after current background prerequisites exit.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCIPLEX_SESSION="${SCIPLEX_SESSION:-cytobridge-sciplex-cpu-cityu}"
EXTERNAL_SESSION="${EXTERNAL_SESSION:-cytobridge-external-env-cityu}"
SCIPLEX_SUMMARY="${SCIPLEX_SUMMARY:-/tmp/0accept-revision-data/cpu_prep_summary.json}"
EXTERNAL_EVIDENCE="${EXTERNAL_EVIDENCE:-/tmp/${USER}/cytobridge-accept-external/state/checkout_verification.json}"

notify() {
  local level="$1"
  local message="$2"
  if [[ -x "$HOME/scripts/notify.py" ]]; then
    "$HOME/scripts/notify.py" --title "CytoBridge CPU coordinator" \
      --text "$message" --level "$level" || true
  fi
}
trap 'notify error "CPU coordinator failed; no automatic retry"' ERR

while tmux has-session -t "$SCIPLEX_SESSION" 2>/dev/null || \
      tmux has-session -t "$EXTERNAL_SESSION" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "$SCIPLEX_SUMMARY" ]]; then
  echo "sci-Plex CPU summary is missing: $SCIPLEX_SUMMARY" >&2
  exit 3
fi
if [[ ! -f "$EXTERNAL_EVIDENCE" ]]; then
  echo "external checkout evidence is missing: $EXTERNAL_EVIDENCE" >&2
  exit 3
fi

cd "$ROOT"
bash scripts/prepare_external_inputs.sh
bash scripts/prepare_tahoe_cpu.sh

trap - ERR
notify info "external inputs and Tahoe CPU preparation passed"
echo "[cpu-coordinator] PASS"
