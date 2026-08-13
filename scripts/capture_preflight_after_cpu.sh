#!/usr/bin/env bash
# Capture a draft cityu resource estimate after every CPU prerequisite passes.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPU_SESSION="${CPU_SESSION:-cytobridge-cpu-coordinator-cityu}"
SCIPLEX_SUMMARY="${SCIPLEX_SUMMARY:-/tmp/0accept-revision-data/cpu_prep_summary.json}"
EXTERNAL_STATE="${EXTERNAL_STATE:-/tmp/${USER}/cytobridge-accept-external/state/checkout_verification.json}"
EXTERNAL_INPUT="${EXTERNAL_INPUT:-$ROOT/data/processed/external/drug_disjoint_v2/selection.manifest.json}"
TAHOE_SOURCE="${TAHOE_SOURCE:-$ROOT/data/processed/tahoe_accept/source_provenance.json}"
PREFLIGHT="${PREFLIGHT:-$ROOT/../experiments/gates/gpu_preflight_draft.json}"

notify() {
  local level="$1"
  local message="$2"
  if [[ -x "$HOME/scripts/notify.py" ]]; then
    "$HOME/scripts/notify.py" --title "CytoBridge preflight draft" \
      --text "$message" --level "$level" || true
  fi
}
trap 'notify error "CPU prerequisites or resource-draft capture failed; no GPU work started"' ERR

while tmux has-session -t "$CPU_SESSION" 2>/dev/null; do
  sleep 60
done

for path in "$SCIPLEX_SUMMARY" "$EXTERNAL_STATE" "$EXTERNAL_INPUT" "$TAHOE_SOURCE"; do
  if [[ ! -f "$path" ]]; then
    echo "CPU prerequisite evidence is missing: $path" >&2
    exit 3
  fi
done

cd "$ROOT"
"$ROOT/env/core_python.sh" -m scripts.capture_gpu_preflight --out "$PREFLIGHT"

trap - ERR
notify info "CPU prerequisites passed and a draft resource estimate is ready; explicit PI confirmation is still required"
echo "[preflight-after-cpu] PASS draft=$PREFLIGHT"
