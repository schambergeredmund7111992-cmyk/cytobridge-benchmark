#!/usr/bin/env bash
# Create and launch one immutable AutoDL campaign phase on a confirmed GPU.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: launch_autodl_campaign_phase.sh PHASE GPU CONFIRMED_PREFLIGHT

Required environment:
  CORE_ENV, SCGPT_CKPT_DIR, CYTOBRIDGE_EXTERNAL_ROOT,
  CYTOBRIDGE_EXTERNAL_PYTHON, CYTOBRIDGE_LAUNCH_CONFIRMED=1
EOF
}

PHASE="${1:-}"
GPU="${2:-}"
PREFLIGHT_INPUT="${3:-}"
if [[ ! "$PHASE" =~ ^P[123]$ || ! "$GPU" =~ ^[0-9]+$ || -z "$PREFLIGHT_INPUT" ]]; then
  usage
  exit 2
fi
if [[ "${CYTOBRIDGE_LAUNCH_CONFIRMED:-0}" != "1" ]]; then
  echo "launch refused: PI confirmation flag is not set" >&2
  exit 3
fi
for name in CORE_ENV SCGPT_CKPT_DIR CYTOBRIDGE_EXTERNAL_ROOT CYTOBRIDGE_EXTERNAL_PYTHON; do
  if [[ -z "${!name:-}" ]]; then
    echo "launch refused: required environment variable is missing: $name" >&2
    exit 4
  fi
done
if [[ ! -f "$PREFLIGHT_INPUT" ]]; then
  echo "confirmed preflight does not exist: $PREFLIGHT_INPUT" >&2
  exit 4
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFLIGHT="$(cd "$(dirname "$PREFLIGHT_INPUT")" && pwd)/$(basename "$PREFLIGHT_INPUT")"
TAG="autodl-${PHASE,,}-5090"
EXPERIMENT_ROOT="${CYTOBRIDGE_LAUNCHER_ROOT:-$ROOT/../experiments/launchers}"
EXPERIMENT_DIR="$(
  CYTOBRIDGE_PYTHON="$CORE_ENV/bin/python" \
    bash "$ROOT/scripts/new_experiment_autodl.sh" "$TAG" --root "$EXPERIMENT_ROOT"
)"
EXPERIMENT_DIR="$(cd "$EXPERIMENT_DIR" && pwd)"
SCHEDULER_REPORT="$EXPERIMENT_DIR/scheduler.json"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  "cd $(printf '%q' "$ROOT")" \
  "$(printf '%q' "$ROOT/env/core_python.sh") -m scripts.campaign_scheduler \\" \
  "  --manifest $(printf '%q' "$ROOT/../experiments/campaign_manifest.json") \\" \
  "  --phase $(printf '%q' "$PHASE") \\" \
  "  --preflight $(printf '%q' "$PREFLIGHT") \\" \
  '  --expected-host autodl \' \
  "  --cpu-parallelism $(printf '%q' "${CYTOBRIDGE_CPU_PARALLELISM:-4}") \\" \
  "  --out $(printf '%q' "$SCHEDULER_REPORT")" \
  >"$EXPERIMENT_DIR/run.sh"
chmod +x "$EXPERIMENT_DIR/run.sh"

printf '%s\n' \
  "tag: $TAG" \
  "phase: $PHASE" \
  'execution_target: autodl' \
  "gpu: $GPU" \
  "preflight: $PREFLIGHT" \
  'resource_confirmation: PI_CONFIRMED' \
  'automatic_retries: 0' \
  >"$EXPERIMENT_DIR/config.yaml"

export CYTOBRIDGE_CORE_ENV="$CORE_ENV"
export CYTOBRIDGE_EXPECTED_HOST=autodl
export CYTOBRIDGE_PYTHON="$CORE_ENV/bin/python"
bash "$ROOT/scripts/launch_monitored_autodl.sh" \
  --tag "$TAG" \
  --gpu "$GPU" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --ready-pattern 'FIRST_FINITE_LOSS' \
  -- "$EXPERIMENT_DIR/run.sh"

echo "AutoDL campaign phase detached: phase=$PHASE experiment=$EXPERIMENT_DIR"
