#!/usr/bin/env bash
# Create a tracked experiment directory on an AutoDL data disk.
set -euo pipefail

usage() {
  echo "Usage: bash scripts/new_experiment_autodl.sh TAG [--root DIR]" >&2
}

TAG="${1:-}"
if [[ -z "$TAG" || "$TAG" == --* ]]; then
  usage
  exit 2
fi
shift
ROOT="../experiments"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
if ! [[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "TAG contains unsupported characters: $TAG" >&2
  exit 2
fi

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"
mkdir -p "$ROOT"
EXPERIMENT_DIR="$ROOT/$(date +%Y%m%d)-$TAG"
if [[ -e "$EXPERIMENT_DIR" ]]; then
  echo "Refusing to overwrite existing experiment: $EXPERIMENT_DIR" >&2
  exit 3
fi
mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/ckpts"

{
  echo "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "code_dir=$CODE_DIR"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_status_begin"
    git status --short
    echo "git_status_end"
  else
    echo "git_commit=NOT_A_GIT_WORKTREE"
  fi
  "${CYTOBRIDGE_PYTHON:-python}" --version 2>&1 || true
  nvidia-smi || true
} >"$EXPERIMENT_DIR/env.txt"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  '# Record the exact frozen command here before launch.' \
  'echo "Populate run.sh, then launch through launch_monitored_autodl.sh"' \
  >"$EXPERIMENT_DIR/run.sh"
chmod +x "$EXPERIMENT_DIR/run.sh"
printf '%s\n' \
  "tag: $TAG" \
  'status: created' \
  'resource_confirmation: REQUIRED_BEFORE_TRAINING' \
  >"$EXPERIMENT_DIR/config.yaml"
printf '%s\n' \
  "# $TAG" \
  '' \
  'AutoDL experiment scaffold. Fill in the frozen command/config and retain all logs.' \
  >"$EXPERIMENT_DIR/README.md"

echo "$EXPERIMENT_DIR"
