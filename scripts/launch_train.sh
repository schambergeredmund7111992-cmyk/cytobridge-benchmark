#!/usr/bin/env bash
# Foreground training command for use inside experiments/<tag>/run.sh.
set -euo pipefail

CONFIG="${1:-accept_base}"
if [[ $# -gt 0 ]]; then
  shift
fi

echo "[train] config=train/${CONFIG}"
echo "[train] this command does not detach; launch the experiment via the host protocol"
exec python train.py "--config-name=train/${CONFIG}" "$@"
