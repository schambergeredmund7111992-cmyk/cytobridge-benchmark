#!/usr/bin/env bash
# Run the selected core interpreter with its matching C++ runtime ahead of the host library.
set -euo pipefail

if [[ -n "${CORE_ENV:-}" ]]; then
  CORE_ENV="$CORE_ENV"
elif [[ -n "${CYTOBRIDGE_CORE_ENV:-}" ]]; then
  CORE_ENV="$CYTOBRIDGE_CORE_ENV"
elif [[ -x "$HOME/miniconda3/envs/cytobridge-accept-5090/bin/python" ]]; then
  CORE_ENV="$HOME/miniconda3/envs/cytobridge-accept-5090"
else
  CORE_ENV="$HOME/anaconda3/envs/cytobridge-accept-core"
fi
if [[ ! -x "$CORE_ENV/bin/python" || ! -f "$CORE_ENV/lib/libstdc++.so.6" ]]; then
  echo "core environment is incomplete: $CORE_ENV" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$CORE_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$CORE_ENV/bin/python" "$@"
