#!/usr/bin/env bash
# AutoDL uses the same fail-closed detached launcher; this alias keeps the handoff unambiguous.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CYTOBRIDGE_EXECUTION_TARGET="${CYTOBRIDGE_EXECUTION_TARGET:-autodl}"
exec bash "$SCRIPT_DIR/launch_monitored_l20.sh" "$@"
