#!/usr/bin/env bash
# Build the isolated chemCPA/biolord environment and frozen official checkouts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
EXTERNAL_ENV="${EXTERNAL_ENV:-$HOME/anaconda3/envs/cytobridge-accept-external}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/tmp/${USER}/cytobridge-accept-external}"
SOURCE_ROOT="$EXTERNAL_ROOT/sources"
STATE_ROOT="$EXTERNAL_ROOT/state"
CHEMCPA_SOURCE="$SOURCE_ROOT/chemCPA"
BIOLORD_SOURCE="$SOURCE_ROOT/biolord"
BIOLORD_REPRO_SOURCE="$SOURCE_ROOT/biolord_reproducibility"
CHEMCPA_COMMIT="43e830eb0958c54e4aa64442c17ec0fed19b3f15"
BIOLORD_COMMIT="b7688790e49728d7f8d0906b980a629de484b19b"
BIOLORD_REPRO_COMMIT="16bfefccc0caa013b11d222f9b02aaf535807f85"
HOST_LABEL="${CYTOBRIDGE_HOST_LABEL:-$(hostname -s 2>/dev/null || echo unknown)}"

finish() {
  local rc=$?
  trap - EXIT
  if [[ -x "$HOME/scripts/notify.py" ]]; then
    "$HOME/scripts/notify.py" --title "CytoBridge external environment" \
      --text "${HOST_LABEL} external environment setup exit=${rc}" || \
      echo "external environment completion notification failed" >&2
  fi
  exit "$rc"
}
trap finish EXIT

clone_frozen() {
  local label=$1
  local url=$2
  local commit=$3
  local destination=$4
  if [[ ! -e "$destination" ]]; then
    git clone --quiet "$url" "$destination"
  fi
  if [[ ! -d "$destination/.git" ]]; then
    echo "$label destination is not a git checkout: $destination" >&2
    exit 2
  fi
  local origin
  origin="$(git -C "$destination" remote get-url origin)"
  if [[ "$origin" != "$url" ]]; then
    echo "$label origin mismatch: expected $url, observed $origin" >&2
    exit 2
  fi
  # Recover only the exact empty-worktree state produced by the earlier
  # `git clone --no-checkout` setup bug. This condition cannot match a checkout
  # containing user edits or untracked files.
  local tracked_count head_file_count staged_deletion_count other_change_count
  tracked_count="$(git -C "$destination" ls-files | wc -l)"
  head_file_count="$(git -C "$destination" ls-tree -r --name-only HEAD | wc -l)"
  staged_deletion_count="$(
    git -C "$destination" diff --cached --diff-filter=D --name-only | wc -l
  )"
  other_change_count="$(
    git -C "$destination" status --porcelain | grep -vc '^D  ' || true
  )"
  if (( tracked_count == 0 && head_file_count > 0 && staged_deletion_count == head_file_count && other_change_count == 0 )); then
    local preserved="${destination}.incomplete-no-checkout"
    if [[ -e "$preserved" ]]; then
      echo "$label preserved incomplete checkout already exists: $preserved" >&2
      exit 2
    fi
    mv "$destination" "$preserved"
    git clone --quiet "$url" "$destination"
  fi
  if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    echo "$label checkout is dirty before commit verification" >&2
    exit 2
  fi
  if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
    git -C "$destination" fetch --quiet origin "$commit"
    git -C "$destination" checkout --quiet --detach "$commit"
  fi
  if [[ "$(git -C "$destination" rev-parse HEAD)" != "$commit" ]] || \
     [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    echo "$label failed frozen checkout verification" >&2
    exit 2
  fi
}

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "missing conda executable: $CONDA_BIN" >&2
  exit 2
fi
mkdir -p "$SOURCE_ROOT" "$STATE_ROOT" "/tmp/${USER}/cytobridge-pip-cache"

clone_frozen chemCPA https://github.com/theislab/chemCPA.git \
  "$CHEMCPA_COMMIT" "$CHEMCPA_SOURCE"
clone_frozen biolord https://github.com/nitzanlab/biolord.git \
  "$BIOLORD_COMMIT" "$BIOLORD_SOURCE"
clone_frozen biolord-repro https://github.com/nitzanlab/biolord_reproducibility.git \
  "$BIOLORD_REPRO_COMMIT" "$BIOLORD_REPRO_SOURCE"

if [[ ! -x "$EXTERNAL_ENV/bin/python" ]]; then
  # The classic solver cannot resolve this environment: it spent 10h41m on
  # "Collecting package metadata (repodata.json)" and was killed without producing an
  # env. The network is not the cause (conda-forge repodata fetches at 5-10 MB/s from
  # cityu); conda 25.7 ships conda-libmamba-solver, which CONDA_NO_PLUGINS disabled.
  # The environment file is unchanged, so its frozen hash still holds.
  env PIP_CACHE_DIR="/tmp/${USER}/cytobridge-pip-cache" \
    "$CONDA_BIN" env create -p "$EXTERNAL_ENV" \
    -f "$ROOT/env/environment-accept-external.yml"
fi

# Install package metadata from the verified local commits without resolving or
# changing their dependencies. biolord imports its version through
# importlib.metadata, so PYTHONPATH alone is insufficient.
"$EXTERNAL_ENV/bin/python" -m pip install --no-deps \
  "$CHEMCPA_SOURCE" "$BIOLORD_SOURCE"

LD_LIBRARY_PATH="$EXTERNAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH="$ROOT:$CHEMCPA_SOURCE:$BIOLORD_SOURCE/src" \
  "$EXTERNAL_ENV/bin/python" -c \
  'import anndata, biolord, lightning, scanpy, scvi, torch; from chemCPA.lightning_module import ChemCPA; from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator; print("external imports ok", torch.__version__, lightning.__version__, scvi.__version__, torch.cuda.is_available())'

LD_LIBRARY_PATH="$EXTERNAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$EXTERNAL_ENV/bin/python" "$ROOT/external/verify_checkout.py" \
  --source-manifest "$ROOT/external/source_manifest.json" \
  --chemcpa "$CHEMCPA_SOURCE" \
  --biolord "$BIOLORD_SOURCE" \
  --biolord-repro "$BIOLORD_REPRO_SOURCE" \
  --out "$STATE_ROOT/checkout_verification.json"
LD_LIBRARY_PATH="$EXTERNAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$EXTERNAL_ENV/bin/python" -m pip freeze >"$STATE_ROOT/pip-freeze.txt"
"$CONDA_BIN" list -p "$EXTERNAL_ENV" --json \
  >"$STATE_ROOT/conda-list.json"
printf '%s\n' \
  "CHEMCPA_SOURCE=$CHEMCPA_SOURCE" \
  "BIOLORD_SOURCE=$BIOLORD_SOURCE" \
  "BIOLORD_REPRO_SOURCE=$BIOLORD_REPRO_SOURCE" \
  >"$STATE_ROOT/paths.env"
echo "external baseline environment ready: $EXTERNAL_ENV"
