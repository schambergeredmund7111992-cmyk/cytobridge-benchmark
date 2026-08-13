#!/usr/bin/env bash
# Download and preprocess official sci-Plex inputs without using a GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${CORE_PY:-$HOME/anaconda3/envs/cytobridge-accept-core/bin/python}"
DATA_ROOT="${CYTOBRIDGE_DATA_ROOT:-/tmp/0accept-revision-data}"
EXPECTED_SCIPLEX_BYTES=2526631614
EXPECTED_SCIPLEX_MD5=c9e70629505d98c7ca1a837f62b14e89
EXPECTED_HALLMARK_BYTES=48690
EXPECTED_HALLMARK_SHA256=ee2463540042078bfa3f67828e1e223bb354446d9fbb4d22845866835ba5c772

finish() {
  local rc=$?
  local level=error
  trap - EXIT
  if [[ $rc -eq 0 ]]; then
    level=info
  fi
  if [[ -x "$HOME/scripts/notify.py" ]]; then
    "$HOME/scripts/notify.py" --title "CytoBridge sci-Plex CPU" \
      --text "cityu CPU preparation exit=${rc}" \
      --level "$level" || true
  fi
  exit "$rc"
}
trap finish EXIT

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing core Python: $PYTHON_BIN" >&2
  exit 2
fi
PYTHON_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/processed" "$DATA_ROOT/cache"
for name in raw processed cache; do
  target="$ROOT/data/$name"
  if [[ -e "$target" && ! -L "$target" ]]; then
    echo "refusing to replace non-symlink data path: $target" >&2
    exit 3
  fi
  if [[ ! -L "$target" ]]; then
    ln -s "$DATA_ROOT/$name" "$target"
  fi
done

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
cd "$ROOT"
"$PYTHON_BIN" -m data.download --target sciplex --out data/raw

SCIPLEX=data/raw/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad
if [[ "$(stat -c %s "$SCIPLEX")" != "$EXPECTED_SCIPLEX_BYTES" ]]; then
  echo "official sci-Plex byte count mismatch" >&2
  exit 4
fi
if [[ "$(md5sum "$SCIPLEX" | awk '{print $1}')" != "$EXPECTED_SCIPLEX_MD5" ]]; then
  echo "official sci-Plex MD5 mismatch" >&2
  exit 4
fi

HALLMARK=data/raw/msigdb/h.all.v2024.1.Hs.symbols.gmt
if [[ ! -f "$HALLMARK" ]]; then
  mkdir -p "$(dirname "$HALLMARK")"
  env -u LD_LIBRARY_PATH curl -L --fail --retry 3 -o "${HALLMARK}.part" \
    https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt
  if [[ "$(stat -c %s "${HALLMARK}.part")" != "$EXPECTED_HALLMARK_BYTES" ]] || \
     [[ "$(sha256sum "${HALLMARK}.part" | awk '{print $1}')" != "$EXPECTED_HALLMARK_SHA256" ]]; then
    echo "official Hallmark identity mismatch" >&2
    exit 5
  fi
  mv "${HALLMARK}.part" "$HALLMARK"
fi

if [[ ! -f data/raw/sciplex/sciplex3.h5ad && \
      ! -f data/raw/sciplex/sciplex3_drugs.csv ]]; then
  "$PYTHON_BIN" -m data.prepare_sciplex_scperturb --fetch_pubchem
elif [[ ! -f data/raw/sciplex/sciplex3.h5ad || \
        ! -f data/raw/sciplex/sciplex3_drugs.csv ]]; then
  echo "partial prepared sci-Plex outputs require review; refusing to continue" >&2
  exit 6
fi

bash scripts/run_pipeline.sh sciplex
"$PYTHON_BIN" scripts/summarize_prepared_data.py \
  --data-root "$DATA_ROOT" \
  --out "$DATA_ROOT/cpu_prep_summary.json"
