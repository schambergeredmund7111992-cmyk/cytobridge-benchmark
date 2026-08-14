#!/usr/bin/env python
"""Collect the small analysis inputs that exist only on the Mac.

The training server holds the per-pair prediction matrices and the raw h5ad,
but the E6E7 gene order and the pooled-vehicle analysis artifacts live on the
Mac under /Users/cgxmac/Desktop/CytoBridge. Run this ON THE MAC:

    python3 scripts/collect_mac_inputs.py --root /Users/cgxmac/Desktop/CytoBridge --out mac_inputs.zip

It scans for (a) the 3000-gene list of the E6E7 pipeline, (b) pooled-vehicle
matrices or the script that re-derives them, (c) the frozen split index, and
zips everything small it finds. Upload the zip to the Google Drive shared
folder (or copy it to the training server) and rerun
scripts/export_regeneration_inputs.py with --mac-inputs mac_inputs.zip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

NAME_PATTERNS = [
    # gene lists of any flavour
    "gene", "hvg", "var_names", "features",
    # pooled / vehicle constructions
    "pooled", "vehicle", "control_counts", "control_pb",
    # split definitions
    "internal_splits", "split_assignments", "split_manifest", "e6e7",
    # final analysis outputs (small csv/json/npy)
    "config_metrics", "bootstrap", "permutation", "calibration", "sensitivity",
    "table1", "per_space", "verify_", "logfc_meta", "mean_predictor",
    "oracle", "derangement", "replicate", "crossplate", "targets",
]
SIZE_LIMIT = 200 * 1024 * 1024  # skip anything larger (h5ad etc.)
SKIP_DIRS = {"logs", "lightning_logs", "wandb", ".git", "__pycache__",
             "node_modules", "version_0", "version_1", "version_2",
             "version_3", "version_4", "version_5"}


def interesting(path: Path) -> bool:
    name = path.name.lower()
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if path.suffix.lower() not in (".npy", ".npz", ".csv", ".json", ".txt", ".yaml", ".yml", ".py", ".sh"):
        return False
    if any(pattern in name for pattern in NAME_PATTERNS):
        return True
    if path.name in ("STATUS.json", "README.md", "PROTOCOL.md"):
        return True
    return False


def collect(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not interesting(path):
            continue
        try:
            if path.stat().st_size > SIZE_LIMIT:
                print(f"  skip (large): {path}")
                continue
        except OSError:
            continue
        files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Mac analysis root (e.g. /Users/cgxmac/Desktop/CytoBridge)")
    parser.add_argument("--out", type=Path, default=Path("mac_inputs.zip"))
    args = parser.parse_args()

    root = Path(args.root)
    files = collect(root)
    print(f"collected {len(files)} files")
    for path in files[:60]:
        print("  ", path.relative_to(root) if root in path.parents else path)
    if len(files) > 60:
        print(f"  ... and {len(files) - 60} more")

    manifest = []
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.relative_to(root) if root in path.parents else path.name
            archive.write(path, str(relative))
            manifest.append({"file": str(relative), "sha256": digest,
                             "size": path.stat().st_size})
    (args.out.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
