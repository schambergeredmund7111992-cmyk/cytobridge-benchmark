"""Create immutable train+validation arrays for fixed-epoch final refits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.artifacts import sha256_file

ARRAY_SUFFIXES = (
    "treated_counts",
    "input_control_counts",
    "truth_control_counts",
    "pathway_gsea",
)


def _concatenate_npy(first_path: Path, second_path: Path, output_path: Path) -> None:
    first = np.load(first_path, mmap_mode="r")
    second = np.load(second_path, mmap_mode="r")
    if first.ndim != second.ndim or first.shape[1:] != second.shape[1:]:
        raise ValueError(
            f"Cannot concatenate incompatible arrays {first.shape} and {second.shape}."
        )
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.result_type(first.dtype, second.dtype),
        shape=(len(first) + len(second), *first.shape[1:]),
    )
    for start in range(0, len(first), 1024):
        output[start : start + 1024] = first[start : start + 1024]
    offset = len(first)
    for start in range(0, len(second), 1024):
        output[offset + start : offset + start + 1024] = second[start : start + 1024]
    output.flush()
    del output


def combine_fit_splits(protocol_dir: Path, output_dir: Path, *, prefix: str) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite final-refit data: {output_dir}")
    split_dir = protocol_dir / "splits"
    train_manifest_path = split_dir / f"{prefix}_train.parquet"
    validation_manifest_path = split_dir / f"{prefix}_val.parquet"
    train = pd.read_parquet(train_manifest_path)
    validation = pd.read_parquet(validation_manifest_path)
    if set(train["drug_id"].astype(str)) & set(validation["drug_id"].astype(str)):
        raise ValueError(
            "Train and validation drugs overlap before final-refit combination."
        )
    train = train.copy()
    validation = validation.copy()
    train["source_split"] = "train"
    validation["source_split"] = "validation"
    combined = pd.concat([train, validation], ignore_index=True)
    combined["split"] = "fit_train_validation"

    output_dir.mkdir(parents=True)
    combined_path = output_dir / f"{prefix}_fit.parquet"
    combined.to_parquet(combined_path, index=False)
    sources = {
        "train_manifest": train_manifest_path,
        "validation_manifest": validation_manifest_path,
    }
    for suffix in ARRAY_SUFFIXES:
        train_path = split_dir / f"{prefix}_train_{suffix}.npy"
        validation_path = split_dir / f"{prefix}_val_{suffix}.npy"
        output_path = output_dir / f"{prefix}_fit_{suffix}.npy"
        _concatenate_npy(train_path, validation_path, output_path)
        sources[f"train_{suffix}"] = train_path
        sources[f"validation_{suffix}"] = validation_path
    manifest = {
        "fit_splits": ["train", "validation"],
        "selection_complete_before_combination": True,
        "validation_used_for_checkpoint_selection_during_refit": False,
        "rows": int(len(combined)),
        "drugs": int(combined["drug_id"].astype(str).nunique()),
        "source_hashes": {
            name: sha256_file(path) for name, path in sorted(sources.items())
        },
        "output_hashes": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "fit_manifest.json"
        },
    }
    (output_dir / "fit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", choices=("sciplex", "tahoe"), required=True)
    args = parser.parse_args()
    result = combine_fit_splits(args.protocol_dir, args.out, prefix=args.prefix)
    print(json.dumps({"rows": result["rows"], "drugs": result["drugs"]}))


if __name__ == "__main__":
    main()
