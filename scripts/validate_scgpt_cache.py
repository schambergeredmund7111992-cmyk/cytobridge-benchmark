#!/usr/bin/env python3
"""Validate a scGPT cache, provenance, and manifest-to-AnnData row alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_manifest_alignment(
    manifest: pd.DataFrame,
    obs_names: Sequence[str],
    *,
    manifest_name: str,
) -> dict[str, Any]:
    required = {
        "cell_idx",
        "control_cell_idx",
        "treated_cell_id",
        "input_control_cell_id",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_name} is missing columns: {missing}")
    n_rows = len(obs_names)
    treated_indices = manifest["cell_idx"].to_numpy(dtype=np.int64)
    control_indices = manifest["control_cell_idx"].to_numpy(dtype=np.int64)
    for label, indices in (("cell_idx", treated_indices), ("control_cell_idx", control_indices)):
        if len(indices) and (indices.min() < 0 or indices.max() >= n_rows):
            raise ValueError(f"{manifest_name} has out-of-bounds {label}")

    obs_array = np.asarray([str(value) for value in obs_names], dtype=object)
    treated_ids = manifest["treated_cell_id"].astype(str).to_numpy()
    control_ids = manifest["input_control_cell_id"].astype(str).to_numpy()
    treated_mismatches = np.flatnonzero(obs_array[treated_indices] != treated_ids)
    control_mismatches = np.flatnonzero(obs_array[control_indices] != control_ids)
    if len(treated_mismatches) or len(control_mismatches):
        raise ValueError(
            f"{manifest_name} row alignment failed: "
            f"treated_mismatches={len(treated_mismatches)}, "
            f"control_mismatches={len(control_mismatches)}"
        )
    return {
        "manifest": manifest_name,
        "rows": len(manifest),
        "treated_rows_aligned": True,
        "input_control_rows_aligned": True,
    }


def validate_cache(
    h5ad_path: Path,
    cache_path: Path,
    provenance_path: Path,
    split_dir: Path,
    wrapper_path: Path,
    *,
    prefix: str = "sciplex",
) -> dict[str, Any]:
    import anndata as ad

    for path in (h5ad_path, cache_path, provenance_path, split_dir, wrapper_path):
        if not path.exists():
            raise FileNotFoundError(path)
    provenance = json.loads(provenance_path.read_text())
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs_names = [str(value) for value in adata.obs_names.tolist()]
        n_obs = int(adata.n_obs)
    finally:
        adata.file.close()

    cache = np.load(cache_path, mmap_mode="r")
    expected_shape = (n_obs, 1, 512)
    if cache.shape != expected_shape:
        raise ValueError(f"cache shape is {cache.shape}, expected {expected_shape}")
    if cache.dtype != np.float32:
        raise ValueError(f"cache dtype is {cache.dtype}, expected float32")
    max_norm_error = 0.0
    for start in range(0, n_obs, 8192):
        block = np.asarray(cache[start : start + 8192, 0])
        if not np.isfinite(block).all():
            raise ValueError(f"cache contains non-finite values near row {start}")
        norm_error = np.max(np.abs(np.linalg.norm(block, axis=1) - 1))
        max_norm_error = max(max_norm_error, float(norm_error))
    if max_norm_error > 1e-4:
        raise ValueError(f"cache CLS vectors are not unit normalized: {max_norm_error}")

    observed_hashes = {
        "input_h5ad_sha256": _sha256_file(h5ad_path),
        "output_sha256": _sha256_file(cache_path),
        "ordered_row_ids_sha256": _sha256_strings(obs_names),
        "wrapper_source_sha256": _sha256_file(wrapper_path),
    }
    for key, observed in observed_hashes.items():
        if provenance.get(key) != observed:
            raise ValueError(
                f"provenance {key} mismatch: expected {provenance.get(key)!r}, "
                f"observed {observed!r}"
            )
    coverage = provenance.get("checkpoint_coverage", {})
    if (
        coverage.get("coverage_fraction") != 1.0
        or coverage.get("missing_encoder_keys")
        or coverage.get("shape_mismatches")
        or coverage.get("unequal_after_load")
    ):
        raise ValueError("provenance does not show complete encoder checkpoint coverage")

    manifest_reports = []
    for split in ("train", "val", "test"):
        manifest_path = split_dir / f"{prefix}_{split}.parquet"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest_reports.append(
            validate_manifest_alignment(
                pd.read_parquet(manifest_path),
                obs_names,
                manifest_name=manifest_path.name,
            )
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "h5ad": str(h5ad_path),
        "cache": str(cache_path),
        "provenance": str(provenance_path),
        "cache_shape": list(cache.shape),
        "cache_dtype": str(cache.dtype),
        "finite": True,
        "unit_norm_max_abs_error": max_norm_error,
        "hashes": observed_hashes,
        "checkpoint_encoder_coverage_fraction": 1.0,
        "manifests": manifest_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--prefix", choices=("sciplex", "tahoe"), default="sciplex")
    parser.add_argument(
        "--wrapper",
        type=Path,
        default=Path("cytobridge/encoders/scgpt_wrapper.py"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    provenance = args.provenance or args.cache.with_suffix(
        args.cache.suffix + ".provenance.json"
    )
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {args.out}")
    report = validate_cache(
        args.h5ad.resolve(),
        args.cache.resolve(),
        provenance.resolve(),
        args.split_dir.resolve(),
        args.wrapper.resolve(),
        prefix=args.prefix,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
