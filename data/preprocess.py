"""Leakage-safe sci-Plex benchmark preprocessing.

Order is enforced as metadata/QC -> frozen drug split -> training-only gene fit -> application
to validation/test. Model-input and truth-reference vehicle pools are disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from data.benchmark_splits import (
    MetadataColumns,
    SplitResult,
    build_sciplex_eligibility,
    make_drug_disjoint_v2,
    make_scaffold_disjoint_v2,
    make_vehicle_reference_pools,
)
from eval.aggregation import aggregate_logfc_by_pair
from eval.artifacts import sha256_file
from eval.metrics import derive_context_gene_panels

try:
    from scipy import sparse
except ImportError:  # pragma: no cover - production environment includes scipy
    sparse = None


def _load_scanpy():
    """Import scanpy only for production preprocessing, not lightweight helper tests."""
    try:
        import scanpy
    except ImportError as exc:  # pragma: no cover - production dependency
        raise ImportError("scanpy is required to run sci-Plex preprocessing.") from exc
    return scanpy


def assert_raw_count_matrix(matrix) -> None:
    """Reject normalized/transformed inputs before they contaminate target construction."""
    values = (
        matrix.data
        if sparse is not None and sparse.issparse(matrix)
        else np.asarray(matrix).ravel()
    )
    if values.size == 0:
        raise ValueError("Expression matrix contains no non-zero counts.")
    sample = np.asarray(values[: min(values.size, 100_000)], dtype=float)
    if not np.isfinite(sample).all() or (sample < 0).any():
        raise ValueError(
            "Expression input must contain finite non-negative raw counts."
        )
    if not np.allclose(sample, np.rint(sample), atol=1e-6, rtol=0.0):
        raise ValueError("Expression input is not integer-like raw count data.")


def cell_qc_mask(
    counts,
    gene_ids: np.ndarray,
    *,
    min_genes: int = 200,
    max_mito_percent: float = 20.0,
) -> np.ndarray:
    """Compute label-free per-cell QC without fitting on held-out responses."""
    if sparse is not None and sparse.issparse(counts):
        detected = np.asarray(counts.getnnz(axis=1)).ravel()
        total = np.asarray(counts.sum(axis=1)).ravel()
        mito_mask = np.char.startswith(np.char.upper(gene_ids.astype(str)), "MT-")
        mito = (
            np.asarray(counts[:, mito_mask].sum(axis=1)).ravel()
            if mito_mask.any()
            else 0.0
        )
    else:
        dense = np.asarray(counts)
        detected = (dense > 0).sum(axis=1)
        total = dense.sum(axis=1)
        mito_mask = np.char.startswith(np.char.upper(gene_ids.astype(str)), "MT-")
        mito = (
            dense[:, mito_mask].sum(axis=1) if mito_mask.any() else np.zeros(len(dense))
        )
    mito_percent = np.divide(
        100.0 * mito,
        total,
        out=np.zeros_like(total, dtype=float),
        where=total > 0,
    )
    return (detected >= int(min_genes)) & (mito_percent < float(max_mito_percent))


def load_smiles_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing SMILES table {path}. Provide a CSV with header "
            "drug_id,smiles (or drug_id,canonical_smiles)."
        )
    table = pd.read_csv(path)
    smiles_column = "canonical_smiles" if "canonical_smiles" in table else "smiles"
    required = {"drug_id", smiles_column}
    if missing := required - set(table.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if table["drug_id"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate drug identifiers.")
    return dict(
        zip(table["drug_id"].astype(str), table[smiles_column].astype(str), strict=True)
    )


def _dose_to_um(value: object, unit: object) -> float:
    factors = {"nm": 1.0e-3, "um": 1.0, "µm": 1.0, "mm": 1.0e3}
    normalized = str(unit).strip().lower()
    if normalized not in factors:
        raise ValueError(f"Unsupported sci-Plex dose unit {unit!r}.")
    return float(value) * factors[normalized]


def build_selection_metadata(
    adata,
    smiles_by_drug: Mapping[str, str],
    *,
    drug_col: str,
    context_col: str,
    batch_col: str,
    dose_col: str,
    dose_unit_col: str,
    time_col: str,
    control_label: str,
) -> pd.DataFrame:
    """Normalize raw obs fields to the response-independent eligibility schema."""
    required = {drug_col, context_col, batch_col, dose_col, dose_unit_col, time_col}
    if missing := required - set(adata.obs.columns):
        raise ValueError(f"Input h5ad obs is missing columns: {sorted(missing)}")
    if not adata.obs_names.is_unique:
        raise ValueError("Input h5ad cell identifiers must be unique.")
    output = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "drug_id": adata.obs[drug_col].astype(str).to_numpy(),
            "cell_line": adata.obs[context_col].astype(str).to_numpy(),
            "batch": adata.obs[batch_col].astype(str).to_numpy(),
            "time_h": pd.to_numeric(adata.obs[time_col], errors="raise").to_numpy(),
        }
    )
    output["is_control"] = output["drug_id"].eq(str(control_label))
    output["smiles"] = output["drug_id"].map(smiles_by_drug)
    output.loc[output["is_control"], "smiles"] = None
    output["dose_um"] = [
        _dose_to_um(value, unit)
        for value, unit in zip(adata.obs[dose_col], adata.obs[dose_unit_col])
    ]
    return output


def fit_training_hvg(
    adata,
    training_cell_ids: set[str],
    *,
    n_hvg: int = 3000,
    min_cells_per_gene: int = 3,
) -> list[str]:
    """Fit the feature space on explicitly identified training cells only."""
    sc = _load_scanpy()
    if adata.n_vars <= n_hvg:
        raise ValueError(
            f"Input has only {adata.n_vars} genes for n_hvg={n_hvg}; provide the full raw gene "
            "space rather than the legacy preselected 3000-gene h5ad."
        )
    mask = adata.obs_names.astype(str).isin(training_cell_ids)
    if not mask.any():
        raise ValueError("No cells matched the frozen training set for HVG fitting.")
    training = adata[mask].copy()
    sc.pp.filter_genes(training, min_cells=min_cells_per_gene)
    try:
        sc.pp.highly_variable_genes(
            training,
            n_top_genes=min(n_hvg, training.n_vars),
            flavor="seurat_v3",
            inplace=True,
        )
    except ImportError as exc:
        raise ImportError(
            "Seurat-v3 HVG fitting requires scikit-misc in the preprocessing environment."
        ) from exc
    selected = set(training.var_names[training.var["highly_variable"]].astype(str))
    ordered = [str(gene) for gene in adata.var_names if str(gene) in selected]
    if len(ordered) != min(n_hvg, training.n_vars):
        raise RuntimeError(
            "Training-only HVG selection returned an unexpected gene count."
        )
    return ordered


def _dense_rows(matrix, indices: np.ndarray) -> np.ndarray:
    subset = matrix[indices]
    if sparse is not None and sparse.issparse(subset):
        subset = subset.toarray()
    return np.asarray(subset, dtype=np.float32)


def _save_rows_npy(
    matrix,
    indices: np.ndarray,
    path: Path,
    *,
    chunk_size: int = 1024,
) -> None:
    """Write selected rows without materializing a full dense split in RAM."""
    row_indices = np.asarray(indices, dtype=int)
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(row_indices), int(matrix.shape[1])),
    )
    for start in range(0, len(row_indices), chunk_size):
        stop = min(start + chunk_size, len(row_indices))
        output[start:stop] = _dense_rows(matrix, row_indices[start:stop])
    output.flush()
    del output


def _save_vector_rows_npy(
    keys: list[tuple[str, str]],
    vectors: Mapping[tuple[str, str], np.ndarray],
    path: Path,
    *,
    chunk_size: int = 1024,
) -> None:
    n_genes = int(next(iter(vectors.values())).shape[0])
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(keys), n_genes),
    )
    for start in range(0, len(keys), chunk_size):
        stop = min(start + chunk_size, len(keys))
        output[start:stop] = np.stack([vectors[key] for key in keys[start:stop]])
    output.flush()
    del output


def _choose_input_control(
    treated_cell_id: str, candidate_ids: list[str], seed: int
) -> str:
    if not candidate_ids:
        raise ValueError(
            f"No input-reference controls for treated cell {treated_cell_id!r}."
        )
    return min(
        candidate_ids,
        key=lambda cell_id: hashlib.sha256(
            f"{seed}\0{treated_cell_id}\0{cell_id}".encode()
        ).hexdigest(),
    )


def _build_split_arrays(
    processed,
    treated_table: pd.DataFrame,
    pool_assignments: pd.DataFrame,
    split_assignments: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int,
    prefix: str = "sciplex",
) -> None:
    cell_to_index = {
        str(cell_id): index for index, cell_id in enumerate(processed.obs_names)
    }
    split_by_drug = (
        split_assignments.set_index("drug_id")["split"].astype(str).to_dict()
    )
    pools = pool_assignments.copy()
    input_by_stratum = {
        (str(context), str(batch)): sorted(group["cell_id"].astype(str))
        for (context, batch), group in pools[pools["pool"].eq("vehicle_input")].groupby(
            ["context", "batch"], sort=True
        )
    }
    truth_by_stratum = {
        (str(context), str(batch)): sorted(group["cell_id"].astype(str))
        for (context, batch), group in pools[pools["pool"].eq("vehicle_truth")].groupby(
            ["context", "batch"], sort=True
        )
    }
    raw_counts = processed.layers["counts"]
    truth_pseudobulk = {}
    for stratum, cell_ids in truth_by_stratum.items():
        indices = np.asarray(
            [cell_to_index[cell_id] for cell_id in cell_ids], dtype=int
        )
        truth_pseudobulk[stratum] = _dense_rows(raw_counts, indices).mean(axis=0)

    for split_name in ("train", "val", "test"):
        split_table = treated_table[
            treated_table["drug_id"].map(split_by_drug).eq(split_name)
        ].copy()
        split_table = split_table.sort_values("cell_id").reset_index(drop=True)
        records = []
        treated_indices = []
        input_indices = []
        truth_strata = []
        for row in split_table.itertuples(index=False):
            stratum = (str(row.cell_line), str(row.batch))
            treated_id = str(row.cell_id)
            input_id = _choose_input_control(
                treated_id, input_by_stratum.get(stratum, []), seed
            )
            if stratum not in truth_pseudobulk:
                raise ValueError(
                    f"No truth-reference vehicle pseudobulk for stratum {stratum}."
                )
            treated_index = cell_to_index[treated_id]
            input_index = cell_to_index[input_id]
            records.append(
                {
                    "cell_idx": treated_index,
                    "control_cell_idx": input_index,
                    "drug_id": str(row.drug_id),
                    "cell_line": str(row.cell_line),
                    "context_id": str(row.cell_line),
                    "batch": str(row.batch),
                    "treated_cell_id": treated_id,
                    "input_control_cell_id": input_id,
                    "canonical_smiles": str(row.canonical_smiles),
                    "split": "validation" if split_name == "val" else split_name,
                }
            )
            treated_indices.append(treated_index)
            input_indices.append(input_index)
            truth_strata.append(stratum)
        if not records:
            raise ValueError(f"Frozen split {split_name!r} contains no treated cells.")
        manifest = pd.DataFrame.from_records(records)
        manifest.to_parquet(output_dir / f"{prefix}_{split_name}.parquet", index=False)
        _save_rows_npy(
            raw_counts,
            np.asarray(treated_indices, dtype=int),
            output_dir / f"{prefix}_{split_name}_treated_counts.npy",
        )
        _save_rows_npy(
            raw_counts,
            np.asarray(input_indices, dtype=int),
            output_dir / f"{prefix}_{split_name}_input_control_counts.npy",
        )
        _save_vector_rows_npy(
            truth_strata,
            truth_pseudobulk,
            output_dir / f"{prefix}_{split_name}_truth_control_counts.npy",
        )


def _write_pair_targets_and_panels(
    split_dir: Path,
    gene_ids: list[str],
    *,
    prefix: str = "sciplex",
    dataset_name: str = "sci-plex",
) -> None:
    pair_targets: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for split_name in ("train", "val", "test"):
        manifest = pd.read_parquet(split_dir / f"{prefix}_{split_name}.parquet")
        treated = np.load(
            split_dir / f"{prefix}_{split_name}_treated_counts.npy", mmap_mode="r"
        )
        truth_control = np.load(
            split_dir / f"{prefix}_{split_name}_truth_control_counts.npy", mmap_mode="r"
        )
        _, true_logfc, metadata = aggregate_logfc_by_pair(
            treated,
            treated,
            truth_control,
            manifest["drug_id"],
            manifest["context_id"],
        )
        target_split = "validation" if split_name == "val" else split_name
        metadata["split"] = target_split
        metadata["dataset"] = dataset_name
        np.savez_compressed(
            split_dir / f"{split_name}_targets.npz",
            true=true_logfc.astype(np.float32),
            gene_ids=np.asarray(gene_ids),
        )
        metadata.to_csv(split_dir / f"{split_name}_targets_metadata.csv", index=False)
        pair_targets[split_name] = (true_logfc, metadata)

    train_true, train_metadata = pair_targets["train"]
    panels = derive_context_gene_panels(
        train_true, train_metadata["context_id"], top_k=500
    )
    panel_payload = {
        context: indices.tolist() for context, indices in sorted(panels.items())
    }
    (split_dir / "training_gene_panels.json").write_text(
        json.dumps(panel_payload, indent=2) + "\n"
    )
    gene_scale = np.maximum(np.std(train_true, axis=0, ddof=0), 1.0e-3).astype(
        np.float32
    )
    np.save(split_dir / "training_gene_scale.npy", gene_scale)


def build_protocol_dataset(
    adata,
    selection_metadata: pd.DataFrame,
    eligibility,
    pools,
    split_result: SplitResult,
    output_dir: Path,
    *,
    n_hvg: int,
    seed: int,
) -> None:
    sc = _load_scanpy()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing protocol directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir()

    split_result.assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_result.manifest, indent=2, sort_keys=True) + "\n"
    )
    pools.assignments.to_csv(output_dir / "vehicle_pool_assignments.csv", index=False)
    (output_dir / "vehicle_pool_manifest.json").write_text(
        json.dumps(pools.manifest, indent=2, sort_keys=True) + "\n"
    )
    eligibility.attrition.to_csv(output_dir / "drug_attrition.csv", index=False)
    eligibility.compounds.to_csv(output_dir / "eligible_compounds.csv", index=False)

    split_by_drug = split_result.assignments.set_index("drug_id")["split"].to_dict()
    training_treated = eligibility.eligible_treated_cells[
        eligibility.eligible_treated_cells["drug_id"].map(split_by_drug).eq("train")
    ]
    train_strata = set(
        training_treated[["cell_line", "batch"]].itertuples(index=False, name=None)
    )
    training_vehicle_ids = set(
        selection_metadata.loc[
            selection_metadata["is_control"]
            & [
                (context, batch) in train_strata
                for context, batch in selection_metadata[
                    ["cell_line", "batch"]
                ].itertuples(index=False, name=None)
            ],
            "cell_id",
        ].astype(str)
    )
    training_cell_ids = (
        set(training_treated["cell_id"].astype(str)) | training_vehicle_ids
    )
    gene_ids = fit_training_hvg(adata, training_cell_ids, n_hvg=n_hvg)

    relevant_ids = set(eligibility.eligible_treated_cells["cell_id"].astype(str)) | set(
        eligibility.matching_vehicle_cells["cell_id"].astype(str)
    )
    relevant_mask = adata.obs_names.astype(str).isin(relevant_ids)
    gene_index = adata.var_names.astype(str).get_indexer(gene_ids)
    if (gene_index < 0).any():
        raise RuntimeError(
            "Frozen training genes are missing from the raw AnnData object."
        )
    processed = adata[relevant_mask, gene_index].copy()
    processed.layers["counts"] = processed.X.copy()
    sc.pp.normalize_total(processed, target_sum=1.0e4)
    sc.pp.log1p(processed)
    processed.obs["drug"] = (
        selection_metadata.set_index("cell_id")
        .loc[processed.obs_names.astype(str), "drug_id"]
        .to_numpy()
    )
    processed.obs["cell_line"] = (
        selection_metadata.set_index("cell_id")
        .loc[processed.obs_names.astype(str), "cell_line"]
        .to_numpy()
    )
    processed.obs["batch"] = (
        selection_metadata.set_index("cell_id")
        .loc[processed.obs_names.astype(str), "batch"]
        .to_numpy()
    )
    processed.write_h5ad(output_dir / "sciplex_processed.h5ad")
    (output_dir / "gene_ids.txt").write_text("\n".join(gene_ids) + "\n")
    (output_dir / "hvg_fit.json").write_text(
        json.dumps(
            {
                "fit_split": "train",
                "n_hvg": len(gene_ids),
                "training_cells": len(training_cell_ids),
                "training_cell_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(training_cell_ids)).encode()
                ).hexdigest(),
                "validation_or_test_cells_used": False,
            },
            indent=2,
        )
        + "\n"
    )
    _build_split_arrays(
        processed,
        eligibility.eligible_treated_cells,
        pools.assignments,
        split_result.assignments,
        split_dir,
        seed=seed,
    )
    _write_pair_targets_and_panels(split_dir, gene_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe sci-Plex benchmark data."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    sc = _load_scanpy()
    input_path = Path(cfg["input_h5ad"])
    out_dir = Path(cfg["out_dir"])
    if out_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark output: {out_dir}"
        )
    adata = sc.read_h5ad(input_path)
    if not adata.var_names.is_unique:
        raise ValueError("Input h5ad gene identifiers must be unique.")
    assert_raw_count_matrix(adata.X)
    qc = cfg.get("qc", {})
    mask = cell_qc_mask(
        adata.X,
        adata.var_names.to_numpy(),
        min_genes=int(qc.get("min_genes", 200)),
        max_mito_percent=float(qc.get("max_mito_percent", 20.0)),
    )
    adata = adata[mask].copy()
    columns = cfg["columns"]
    metadata = build_selection_metadata(
        adata,
        load_smiles_map(Path(cfg["smiles_csv"])),
        drug_col=columns["drug"],
        context_col=columns["context"],
        batch_col=columns["batch"],
        dose_col=columns["dose"],
        dose_unit_col=columns["dose_unit"],
        time_col=columns["time"],
        control_label=cfg.get("control_label", "DMSO"),
    )
    eligibility = build_sciplex_eligibility(metadata, columns=MetadataColumns())
    pools = make_vehicle_reference_pools(metadata, columns=MetadataColumns())
    split_results = {
        "drug_disjoint_v2": make_drug_disjoint_v2(eligibility),
        "scaffold_disjoint_v2": make_scaffold_disjoint_v2(eligibility),
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        (staging / "source_provenance.json").write_text(
            json.dumps(
                {
                    "input_h5ad": str(input_path),
                    "input_h5ad_sha256": sha256_file(input_path),
                    "smiles_csv": str(cfg["smiles_csv"]),
                    "smiles_csv_sha256": sha256_file(cfg["smiles_csv"]),
                    "cells_before_qc": int(len(mask)),
                    "cells_after_qc": int(mask.sum()),
                },
                indent=2,
            )
            + "\n"
        )
        for protocol_name, split_result in split_results.items():
            build_protocol_dataset(
                adata,
                metadata,
                eligibility,
                pools,
                split_result,
                staging / protocol_name,
                n_hvg=int(cfg.get("n_hvg", 3000)),
                seed=int(cfg.get("seed", 20260710)),
            )
        staging.replace(out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "complete", "out_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
