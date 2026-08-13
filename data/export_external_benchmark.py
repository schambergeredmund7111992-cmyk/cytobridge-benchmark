"""Export the frozen benchmark to official chemCPA/biolord-compatible AnnData."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from eval.artifacts import sha256_file

OFFICIAL_SPLIT = {"train": "train", "validation": "test", "test": "ood"}
SHORT_SPLIT = {"train": "train", "validation": "val", "test": "test"}
CONTROL_SMILES = "CS(C)=O"


def external_row_id(split: str, role: str, source_cell_id: str) -> str:
    return f"{split}::{role}::{source_cell_id}"


def _log1p_counts(matrix):
    output = matrix.copy()
    if sparse.issparse(output):
        output.data = np.log1p(output.data)
        return output
    return np.log1p(np.asarray(output, dtype=np.float32))


def _append_split_rows(
    adata: ad.AnnData,
    manifest: pd.DataFrame,
    benchmark_split: str,
    *,
    official_split_override: str | None = None,
) -> tuple[list[int], list[dict]]:
    required = {
        "treated_cell_id",
        "input_control_cell_id",
        "drug_id",
        "context_id",
        "batch",
        "canonical_smiles",
    }
    if missing := required - set(manifest.columns):
        raise ValueError(
            f"{benchmark_split} manifest is missing columns: {sorted(missing)}"
        )
    if manifest["treated_cell_id"].astype(str).duplicated().any():
        raise ValueError(
            f"{benchmark_split} manifest has duplicate treated cell identifiers."
        )
    cell_to_index = {
        str(cell_id): index for index, cell_id in enumerate(adata.obs_names)
    }
    indices: list[int] = []
    records: list[dict] = []
    official_split = official_split_override or OFFICIAL_SPLIT[benchmark_split]
    for row in manifest.itertuples(index=False):
        treated_id = str(row.treated_cell_id)
        control_id = str(row.input_control_cell_id)
        if treated_id not in cell_to_index or control_id not in cell_to_index:
            raise ValueError(
                "External export manifest references a cell absent from AnnData."
            )
        indices.append(cell_to_index[treated_id])
        records.append(
            {
                "benchmark_row_id": external_row_id(
                    benchmark_split, "treated", treated_id
                ),
                "source_cell_id": treated_id,
                "input_control_row_id": external_row_id(
                    benchmark_split, "control", control_id
                ),
                "condition": str(row.drug_id),
                "drug": str(row.drug_id),
                "dose": 1.0,
                "dose_val": 1.0,
                "cell_type": str(row.context_id),
                "batch": str(row.batch),
                "canonical_smiles": str(row.canonical_smiles),
                "SMILES": str(row.canonical_smiles),
                "control": 0,
                "split": official_split,
                "split_ood": official_split,
                "benchmark_split": benchmark_split,
            }
        )
    controls = manifest[
        ["input_control_cell_id", "context_id", "batch"]
    ].drop_duplicates("input_control_cell_id")
    for row in controls.sort_values("input_control_cell_id").itertuples(index=False):
        control_id = str(row.input_control_cell_id)
        indices.append(cell_to_index[control_id])
        records.append(
            {
                "benchmark_row_id": external_row_id(
                    benchmark_split, "control", control_id
                ),
                "source_cell_id": control_id,
                "input_control_row_id": "NA",
                "condition": "DMSO",
                "drug": "DMSO",
                "dose": 0.0,
                "dose_val": 0.0,
                "cell_type": str(row.context_id),
                "batch": str(row.batch),
                "canonical_smiles": CONTROL_SMILES,
                "SMILES": CONTROL_SMILES,
                "control": 1,
                "split": official_split,
                "split_ood": official_split,
                "benchmark_split": benchmark_split,
            }
        )
    return indices, records


def _rdkit2d_features(obs: pd.DataFrame) -> tuple[np.ndarray, dict]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except (
        ImportError
    ) as exc:  # pragma: no cover - benchmark environment supplies RDKit
        raise ImportError("--with-rdkit2d requires RDKit.") from exc
    descriptor_names = [name for name, _ in Descriptors.descList]
    functions = [function for _, function in Descriptors.descList]
    unique_smiles = sorted(obs["canonical_smiles"].astype(str).unique())
    by_smiles = {}
    for smiles in unique_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid canonical SMILES in external export: {smiles!r}")
        values = np.asarray([function(molecule) for function in functions], dtype=float)
        by_smiles[smiles] = np.where(np.isfinite(values), values, 0.0)
    raw = np.stack([by_smiles[value] for value in obs["canonical_smiles"].astype(str)])
    training = obs["benchmark_split"].eq("train").to_numpy()
    train_std = raw[training].std(axis=0, ddof=0)
    keep = train_std > 0.001
    if not keep.any():
        raise ValueError("No nonconstant training-derived RDKit2D descriptors remain.")
    train_mean = raw[training][:, keep].mean(axis=0)
    train_std = raw[training][:, keep].std(axis=0, ddof=0)
    normalized = (raw[:, keep] - train_mean) / train_std
    dose = obs["dose_val"].to_numpy(dtype=float).reshape(-1, 1)
    manifest = {
        "descriptor_names": [
            name for name, selected in zip(descriptor_names, keep) if selected
        ],
        "fit_split": "train",
        "validation_or_test_response_used": False,
    }
    return np.concatenate([normalized, dose], axis=1).astype(np.float32), manifest


def export_external_benchmark(
    processed_h5ad: Path,
    split_dir: Path,
    output_h5ad: Path,
    *,
    with_rdkit2d: bool = False,
    selection_only: bool = False,
    final_refit: bool = False,
) -> dict:
    """Write one immutable external-model dataset without truth-reference controls."""
    if output_h5ad.exists():
        raise FileExistsError(
            f"Refusing to overwrite external benchmark: {output_h5ad}"
        )
    manifest_path = output_h5ad.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite external manifest: {manifest_path}"
        )
    source = ad.read_h5ad(processed_h5ad)
    if "counts" not in source.layers:
        raise ValueError(
            "Processed AnnData must retain raw counts in layers['counts']."
        )
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("External export requires unique cell and gene identifiers.")

    all_indices: list[int] = []
    all_records: list[dict] = []
    source_paths = {"processed_h5ad": processed_h5ad}
    if selection_only and final_refit:
        raise ValueError("selection_only and final_refit are mutually exclusive.")
    included_splits = ("train", "validation") if selection_only else (
        "train",
        "validation",
        "test",
    )
    for benchmark_split in included_splits:
        short_name = SHORT_SPLIT[benchmark_split]
        path = split_dir / f"sciplex_{short_name}.parquet"
        source_paths[f"{benchmark_split}_manifest"] = path
        split_override = "train" if final_refit and benchmark_split == "validation" else None
        indices, records = _append_split_rows(
            source,
            pd.read_parquet(path),
            benchmark_split,
            official_split_override=split_override,
        )
        all_indices.extend(indices)
        all_records.extend(records)

    obs = pd.DataFrame.from_records(all_records).set_index(
        "benchmark_row_id", drop=False
    )
    if obs.index.duplicated().any():
        raise ValueError("External benchmark row identifiers must be unique.")
    counts = source.layers["counts"][np.asarray(all_indices, dtype=int)]
    external = ad.AnnData(
        X=_log1p_counts(counts),
        obs=obs,
        var=pd.DataFrame(index=source.var_names.astype(str)),
    )
    external.layers["counts"] = counts.copy()
    external.obs["cov_drug_dose_name"] = (
        external.obs["cell_type"].astype(str)
        + "_"
        + external.obs["condition"].astype(str)
        + "_"
        + external.obs["dose_val"].astype(str)
    )
    categories = sorted(external.obs["cov_drug_dose_name"].astype(str).unique())
    external.uns["benchmark_all_genes"] = {
        category: external.var_names.astype(str).to_numpy() for category in categories
    }
    descriptor_manifest = None
    if with_rdkit2d:
        external.obsm["rdkit2d_dose"], descriptor_manifest = _rdkit2d_features(
            external.obs
        )
    external.uns["cytobridge_external_contract"] = {
        "expression_space": "log1p_raw_counts",
        "truth_reference_controls_included": False,
        "input_reference_controls_only": True,
        "official_split_mapping": OFFICIAL_SPLIT,
    }
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    external.write_h5ad(output_h5ad)
    result = {
        "output_h5ad": str(output_h5ad),
        "output_h5ad_sha256": sha256_file(output_h5ad),
        "source_hashes": {
            name: sha256_file(path) for name, path in sorted(source_paths.items())
        },
        "n_rows": int(external.n_obs),
        "n_genes": int(external.n_vars),
        "rows_by_split_and_control": {
            f"{split}:{int(control)}": int(count)
            for (split, control), count in external.obs.groupby(
                ["benchmark_split", "control"], observed=True
            )
            .size()
            .items()
        },
        "expression_space": "log1p_raw_counts",
        "truth_reference_controls_included": False,
        "rdkit2d": descriptor_manifest,
        "included_benchmark_splits": list(included_splits),
        "test_responses_included": not selection_only,
        "train_validation_refit_union": final_refit,
    }
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-h5ad", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--with-rdkit2d", action="store_true")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Physically omit benchmark test responses during hyperparameter screening.",
    )
    parser.add_argument(
        "--final-refit",
        action="store_true",
        help="Map benchmark train and validation rows to official train; retain test as ood.",
    )
    args = parser.parse_args()
    result = export_external_benchmark(
        args.processed_h5ad,
        args.split_dir,
        args.out,
        with_rdkit2d=args.with_rdkit2d,
        selection_only=args.selection_only,
        final_refit=args.final_refit,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
