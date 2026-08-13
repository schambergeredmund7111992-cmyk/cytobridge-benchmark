"""Build the frozen Tahoe train/validation/test benchmark from a bounded cell panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from data.benchmark_splits import (
    VEHICLE_SEED,
    bemis_murcko_scaffold,
    canonical_pool_sha256,
    canonicalize_smiles,
    make_drug_disjoint_v2,
)
from data.preprocess import (
    _build_split_arrays,
    _write_pair_targets_and_panels,
    assert_raw_count_matrix,
)
from eval.artifacts import sha256_file, sha256_gene_panel


def align_count_matrix(
    counts,
    source_gene_ids: np.ndarray,
    target_gene_ids: np.ndarray,
) -> tuple[sparse.csr_matrix, dict]:
    """Align sparse raw counts to a frozen gene order, inserting zero missing columns."""
    source = np.asarray(source_gene_ids).astype(str)
    target = np.asarray(target_gene_ids).astype(str)
    if len(set(source.tolist())) != len(source) or len(set(target.tolist())) != len(
        target
    ):
        raise ValueError("Source and target gene identifiers must each be unique.")
    source_lookup = {gene: index for index, gene in enumerate(source)}
    present_target = np.asarray(
        [index for index, gene in enumerate(target) if gene in source_lookup], dtype=int
    )
    source_columns = np.asarray(
        [source_lookup[target[index]] for index in present_target]
    )
    matrix = sparse.csr_matrix(counts)
    subset = matrix[:, source_columns].tocoo()
    aligned = sparse.csr_matrix(
        (subset.data, (subset.row, present_target[subset.col])),
        shape=(matrix.shape[0], len(target)),
        dtype=np.float32,
    )
    coverage = float(len(present_target) / len(target))
    return aligned, {
        "source_genes": int(len(source)),
        "target_genes": int(len(target)),
        "matched_genes": int(len(present_target)),
        "coverage": coverage,
        "missing_target_genes": target[
            ~np.isin(np.arange(len(target)), present_target)
        ].tolist(),
    }


def make_tahoe_vehicle_pools(
    controls: pd.DataFrame,
    *,
    seed: int = VEHICLE_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Split plate/context controls into deterministic, disjoint reference pools."""
    required = {"cell_id", "cell_line", "batch"}
    if missing := required - set(controls.columns):
        raise ValueError(f"Tahoe controls are missing columns: {sorted(missing)}")
    records = []
    for (context, batch), group in controls.groupby(["cell_line", "batch"], sort=True):
        cell_ids = sorted(group["cell_id"].astype(str))
        if len(cell_ids) < 2:
            raise ValueError(
                f"Tahoe control stratum {(context, batch)} has fewer than two cells."
            )
        ordered = sorted(
            cell_ids,
            key=lambda value: hashlib.sha256(
                f"{seed}|tahoe-vehicle|{context}|{batch}|{value}".encode()
            ).hexdigest(),
        )
        cut = len(ordered) // 2
        for index, cell_id in enumerate(ordered):
            records.append(
                {
                    "cell_id": cell_id,
                    "context": str(context),
                    "batch": str(batch),
                    "pool": "vehicle_input" if index < cut else "vehicle_truth",
                }
            )
    assignments = pd.DataFrame(records).sort_values("cell_id").reset_index(drop=True)
    manifest = {
        "protocol_name": "tahoe_vehicle_input_truth_v1",
        "seed": int(seed),
        "strata": ["cell_line_id", "plate"],
        "assignment_sha256": canonical_pool_sha256(
            assignments,
            protocol_name="tahoe_vehicle_input_truth_v1",
            seed=seed,
        ),
        "pool_counts": {
            name: int(assignments["pool"].eq(name).sum())
            for name in ("vehicle_input", "vehicle_truth")
        },
    }
    return assignments, manifest


def _normalized_log1p(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if (totals <= 0).any():
        raise ValueError("Tahoe selected panel contains a zero-library cell.")
    scaled = sparse.diags(1.0e4 / totals) @ counts
    scaled = scaled.tocsr().astype(np.float32)
    scaled.data = np.log1p(scaled.data)
    return scaled


def _compound_table(treated: pd.DataFrame) -> pd.DataFrame:
    pairs = treated[["drug_id", "canonical_smiles"]].drop_duplicates()
    if pairs["drug_id"].astype(str).duplicated().any():
        raise ValueError("A Tahoe drug maps to multiple SMILES values.")
    records = []
    for row in pairs.itertuples(index=False):
        canonical = canonicalize_smiles(row.canonical_smiles)
        if canonical is None:
            raise ValueError(
                f"Invalid Tahoe SMILES survived selection for {row.drug_id!r}."
            )
        records.append(
            {
                "drug_id": str(row.drug_id),
                "canonical_smiles": canonical,
                "scaffold_smiles": bemis_murcko_scaffold(canonical),
                "eligible": True,
            }
        )
    return pd.DataFrame(records)


def build_tahoe_protocol_dataset(
    source: ad.AnnData,
    target_gene_ids: list[str],
    output_dir: Path,
    *,
    seed: int = 20260710,
    minimum_gene_coverage: float = 0.90,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite Tahoe protocol output: {output_dir}"
        )
    required_obs = {
        "cell_id",
        "role",
        "drug",
        "cell_line_id",
        "plate",
        "canonical_smiles",
    }
    if missing := required_obs - set(source.obs.columns):
        raise ValueError(f"Tahoe AnnData is missing columns: {sorted(missing)}")
    if "counts" not in source.layers:
        raise ValueError("Tahoe AnnData must retain raw counts in layers['counts'].")
    assert_raw_count_matrix(source.layers["counts"])
    cell_ids = source.obs["cell_id"].astype(str)
    if cell_ids.duplicated().any():
        raise ValueError("Tahoe cell_id values must be unique.")
    source = source.copy()
    source.obs_names = cell_ids
    target = np.asarray(target_gene_ids, dtype=str)
    aligned_counts, coverage = align_count_matrix(
        source.layers["counts"], source.var_names.to_numpy(), target
    )
    if coverage["coverage"] < minimum_gene_coverage:
        raise ValueError(
            f"Tahoe gene overlap {coverage['coverage']:.3f} is below "
            f"{minimum_gene_coverage:.3f}."
        )
    processed_obs = source.obs.copy()
    processed_obs["drug"] = processed_obs["drug"].astype(str)
    processed_obs["cell_line"] = processed_obs["cell_line_id"].astype(str)
    processed_obs["batch"] = processed_obs["plate"].astype(str)
    processed = ad.AnnData(
        X=_normalized_log1p(aligned_counts),
        obs=processed_obs,
        var=pd.DataFrame(index=pd.Index(target, name="gene_id")),
    )
    processed.layers["counts"] = aligned_counts
    treated_obs = processed.obs[processed.obs["role"].astype(str).eq("treated")].copy()
    treated_table = pd.DataFrame(
        {
            "cell_id": treated_obs.index.astype(str),
            "drug_id": treated_obs["drug"].astype(str).to_numpy(),
            "cell_line": treated_obs["cell_line"].astype(str).to_numpy(),
            "batch": treated_obs["batch"].astype(str).to_numpy(),
            "canonical_smiles": treated_obs["canonical_smiles"].astype(str).to_numpy(),
        }
    )
    compounds = _compound_table(treated_table)
    split_result = make_drug_disjoint_v2(compounds, seed=seed)
    canonical_by_drug = split_result.assignments.set_index("drug_id")[
        "canonical_smiles"
    ]
    treated_table["canonical_smiles"] = treated_table["drug_id"].map(canonical_by_drug)
    control_obs = processed.obs[processed.obs["role"].astype(str).eq("control")]
    control_table = pd.DataFrame(
        {
            "cell_id": control_obs.index.astype(str),
            "cell_line": control_obs["cell_line"].astype(str).to_numpy(),
            "batch": control_obs["batch"].astype(str).to_numpy(),
        }
    )
    pools, pool_manifest = make_tahoe_vehicle_pools(control_table)

    output_dir.mkdir(parents=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir()
    processed.write_h5ad(output_dir / "tahoe_processed.h5ad")
    (output_dir / "gene_ids.txt").write_text("\n".join(target.tolist()) + "\n")
    (output_dir / "gene_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n"
    )
    split_result.assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    drug_smiles = compounds[["drug_id", "canonical_smiles"]].copy()
    drug_smiles["smiles"] = drug_smiles["canonical_smiles"]
    drug_smiles = drug_smiles[
        ["drug_id", "smiles", "canonical_smiles"]
    ].sort_values("drug_id", kind="mergesort")
    drug_smiles.to_csv(output_dir / "drug_smiles.csv", index=False)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_result.manifest, indent=2, sort_keys=True) + "\n"
    )
    pools.to_csv(output_dir / "vehicle_pool_assignments.csv", index=False)
    (output_dir / "vehicle_pool_manifest.json").write_text(
        json.dumps(pool_manifest, indent=2, sort_keys=True) + "\n"
    )
    _build_split_arrays(
        processed,
        treated_table,
        pools,
        split_result.assignments,
        split_dir,
        seed=seed,
        prefix="tahoe",
    )
    _write_pair_targets_and_panels(
        split_dir,
        target.tolist(),
        prefix="tahoe",
        dataset_name="tahoe-100m",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-h5ad", type=Path, required=True)
    parser.add_argument("--sciplex-gene-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--minimum-gene-coverage", type=float, default=0.90)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(
            f"Refusing to overwrite Tahoe benchmark output: {args.out}"
        )
    target_genes = [
        line.strip() for line in args.sciplex_gene_ids.read_text().splitlines() if line
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.out.name}.tmp-", dir=args.out.parent)
    )
    try:
        build_tahoe_protocol_dataset(
            ad.read_h5ad(args.selected_h5ad),
            target_genes,
            staging / "benchmark",
            seed=args.seed,
            minimum_gene_coverage=args.minimum_gene_coverage,
        )
        benchmark = staging / "benchmark"
        (benchmark / "source_provenance.json").write_text(
            json.dumps(
                {
                    "selected_h5ad": str(args.selected_h5ad),
                    "selected_h5ad_sha256": sha256_file(args.selected_h5ad),
                    "sciplex_gene_ids": str(args.sciplex_gene_ids),
                    "sciplex_gene_ids_sha256": sha256_file(args.sciplex_gene_ids),
                    "ordered_gene_panel_sha256": sha256_gene_panel(target_genes),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        benchmark.replace(args.out)
        shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "complete", "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
