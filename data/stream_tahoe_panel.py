"""Stream only preregistered Tahoe-100M cells into a compact sparse AnnData file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse


def rows_to_sparse_counts(
    rows: list[Mapping[str, object]],
    token_to_column: Mapping[int, int],
) -> sparse.csr_matrix:
    """Convert official sparse token/count rows, ignoring the unmapped CLS marker."""
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    for row_number, row in enumerate(rows):
        tokens = np.asarray(row["genes"], dtype=np.int64)
        counts = np.asarray(row["expressions"], dtype=np.float32)
        if tokens.ndim != 1 or counts.ndim != 1 or len(tokens) != len(counts):
            raise ValueError(
                f"Tahoe expression row {row_number} has misaligned sparse fields."
            )
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(
                f"Tahoe expression row {row_number} has invalid raw counts."
            )
        aggregated: dict[int, float] = {}
        for token, count in zip(tokens.tolist(), counts.tolist()):
            if int(token) not in token_to_column:
                continue
            column = int(token_to_column[int(token)])
            aggregated[column] = aggregated.get(column, 0.0) + float(count)
        for column in sorted(aggregated):
            indices.append(column)
            values.append(aggregated[column])
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(rows), len(token_to_column)),
    )


def take_selected_rows(
    stream: Iterable[Mapping[str, object]],
    selected_ids: set[str],
    *,
    max_records: int | None = None,
) -> tuple[list[Mapping[str, object]], int]:
    """Scan until every selected cell is found, with an optional hard scan limit."""
    if not selected_ids:
        raise ValueError("selected_ids must not be empty.")
    found: dict[str, Mapping[str, object]] = {}
    scanned = 0
    for row in stream:
        scanned += 1
        cell_id = str(row.get("BARCODE_SUB_LIB_ID", ""))
        if cell_id in selected_ids:
            if cell_id in found:
                raise ValueError(
                    f"Tahoe stream contains duplicate cell identifier {cell_id!r}."
                )
            found[cell_id] = row
            if len(found) == len(selected_ids):
                break
        if max_records is not None and scanned >= max_records:
            break
    missing = sorted(selected_ids - set(found))
    if missing:
        raise ValueError(
            f"Tahoe stream ended/limit reached with {len(missing)} selected cells missing; "
            f"examples={missing[:5]}."
        )
    return [found[cell_id] for cell_id in sorted(found)], scanned


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded Tahoe expression streaming.")
    parser.add_argument("--selected-cells", type=Path, required=True)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="2dc5790")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    selected = pd.read_parquet(args.selected_cells)
    if "cell_id" not in selected or selected["cell_id"].astype(str).duplicated().any():
        raise ValueError("selected-cells must contain unique cell_id values.")
    genes = pd.read_parquet(args.gene_metadata)
    required_gene = {"token_id", "gene_symbol", "ensembl_id"}
    if missing := required_gene - set(genes.columns):
        raise ValueError(f"gene metadata is missing columns: {sorted(missing)}")
    if (
        genes["token_id"].duplicated().any()
        or genes["gene_symbol"].astype(str).duplicated().any()
    ):
        raise ValueError("Tahoe token IDs and gene symbols must be unique.")
    genes = genes.sort_values("token_id").reset_index(drop=True)
    token_to_column = {
        int(token): column for column, token in enumerate(genes["token_id"].tolist())
    }

    from datasets import load_dataset

    stream = load_dataset(
        "tahoebio/Tahoe-100M",
        "expression_data",
        split="train",
        streaming=True,
        revision=args.dataset_revision,
    )
    rows, scanned = take_selected_rows(
        stream,
        set(selected["cell_id"].astype(str)),
        max_records=args.max_records,
    )
    counts = rows_to_sparse_counts(rows, token_to_column)
    by_id = selected.set_index(selected["cell_id"].astype(str), drop=False)
    ordered_ids = [str(row["BARCODE_SUB_LIB_ID"]) for row in rows]
    obs = by_id.loc[ordered_ids].reset_index(drop=True)

    import anndata as ad

    var = pd.DataFrame(
        {"ensembl_id": genes["ensembl_id"].astype(str).to_numpy()},
        index=pd.Index(genes["gene_symbol"].astype(str), name="gene_symbol"),
    )
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    selection_hash = hashlib.sha256(args.selected_cells.read_bytes()).hexdigest()
    metadata = {
        "dataset": "tahoebio/Tahoe-100M",
        "dataset_revision": args.dataset_revision,
        "selected_cells_sha256": selection_hash,
        "selected_cells": len(rows),
        "records_scanned": scanned,
        "shape": list(counts.shape),
    }
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
