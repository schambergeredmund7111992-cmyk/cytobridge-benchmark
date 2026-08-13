"""Convert row-aligned external-model output through the shared target pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.export_external_benchmark import external_row_id
from eval.aggregation import aggregate_logfc_by_pair
from eval.package_artifact import load_targets


def import_external_predictions(
    predicted_log1p_path: Path,
    eval_manifest_path: Path,
    true_treated_path: Path,
    truth_control_path: Path,
    frozen_targets_path: Path,
    frozen_metadata_path: Path,
    output_path: Path,
) -> dict:
    """Validate external rows, aggregate, and align to the frozen pair ordering."""
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite imported predictions: {output_path}"
        )
    manifest = pd.read_parquet(eval_manifest_path)
    with np.load(predicted_log1p_path, allow_pickle=True) as payload:
        required = {"pred_log1p", "row_ids", "gene_ids"}
        if missing := required - set(payload.files):
            raise ValueError(
                f"External prediction NPZ is missing arrays: {sorted(missing)}"
            )
        pred_log1p = np.asarray(payload["pred_log1p"], dtype=float)
        row_ids = np.asarray(payload["row_ids"]).astype(str)
        prediction_genes = np.asarray(payload["gene_ids"]).astype(str)
    frozen_true, frozen_genes = load_targets(frozen_targets_path)
    frozen_metadata = pd.read_csv(frozen_metadata_path)
    expected_rows = np.asarray(
        [
            external_row_id("test", "treated", str(cell_id))
            for cell_id in manifest["treated_cell_id"]
        ],
        dtype=str,
    )
    if not np.array_equal(row_ids, expected_rows):
        raise ValueError(
            "External prediction row_ids differ from the frozen test manifest."
        )
    if not np.array_equal(prediction_genes, frozen_genes):
        raise ValueError(
            "External prediction genes differ from the frozen benchmark order."
        )
    expected_shape = (len(manifest), len(frozen_genes))
    if pred_log1p.shape != expected_shape or not np.isfinite(pred_log1p).all():
        raise ValueError(
            "External pred_log1p must be finite and match test rows by genes."
        )
    negative_fraction = float((pred_log1p < 0).mean())
    predicted_counts = np.expm1(np.maximum(pred_log1p, 0.0))
    true_treated = np.load(true_treated_path)
    truth_control = np.load(truth_control_path)
    pair_pred, reconstructed_true, generated_metadata = aggregate_logfc_by_pair(
        predicted_counts,
        true_treated,
        truth_control,
        manifest["drug_id"],
        manifest["context_id"],
    )
    generated_index = generated_metadata.set_index("pair_id")
    frozen_pair_ids = frozen_metadata["pair_id"].astype(str).tolist()
    if set(generated_index.index.astype(str)) != set(frozen_pair_ids):
        raise ValueError("Aggregated external pairs differ from frozen target pairs.")
    order = generated_index.loc[frozen_pair_ids, "pair_index"].to_numpy(dtype=int)
    pair_pred = pair_pred[order]
    reconstructed_true = reconstructed_true[order]
    if reconstructed_true.shape != frozen_true.shape or not np.allclose(
        reconstructed_true, frozen_true, rtol=1.0e-5, atol=1.0e-6
    ):
        raise ValueError(
            "True-count reconstruction does not match the frozen target artifact."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pred=pair_pred.astype(np.float32),
        pair_ids=np.asarray(frozen_pair_ids, dtype=str),
        gene_ids=frozen_genes,
    )
    result = {
        "rows": int(len(manifest)),
        "pairs": int(len(pair_pred)),
        "genes": int(len(frozen_genes)),
        "prediction_space_received": "log1p_raw_counts",
        "negative_prediction_fraction_clipped_to_zero": negative_fraction,
        "frozen_truth_reconstruction_verified": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-log1p", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--true-treated", type=Path, required=True)
    parser.add_argument("--truth-control", type=Path, required=True)
    parser.add_argument("--frozen-targets", type=Path, required=True)
    parser.add_argument("--frozen-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = import_external_predictions(
        args.predicted_log1p,
        args.eval_manifest,
        args.true_treated,
        args.truth_control,
        args.frozen_targets,
        args.frozen_metadata,
        args.out,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
