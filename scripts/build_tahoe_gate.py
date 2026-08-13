"""Build the P3 Tahoe readiness gate from pinned-source, split, cache, and control evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.artifacts import sha256_file
from scripts.control_calibration import calibrate_controls

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
GATE_ROOT = PROJECT_ROOT / "experiments/gates"
TAHOE_RAW = CODE_ROOT / "data/raw/tahoe"
TAHOE_SELECTION = CODE_ROOT / "data/processed/tahoe_selection"
TAHOE = CODE_ROOT / "data/processed/tahoe_accept"
TAHOE_CACHE = CODE_ROOT / "data/cache/tahoe_accept"
PINNED_REVISION = "2dc5790"


def build_tahoe_gate(output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Tahoe gate: {output_path}")
    metadata_provenance = json.loads(
        (TAHOE_RAW / "metadata_provenance.json").read_text()
    )
    selection = json.loads((TAHOE_SELECTION / "selection.json").read_text())
    stream = json.loads((TAHOE_RAW / "selected_panel.provenance.json").read_text())
    coverage = json.loads((TAHOE / "gene_coverage.json").read_text())
    fit = json.loads((TAHOE / "fit/fit_manifest.json").read_text())
    scgpt = json.loads((TAHOE_CACHE / "scgpt_cache_validation.json").read_text())
    molformer = json.loads(
        (TAHOE_CACHE / "molformer_emb.npz.provenance.json").read_text()
    )
    if (
        metadata_provenance.get("revision") != PINNED_REVISION
        or metadata_provenance.get("expression_data_downloaded") is not False
        or stream.get("dataset_revision") != PINNED_REVISION
    ):
        raise ValueError("Tahoe source does not match pinned revision 2dc5790.")
    if (
        selection.get("highest_dose_only") is not True
        or selection.get("n_contexts") != 10
        or not 1 <= int(selection.get("n_drugs", 0)) <= 120
        or selection.get("cap_per_drug_context_dose_sample") != 64
        or selection.get("plate_matched_controls") is not True
        or selection.get("metadata_passes") != 2
    ):
        raise ValueError("Tahoe bounded-panel selection differs from protocol 1.4.")
    if float(coverage.get("coverage", 0.0)) < 0.90:
        raise ValueError("Tahoe gene coverage is below 90 percent.")
    if (
        fit.get("fit_splits") != ["train", "validation"]
        or fit.get("validation_used_for_checkpoint_selection_during_refit") is not False
    ):
        raise ValueError("Tahoe final-fit data does not use the frozen refit boundary.")
    if (
        scgpt.get("status") != "passed"
        or scgpt.get("checkpoint_encoder_coverage_fraction") != 1.0
        or len(scgpt.get("manifests", [])) != 3
        or any(
            not str(row.get("manifest", "")).startswith("tahoe_")
            for row in scgpt.get("manifests", [])
        )
    ):
        raise ValueError("Tahoe scGPT cache validation did not pass.")
    drug_count = int(selection["n_drugs"])
    if (
        molformer.get("revision")
        != "a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8"
        or molformer.get("all_finite") is not True
        or int(molformer.get("distinct_drug_rows", -1)) != drug_count
    ):
        raise ValueError("Tahoe MolFormer cache provenance did not pass.")
    pathway_names = [
        line
        for line in (TAHOE / "pathway_names.txt").read_text().splitlines()
        if line
    ]
    if len(pathway_names) != 50 or len(set(pathway_names)) != 50:
        raise ValueError("Tahoe Hallmark pathway set is not the frozen 50-set panel.")

    required = [
        TAHOE_RAW / "metadata_provenance.json",
        TAHOE_RAW / "selected_panel.h5ad",
        TAHOE_RAW / "selected_panel.provenance.json",
        TAHOE_SELECTION / "selected_cells.parquet",
        TAHOE_SELECTION / "selection.json",
        TAHOE / "source_provenance.json",
        TAHOE / "split_manifest.json",
        TAHOE / "vehicle_pool_manifest.json",
        TAHOE / "drug_smiles.csv",
        TAHOE / "pathway_names.txt",
        TAHOE / "fit/fit_manifest.json",
        TAHOE_CACHE / "scgpt_emb.npy",
        TAHOE_CACHE / "scgpt_emb.npy.provenance.json",
        TAHOE_CACHE / "scgpt_cache_validation.json",
        TAHOE_CACHE / "molformer_emb.npz",
        TAHOE_CACHE / "molformer_emb.npz.provenance.json",
    ]
    for split in ("train", "val", "test"):
        required.extend(
            [
                TAHOE / f"splits/tahoe_{split}.parquet",
                TAHOE / f"splits/tahoe_{split}_pathway_gsea.npy",
                TAHOE / f"splits/{split}_targets.npz",
                TAHOE / f"splits/{split}_targets_metadata.csv",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Tahoe readiness files are missing: {missing}")
    controls = calibrate_controls(TAHOE)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "dataset": "tahoebio/Tahoe-100M",
        "revision": PINNED_REVISION,
        "test_targets_opened": False,
        "n_drugs": drug_count,
        "n_contexts": 10,
        "gene_coverage": float(coverage["coverage"]),
        "control_calibration": controls,
        "source_hashes": {
            str(path.relative_to(CODE_ROOT)): sha256_file(path) for path in required
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=GATE_ROOT / "tahoe_ready.json")
    args = parser.parse_args()
    result = build_tahoe_gate(args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
