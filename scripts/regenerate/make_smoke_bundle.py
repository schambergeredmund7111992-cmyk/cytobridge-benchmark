"""Build a synthetic regeneration-inputs bundle for pipeline tests.

Layout mirrors the real bundle exactly (see scripts/export_regeneration_inputs.py),
with random matrices small enough to run in seconds. The synthetic truths obey
the construction invariants (pooled no-drug-info AUC == 0.5; per-pair anchor
inflated above chance).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.regenerate.inputs import CONFIGS

N_PAIRS = 27
N_GENES = 300
CELLS = ["A549", "K562", "MCF7"]
DRUGS = [
    "AG-490 (Tyrphostin B42)", "Celecoxib", "Fulvestrant",
    "Ramelteon", "SL-327", "SRT3025 HCl", "Thalidomide",
    "Tofacitinib Citrate", "Zileuton",
]
TRAIN_N = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(out: Path, *, zip_it: bool = True) -> Path:
    out = Path(out)
    for sub in ("e6e7", "logs", "baselines/ridge", "baselines/chemcpa",
                "baselines/biolord", "tahoe", "oracle", "replicates", "protocol"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    meta = pd.DataFrame(
        {"drug": DRUGS * 3, "cell_line": [c for c in CELLS for _ in DRUGS]}
    )
    base = rng.normal(size=(3, N_GENES))          # per-cell-line shared structure
    true_pooled = np.stack(
        [base[i // 9] + rng.normal(0, 0.05, N_GENES) for i in range(N_PAIRS)]
    ).astype(np.float32)
    delta = rng.normal(0, 0.05, size=(N_PAIRS, N_GENES)).astype(np.float32)
    true_perpair = true_pooled + delta

    manifest_files = []
    for tag in CONFIGS.values():
        pred_perpair = (
            0.1 * true_pooled + rng.normal(0, 0.05, (N_PAIRS, N_GENES))
        ).astype(np.float32)
        np.save(out / "e6e7" / f"logfc_pred_{tag}.npy", pred_perpair)
        np.save(out / "e6e7" / f"logfc_true_{tag}.npy", true_perpair)
        meta.to_csv(out / "e6e7" / f"logfc_meta_{tag}.csv", index=False)
        manifest_files.append(
            {"file": f"e6e7/logfc_pred_{tag}.npy",
             "sha256": _sha256(out / "e6e7" / f"logfc_pred_{tag}.npy"),
             "shape": [N_PAIRS, N_GENES]}
        )
        manifest_files.append(
            {"file": f"e6e7/logfc_true_{tag}.npy",
             "sha256": _sha256(out / "e6e7" / f"logfc_true_{tag}.npy"),
             "shape": [N_PAIRS, N_GENES]}
        )
        # training logs
        log_dir = out / "logs" / tag
        log_dir.mkdir(exist_ok=True)
        steps = np.arange(1, 11)
        pd.DataFrame(
            {"step": steps, "train/L_recon_step": 1000 * np.exp(-steps / 5),
             "train/L_contrast_step": 2.0 + 0.1 * steps}
        ).to_csv(log_dir / "metrics.csv", index=False)

    np.save(out / "e6e7" / "true_pooled.npy", true_pooled)
    np.save(out / "e6e7" / "delta.npy", delta)
    meta.to_csv(out / "e6e7" / "true_pooled_meta.csv", index=False)
    mean_train = np.stack(
        [base[i // 9] + rng.normal(0, 0.02, N_GENES) for i in range(N_PAIRS)]
    ).astype(np.float32)
    np.save(out / "e6e7" / "mean_predictor_train.npy", mean_train)
    for name, array in (("true_pooled", true_pooled), ("delta", delta),
                        ("mean_predictor_train", mean_train)):
        manifest_files.append(
            {"file": f"e6e7/{name}.npy",
             "sha256": _sha256(out / "e6e7" / f"{name}.npy"),
             "shape": list(array.shape)}
        )

    # baselines: synthetic self-space matrices with distinct signal levels
    for key, signal in (("ridge", 0.4), ("chemcpa", 0.0), ("biolord", 0.02)):
        pred = (signal * true_pooled + (1 - signal) * mean_train).astype(np.float32)
        true = true_pooled.astype(np.float32)
        np.save(out / "baselines" / key / "pred.npy", pred)
        np.save(out / "baselines" / key / "true.npy", true)
        meta.to_csv(out / "baselines" / key / "meta.csv", index=False)
        (out / "baselines" / key / "metrics.json").write_text(
            json.dumps({"note": "synthetic smoke baseline"}) + "\n"
        )

    # oracle inputs
    train_responses = rng.normal(size=(TRAIN_N, 3, N_GENES)).astype(np.float32)
    np.save(out / "oracle" / "training_responses.npy", train_responses)
    pd.DataFrame(
        {"drug_id": [f"train_{i}" for i in range(TRAIN_N)],
         "canonical_smiles": ["CCO" if i % 2 else "CCC" for i in range(TRAIN_N)],
         "vendor_target": [f"T{i % 5}" for i in range(TRAIN_N)]}
    ).to_csv(out / "oracle" / "training_drugs.csv", index=False)
    pd.DataFrame(
        {"drug_id": DRUGS, "canonical_smiles": ["CCO"] * 9,
         "vendor_target": [f"T{i}" for i in range(9)]}
    ).to_csv(out / "oracle" / "drugs_172.csv", index=False)

    # replicates
    np.save(out / "replicates" / "crossplate_rep1.npy",
            (true_pooled + rng.normal(0, 0.05, (N_PAIRS, N_GENES))).astype(np.float32))
    np.save(out / "replicates" / "crossplate_rep2.npy",
            (true_pooled + rng.normal(0, 0.05, (N_PAIRS, N_GENES))).astype(np.float32))

    # misc
    pd.DataFrame(
        [{"drug": drug, "cell_line": cell, "n_cells": 150 + i}
         for i, (drug, cell) in enumerate(zip(meta["drug"], meta["cell_line"]))]
        + [{"drug": "Vehicle", "cell_line": cell, "n_cells": 2000}
           for cell in CELLS]
    ).to_csv(out / "table3_cell_counts.csv", index=False)
    pd.DataFrame(
        {"component": ["L_recon", "L_contrast"], "value": [1003.4, 4.2]}
    ).to_csv(out / "loss_components.csv", index=False)
    (out / "pathway").mkdir(exist_ok=True)
    (out / "pathway" / "pathway_illusion.json").write_text(
        json.dumps({"on_diag_mean": 0.9484, "off_diag_mean": 0.9483,
                    "gap": 0.00006, "specificity_auc": 0.509}) + "\n"
    )
    pd.DataFrame({"drug_id": DRUGS[:5], "split": ["train"] * 5}
                 ).to_csv(out / "protocol" / "split_assignments.csv", index=False)
    (out / "protocol" / "gene_ids.txt").write_text(
        "\n".join(f"gene_{i}" for i in range(N_GENES)) + "\n"
    )

    manifest = {"schema_version": 1, "files": manifest_files,
                "selected_pooled_truth": "synthetic", "note": "smoke bundle"}
    (out / "inputs_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if zip_it:
        zip_path = out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(out))
        return zip_path
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("smoke_bundle"))
    args = parser.parse_args()
    print(build(args.out))
