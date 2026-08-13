"""Leakage-safe Morgan-fingerprint Ridge baseline.

Target construction is performed by the shared benchmark pipeline. This module receives aligned
pseudobulk logFC targets, selects ``alpha`` on validation drugs only, and never accepts test
targets in a fitting function.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from eval.artifacts import sha256_file
from eval.metrics import conditional_rank_score, pair_own_spearman
from eval.package_artifact import load_targets

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover - remote benchmark environment provides RDKit
    Chem = None
    DataStructs = None
    AllChem = None


DEFAULT_ALPHA_GRID = (
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
)


@dataclass(frozen=True)
class RidgeSelection:
    alpha: float
    trials: pd.DataFrame


def canonicalize_smiles(smiles: str) -> str:
    if Chem is None:
        raise ImportError("RDKit is required for Morgan Ridge featurization.")
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES for Ridge baseline: {smiles!r}")
    return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a radius-2, 2048-bit Morgan fingerprint by default."""
    if Chem is None or AllChem is None or DataStructs is None:
        raise ImportError("RDKit is required for Morgan Ridge featurization.")
    canonical = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    fingerprint = AllChem.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    ).GetFingerprint(molecule)
    array = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def load_drug_smiles(path: Path) -> dict[str, str]:
    table = pd.read_csv(path)
    required = {"drug_id", "canonical_smiles"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if table["drug_id"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate drug_id values.")
    return dict(
        zip(
            table["drug_id"].astype(str),
            table["canonical_smiles"].astype(str),
            strict=True,
        )
    )


def _normalise_split_entries(
    entries: Sequence[object],
    valid_drugs: set[str],
    smiles_to_drugs: Mapping[str, set[str]],
    control_label: str,
) -> set[str]:
    output: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict):
            value = next(
                (
                    entry.get(key)
                    for key in (
                        "drug_id",
                        "drug",
                        "compound",
                        "name",
                        "canonical_smiles",
                    )
                    if entry.get(key) is not None
                ),
                None,
            )
        else:
            value = entry
        if value is None or str(value) == control_label:
            continue
        value = str(value)
        if value in valid_drugs:
            output.add(value)
        elif value in smiles_to_drugs:
            output.update(smiles_to_drugs[value])
        else:
            raise ValueError(
                f"Split entry {value!r} does not match a valid drug or SMILES."
            )
    return output


def read_split_drugs(
    splits: dict,
    split_name: str,
    valid_drugs: set[str],
    drug_smiles: Mapping[str, str],
    control_label: str,
) -> set[str]:
    """Read either ``train`` or ``train_drugs`` style split files."""
    raw = None
    for key in (f"{split_name}_drugs", split_name):
        if key in splits:
            raw = splits[key]
            break
    if isinstance(raw, dict):
        raw = next(
            (raw[key] for key in ("drugs", "drug_ids", "smiles") if key in raw), None
        )
    if raw is None:
        raise ValueError(f"Split file does not define {split_name!r}.")
    smiles_to_drugs: dict[str, set[str]] = {}
    for drug_id, smiles in drug_smiles.items():
        smiles_to_drugs.setdefault(str(smiles), set()).add(str(drug_id))
    return _normalise_split_entries(raw, valid_drugs, smiles_to_drugs, control_label)


def validate_split_metadata(
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame | None = None,
) -> None:
    """Fail on drug or canonical-SMILES overlap across benchmark splits."""
    tables = {"train": train_metadata, "validation": validation_metadata}
    if test_metadata is not None:
        tables["test"] = test_metadata
    required = {"drug_id", "context_id", "canonical_smiles"}
    for name, table in tables.items():
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} metadata is missing columns: {sorted(missing)}")
        if table[["drug_id", "context_id"]].astype(str).duplicated().any():
            raise ValueError(
                f"{name} metadata has duplicate drug/context pseudobulk rows."
            )

    names = list(tables)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = tables[left_name]
            right = tables[right_name]
            drug_overlap = set(left["drug_id"].astype(str)) & set(
                right["drug_id"].astype(str)
            )
            smiles_overlap = set(left["canonical_smiles"].astype(str)) & set(
                right["canonical_smiles"].astype(str)
            )
            if drug_overlap or smiles_overlap:
                raise ValueError(
                    f"{left_name}/{right_name} leakage: drug_overlap={sorted(drug_overlap)}, "
                    f"canonical_smiles_overlap={sorted(smiles_overlap)}"
                )


def build_fingerprint_map(
    metadata_tables: Sequence[pd.DataFrame],
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[str, np.ndarray]:
    """Featurize unique drugs without using response values."""
    combined = pd.concat(
        [table[["drug_id", "canonical_smiles"]] for table in metadata_tables],
        ignore_index=True,
    ).drop_duplicates()
    if combined["drug_id"].astype(str).duplicated().any():
        raise ValueError("A drug_id maps to multiple canonical SMILES values.")
    return {
        str(row.drug_id): smiles_to_morgan(str(row.canonical_smiles), radius, n_bits)
        for row in combined.itertuples(index=False)
    }


def build_design_matrix(
    metadata: pd.DataFrame,
    fingerprints: Mapping[str, np.ndarray],
    context_categories: Sequence[object],
) -> np.ndarray:
    """Concatenate a frozen drug fingerprint with a training-defined context one-hot."""
    required = {"drug_id", "context_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    categories = [str(value) for value in context_categories]
    if len(categories) != len(set(categories)) or not categories:
        raise ValueError("context_categories must be a non-empty unique sequence.")
    category_index = {value: index for index, value in enumerate(categories)}

    rows = []
    fingerprint_size = None
    for row in metadata.itertuples(index=False):
        drug_id = str(row.drug_id)
        context_id = str(row.context_id)
        if drug_id not in fingerprints:
            raise ValueError(f"Missing fingerprint for drug {drug_id!r}.")
        if context_id not in category_index:
            raise ValueError(
                f"Context {context_id!r} was absent from training categories."
            )
        fingerprint = np.asarray(fingerprints[drug_id], dtype=np.float32)
        if fingerprint.ndim != 1 or not np.isfinite(fingerprint).all():
            raise ValueError(
                f"Fingerprint for drug {drug_id!r} is not a finite vector."
            )
        fingerprint_size = (
            fingerprint.size if fingerprint_size is None else fingerprint_size
        )
        if fingerprint.size != fingerprint_size:
            raise ValueError("All fingerprints must have the same length.")
        context_one_hot = np.zeros(len(categories), dtype=np.float32)
        context_one_hot[category_index[context_id]] = 1.0
        rows.append(np.concatenate([fingerprint, context_one_hot]))
    if not rows:
        return np.empty(
            (0, int(fingerprint_size or 0) + len(categories)), dtype=np.float32
        )
    return np.stack(rows)


def fit_ridge(train_design: np.ndarray, train_true: np.ndarray, alpha: float) -> Ridge:
    """Fit a multi-output Ridge model. This API deliberately has no test argument."""
    design = np.asarray(train_design, dtype=float)
    targets = np.asarray(train_true, dtype=float)
    if design.ndim != 2 or targets.ndim != 2 or len(design) != len(targets):
        raise ValueError(
            "train_design and train_true must be aligned two-dimensional arrays."
        )
    if (
        len(design) == 0
        or not np.isfinite(design).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("Ridge training arrays must be non-empty and finite.")
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive.")
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(design, targets)
    return model


def _drug_macro(values: np.ndarray, metadata: pd.DataFrame) -> float:
    frame = pd.DataFrame(
        {"drug_id": metadata["drug_id"].astype(str).to_numpy(), "value": values}
    )
    return float(frame.groupby("drug_id", sort=True)["value"].mean().mean())


def select_ridge_alpha(
    train_design: np.ndarray,
    train_true: np.ndarray,
    validation_design: np.ndarray,
    validation_true: np.ndarray,
    validation_metadata: pd.DataFrame,
    gene_panels: Mapping[str, Sequence[int]],
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> RidgeSelection:
    """Choose alpha by validation conditional accuracy, then pair-own Spearman."""
    required = {"drug_id", "context_id"}
    if missing := required - set(validation_metadata.columns):
        raise ValueError(f"Validation metadata is missing columns: {sorted(missing)}")
    if len(validation_design) != len(validation_true) or len(validation_true) != len(
        validation_metadata
    ):
        raise ValueError("Validation design, targets, and metadata rows do not align.")
    candidates = sorted({float(alpha) for alpha in alphas})
    if not candidates or any(alpha <= 0 for alpha in candidates):
        raise ValueError("alphas must contain positive values.")

    records = []
    for alpha in candidates:
        model = fit_ridge(train_design, train_true, alpha)
        prediction = np.asarray(model.predict(validation_design), dtype=float)
        conditional = conditional_rank_score(
            prediction,
            validation_true,
            validation_metadata["context_id"],
            validation_metadata["drug_id"],
            gene_panels,
        )
        spearman = pair_own_spearman(validation_true, prediction, top_k=50)
        records.append(
            {
                "config_id": f"alpha={alpha:.12g}",
                "split": "validation",
                "alpha": alpha,
                "conditional_accuracy_drug_macro": conditional.summary[
                    "conditional_accuracy_drug_macro"
                ],
                "pair_own_spearman_top50_drug_macro": _drug_macro(
                    spearman, validation_metadata
                ),
            }
        )
    trials = pd.DataFrame.from_records(records)
    ranked = trials.sort_values(
        [
            "conditional_accuracy_drug_macro",
            "pair_own_spearman_top50_drug_macro",
            "config_id",
        ],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return RidgeSelection(alpha=float(ranked.iloc[0]["alpha"]), trials=trials)


def fit_final_ridge(
    train_design: np.ndarray,
    train_true: np.ndarray,
    validation_design: np.ndarray,
    validation_true: np.ndarray,
    alpha: float,
) -> Ridge:
    """Refit the frozen alpha on train+validation; no test array enters this function."""
    design = np.concatenate([train_design, validation_design], axis=0)
    targets = np.concatenate([train_true, validation_true], axis=0)
    return fit_ridge(design, targets, alpha)


def _attach_smiles(metadata: pd.DataFrame, smiles: Mapping[str, str]) -> pd.DataFrame:
    output = metadata.copy()
    if "canonical_smiles" not in output:
        output["canonical_smiles"] = output["drug_id"].astype(str).map(smiles)
    if output["canonical_smiles"].isna().any():
        missing = output.loc[output["canonical_smiles"].isna(), "drug_id"].unique()
        raise ValueError(f"Missing canonical SMILES for drugs {missing.tolist()}.")
    return output


def _rank_trials(trials: pd.DataFrame) -> pd.DataFrame:
    ranked = trials.sort_values(
        [
            "conditional_accuracy_drug_macro",
            "pair_own_spearman_top50_drug_macro",
            "config_id",
        ],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "selection_rank", np.arange(1, len(ranked) + 1))
    return ranked


def run_ridge_selection(
    *,
    train_targets_path: Path,
    train_metadata_path: Path,
    validation_targets_path: Path,
    validation_metadata_path: Path,
    smiles_csv_path: Path,
    gene_panels_path: Path,
    output_selection_path: Path,
    output_trials_path: Path,
) -> dict:
    """Select alpha without accepting or opening any test artifact."""
    if output_selection_path.exists() or output_trials_path.exists():
        raise FileExistsError("Refusing to overwrite frozen Ridge selection outputs.")
    metadata = {
        "train": pd.read_csv(train_metadata_path),
        "validation": pd.read_csv(validation_metadata_path),
    }
    targets_with_genes = {
        "train": load_targets(train_targets_path),
        "validation": load_targets(validation_targets_path),
    }
    targets = {name: values[0] for name, values in targets_with_genes.items()}
    if not np.array_equal(
        targets_with_genes["train"][1], targets_with_genes["validation"][1]
    ):
        raise ValueError("Ridge train and validation gene orders differ.")
    smiles = load_drug_smiles(smiles_csv_path)
    metadata = {name: _attach_smiles(table, smiles) for name, table in metadata.items()}
    validate_split_metadata(metadata["train"], metadata["validation"])
    for split in metadata:
        if len(metadata[split]) != len(targets[split]):
            raise ValueError(f"{split} target rows and metadata rows do not align.")

    fingerprints = build_fingerprint_map(
        [metadata["train"], metadata["validation"]], radius=2, n_bits=2048
    )
    contexts = sorted(metadata["train"]["context_id"].astype(str).unique())
    design = {
        split: build_design_matrix(table, fingerprints, contexts)
        for split, table in metadata.items()
    }
    panel_payload = json.loads(gene_panels_path.read_text())
    panels = {
        str(key): np.asarray(value, dtype=int) for key, value in panel_payload.items()
    }
    selection = select_ridge_alpha(
        design["train"],
        targets["train"],
        design["validation"],
        targets["validation"],
        metadata["validation"],
        panels,
    )
    ranked = _rank_trials(selection.trials)
    selected = ranked.iloc[0]
    output_trials_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_trials_path, index=False)
    payload = {
        "schema_version": 1,
        "model": "ridge",
        "selected_alpha": float(selected["alpha"]),
        "selected_config_id": str(selected["config_id"]),
        "selected_validation_metrics": {
            "conditional_accuracy_drug_macro": float(
                selected["conditional_accuracy_drug_macro"]
            ),
            "pair_own_spearman_top50_drug_macro": float(
                selected["pair_own_spearman_top50_drug_macro"]
            ),
        },
        "alpha_grid": list(DEFAULT_ALPHA_GRID),
        "fingerprint": {"kind": "Morgan", "radius": 2, "bits": 2048},
        "fit_splits_during_selection": ["train"],
        "selection_split": "validation",
        "test_artifacts_opened_during_selection": False,
        "context_categories": contexts,
        "trials_sha256": sha256_file(output_trials_path),
        "source_hashes": {
            "train_targets": sha256_file(train_targets_path),
            "train_metadata": sha256_file(train_metadata_path),
            "validation_targets": sha256_file(validation_targets_path),
            "validation_metadata": sha256_file(validation_metadata_path),
            "smiles_csv": sha256_file(smiles_csv_path),
            "gene_panels": sha256_file(gene_panels_path),
        },
    }
    output_selection_path.parent.mkdir(parents=True, exist_ok=True)
    output_selection_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    return payload


def run_ridge_refit_predict(
    *,
    selection_path: Path,
    train_targets_path: Path,
    train_metadata_path: Path,
    validation_targets_path: Path,
    validation_metadata_path: Path,
    test_metadata_path: Path,
    smiles_csv_path: Path,
    output_predictions_path: Path,
) -> dict:
    """Refit a frozen alpha and predict from test metadata without reading test truth."""
    if output_predictions_path.exists():
        raise FileExistsError("Refusing to overwrite Ridge final predictions.")
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("model") != "ridge"
        or selection.get("test_artifacts_opened_during_selection") is not False
    ):
        raise ValueError("Ridge selection evidence is not admissible.")
    alpha = float(selection["selected_alpha"])
    if alpha not in DEFAULT_ALPHA_GRID:
        raise ValueError("Frozen Ridge alpha is outside the preregistered grid.")

    metadata = {
        "train": pd.read_csv(train_metadata_path),
        "validation": pd.read_csv(validation_metadata_path),
        "test": pd.read_csv(test_metadata_path),
    }
    if "pair_id" not in metadata["test"]:
        raise ValueError("Ridge test metadata is missing pair_id alignment keys.")
    train_true, train_gene_ids = load_targets(train_targets_path)
    validation_true, validation_gene_ids = load_targets(validation_targets_path)
    if not np.array_equal(train_gene_ids, validation_gene_ids):
        raise ValueError("Ridge train and validation gene orders differ.")
    if len(metadata["train"]) != len(train_true) or len(metadata["validation"]) != len(
        validation_true
    ):
        raise ValueError("Ridge fit target rows and metadata rows do not align.")

    smiles = load_drug_smiles(smiles_csv_path)
    metadata = {name: _attach_smiles(table, smiles) for name, table in metadata.items()}
    validate_split_metadata(metadata["train"], metadata["validation"], metadata["test"])
    fingerprints = build_fingerprint_map(list(metadata.values()), radius=2, n_bits=2048)
    contexts = sorted(metadata["train"]["context_id"].astype(str).unique())
    design = {
        split: build_design_matrix(table, fingerprints, contexts)
        for split, table in metadata.items()
    }
    final_model = fit_final_ridge(
        design["train"],
        train_true,
        design["validation"],
        validation_true,
        alpha,
    )
    prediction = np.asarray(final_model.predict(design["test"]), dtype=float)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Ridge produced NaN or Inf predictions.")
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_predictions_path,
        pred=prediction.astype(np.float32),
        pair_ids=np.asarray(
            metadata["test"]["pair_id"].astype(str).tolist(), dtype=str
        ),
        gene_ids=train_gene_ids,
    )
    return {
        "schema_version": 1,
        "selected_alpha": alpha,
        "selection_sha256": sha256_file(selection_path),
        "fit_splits": ["train", "validation"],
        "test_targets_opened": False,
        "prediction_rows": int(len(prediction)),
        "prediction_genes": int(prediction.shape[1]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protocol-locked Morgan-2048 Ridge.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    selection = subparsers.add_parser(
        "select", help="Select alpha from train and validation only."
    )
    for split in ("train", "validation"):
        selection.add_argument(f"--{split}-targets", type=Path, required=True)
        selection.add_argument(f"--{split}-metadata", type=Path, required=True)
    selection.add_argument("--smiles-csv", type=Path, required=True)
    selection.add_argument("--gene-panels", type=Path, required=True)
    selection.add_argument("--out-selection", type=Path, required=True)
    selection.add_argument("--out-trials", type=Path, required=True)

    final = subparsers.add_parser(
        "refit-predict", help="Refit frozen alpha and predict without test truth."
    )
    final.add_argument("--selection", type=Path, required=True)
    for split in ("train", "validation"):
        final.add_argument(f"--{split}-targets", type=Path, required=True)
        final.add_argument(f"--{split}-metadata", type=Path, required=True)
    final.add_argument("--test-metadata", type=Path, required=True)
    final.add_argument("--smiles-csv", type=Path, required=True)
    final.add_argument("--out-predictions", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "select":
        result = run_ridge_selection(
            train_targets_path=args.train_targets,
            train_metadata_path=args.train_metadata,
            validation_targets_path=args.validation_targets,
            validation_metadata_path=args.validation_metadata,
            smiles_csv_path=args.smiles_csv,
            gene_panels_path=args.gene_panels,
            output_selection_path=args.out_selection,
            output_trials_path=args.out_trials,
        )
    else:
        result = run_ridge_refit_predict(
            selection_path=args.selection,
            train_targets_path=args.train_targets,
            train_metadata_path=args.train_metadata,
            validation_targets_path=args.validation_targets,
            validation_metadata_path=args.validation_metadata,
            test_metadata_path=args.test_metadata,
            smiles_csv_path=args.smiles_csv,
            output_predictions_path=args.out_predictions,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
