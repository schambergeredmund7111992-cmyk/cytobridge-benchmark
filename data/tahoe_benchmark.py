"""Bounded, metadata-first Tahoe-100M benchmark selection.

The full 95.6M-cell expression table is never materialized by this module. It first selects a
balanced panel from official metadata, records exact cell identifiers, and leaves expression
streaming to a separate bounded extraction step.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from data.benchmark_splits import canonicalize_smiles

CONTROL_DRUG = "DMSO_TF"


def concentration_to_um(value: float, unit: str) -> float:
    conversions = {"nm": 1.0e-3, "um": 1.0, "µm": 1.0, "mm": 1.0e3}
    normalized = str(unit).strip().lower()
    if normalized not in conversions:
        raise ValueError(f"Unsupported Tahoe concentration unit: {unit!r}")
    concentration = float(value) * conversions[normalized]
    if not np.isfinite(concentration) or concentration < 0:
        raise ValueError(f"Invalid Tahoe concentration: {value!r} {unit!r}")
    return concentration


def parse_drug_concentration(value: object) -> tuple[str, float]:
    """Parse official ``drugname_drugconc`` into drug name and concentration in uM."""
    try:
        payload = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Cannot parse drugname_drugconc value {value!r}.") from exc
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise ValueError(f"Expected one Tahoe treatment tuple; got {payload!r}.")
    treatment = payload[0]
    if not isinstance(treatment, (list, tuple)) or len(treatment) != 3:
        raise ValueError(f"Expected (drug, concentration, unit); got {treatment!r}.")
    drug, concentration, unit = treatment
    return str(drug), concentration_to_um(float(concentration), str(unit))


def normalize_sample_metadata(sample_metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"sample", "plate", "drug", "drugname_drugconc"}
    if missing := required - set(sample_metadata.columns):
        raise ValueError(f"Tahoe sample metadata is missing columns: {sorted(missing)}")
    if sample_metadata["sample"].astype(str).duplicated().any():
        raise ValueError("Tahoe sample metadata must have unique sample identifiers.")
    output = sample_metadata.copy()
    parsed_drug = []
    dose_um = []
    for row in output.itertuples(index=False):
        if str(row.drug) == CONTROL_DRUG:
            parsed_drug.append(CONTROL_DRUG)
            dose_um.append(0.0)
            continue
        name, concentration = parse_drug_concentration(row.drugname_drugconc)
        if name != str(row.drug):
            raise ValueError(
                f"Sample {row.sample!r} drug mismatch: {row.drug!r} versus parsed {name!r}."
            )
        parsed_drug.append(name)
        dose_um.append(concentration)
    output["drug"] = parsed_drug
    output["dose_um"] = np.asarray(dose_um, dtype=float)
    output["sample"] = output["sample"].astype(str)
    output["plate"] = output["plate"].astype(str)
    return output


def _stable_order(values: Sequence[str], seed: int) -> list[str]:
    return sorted(
        map(str, values),
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
    )


def _choose_balanced_contexts(
    eligible_pairs: pd.DataFrame,
    n_contexts: int,
) -> tuple[list[str], set[str]]:
    by_context = {
        str(context): set(group["drug"].astype(str))
        for context, group in eligible_pairs.groupby("cell_line_id", sort=True)
    }
    if len(by_context) < n_contexts:
        raise ValueError(
            f"Only {len(by_context)} Tahoe contexts meet coverage, fewer than {n_contexts}."
        )
    selected: list[str] = []
    intersection: set[str] | None = None
    remaining = set(by_context)
    while len(selected) < n_contexts:
        candidates = []
        for context in sorted(remaining):
            candidate_drugs = (
                by_context[context]
                if intersection is None
                else intersection & by_context[context]
            )
            candidates.append(
                (
                    len(candidate_drugs),
                    len(by_context[context]),
                    context,
                    candidate_drugs,
                )
            )
        _, _, chosen, chosen_drugs = max(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        if len(chosen_drugs) < 2:
            raise ValueError(
                "Tahoe context selection would leave fewer than two balanced drugs."
            )
        selected.append(chosen)
        remaining.remove(chosen)
        intersection = set(chosen_drugs)
    assert intersection is not None
    return selected, intersection


def select_tahoe_panel(
    obs_metadata: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    drug_metadata: pd.DataFrame,
    *,
    cell_id_col: str = "BARCODE_SUB_LIB_ID",
    n_contexts: int = 10,
    max_drugs: int = 120,
    min_cells_per_pair: int = 64,
    seed: int = 20260710,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select a highest-dose, plate-controlled, balanced Tahoe panel from metadata."""
    required_obs = {cell_id_col, "sample", "cell_line_id"}
    if missing := required_obs - set(obs_metadata.columns):
        raise ValueError(f"Tahoe obs metadata is missing columns: {sorted(missing)}")
    required_drug = {"drug", "canonical_smiles"}
    if missing := required_drug - set(drug_metadata.columns):
        raise ValueError(f"Tahoe drug metadata is missing columns: {sorted(missing)}")
    if obs_metadata[cell_id_col].astype(str).duplicated().any():
        raise ValueError("Tahoe cell identifiers must be unique.")
    if drug_metadata["drug"].astype(str).duplicated().any():
        raise ValueError("Tahoe drug metadata must have unique drug names.")

    samples = normalize_sample_metadata(sample_metadata)
    drugs = drug_metadata[["drug", "canonical_smiles"]].copy()
    drugs["drug"] = drugs["drug"].astype(str)
    drugs["canonical_smiles"] = [
        "" if str(drug) == CONTROL_DRUG else (canonicalize_smiles(smiles) or "")
        for drug, smiles in zip(drugs["drug"], drugs["canonical_smiles"])
    ]
    cells = obs_metadata[[cell_id_col, "sample", "cell_line_id"]].copy()
    cells.columns = ["cell_id", "sample", "cell_line_id"]
    cells = cells.merge(
        samples[["sample", "plate", "drug", "dose_um"]],
        on="sample",
        validate="many_to_one",
    )
    cells = cells.merge(drugs, on="drug", how="left", validate="many_to_one")
    if cells[["plate", "drug", "dose_um"]].isna().any().any():
        raise ValueError("Tahoe obs rows failed to map to complete sample metadata.")

    treated = cells[cells["drug"] != CONTROL_DRUG].copy()
    treated["valid_smiles"] = treated["canonical_smiles"].fillna("").str.len() > 0
    drug_status = (
        treated.groupby("drug", sort=True)
        .agg(valid_smiles=("valid_smiles", "all"), n_cells_all=("cell_id", "size"))
        .reset_index()
    )
    highest_dose = treated.groupby("drug", sort=True)["dose_um"].transform("max")
    treated = treated[
        np.isclose(treated["dose_um"], highest_dose) & treated["valid_smiles"]
    ]
    pair_counts = (
        treated.groupby(["drug", "cell_line_id"], sort=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    covered_pairs = pair_counts[pair_counts["n_cells"] >= min_cells_per_pair]
    contexts, balanced_drugs = _choose_balanced_contexts(covered_pairs, n_contexts)

    controls = cells[cells["drug"] == CONTROL_DRUG]
    control_strata = set(
        zip(controls["cell_line_id"].astype(str), controls["plate"].astype(str))
    )
    plate_valid_drugs = set()
    for drug in sorted(balanced_drugs):
        rows = treated[
            (treated["drug"].astype(str) == drug)
            & treated["cell_line_id"].astype(str).isin(contexts)
        ]
        needed = set(zip(rows["cell_line_id"].astype(str), rows["plate"].astype(str)))
        if needed and needed <= control_strata:
            plate_valid_drugs.add(drug)
    if len(plate_valid_drugs) < 2:
        raise ValueError(
            "Fewer than two balanced Tahoe drugs have plate-matched DMSO controls."
        )
    selected_drugs = _stable_order(sorted(plate_valid_drugs), seed)[:max_drugs]
    selected_treated = treated[
        treated["drug"].astype(str).isin(selected_drugs)
        & treated["cell_line_id"].astype(str).isin(contexts)
    ].copy()
    needed_control_strata = set(
        zip(
            selected_treated["cell_line_id"].astype(str),
            selected_treated["plate"].astype(str),
        )
    )
    selected_control = controls[
        [
            (str(context), str(plate)) in needed_control_strata
            for context, plate in zip(controls["cell_line_id"], controls["plate"])
        ]
    ].copy()
    panel = pd.concat([selected_treated, selected_control], ignore_index=True)
    panel["role"] = np.where(panel["drug"] == CONTROL_DRUG, "control", "treated")
    panel = panel.sort_values(
        ["role", "drug", "cell_line_id", "sample", "cell_id"]
    ).reset_index(drop=True)

    selected_set = set(selected_drugs)
    status = drug_status.assign(
        selected=lambda frame: frame["drug"].astype(str).isin(selected_set),
        exclusion_reason=lambda frame: np.where(
            frame["drug"].astype(str).isin(selected_set),
            "included",
            np.where(
                ~frame["valid_smiles"],
                "invalid_or_missing_smiles",
                "coverage_or_control",
            ),
        ),
    )
    summary = {
        "seed": seed,
        "highest_dose_only": True,
        "contexts": contexts,
        "n_contexts": len(contexts),
        "drugs": selected_drugs,
        "n_drugs": len(selected_drugs),
        "min_cells_per_pair": min_cells_per_pair,
        "max_drugs": max_drugs,
        "control_label": CONTROL_DRUG,
        "plate_matched_controls": True,
    }
    return panel, status, summary


def cap_cells_per_sample(
    panel: pd.DataFrame,
    cap: int = 64,
    seed: int = 20260710,
) -> pd.DataFrame:
    """Deterministically cap cells per drug/context/dose/sample without response values."""
    if cap <= 0:
        raise ValueError("cap must be positive.")
    required = {"cell_id", "drug", "cell_line_id", "dose_um", "sample"}
    if missing := required - set(panel.columns):
        raise ValueError(f"Tahoe panel is missing columns: {sorted(missing)}")
    kept = []
    group_columns = ["drug", "cell_line_id", "dose_um", "sample"]
    for _, group in panel.groupby(group_columns, sort=True, dropna=False):
        ordered_ids = _stable_order(group["cell_id"].astype(str).tolist(), seed)
        keep_ids = set(ordered_ids[:cap])
        kept.append(group[group["cell_id"].astype(str).isin(keep_ids)])
    return (
        pd.concat(kept, ignore_index=True).sort_values("cell_id").reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the bounded Tahoe benchmark panel."
    )
    parser.add_argument("--obs-metadata", type=Path, required=True)
    parser.add_argument("--sample-metadata", type=Path, required=True)
    parser.add_argument("--drug-metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-contexts", type=int, default=10)
    parser.add_argument("--max-drugs", type=int, default=120)
    parser.add_argument("--min-cells", type=int, default=64)
    parser.add_argument("--cap", type=int, default=64)
    args = parser.parse_args()

    panel, attrition, summary = select_tahoe_panel(
        pd.read_parquet(args.obs_metadata),
        pd.read_parquet(args.sample_metadata),
        pd.read_parquet(args.drug_metadata),
        n_contexts=args.n_contexts,
        max_drugs=args.max_drugs,
        min_cells_per_pair=args.min_cells,
    )
    capped = cap_cells_per_sample(panel, cap=args.cap)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    capped.to_parquet(args.out_dir / "selected_cells.parquet", index=False)
    attrition.to_csv(args.out_dir / "drug_attrition.csv", index=False)
    (args.out_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({**summary, "n_selected_cells": len(capped)}, sort_keys=True))


if __name__ == "__main__":
    main()
