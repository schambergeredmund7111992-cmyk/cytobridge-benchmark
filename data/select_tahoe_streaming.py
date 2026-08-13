"""Select a balanced Tahoe panel with two metadata passes and bounded reservoirs."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads

from data.benchmark_splits import canonicalize_smiles
from data.tahoe_benchmark import (
    CONTROL_DRUG,
    _choose_balanced_contexts,
    _stable_order,
    normalize_sample_metadata,
)
from eval.artifacts import sha256_file

OBS_COLUMNS = ("BARCODE_SUB_LIB_ID", "sample", "cell_line_id")


def _obs_batches(path: Path, batch_size: int):
    dataset = pads.dataset(path, format="parquet")
    missing = set(OBS_COLUMNS) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Tahoe obs metadata is missing columns: {sorted(missing)}")
    for batch in dataset.to_batches(columns=list(OBS_COLUMNS), batch_size=batch_size):
        yield from dataset.to_batches(columns=list(OBS_COLUMNS), batch_size=batch_size)


def _prepare_sample_and_drug_tables(
    sample_metadata_path: Path,
    drug_metadata_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = normalize_sample_metadata(pd.read_parquet(sample_metadata_path))
    drugs = pd.read_parquet(drug_metadata_path)[["drug", "canonical_smiles"]].copy()
    if drugs["drug"].astype(str).duplicated().any():
        raise ValueError("Tahoe drug metadata must have unique drug names.")
    drugs["drug"] = drugs["drug"].astype(str)
    drugs["canonical_smiles"] = [
        "" if drug == CONTROL_DRUG else (canonicalize_smiles(smiles) or "")
        for drug, smiles in zip(drugs["drug"], drugs["canonical_smiles"])
    ]
    samples = samples.merge(drugs, on="drug", how="left", validate="many_to_one")
    highest = (
        samples.loc[samples["drug"].ne(CONTROL_DRUG)]
        .groupby("drug")["dose_um"]
        .transform("max")
    )
    treated_mask = samples["drug"].ne(CONTROL_DRUG)
    samples["eligible_sample"] = False
    samples.loc[treated_mask, "eligible_sample"] = samples.loc[
        treated_mask, "dose_um"
    ].eq(highest) & samples.loc[treated_mask, "canonical_smiles"].fillna("").ne("")
    samples.loc[samples["drug"].eq(CONTROL_DRUG), "eligible_sample"] = True
    return samples, drugs


def _first_pass(
    obs_path: Path,
    sample_lookup: pd.DataFrame,
    *,
    batch_size: int,
) -> tuple[Counter, Counter, Counter, int]:
    pair_counts: Counter = Counter()
    treated_plate_counts: Counter = Counter()
    control_counts: Counter = Counter()
    sample_map = sample_lookup.set_index("sample")[
        ["drug", "plate", "dose_um", "eligible_sample"]
    ].to_dict(orient="index")
    scanned = 0
    for batch in _obs_batches(obs_path, batch_size):
        frame = batch.to_pandas()
        scanned += len(frame)
        frame = frame[frame["sample"].astype(str).isin(sample_map)]
        for sample, context in frame[["sample", "cell_line_id"]].itertuples(
            index=False
        ):
            info = sample_map[str(sample)]
            if not bool(info["eligible_sample"]):
                continue
            context = str(context)
            drug = str(info["drug"])
            plate = str(info["plate"])
            if drug == CONTROL_DRUG:
                control_counts[(context, plate)] += 1
            else:
                pair_counts[(drug, context)] += 1
                treated_plate_counts[(drug, context, plate)] += 1
    return pair_counts, treated_plate_counts, control_counts, scanned


def _select_design(
    pair_counts: Counter,
    treated_plate_counts: Counter,
    control_counts: Counter,
    *,
    n_contexts: int,
    max_drugs: int,
    min_cells_per_pair: int,
    seed: int,
) -> tuple[list[str], list[str], set[tuple[str, str]]]:
    covered = pd.DataFrame(
        [
            {"drug": drug, "cell_line_id": context, "n_cells": count}
            for (drug, context), count in pair_counts.items()
            if count >= min_cells_per_pair
        ]
    )
    if covered.empty:
        raise ValueError("No Tahoe drug/context pair meets the minimum cell count.")
    contexts, balanced_drugs = _choose_balanced_contexts(covered, n_contexts)
    plate_valid = []
    for drug in sorted(balanced_drugs):
        needed = {
            (context, plate)
            for (candidate, context, plate), count in treated_plate_counts.items()
            if candidate == drug and context in contexts and count > 0
        }
        if needed and all(control_counts[stratum] >= 2 for stratum in needed):
            plate_valid.append(drug)
    selected_drugs = _stable_order(plate_valid, seed)[:max_drugs]
    if len(selected_drugs) < 2:
        raise ValueError(
            "Fewer than two balanced Tahoe drugs have usable plate controls."
        )
    needed_control_strata = {
        (context, plate)
        for (drug, context, plate), count in treated_plate_counts.items()
        if drug in selected_drugs and context in contexts and count > 0
    }
    return contexts, selected_drugs, needed_control_strata


def _reservoir_add(heap: list, record: dict, seed: int, cap: int) -> None:
    cell_id = str(record["cell_id"])
    digest = int(hashlib.sha256(f"{seed}\0{cell_id}".encode()).hexdigest(), 16)
    item = (-digest, cell_id, record)
    if len(heap) < cap:
        heapq.heappush(heap, item)
    elif digest < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _second_pass(
    obs_path: Path,
    samples: pd.DataFrame,
    contexts: list[str],
    selected_drugs: list[str],
    needed_control_strata: set[tuple[str, str]],
    *,
    cap: int,
    seed: int,
    batch_size: int,
) -> tuple[pd.DataFrame, int]:
    sample_map = samples.set_index("sample")[
        ["drug", "plate", "dose_um", "canonical_smiles", "eligible_sample"]
    ].to_dict(orient="index")
    context_set = set(contexts)
    drug_set = set(selected_drugs)
    reservoirs: dict[tuple, list] = {}
    scanned = 0
    for batch in _obs_batches(obs_path, batch_size):
        frame = batch.to_pandas()
        scanned += len(frame)
        for cell_id, sample, context in frame.itertuples(index=False):
            sample = str(sample)
            context = str(context)
            if sample not in sample_map or context not in context_set:
                continue
            info = sample_map[sample]
            if not bool(info["eligible_sample"]):
                continue
            drug = str(info["drug"])
            plate = str(info["plate"])
            if drug == CONTROL_DRUG:
                if (context, plate) not in needed_control_strata:
                    continue
                role = "control"
            elif drug in drug_set:
                role = "treated"
            else:
                continue
            record = {
                "cell_id": str(cell_id),
                "sample": sample,
                "cell_line_id": context,
                "plate": plate,
                "drug": drug,
                "dose_um": float(info["dose_um"]),
                "canonical_smiles": str(info["canonical_smiles"] or ""),
                "role": role,
            }
            key = (drug, context, float(info["dose_um"]), sample)
            _reservoir_add(reservoirs.setdefault(key, []), record, seed, cap)
    records = [item[2] for heap in reservoirs.values() for item in heap]
    panel = pd.DataFrame(records)
    if panel.empty or panel["cell_id"].duplicated().any():
        raise ValueError(
            "Tahoe bounded selection produced no cells or duplicate cell IDs."
        )
    return (
        panel.sort_values(
            ["role", "drug", "cell_line_id", "sample", "cell_id"]
        ).reset_index(drop=True),
        scanned,
    )


def select_tahoe_from_parquet(
    obs_metadata_path: Path,
    sample_metadata_path: Path,
    drug_metadata_path: Path,
    output_dir: Path,
    *,
    n_contexts: int = 10,
    max_drugs: int = 120,
    min_cells_per_pair: int = 64,
    cap: int = 64,
    seed: int = 20260710,
    batch_size: int = 250_000,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Tahoe selection: {output_dir}")
    if cap < min_cells_per_pair:
        raise ValueError(
            "cap must be at least min_cells_per_pair for the balanced panel."
        )
    samples, drugs = _prepare_sample_and_drug_tables(
        sample_metadata_path, drug_metadata_path
    )
    pair_counts, treated_plate_counts, control_counts, first_scanned = _first_pass(
        obs_metadata_path, samples, batch_size=batch_size
    )
    contexts, selected_drugs, needed_control_strata = _select_design(
        pair_counts,
        treated_plate_counts,
        control_counts,
        n_contexts=n_contexts,
        max_drugs=max_drugs,
        min_cells_per_pair=min_cells_per_pair,
        seed=seed,
    )
    panel, second_scanned = _second_pass(
        obs_metadata_path,
        samples,
        contexts,
        selected_drugs,
        needed_control_strata,
        cap=cap,
        seed=seed,
        batch_size=batch_size,
    )
    treated_counts = (
        panel[panel["role"].eq("treated")].groupby(["drug", "cell_line_id"]).size()
    )
    if (
        len(treated_counts) != len(selected_drugs) * len(contexts)
        or (treated_counts < min_cells_per_pair).any()
    ):
        raise ValueError("Capped Tahoe panel is no longer a complete balanced matrix.")
    control_counts_final = (
        panel[panel["role"].eq("control")].groupby(["cell_line_id", "plate"]).size()
    )
    if any(
        control_counts_final.get(stratum, 0) < 2 for stratum in needed_control_strata
    ):
        raise ValueError(
            "Capped Tahoe panel lacks two controls in a required plate/context."
        )
    selected = set(selected_drugs)
    attrition = drugs[drugs["drug"].ne(CONTROL_DRUG)].copy()
    attrition["selected"] = attrition["drug"].isin(selected)
    attrition["exclusion_reason"] = attrition.apply(
        lambda row: (
            "included"
            if row["selected"]
            else (
                "invalid_or_missing_smiles"
                if not str(row["canonical_smiles"])
                else "coverage_or_control"
            )
        ),
        axis=1,
    )
    summary = {
        "seed": seed,
        "highest_dose_only": True,
        "contexts": contexts,
        "drugs": selected_drugs,
        "n_contexts": len(contexts),
        "n_drugs": len(selected_drugs),
        "min_cells_per_pair": min_cells_per_pair,
        "cap_per_drug_context_dose_sample": cap,
        "control_label": CONTROL_DRUG,
        "plate_matched_controls": True,
        "metadata_passes": 2,
        "rows_scanned_first_pass": first_scanned,
        "rows_scanned_second_pass": second_scanned,
        "obs_metadata_sha256": sha256_file(obs_metadata_path),
        "sample_metadata_sha256": sha256_file(sample_metadata_path),
        "drug_metadata_sha256": sha256_file(drug_metadata_path),
    }
    output_dir.mkdir(parents=True)
    panel.to_parquet(output_dir / "selected_cells.parquet", index=False)
    attrition.to_csv(output_dir / "drug_attrition.csv", index=False)
    (output_dir / "selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-metadata", type=Path, required=True)
    parser.add_argument("--sample-metadata", type=Path, required=True)
    parser.add_argument("--drug-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-contexts", type=int, default=10)
    parser.add_argument("--max-drugs", type=int, default=120)
    parser.add_argument("--min-cells", type=int, default=64)
    parser.add_argument("--cap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=250_000)
    args = parser.parse_args()
    result = select_tahoe_from_parquet(
        args.obs_metadata,
        args.sample_metadata,
        args.drug_metadata,
        args.out,
        n_contexts=args.n_contexts,
        max_drugs=args.max_drugs,
        min_cells_per_pair=args.min_cells,
        cap=args.cap,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
