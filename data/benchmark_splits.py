"""Frozen sci-Plex benchmark eligibility, split, and reference-pool helpers.

The functions in this module implement protocol ``1.1.0`` without reading response
values.  They operate on one-row-per-cell metadata and return tables plus canonical
SHA256 manifests; callers are responsible for persisting the returned artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Optional at import time so non-chemistry tooling can still import the module.
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - exercised by an explicit monkeypatch test
    Chem = None
    MurckoScaffold = None


PROTOCOL_VERSION = "1.1.0"
TARGET_CONTEXTS = ("A549", "K562", "MCF7")
TARGET_DOSE_UM = 10.0
TARGET_TIME_H = 24.0
MIN_TREATED_CELLS = 50
SPLIT_SEED = 20260710
VEHICLE_SEED = 1729
SPLIT_NAMES = ("train", "val", "test")
TARGET_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

_RESPONSE_COLUMN = re.compile(
    r"(?:^|_)(?:response|logfc|fold_?change|effect_?size|de_?score|"
    r"differential_?expression)(?:$|_)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MetadataColumns:
    """Column names for one-row-per-cell benchmark metadata."""

    cell_id: str = "cell_id"
    drug_id: str = "drug_id"
    smiles: str = "smiles"
    context: str = "cell_line"
    batch: str = "batch"
    dose_um: str = "dose_um"
    time_h: str = "time_h"
    is_control: str = "is_control"

    def selection_columns(self) -> tuple[str, ...]:
        return (
            self.cell_id,
            self.drug_id,
            self.smiles,
            self.context,
            self.batch,
            self.dose_um,
            self.time_h,
            self.is_control,
        )


@dataclass(frozen=True)
class EligibilityResult:
    """Eligibility tables and the exact label-free selection contract."""

    compounds: pd.DataFrame
    attrition: pd.DataFrame
    eligible_treated_cells: pd.DataFrame
    matching_vehicle_cells: pd.DataFrame
    selection_columns: tuple[str, ...]
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class SplitResult:
    assignments: pd.DataFrame
    manifest: Mapping[str, Any]

    @property
    def split_hash(self) -> str:
        return str(self.manifest["split_sha256"])


@dataclass(frozen=True)
class PoolResult:
    assignments: pd.DataFrame
    manifest: Mapping[str, Any]

    @property
    def assignment_hash(self) -> str:
        return str(self.manifest["assignment_sha256"])


def _require_rdkit() -> None:
    if Chem is None or MurckoScaffold is None:
        raise ImportError(
            "RDKit is required for benchmark SMILES canonicalization and "
            "Bemis-Murcko scaffold construction. Install `rdkit>=2024.3.3`."
        )


def canonicalize_smiles(smiles: object) -> str | None:
    """Return an isomeric canonical SMILES, or ``None`` for an invalid value."""
    _require_rdkit()
    if smiles is None or pd.isna(smiles):
        return None
    value = str(smiles).strip()
    if not value:
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def bemis_murcko_scaffold(canonical_smiles: object) -> str | None:
    """Return a non-empty canonical Bemis-Murcko scaffold.

    Acyclic compounds yield ``None`` here. They remain eligible for the primary
    drug split and are grouped together as ``__ACYCLIC__`` in the scaffold stress test.
    """
    _require_rdkit()
    canonical = canonicalize_smiles(canonical_smiles)
    if canonical is None:
        return None
    mol = Chem.MolFromSmiles(canonical)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )
    if not scaffold:
        return None
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    if scaffold_mol is None:
        return None
    value = str(Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=False))
    return value or None


def _validate_contexts(contexts: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in contexts)
    if len(normalized) < 1 or len(set(normalized)) != len(normalized):
        raise ValueError("target_contexts must contain distinct, non-empty values.")
    if any(not value for value in normalized):
        raise ValueError("target_contexts must contain distinct, non-empty values.")
    return normalized


def _validate_label_free_columns(columns: MetadataColumns) -> None:
    names = columns.selection_columns()
    if len(set(names)) != len(names):
        raise ValueError("MetadataColumns entries must refer to distinct columns.")
    forbidden = sorted(name for name in names if _RESPONSE_COLUMN.search(name))
    if forbidden:
        raise ValueError(
            "Eligibility columns may not be response-derived; forbidden mapping(s): "
            f"{forbidden}."
        )


def _normalize_metadata(
    metadata: pd.DataFrame, columns: MetadataColumns
) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    if metadata.empty:
        raise ValueError("metadata must contain at least one row.")
    _validate_label_free_columns(columns)
    missing = sorted(set(columns.selection_columns()) - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing required column(s): {missing}.")

    work = metadata.copy()
    required_non_null = (
        columns.cell_id,
        columns.drug_id,
        columns.context,
        columns.batch,
        columns.time_h,
        columns.is_control,
    )
    null_columns = [name for name in required_non_null if work[name].isna().any()]
    if null_columns:
        raise ValueError(
            f"metadata contains null values in required column(s): {null_columns}."
        )

    for name in (columns.cell_id, columns.drug_id, columns.context, columns.batch):
        work[name] = work[name].astype(str)
        if work[name].str.len().eq(0).any():
            raise ValueError(f"metadata column {name!r} contains empty identifiers.")
    duplicates = (
        work[columns.cell_id][work[columns.cell_id].duplicated()].unique().tolist()
    )
    if duplicates:
        raise ValueError(
            f"cell IDs must be unique; duplicate example(s): {duplicates[:3]}."
        )

    control_values = set(work[columns.is_control].dropna().tolist())
    if not control_values <= {True, False, 0, 1}:
        raise ValueError(f"{columns.is_control!r} must contain only booleans or 0/1.")
    work[columns.is_control] = work[columns.is_control].astype(bool)

    work["_benchmark_time_h"] = pd.to_numeric(work[columns.time_h], errors="coerce")
    if work["_benchmark_time_h"].isna().any():
        raise ValueError(f"{columns.time_h!r} must be numeric for every row.")
    work["_benchmark_dose_um"] = pd.to_numeric(work[columns.dose_um], errors="coerce")
    bad_treated_dose = ~work[columns.is_control] & work["_benchmark_dose_um"].isna()
    if bad_treated_dose.any():
        raise ValueError(f"{columns.dose_um!r} must be numeric for every treated row.")
    return work


def _reason(
    drug_id: str,
    code: str,
    *,
    canonical_smiles: str | None = None,
    context: str | None = None,
    batch: str | None = None,
    observed: int | None = None,
    required: int | None = None,
) -> dict[str, Any]:
    return {
        "drug_id": drug_id,
        "canonical_smiles": canonical_smiles,
        "reason_code": code,
        "context": context,
        "batch": batch,
        "observed": observed,
        "required": required,
    }


def build_sciplex_eligibility(
    metadata: pd.DataFrame,
    *,
    columns: MetadataColumns = MetadataColumns(),
    target_contexts: Sequence[str] = TARGET_CONTEXTS,
    dose_um: float = TARGET_DOSE_UM,
    time_h: float = TARGET_TIME_H,
    min_treated_cells: int = MIN_TREATED_CELLS,
) -> EligibilityResult:
    """Apply the frozen, response-independent sci-Plex eligibility rules.

    Treated-cell counts use exact numeric equality at ``dose_um`` and ``time_h``.
    A matching vehicle must exist at the same context, acquisition batch, and time;
    vehicle dose is intentionally not used because vehicle rows normally have dose 0.
    """
    _require_rdkit()
    contexts = _validate_contexts(target_contexts)
    if not isinstance(min_treated_cells, int) or min_treated_cells < 1:
        raise ValueError("min_treated_cells must be a positive integer.")
    if not math.isfinite(float(dose_um)) or not math.isfinite(float(time_h)):
        raise ValueError("dose_um and time_h must be finite.")

    work = _normalize_metadata(metadata, columns)
    controls = work[
        work[columns.is_control]
        & work[columns.context].isin(contexts)
        & work["_benchmark_time_h"].eq(float(time_h))
    ]
    treated = work[~work[columns.is_control]]
    summaries: list[dict[str, Any]] = []
    attrition: list[dict[str, Any]] = []

    for drug_id in sorted(treated[columns.drug_id].unique()):
        drug_rows = treated[treated[columns.drug_id].eq(drug_id)]
        raw_smiles = sorted(
            {
                str(value).strip()
                for value in drug_rows[columns.smiles]
                if not pd.isna(value)
            }
        )
        canonical_values = [canonicalize_smiles(value) for value in raw_smiles]
        valid_canonical = sorted(
            {value for value in canonical_values if value is not None}
        )
        drug_reasons: list[dict[str, Any]] = []

        canonical: str | None = None
        scaffold: str | None = None
        if not raw_smiles or any(value is None for value in canonical_values):
            drug_reasons.append(_reason(drug_id, "invalid_smiles"))
        elif len(valid_canonical) != 1:
            drug_reasons.append(_reason(drug_id, "conflicting_canonical_smiles"))
        else:
            canonical = valid_canonical[0]
            scaffold = bemis_murcko_scaffold(canonical)

        exact = drug_rows[
            drug_rows["_benchmark_dose_um"].eq(float(dose_um))
            & drug_rows["_benchmark_time_h"].eq(float(time_h))
            & drug_rows[columns.context].isin(contexts)
        ]
        if exact.empty:
            drug_reasons.append(
                _reason(
                    drug_id,
                    "no_exact_dose_time_rows",
                    canonical_smiles=canonical,
                    observed=0,
                    required=min_treated_cells * len(contexts),
                )
            )

        counts = exact.groupby(columns.context).size().to_dict()
        for context in contexts:
            count = int(counts.get(context, 0))
            if count == 0:
                drug_reasons.append(
                    _reason(
                        drug_id,
                        "missing_required_context",
                        canonical_smiles=canonical,
                        context=context,
                        observed=0,
                        required=min_treated_cells,
                    )
                )
            elif count < min_treated_cells:
                drug_reasons.append(
                    _reason(
                        drug_id,
                        "insufficient_treated_cells",
                        canonical_smiles=canonical,
                        context=context,
                        observed=count,
                        required=min_treated_cells,
                    )
                )

        for (context, batch), _rows in exact.groupby([columns.context, columns.batch]):
            has_vehicle = (
                controls[columns.context].eq(context)
                & controls[columns.batch].eq(batch)
            ).any()
            if not has_vehicle:
                drug_reasons.append(
                    _reason(
                        drug_id,
                        "missing_vehicle_control",
                        canonical_smiles=canonical,
                        context=str(context),
                        batch=str(batch),
                        observed=0,
                        required=1,
                    )
                )

        reason_codes = tuple(sorted({row["reason_code"] for row in drug_reasons}))
        summary: dict[str, Any] = {
            "drug_id": str(drug_id),
            "raw_smiles": tuple(raw_smiles),
            "canonical_smiles": canonical,
            "scaffold_smiles": scaffold,
            "eligible": not drug_reasons,
            "attrition_reasons": reason_codes,
            "exact_treated_total": int(len(exact)),
        }
        for context in contexts:
            summary[f"treated_count_{context}"] = int(counts.get(context, 0))
        summaries.append(summary)
        attrition.extend(drug_reasons)

    compounds = pd.DataFrame(summaries).sort_values("drug_id").reset_index(drop=True)
    attrition_columns = (
        "drug_id",
        "canonical_smiles",
        "reason_code",
        "context",
        "batch",
        "observed",
        "required",
    )
    attrition_frame = pd.DataFrame(attrition, columns=attrition_columns)
    if not attrition_frame.empty:
        attrition_frame = attrition_frame.sort_values(
            ["drug_id", "reason_code", "context", "batch"],
            na_position="first",
        ).reset_index(drop=True)

    eligible = compounds[compounds["eligible"]]
    eligible_ids = set(eligible["drug_id"])
    chemistry = eligible.set_index("drug_id")[["canonical_smiles", "scaffold_smiles"]]
    eligible_treated = treated[
        treated[columns.drug_id].isin(eligible_ids)
        & treated["_benchmark_dose_um"].eq(float(dose_um))
        & treated["_benchmark_time_h"].eq(float(time_h))
        & treated[columns.context].isin(contexts)
    ].copy()
    if not eligible_treated.empty:
        eligible_treated = eligible_treated.join(chemistry, on=columns.drug_id)
        eligible_treated = eligible_treated.sort_values(columns.cell_id).reset_index(
            drop=True
        )

    needed_vehicle_strata = set(
        eligible_treated[[columns.context, columns.batch]].itertuples(
            index=False, name=None
        )
    )
    if needed_vehicle_strata:
        vehicle_mask = [
            (context, batch) in needed_vehicle_strata
            for context, batch in controls[[columns.context, columns.batch]].itertuples(
                index=False,
                name=None,
            )
        ]
        matching_vehicles = controls.loc[vehicle_mask].copy()
        matching_vehicles = matching_vehicles.sort_values(columns.cell_id).reset_index(
            drop=True
        )
    else:
        matching_vehicles = controls.iloc[0:0].copy()

    return EligibilityResult(
        compounds=compounds,
        attrition=attrition_frame,
        eligible_treated_cells=eligible_treated,
        matching_vehicle_cells=matching_vehicles,
        selection_columns=columns.selection_columns(),
        parameters={
            "protocol_version": PROTOCOL_VERSION,
            "dose_um": float(dose_um),
            "time_h": float(time_h),
            "min_treated_cells": min_treated_cells,
            "target_contexts": list(contexts),
            "response_columns_used": [],
        },
    )


def _normalize_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"fractions must have exactly the keys {SPLIT_NAMES}.")
    normalized = {name: float(fractions[name]) for name in SPLIT_NAMES}
    if any(not math.isfinite(value) or value <= 0 for value in normalized.values()):
        raise ValueError("all split fractions must be finite and positive.")
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split fractions must sum to 1.0.")
    return normalized


def _seeded_digest(seed: int, domain: str, value: object) -> str:
    return hashlib.sha256(f"{seed}|{domain}|{value}".encode("utf-8")).hexdigest()


def _eligible_compounds_frame(
    compounds: EligibilityResult | pd.DataFrame,
) -> pd.DataFrame:
    frame = (
        compounds.compounds.copy()
        if isinstance(compounds, EligibilityResult)
        else compounds.copy()
    )
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("compounds must be an EligibilityResult or pandas DataFrame.")
    if "eligible" in frame:
        frame = frame[
            frame["eligible"].eq(True)
        ].copy()  # noqa: E712 - strict bool match
    required = {"drug_id", "canonical_smiles", "scaffold_smiles"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"compound table is missing required column(s): {missing}.")
    if frame.empty:
        raise ValueError("no eligible compounds are available for splitting.")
    if frame[["drug_id", "canonical_smiles"]].isna().any().any():
        raise ValueError(
            "eligible compounds must have drug and canonical SMILES values."
        )
    for column in ("drug_id", "canonical_smiles"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.len().eq(0).any():
            raise ValueError(
                f"eligible compound column {column!r} contains an empty value."
            )
    if frame["drug_id"].duplicated().any():
        raise ValueError("eligible compound table must contain one row per drug_id.")

    scaffold_counts = frame.groupby("canonical_smiles")["scaffold_smiles"].nunique(
        dropna=True
    )
    conflicts = scaffold_counts[scaffold_counts.gt(1)]
    if not conflicts.empty:
        raise ValueError(
            "a canonical SMILES maps to multiple scaffolds: "
            f"{conflicts.index.tolist()[:3]}."
        )
    return frame.sort_values("drug_id").reset_index(drop=True)


def _greedy_group_assignment(
    group_sizes: Mapping[str, int],
    *,
    fractions: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    if len(group_sizes) < len(SPLIT_NAMES):
        raise ValueError(
            f"at least {len(SPLIT_NAMES)} distinct groups are required for non-empty splits; "
            f"got {len(group_sizes)}."
        )
    if any(int(size) < 1 for size in group_sizes.values()):
        raise ValueError(
            "every split group must contain at least one canonical compound."
        )
    total = int(sum(group_sizes.values()))
    targets = {name: fractions[name] * total for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    group_counts = {name: 0 for name in SPLIT_NAMES}
    assignments: dict[str, str] = {}
    ordered = sorted(
        group_sizes,
        key=lambda key: (
            -int(group_sizes[key]),
            _seeded_digest(seed, "group-order", key),
        ),
    )

    for index, key in enumerate(ordered):
        remaining_after = len(ordered) - index - 1
        empty = [name for name in SPLIT_NAMES if group_counts[name] == 0]
        candidates = empty if len(empty) > remaining_after else list(SPLIT_NAMES)

        def candidate_score(name: str) -> tuple[float, str]:
            projected = dict(counts)
            projected[name] += int(group_sizes[key])
            error = sum(
                ((projected[split] - targets[split]) / targets[split]) ** 2
                for split in SPLIT_NAMES
            )
            return error, _seeded_digest(seed, f"split-tie:{key}", name)

        chosen = min(candidates, key=candidate_score)
        assignments[key] = chosen
        counts[chosen] += int(group_sizes[key])
        group_counts[chosen] += 1

    if any(group_counts[name] == 0 for name in SPLIT_NAMES):  # defensive invariant
        raise RuntimeError("deterministic group assignment produced an empty split.")
    return assignments


def canonical_split_sha256(
    assignments: pd.DataFrame,
    *,
    protocol_name: str,
    seed: int,
    fractions: Mapping[str, float] = TARGET_FRACTIONS,
) -> str:
    """Hash a split independently of DataFrame row order or JSON whitespace."""
    normalized_fractions = _normalize_fractions(fractions)
    required = {"drug_id", "canonical_smiles", "scaffold_smiles", "group_key", "split"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"split assignments are missing column(s): {missing}.")
    records = [
        {name: str(row[name]) for name in sorted(required)}
        for row in assignments.to_dict(orient="records")
    ]
    records.sort(
        key=lambda row: (
            row["canonical_smiles"],
            row["drug_id"],
            row["split"],
            row["group_key"],
        )
    )
    payload = {
        "protocol_name": str(protocol_name),
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "target_fractions": normalized_fractions,
        "assignments": records,
    }
    canonical_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _make_group_split(
    compounds: EligibilityResult | pd.DataFrame,
    *,
    protocol_name: str,
    group_column: str,
    seed: int,
    fractions: Mapping[str, float],
) -> SplitResult:
    frame = _eligible_compounds_frame(compounds)
    if group_column == "scaffold_smiles":
        frame["scaffold_smiles"] = (
            frame["scaffold_smiles"].replace("", np.nan).fillna("__ACYCLIC__")
        )
    normalized_fractions = _normalize_fractions(fractions)
    units = frame[["canonical_smiles", "scaffold_smiles"]].drop_duplicates(
        subset=["canonical_smiles"]
    )
    group_sizes = units.groupby(group_column)["canonical_smiles"].nunique().to_dict()
    group_assignment = _greedy_group_assignment(
        {str(key): int(value) for key, value in group_sizes.items()},
        fractions=normalized_fractions,
        seed=int(seed),
    )
    frame["group_key"] = frame[group_column].astype(str)
    frame["split"] = frame["group_key"].map(group_assignment)
    frame["split_protocol"] = protocol_name
    assignments = (
        frame[
            [
                "drug_id",
                "canonical_smiles",
                "scaffold_smiles",
                "group_key",
                "split",
                "split_protocol",
            ]
        ]
        .sort_values("drug_id")
        .reset_index(drop=True)
    )

    canonical_split = assignments.drop_duplicates("canonical_smiles")
    canonical_counts = {
        name: int(canonical_split["split"].eq(name).sum()) for name in SPLIT_NAMES
    }
    total_canonical = int(len(canonical_split))
    group_counts = {name: 0 for name in SPLIT_NAMES}
    for split in set(group_assignment.values()):
        group_counts[split] = sum(value == split for value in group_assignment.values())
    split_hash = canonical_split_sha256(
        assignments,
        protocol_name=protocol_name,
        seed=int(seed),
        fractions=normalized_fractions,
    )
    manifest = {
        "protocol_name": protocol_name,
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "group_column": group_column,
        "target_fractions": normalized_fractions,
        "actual_canonical_counts": canonical_counts,
        "actual_canonical_fractions": {
            name: canonical_counts[name] / total_canonical for name in SPLIT_NAMES
        },
        "actual_group_counts": group_counts,
        "n_drug_ids": int(len(assignments)),
        "n_canonical_compounds": total_canonical,
        "n_groups": int(len(group_assignment)),
        "split_sha256": split_hash,
    }
    return SplitResult(assignments=assignments, manifest=manifest)


def make_drug_disjoint_v2(
    compounds: EligibilityResult | pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
    fractions: Mapping[str, float] = TARGET_FRACTIONS,
) -> SplitResult:
    """Assign canonical-SMILES groups to deterministic 70/15/15 splits."""
    return _make_group_split(
        compounds,
        protocol_name="drug_disjoint_v2",
        group_column="canonical_smiles",
        seed=seed,
        fractions=fractions,
    )


def make_scaffold_disjoint_v2(
    compounds: EligibilityResult | pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
    fractions: Mapping[str, float] = TARGET_FRACTIONS,
) -> SplitResult:
    """Assign Bemis-Murcko scaffold groups to deterministic 70/15/15 splits."""
    return _make_group_split(
        compounds,
        protocol_name="scaffold_disjoint_v2",
        group_column="scaffold_smiles",
        seed=seed,
        fractions=fractions,
    )


def _balanced_binary_assignment(
    cell_ids: Iterable[str],
    *,
    seed: int,
    domain: str,
    labels: tuple[str, str],
) -> dict[str, str]:
    identifiers = [str(cell_id) for cell_id in cell_ids]
    if len(identifiers) < 2:
        raise ValueError(
            f"stratum {domain!r} needs at least two cells for non-empty pools."
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"stratum {domain!r} contains duplicate cell IDs.")
    ordered = sorted(
        identifiers,
        key=lambda value: (_seeded_digest(seed, domain, value), value),
    )
    cut = len(ordered) // 2
    return {
        cell_id: labels[0] if index < cut else labels[1]
        for index, cell_id in enumerate(ordered)
    }


def canonical_pool_sha256(
    assignments: pd.DataFrame,
    *,
    protocol_name: str,
    seed: int,
) -> str:
    required = {"cell_id", "context", "batch", "pool"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"pool assignments are missing column(s): {missing}.")
    optional = [name for name in ("drug_id", "kind", "half") if name in assignments]
    fields = sorted(required | set(optional))
    records = [
        {name: str(row[name]) for name in fields}
        for row in assignments.to_dict("records")
    ]
    records.sort(key=lambda row: tuple(row[name] for name in fields))
    payload = {
        "protocol_name": protocol_name,
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "assignments": records,
    }
    canonical_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def make_vehicle_reference_pools(
    metadata: pd.DataFrame,
    *,
    columns: MetadataColumns = MetadataColumns(),
    target_contexts: Sequence[str] = TARGET_CONTEXTS,
    time_h: float = TARGET_TIME_H,
    seed: int = VEHICLE_SEED,
) -> PoolResult:
    """Split vehicles into non-overlapping model-input and truth-reference pools."""
    contexts = _validate_contexts(target_contexts)
    work = _normalize_metadata(metadata, columns)
    vehicles = work[
        work[columns.is_control]
        & work[columns.context].isin(contexts)
        & work["_benchmark_time_h"].eq(float(time_h))
    ].copy()
    if vehicles.empty:
        raise ValueError(
            "no matching vehicle rows are available at the registered time."
        )

    rows: list[dict[str, str]] = []
    for (context, batch), stratum in vehicles.groupby([columns.context, columns.batch]):
        domain = f"vehicle-reference|{context}|{batch}"
        assigned = _balanced_binary_assignment(
            stratum[columns.cell_id],
            seed=int(seed),
            domain=domain,
            labels=("vehicle_input", "vehicle_truth"),
        )
        for cell_id in stratum[columns.cell_id]:
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "context": str(context),
                    "batch": str(batch),
                    "pool": assigned[str(cell_id)],
                }
            )
    assignments = pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)
    for (_context, _batch), stratum in assignments.groupby(["context", "batch"]):
        if set(stratum["pool"]) != {"vehicle_input", "vehicle_truth"}:
            raise RuntimeError(
                "a vehicle stratum did not receive both reference pools."
            )
    assignment_hash = canonical_pool_sha256(
        assignments,
        protocol_name="vehicle_input_truth_v1",
        seed=int(seed),
    )
    manifest = {
        "protocol_name": "vehicle_input_truth_v1",
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "strata": [columns.context, columns.batch],
        "time_h": float(time_h),
        "pool_counts": {
            name: int(assignments["pool"].eq(name).sum())
            for name in ("vehicle_input", "vehicle_truth")
        },
        "assignment_sha256": assignment_hash,
    }
    return PoolResult(assignments=assignments, manifest=manifest)


def make_technical_split_half_pools(
    metadata: pd.DataFrame,
    *,
    columns: MetadataColumns = MetadataColumns(),
    eligible_drug_ids: Iterable[str] | None = None,
    target_contexts: Sequence[str] = TARGET_CONTEXTS,
    dose_um: float = TARGET_DOSE_UM,
    time_h: float = TARGET_TIME_H,
    seed: int = VEHICLE_SEED,
) -> PoolResult:
    """Create independent A/B treated and A/B vehicle technical pools.

    Treated rows are stratified by drug, context, and batch.  Vehicle rows are
    independently stratified by context and batch, so half A and half B never use
    the same vehicle cell.
    """
    contexts = _validate_contexts(target_contexts)
    work = _normalize_metadata(metadata, columns)
    eligible_ids = (
        None
        if eligible_drug_ids is None
        else {str(value) for value in eligible_drug_ids}
    )
    treated_mask = (
        ~work[columns.is_control]
        & work[columns.context].isin(contexts)
        & work["_benchmark_dose_um"].eq(float(dose_um))
        & work["_benchmark_time_h"].eq(float(time_h))
    )
    if eligible_ids is not None:
        treated_mask &= work[columns.drug_id].isin(eligible_ids)
    treated = work[treated_mask].copy()
    vehicles = work[
        work[columns.is_control]
        & work[columns.context].isin(contexts)
        & work["_benchmark_time_h"].eq(float(time_h))
    ].copy()
    if treated.empty:
        raise ValueError(
            "no exact-dose/time treated rows are available for technical halves."
        )
    if vehicles.empty:
        raise ValueError(
            "no registered-time vehicle rows are available for technical halves."
        )

    treated_strata = set(
        treated[[columns.context, columns.batch]].itertuples(index=False, name=None)
    )
    vehicle_strata = set(
        vehicles[[columns.context, columns.batch]].itertuples(index=False, name=None)
    )
    missing_vehicle_strata = sorted(treated_strata - vehicle_strata)
    if missing_vehicle_strata:
        raise ValueError(
            "treated strata lack matching vehicle controls: "
            f"{missing_vehicle_strata[:3]}."
        )

    rows: list[dict[str, str]] = []
    for (drug_id, context, batch), stratum in treated.groupby(
        [columns.drug_id, columns.context, columns.batch]
    ):
        domain = f"technical-treated|{drug_id}|{context}|{batch}"
        assigned = _balanced_binary_assignment(
            stratum[columns.cell_id],
            seed=int(seed),
            domain=domain,
            labels=("A", "B"),
        )
        for cell_id in stratum[columns.cell_id]:
            half = assigned[str(cell_id)]
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "drug_id": str(drug_id),
                    "context": str(context),
                    "batch": str(batch),
                    "kind": "treated",
                    "half": half,
                    "pool": f"treated_{half}",
                }
            )

    for (context, batch), stratum in vehicles.groupby([columns.context, columns.batch]):
        domain = f"technical-vehicle|{context}|{batch}"
        assigned = _balanced_binary_assignment(
            stratum[columns.cell_id],
            seed=int(seed),
            domain=domain,
            labels=("A", "B"),
        )
        for cell_id in stratum[columns.cell_id]:
            half = assigned[str(cell_id)]
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "drug_id": str(stratum.iloc[0][columns.drug_id]),
                    "context": str(context),
                    "batch": str(batch),
                    "kind": "vehicle",
                    "half": half,
                    "pool": f"vehicle_{half}",
                }
            )

    assignments = (
        pd.DataFrame(rows).sort_values(["kind", "cell_id"]).reset_index(drop=True)
    )
    if assignments["cell_id"].duplicated().any():
        raise RuntimeError("technical A/B pools contain an overlapping cell ID.")
    for (_kind, *_stratum), group in assignments.groupby(["kind", "context", "batch"]):
        if set(group["half"]) != {"A", "B"}:
            raise RuntimeError("a technical stratum did not receive both halves.")
    assignment_hash = canonical_pool_sha256(
        assignments,
        protocol_name="technical_split_half_v1",
        seed=int(seed),
    )
    manifest = {
        "protocol_name": "technical_split_half_v1",
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "treated_strata": [columns.drug_id, columns.context, columns.batch],
        "vehicle_strata": [columns.context, columns.batch],
        "dose_um": float(dose_um),
        "time_h": float(time_h),
        "pool_counts": {
            name: int(assignments["pool"].eq(name).sum())
            for name in ("treated_A", "treated_B", "vehicle_A", "vehicle_B")
        },
        "assignment_sha256": assignment_hash,
    }
    return PoolResult(assignments=assignments, manifest=manifest)
