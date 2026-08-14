"""Schema and validation for the frozen expected-values manifest.

The manifest (manuscript/analysis/expected_values.json) pins every printed
number of the paper. The regeneration pipeline computes the same quantities
from shipped inputs and `verify.py` reconciles the two.
"""
from __future__ import annotations

import math
from typing import Any

TOL_KEYS = (
    "exact",
    "round0",
    "round2",
    "round3",
    "round4",
    "ci_endpoint",
    "p_mc",
    "relative_1pct",
)
KINDS = ("deterministic", "stochastic")
_REQUIRED_ENTRY_FIELDS = {"id", "label", "expect", "tol", "kind", "producer"}


def tolerance_value(tol_key: str, expect: float, tolerances: dict) -> float:
    """Absolute tolerance for a tolerance key and expected value."""
    if tol_key not in TOL_KEYS:
        raise ValueError(f"unknown tolerance key {tol_key!r}")
    if tol_key == "relative_1pct":
        return abs(float(expect)) * 0.01
    return float(tolerances.get(tol_key, 0.0))


def _default_tolerances() -> dict:
    return {
        "exact": 1e-9,
        "round0": 0.5,      # integer-printed counts
        "round2": 0.005,    # 2 decimal places
        "round3": 0.0005,   # 3 decimal places
        "round4": 0.00005,  # 4 decimal places
        "ci_endpoint": 0.02,
        "p_mc": 0.05,
        "relative_1pct": "1%",
    }


def validate_manifest(data: dict) -> list[str]:
    """Return a list of problems; empty means the manifest is well formed."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["manifest root must be an object"]
    if "entries" not in data or not isinstance(data["entries"], list):
        return ["manifest must contain an 'entries' list"]
    tolerances = dict(_default_tolerances())
    tolerances.update(data.get("tolerances", {}) or {})
    for key in tolerances:
        if key not in TOL_KEYS:
            problems.append(f"tolerances: unknown key {key!r}")
    seen_ids: set[str] = set()
    for index, entry in enumerate(data["entries"]):
        where = f"entries[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: not an object")
            continue
        missing = sorted(_REQUIRED_ENTRY_FIELDS - set(entry))
        if missing:
            problems.append(f"{where}: missing field(s) {missing}")
            continue
        entry_id = entry["id"]
        if not isinstance(entry_id, str) or not entry_id:
            problems.append(f"{where}: id must be a non-empty string")
        elif entry_id in seen_ids:
            problems.append(f"{where}: duplicate id {entry_id!r}")
        else:
            seen_ids.add(entry_id)
        expect = entry["expect"]
        if isinstance(expect, bool) or not isinstance(expect, (int, float)):
            problems.append(f"{where}: expect must be numeric, got {expect!r}")
        elif not math.isfinite(float(expect)):
            problems.append(f"{where}: expect must be finite")
        if entry["tol"] not in TOL_KEYS:
            problems.append(f"{where}: unknown tol {entry['tol']!r}")
        if entry["kind"] not in KINDS:
            problems.append(f"{where}: unknown kind {entry['kind']!r}")
        if not isinstance(entry["producer"], str) or not entry["producer"]:
            problems.append(f"{where}: producer must be a non-empty string")
        random = entry.get("random")
        if random is not None and not isinstance(random, dict):
            problems.append(f"{where}: 'random' must be an object when present")
        requires = entry.get("requires")
        if requires is not None and not isinstance(requires, list):
            problems.append(f"{where}: 'requires' must be a list when present")
    return problems


def iter_entries(data: dict):
    """Yield (entry, tolerance) pairs in manifest order."""
    tolerances = dict(_default_tolerances())
    tolerances.update(data.get("tolerances", {}) or {})
    for entry in data["entries"]:
        yield entry, tolerance_value(entry["tol"], entry["expect"], tolerances)
