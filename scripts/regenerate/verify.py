"""Reconcile computed results against the frozen expected-values manifest.

Reads a results dict {entry_id: value} and emits a markdown report classifying
every manifest entry as PASS / FAIL / SKIP / INFO.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.regenerate.manifest_spec import iter_entries, validate_manifest

VERDICTS = ("PASS", "FAIL", "SKIP", "INFO")


def _status_for(
    entry: dict,
    tolerance: float,
    computed: Any,
    missing_inputs: set[str],
) -> tuple[str, str]:
    requires = set(entry.get("requires", []) or [])
    if requires and not requires.isdisjoint(missing_inputs):
        return "SKIP", f"required input missing: {sorted(requires & missing_inputs)}"
    if computed is None:
        return "INFO", "not computed (producer unavailable)"
    if isinstance(computed, bool) or not isinstance(computed, (int, float)):
        return "INFO", f"non-numeric computed value {computed!r}"
    expect = float(entry["expect"])
    value = float(computed)
    delta = value - expect
    if abs(delta) <= tolerance:
        return "PASS", f"delta={delta:+.4g} (tol {tolerance:g})"
    return "FAIL", f"delta={delta:+.4g} (tol {tolerance:g})"


def run_reconciliation(
    expected: dict,
    results: dict,
    *,
    missing_inputs: set[str] | None = None,
    figures_data_dir: Path | None = None,
) -> dict:
    """Return {verdict, report_md, failures, skipped, info}."""
    problems = validate_manifest(expected)
    if problems:
        raise ValueError(
            "expected-values manifest is invalid:\n  " + "\n  ".join(problems)
        )
    missing_inputs = set(missing_inputs or ())
    if figures_data_dir is not None:
        figures_data_dir = Path(figures_data_dir)
        for key in ("bootstrap_auc.npy", "bootstrap_meta.json"):
            if not (figures_data_dir / key).exists():
                missing_inputs.add(f"figures_data/{key}")

    rows: list[str] = []
    counts = {verdict: 0 for verdict in VERDICTS}
    for entry, tolerance in iter_entries(expected):
        computed = results.get(entry["id"])
        verdict, note = _status_for(entry, tolerance, computed, missing_inputs)
        counts[verdict] += 1
        rows.append(
            f"| {entry['id']} | {entry['label']} | {entry['expect']} | "
            f"{computed if computed is not None else '-'} | {entry['tol']} | "
            f"{entry['kind']} | {verdict} | {note} |"
        )

    summary = (
        f"PASS {counts['PASS']} / FAIL {counts['FAIL']} / "
        f"SKIP {counts['SKIP']} / INFO {counts['INFO']} "
        f"(of {sum(counts.values())} entries)"
    )
    header = (
        "| entry | label | expected | computed | tol | kind | verdict | note |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    report = (
        "# Reconciliation report\n\n"
        f"**{summary}**\n\n"
        + header
        + "\n"
        + "\n".join(rows)
        + "\n"
    )
    return {
        "verdict": "PASS" if counts["FAIL"] == 0 else "FAIL",
        "report_md": report,
        "failures": counts["FAIL"],
        "skipped": counts["SKIP"],
        "info": counts["INFO"],
    }


def load_expected(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
