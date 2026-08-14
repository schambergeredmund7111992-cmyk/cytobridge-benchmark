"""Tests for the expected-values manifest schema and the reconciliation logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.regenerate.manifest_spec import validate_manifest
from scripts.regenerate.verify import run_reconciliation

REPO = Path(__file__).resolve().parents[1]
EXPECTED_PATH = REPO / "manuscript" / "analysis" / "expected_values.json"


def _entry(**overrides):
    base = {
        "id": "t4.mean",
        "label": "test",
        "expect": 0.491,
        "tol": "round3",
        "kind": "deterministic",
        "producer": "tables.table4",
    }
    base.update(overrides)
    return base


def test_shipped_manifest_is_valid():
    data = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    assert validate_manifest(data) == []


def test_manifest_rejects_duplicate_ids():
    problems = validate_manifest({"entries": [_entry(), _entry()]})
    assert any("duplicate id" in p for p in problems)


def test_manifest_rejects_unknown_tol():
    problems = validate_manifest({"entries": [_entry(tol="nonsense")]})
    assert any("unknown tol" in p for p in problems)


def test_manifest_rejects_missing_field():
    entry = _entry()
    del entry["producer"]
    problems = validate_manifest({"entries": [entry]})
    assert any("missing field" in p for p in problems)


def test_reconciliation_pass_fail_skip():
    manifest = {
        "entries": [
            _entry(id="a", expect=0.5, tol="round3"),
            _entry(id="b", expect=0.5, tol="round3"),
            _entry(id="c", expect=0.5, tol="round3", requires=["h5ad"]),
        ]
    }
    results = {"a": 0.5001, "b": 0.52}
    out = run_reconciliation(
        manifest, results, missing_inputs={"h5ad"}
    )
    report = out["report_md"]
    assert "| a |" in report and "PASS" in report
    assert "| b |" in report and "FAIL" in report
    assert "| c |" in report and "SKIP" in report
    assert out["verdict"] == "FAIL"
    assert out["failures"] == 1 and out["skipped"] == 1


def test_reconciliation_missing_result_is_info():
    manifest = {"entries": [_entry(id="a", expect=0.5)]}
    out = run_reconciliation(manifest, {})
    assert "INFO" in out["report_md"]
    assert out["verdict"] == "PASS"


def test_invalid_manifest_raises():
    with pytest.raises(ValueError):
        run_reconciliation({"entries": [_entry(tol="bad")]}, {})


def test_tolerances_map_to_printed_precision():
    manifest = {"entries": [_entry(id="a", expect=0.509, tol="round3")]}
    out = run_reconciliation(manifest, {"a": 0.5094})
    assert out["verdict"] == "PASS"
    out = run_reconciliation(manifest, {"a": 0.5105})
    assert out["verdict"] == "FAIL"
