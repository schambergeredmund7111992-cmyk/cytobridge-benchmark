"""End-to-end smoke test: run the full pipeline on a synthetic bundle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def test_pipeline_runs_on_smoke_bundle(tmp_path):
    from regenerate.make_smoke_bundle import build
    from regenerate_paper_numbers import main

    bundle = build(tmp_path / "smoke", zip_it=True)
    out = tmp_path / "out"
    exit_code = main(["--bundle", str(bundle), "--out", str(out),
                      "--skip-figures"])
    # Synthetic data cannot reproduce the paper's numbers, so FAILs are
    # expected; the pipeline itself must complete and write both artifacts.
    assert exit_code in (0, 1)
    assert (out / "results.json").is_file()
    assert (out / "reconciliation_report.md").is_file()

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    for key in ("t4.cb_loss_only", "t5.loss_only.auc", "t5.loss_only.gap",
                "t5.no_drug_info.pooled.auc", "fig4e.ci_lo", "fig4e.ci_hi",
                "sec45.perm_p_loss_only", "sec45.null_mean", "fig4b.ladder.mean",
                "fig4b.ladder.oracle", "fig5.hindsight", "t8.biological.auc"):
        assert key in results, f"missing computed entry {key}"
    # pooled no-drug-info anchor must be exactly chance (the construction invariant)
    assert abs(results["t5.no_drug_info.pooled.auc"] - 0.5) < 1e-6
    assert abs(results["fig4b.ladder.mean"] - 0.5) < 1e-6
    assert results["fig4b.ladder.oracle"] == 1.0
    report = (out / "reconciliation_report.md").read_text(encoding="utf-8")
    assert "PASS" in report or "FAIL" in report


def test_smoke_bundle_manifest_verifies(tmp_path):
    from scripts.regenerate.inputs import BundleLoader

    bundle = Path(__import__("regenerate.make_smoke_bundle", fromlist=["build"]).build(
        tmp_path / "smoke2", zip_it=True
    ))
    loader = BundleLoader.from_path(bundle)
    config = loader.e6e7_config("loss-only")
    assert config is not None
    assert config["pred"].shape == (27, 300)
    assert loader.pooled_truth() is not None
    assert loader.oracle_inputs() is not None
    assert loader.replicates() is not None
    assert loader.logs_metrics("loss-only") is not None
