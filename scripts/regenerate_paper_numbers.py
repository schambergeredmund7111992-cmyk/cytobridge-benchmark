#!/usr/bin/env python
"""Regenerate every table/figure number of the paper from the shipped inputs
and reconcile them against manuscript/analysis/expected_values.json.

Usage:
    python scripts/regenerate_paper_numbers.py --bundle regeneration_inputs.zip
    python scripts/regenerate_paper_numbers.py --bundle regeneration_inputs/ \
        --h5ad /path/to/SrivatsanTrapnell2020_sciplex3.h5ad   # Table 8 technical

The pipeline aborts if the construction invariants fail, so a wrong vehicle
construction can never produce a green reconciliation report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.regenerate.calibration import calibration_curve, effective_alpha
from scripts.regenerate.constructions import (
    cell_line_means,
    check_invariants,
    conversion_delta,
    derive_pooled,
)
from scripts.regenerate.drug_clustered_auc_bootstrap import (
    between_drug_sd,
    delete_one_drug_sd,
    drug_clustered_auc_bootstrap,
    permutation_null,
    power_analysis,
)
from scripts.regenerate.figures import write_figures_data
from scripts.regenerate.inputs import CONFIG_ORDER, BundleLoader
from scripts.regenerate.oracle_ladder import build_oracle_ladder
from scripts.regenerate.selfspace_audit import audit_all
from scripts.regenerate.table8_ceiling import biological_ceiling, technical_ceiling
from scripts.regenerate.tables import table3, table4, table5, table7
from scripts.regenerate.verify import load_expected, run_reconciliation

SEED = 7301

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = REPO / "manuscript" / "analysis" / "expected_values.json"


def _pooled_predictions(loader: BundleLoader, delta: np.ndarray) -> dict:
    pooled: dict = {}
    for name in CONFIG_ORDER:
        config = loader.e6e7_config(name)
        if config is None:
            continue
        pooled[name] = derive_pooled(config["pred"], delta)
    return pooled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="regeneration_inputs.zip or extracted directory")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--h5ad", type=Path, default=None,
                        help="cell-level sci-Plex h5ad for the Table 8 technical ceiling")
    parser.add_argument("--strict", action="store_true",
                        help="treat SKIP entries as failures")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loader = BundleLoader.from_path(args.bundle)
    results: dict = {}
    artifacts: dict = {}

    # ---------------- load + construct ----------------
    truth = loader.pooled_truth()
    if truth is None:
        print("[load] pooled truth missing; cannot construct pooled space")
        return 1
    pooled_true = truth["true"].astype(np.float64)
    meta = truth["meta"]
    cl = meta["cell_line"].astype(str).to_numpy()
    drugs = meta["drug"].astype(str).to_numpy()

    reference = loader.e6e7_config("loss-only")
    if reference is None:
        print("[load] loss-only per-pair matrices missing")
        return 1
    true_perpair = reference["true"].astype(np.float64)
    delta = conversion_delta(true_perpair, pooled_true)
    pooled_preds = _pooled_predictions(loader, delta)
    if not pooled_preds:
        print("[load] no per-pair config matrices found")
        return 1

    problems = check_invariants(
        true_perpair=true_perpair,
        true_pooled=pooled_true,
        pooled_predictions=pooled_preds,
        cell_lines=cl,
    )
    if problems:
        print("[construct] invariant failures; aborting:")
        for problem in problems:
            print("  -", problem)
        return 1
    artifacts["delta"] = delta
    print(f"[construct] pooled construction OK "
          f"({len(pooled_preds)} configs, delta max |.|="
          f"{float(np.abs(delta).max()):.4f})")

    # ---------------- tables 3 / 4 / 5 / 7 ----------------
    results.update(table3(loader))
    results.update(table4(loader, pooled_preds, pooled_true, cl))
    results.update(table5(loader, pooled_preds, pooled_true, cl, true_perpair))
    results.update(table7(loader, pooled_preds, pooled_true, cl))

    cb_aucs = [v for k, v in results.items()
               if k.startswith("t5.") and k.endswith(".auc")]
    best_auc = max(cb_aucs) if cb_aucs else None

    # ---------------- uncertainty (Fig 4e / Sec 4.5) ----------------
    loss_only = pooled_preds.get("loss-only")
    if loss_only is not None:
        boot = drug_clustered_auc_bootstrap(
            loss_only, pooled_true, cl, drugs, n_boot=1000, seed=SEED
        )
        results["fig4e.ci_lo"] = boot["ci_lo"]
        results["fig4e.ci_hi"] = boot["ci_hi"]
        artifacts["bootstrap"] = boot

        perm = permutation_null(loss_only, pooled_true, cl, n_perm=1000, seed=SEED)
        results["sec45.perm_p_loss_only"] = perm["p_value"]
        results["sec45.null_mean"] = perm["null_mean"]
        results["sec45.null_sd"] = perm["null_sd"]
        artifacts["permutation"] = perm

        best_name = max(pooled_preds, key=lambda n: results.get(f"t5.{_id(n)}.auc", 0.0))
        perm_best = permutation_null(
            pooled_preds[best_name], pooled_true, cl, n_perm=1000, seed=SEED
        )
        results["sec45.perm_p_best"] = perm_best["p_value"]

        results["sec45.loo_sd"] = delete_one_drug_sd(
            loss_only, pooled_true, cl, drugs
        )
        between = between_drug_sd(loss_only, pooled_true, cl, drugs)
        results["sec45.between_drug_sd"] = between
        power = power_analysis(between, n_drugs=9)
        results["sec45.power70"] = power["power70"]
        results["sec45.power60"] = power["power60"]
        results["sec45.power55"] = power["power55"]
        artifacts["power"] = power

    # ---------------- calibration / ladder (Fig 4a/b) ----------------
    curve = calibration_curve(pooled_true, cl)
    if best_auc is not None:
        results["fig4a.eff_alpha"] = effective_alpha(curve, best_auc)
    artifacts["curve"] = curve
    artifacts["best_auc"] = best_auc

    rng = np.random.default_rng(SEED)
    random_aucs = []
    shuffled = pooled_true.copy()
    for _ in range(50):
        for cell in np.unique(cl):
            mask = cl == cell
            shuffled[mask] = rng.permutation(shuffled[mask])
        from eval.metrics import drug_discrimination_score
        random_aucs.append(
            float(
                drug_discrimination_score(
                    shuffled, pooled_true, cl, top_k=50, metric="pearson"
                )["specificity_auc"]
            )
        )
    results["fig4b.ladder.random"] = float(np.mean(random_aucs))
    mean_pooled = cell_line_means(pooled_true, cl)
    from eval.metrics import drug_discrimination_score as ddc
    results["fig4b.ladder.mean"] = float(
        ddc(mean_pooled, pooled_true, cl, top_k=50, metric="pearson")["specificity_auc"]
    )
    results["fig4b.ladder.oracle"] = 1.0
    results["fig4b.ladder.cb"] = best_auc
    results["fig4b.ladder.ridge"] = results.get("t6.ridge.auc") or results.get("t4.ridge")
    results["fig4b.ladder.chemcpa"] = results.get("t5.chemcpa.auc")
    results["fig4b.ladder.biolord"] = results.get("t5.biolord.auc")

    # ---------------- oracles (Fig 5) ----------------
    results.update(build_oracle_ladder(loader, pooled_true, meta))

    # ---------------- ceilings (Table 8) ----------------
    replicates = loader.replicates()
    if replicates is not None:
        results.update(
            biological_ceiling(replicates["rep1"], replicates["rep2"], meta, drugs)
        )
        results["fig5.ceiling"] = results.get("t8.biological.auc")
    if args.h5ad is not None:
        results.update(technical_ceiling(Path(args.h5ad), meta, n_splits=20, seed0=0))

    # ---------------- self-space audit (Table 6) ----------------
    results.update(audit_all(loader, reference, {"pred": loss_only}))

    # ---------------- figures_data ----------------
    if not args.skip_figures:
        write_figures_data(
            out,
            loader=loader,
            results=results,
            pooled_preds=pooled_preds,
            pooled_true=pooled_true,
            true_perpair=true_perpair,
            meta=meta,
            curve=curve,
            best_auc=best_auc,
            bootstrap=artifacts.get("bootstrap"),
            permutation=artifacts.get("permutation"),
        )

    # ---------------- reconcile ----------------
    (out / "results.json").write_text(
        json.dumps(
            {key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
             for key, value in sorted(results.items())},
            indent=2,
        )
        + "\n"
    )
    reconciliation = run_reconciliation(
        load_expected(args.expected),
        results,
        missing_inputs=loader.missing,
        figures_data_dir=out / "figures_data",
    )
    report_path = out / "reconciliation_report.md"
    report_path.write_text(reconciliation["report_md"], encoding="utf-8")
    print(reconciliation["report_md"].splitlines()[2])
    print(f"report written to {report_path}")

    failed = reconciliation["failures"] > 0
    if args.strict and reconciliation["skipped"] > 0:
        failed = True
    return 1 if failed else 0


def _id(name: str) -> str:
    return {
        "loss-only": "loss_only",
        "drug-spec x1": "drugspec1",
        "drug-spec x3": "drugspec3",
        "drug-spec x5": "drugspec5",
        "norm-only": "norm_only",
        "low recon weight": "low_recon",
        "recovery baseline": "recovery_base",
    }[name]


if __name__ == "__main__":
    sys.exit(main())
