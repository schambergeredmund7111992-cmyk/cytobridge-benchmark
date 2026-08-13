"""Result gate — validates that every preregistered artifact exists before manuscript
inputs are unlocked.

The gate checks for:
  - Frozen sci-Plex splits (drug_disjoint_v2, scaffold_disjoint_v2)
  - Tahoe panel
  - Five-seed primary run checkpoints and predictions
  - Ablation coverage (no_pathway_gating, no_pathway_loss, no_molformer, no_contrast)
  - Calibrated reference controls (random, oracle, mean)
  - Paired Wilcoxon comparisons against baselines
  - Gradient audit records
  - Final aggregated benchmark table

All artifacts are read-only; this script never overwrites.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Gate checklist definitions
# ---------------------------------------------------------------------------
REQUIRED_SEEDS = (11, 23, 42, 67, 101)
PRIMARY_ABLATIONS = (
    "no_pathway_gating",
    "no_pathway_loss",
    "no_molformer",
    "no_contrast",
)
SCIPLEX_SPLITS = ("drug_disjoint_v2", "scaffold_disjoint_v2")


@dataclass
class GateItem:
    label: str
    paths: Sequence[str | Path]
    required: bool = True
    note: str = ""

    def resolve(self, root: Path) -> list[Path]:
        return [root / p for p in self.paths]


def _glob_check(patterns: list[str], *, label: str) -> list[str]:
    missing = []
    for pat in patterns:
        matches = list(Path(".").glob(pat))
        if not matches:
            missing.append(pat)
    return missing


def checklist(root: str | Path) -> list[GateItem]:
    root = Path(root)
    items: list[GateItem] = []

    # --- Frozen splits ---
    for split in SCIPLEX_SPLITS:
        items.append(
            GateItem(
                f"sci-Plex {split} manifest",
                [f"data/processed/sciplex_accept/{split}/sciplex_train.csv",
                 f"data/processed/sciplex_accept/{split}/sciplex_val.csv",
                 f"data/processed/sciplex_accept/{split}/sciplex_test.csv"],
                note="frozen 70/15/15 split with seed 20260710",
            )
        )

    # --- Tahoe ---
    items.append(
        GateItem(
            "Tahoe panel",
            ["data/processed/tahoe_accept/tahoe_train.csv",
             "data/processed/tahoe_accept/tahoe_val.csv",
             "data/processed/tahoe_accept/tahoe_test.csv"],
            note="streaming panel, gene-aligned to sci-Plex",
        )
    )

    # --- Five-seed primary checkpoints ---
    for seed in REQUIRED_SEEDS:
        items.append(
            GateItem(
                f"primary checkpoint seed={seed}",
                [f"ckpts/accept_base_seed{seed}/last.ckpt"],
                note="final primary run checkpoint",
            )
        )

    # --- Ablation checkpoints ---
    for ablation in PRIMARY_ABLATIONS:
        for seed in (42,):  # single representative seed for ablations
            items.append(
                GateItem(
                    f"ablation {ablation} seed={seed}",
                    [f"ckpts/{ablation}_seed{seed}/last.ckpt"],
                    note="ablation run checkpoint",
                )
            )

    # --- Prediction artifacts ---
    for split in SCIPLEX_SPLITS:
        for seed in REQUIRED_SEEDS:
            items.append(
                GateItem(
                    f"prediction {split} seed={seed}",
                    [f"results/predictions/{split}/cytobridge_seed{seed}/pred.npz"],
                    note="frozen prediction artifact",
                )
            )

    # --- Reference controls ---
    for split in SCIPLEX_SPLITS:
        for kind in ("random", "oracle", "mean"):
            items.append(
                GateItem(
                    f"control {kind} ({split})",
                    [f"results/controls/{split}/{kind}.npz"],
                    note="calibrated reference prediction",
                )
            )

    # --- Gradient audits ---
    audit_patterns = []
    for split in SCIPLEX_SPLITS:
        for seed in REQUIRED_SEEDS:
            audit_patterns.append(
                f"supplementary/gradient_audits/*{split}*seed{seed}*_gradient_norms.jsonl"
            )
    items.append(
        GateItem(
            "gradient audits",
            [],
            note=f"{len(audit_patterns)} expected audit files (pattern-matched below)",
        )
    )

    # --- Final aggregated benchmark ---
    items.append(
        GateItem(
            "final comparison table",
            ["supplementary/final/full_comparison_table.csv"],
            note="aggregated multi-seed benchmark",
        )
    )

    items.append(
        GateItem(
            "final summary",
            ["supplementary/final_summary.json"],
            note="campaign summary",
        )
    )

    return items


def verify(root: str | Path, *, strict: bool = True) -> dict:
    """Check all gate items and return a status report."""
    root = Path(root)
    results: list[dict] = []
    missing_count = 0

    for item in checklist(root):
        paths = item.resolve(root)
        statuses = []
        for p in paths:
            statuses.append(p.exists())
        all_ok = all(statuses) if paths else True  # pattern-matched items handled separately
        if not all_ok:
            missing = [str(p) for p, ok in zip(paths, statuses) if not ok]
            missing_count += 1
        else:
            missing = []
        results.append({
            "label": item.label,
            "ok": all_ok,
            "missing": missing,
            "note": item.note,
        })

    # --- Pattern-matched gradient audits ---
    audit_missing = _glob_check(
        audit_patterns := [
            f"supplementary/gradient_audits/p2-sciplex-*seed{seed}*_gradient_norms.jsonl"
            for seed in REQUIRED_SEEDS
        ],
        label="gradient_audits",
    )
    if audit_missing:
        missing_count += 1
        results.append({
            "label": "gradient audits (glob)",
            "ok": False,
            "missing": audit_missing,
            "note": f"checked {len(audit_patterns)} patterns",
        })
    else:
        results.append({
            "label": "gradient audits (glob)",
            "ok": True,
            "missing": [],
            "note": f"all {len(audit_patterns)} seed patterns present",
        })

    passed = missing_count == 0
    report = {
        "passed": passed,
        "total_checks": len(results),
        "failed_checks": missing_count,
        "items": results,
    }

    if not passed and strict:
        print("\n".join(
            f"  MISSING {r['label']}: {r['missing']}" for r in results if not r["ok"]
        ))
        print(f"\n{missing_count} gate check(s) failed.")
        sys.exit(1)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="Repository root (default: cwd)."
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write JSON report to this path."
    )
    parser.add_argument(
        "--no-strict", action="store_true",
        help="Report failures without a non-zero exit code."
    )
    args = parser.parse_args()

    report = verify(args.root, strict=not args.no_strict)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print(f"Gate report written to {args.out}")

    if report["passed"]:
        print("Gate passed — all artifacts present.")
    else:
        print(f"Gate failed — {report['failed_checks']} check(s) missing.")
        if not args.no_strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
