#!/usr/bin/env python
"""
eval/verify_drug_specificity.py  (the decisive "is it real" gate)
-----------------------------------------------------------------------------
Runs the off-diagonal drug-shuffle control on a saved (pred, true) artifact, for
EITHER pathway-attribution vectors OR expression logFC. Answers the one question
that motivated this control: is a high per-pair correlation DRUG-SPECIFIC, or just
the shared structure that every pair has in common (a collapsed/constant output)?

Smoking gun this guards against: in the model's pathway_gate was numerically
IDENTICAL for every drug (incl. an empty SMILES), yet per-pair Pearson vs GSEA NES
was 0.95 ("27/27 positive"). drug_discrimination_score exposes that: gap ~= 0 and
specificity_auc ~= 0.5 => the 0.95 is an artifact, NOT a faithful attribution.

Inputs (per-sample OR per-pair; --aggregate means per (drug, cell_line)):
    --pred  pred.npy   [N, D]   predicted vectors
    --true  true.npy   [N, D]   ground-truth vectors
    --meta  meta.csv   columns including a drug column and a cell-line column
Convenience: --attr_dir <dir> loads <dir>/pred.npy, true.npy, meta.csv at once
(matches the layout of results/pathway_attribution/).

Verdict (printed + JSON):
    PASS  gap > 0, specificity_auc > 0.7, wilcoxon p < 0.01  -> drug-specific signal
    FAIL  otherwise                                           -> collapse artifact

Examples:
    # pathway attribution (all 50 dims):
    python eval/verify_drug_specificity.py --attr_dir results/pathway_attribution \
        --modality pathway --aggregate --out results/verify_pathway.json
    # expression logFC (top-50 DEG), from a saved per-pair npz:
    python eval/verify_drug_specificity.py --pred results/logfc_pred.npy \
        --true results/logfc_true.npy --meta results/logfc_meta.csv \
        --modality logfc --out results/verify_logfc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import drug_discrimination_score, inter_drug_pearson  # noqa: E402


DRUG_COLS = ["drug", "drug_id", "product_name", "compound", "perturbation"]
CELL_COLS = ["cell_line", "cell", "cell_type", "celltype"]


def _pick(cols, candidates, what):
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(f"ERROR: no {what} column in meta (looked for {candidates}); got {list(cols)}")


def aggregate_per_pair(pred, true, meta, drug_col, cell_col):
    """Mean pred/true over samples within each (drug, cell_line) pair."""
    keys = list(zip(meta[drug_col].astype(str), meta[cell_col].astype(str)))
    order, seen = [], set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            order.append(k)
    idx_of = {k: i for i, k in enumerate(order)}
    sums_p = np.zeros((len(order), pred.shape[1]))
    sums_t = np.zeros((len(order), true.shape[1]))
    cnt = np.zeros(len(order))
    for r, k in enumerate(keys):
        i = idx_of[k]
        sums_p[i] += pred[r]
        sums_t[i] += true[r]
        cnt[i] += 1
    P = sums_p / cnt[:, None]
    T = sums_t / cnt[:, None]
    cls = np.array([k[1] for k in order])
    drugs = np.array([k[0] for k in order])
    return P, T, cls, drugs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--true")
    ap.add_argument("--meta")
    ap.add_argument("--attr_dir", help="dir with pred.npy/true.npy/meta.csv")
    ap.add_argument("--modality", choices=["pathway", "logfc"], default="pathway")
    ap.add_argument("--aggregate", action="store_true",
                    help="mean per (drug, cell_line) before the control (use for per-sample inputs)")
    ap.add_argument("--top_k", type=int, default=None,
                    help="dims to use; default = all for pathway, 50 for logfc")
    ap.add_argument("--metric", choices=["pearson", "spearman"], default="pearson")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.attr_dir:
        d = Path(args.attr_dir)
        pred = np.load(d / "pred.npy")
        true = np.load(d / "true.npy")
        meta = pd.read_csv(d / "meta.csv")
    else:
        if not (args.pred and args.true and args.meta):
            raise SystemExit("ERROR: provide --attr_dir OR (--pred --true --meta)")
        pred = np.load(args.pred)
        true = np.load(args.true)
        meta = pd.read_csv(args.meta)

    drug_col = _pick(meta.columns, DRUG_COLS, "drug")
    cell_col = _pick(meta.columns, CELL_COLS, "cell-line")
    print(f"[verify] modality={args.modality} pred={pred.shape} true={true.shape} "
          f"n_meta={len(meta)} drug_col={drug_col} cell_col={cell_col}")

    if args.aggregate or len(meta) != pred.shape[0] or len(np.unique(meta[drug_col])) < pred.shape[0]:
        P, T, cls, _drugs = aggregate_per_pair(pred, true, meta, drug_col, cell_col)
        print(f"[verify] aggregated to {P.shape[0]} (drug, cell_line) pairs")
    else:
        P, T = pred, true
        cls = meta[cell_col].astype(str).to_numpy()

    top_k = args.top_k if args.top_k is not None else (P.shape[1] if args.modality == "pathway" else 50)
    res = drug_discrimination_score(P, T, cls, top_k=top_k, metric=args.metric)
    res["inter_drug_pred_pearson"] = inter_drug_pearson(P, cls)
    res["n_pairs"] = int(P.shape[0])
    res["modality"] = args.modality

    passed = (
        res["gap"] is not None and not np.isnan(res["gap"]) and res["gap"] > 0
        and res["specificity_auc"] > 0.7
        and not np.isnan(res.get("wilcoxon_p_on_gt_off", np.nan))
        and res["wilcoxon_p_on_gt_off"] < 0.01
    )
    res["VERDICT"] = "PASS: drug-specific signal" if passed else "FAIL: collapse artifact (per-pair r is shared structure)"

    print(json.dumps(res, indent=2))
    print("\n" + "=" * 70)
    print(f"on_diag (usual per-pair r): {res['on_diag_mean']:.4f}")
    print(f"off_diag (drug-shuffled)  : {res['off_diag_mean']:.4f}")
    print(f"gap (drug-specific signal): {res['gap']:.4f}   "
          f"specificity_auc: {res['specificity_auc']:.4f}   "
          f"p(on>off): {res.get('wilcoxon_p_on_gt_off', float('nan')):.2e}")
    print(f"inter-drug pred Pearson   : {res['inter_drug_pred_pearson']:.4f}  (collapse if > 0.7)")
    print(f">>> {res['VERDICT']}")
    print("=" * 70)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
