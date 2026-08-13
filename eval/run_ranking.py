"""eval/run_ranking.py
--------------------
Drug-ranking decision-support eval (the quantitative decision-support hook).

Default ranking task (DEFENSIBLE DEFAULT — confirm framing with CGX before it
becomes the paper's headline number):
  For each cell line, score every UNSEEN test drug by predicted perturbation
  STRENGTH = ||pred_logfc||_2 over the pair's top-K DEGs, and rank drugs by it.
  Ground-truth ranking = ||true_logfc||_2. Report:
    - per-cell-line Spearman(pred drug-ranking, true drug-ranking)
    - Hit@K / NDCG@K for recovering the strongest-perturbing drugs
    - cross-cell-line consistency of the predicted ranking
  The Mean(per-cell-line) baseline is drug-AGNOSTIC, so it scores ~0 here by
  construction — this is exactly the capability gap Option E must demonstrate.

Run (after the anti-collapse gate passes):
  cd code
  python eval/run_ranking.py --ckpt ckpts/t6_collapse_fix/last.ckpt

Reuses run_internal's model load + prediction + pseudobulk aggregation so the
predictions are identical to the headline eval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cytobridge.data import CytoBridgeDataset
from eval.metrics import hit_at_k, ndcg_at_k
from eval.run_internal import aggregate_by_pair, load_model, predict_all


def _effect_strength(logfc: np.ndarray, top_k: int = 50) -> float:
    """L2 norm of the logFC restricted to that pair's own top-K |logFC| genes."""
    idx = np.argsort(-np.abs(logfc))[:top_k]
    return float(np.linalg.norm(logfc[idx]))


def rank_drugs_within_cell_line(delta: np.ndarray, drugs, cells, top_k=50):
    """Return {cell_line: DataFrame(drug, strength)} ranked desc by strength."""
    meta = pd.DataFrame({"drug": drugs, "cell": cells})
    out = {}
    for cl, sub in meta.groupby("cell"):
        rows = [(meta.loc[i, "drug"], _effect_strength(delta[i], top_k)) for i in sub.index]
        df = pd.DataFrame(rows, columns=["drug", "strength"]).sort_values(
            "strength", ascending=False).reset_index(drop=True)
        out[str(cl)] = df
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/sciplex/splits/sciplex_test.parquet"))
    ap.add_argument("--cell_emb", type=Path, default=Path("data/cache/sciplex_scgpt_emb.npy"))
    ap.add_argument("--drug_emb", type=Path, default=Path("data/cache/sciplex_molformer_emb.npz"))
    ap.add_argument("--counts", type=Path,
                    default=Path("data/processed/sciplex/splits/sciplex_test_treated_counts.npy"))
    ap.add_argument("--control_counts", type=Path,
                    default=Path("data/processed/sciplex/splits/sciplex_test_control_counts.npy"))
    ap.add_argument("--gsea", type=Path,
                    default=Path("data/processed/sciplex/splits/sciplex_test_pathway_gsea.npy"))
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("results/ranking_t6.json"))
    args = ap.parse_args()

    model = load_model(args.ckpt)
    ds = CytoBridgeDataset(
        manifest_path=args.manifest, cell_emb_path=args.cell_emb,
        drug_emb_path=args.drug_emb, treated_counts_path=args.counts,
        pathway_gsea_path=args.gsea, control_counts_path=args.control_counts,
        n_hard_same_drug=0, n_hard_same_cell=0,
    )
    pred_mu, true_counts, ctrl_counts, drugs, cells = predict_all(model, ds)
    preds, trues, drugs, cells = aggregate_by_pair(pred_mu, true_counts, ctrl_counts, drugs, cells)

    pred_rank = rank_drugs_within_cell_line(preds, drugs, cells, args.top_k)
    true_rank = rank_drugs_within_cell_line(trues, drugs, cells, args.top_k)

    # per-cell-line Spearman of predicted vs true drug ranking (by strength)
    report = {"per_cell_line": {}, "top_k": args.top_k}
    score_mats = []
    cl_order = sorted(pred_rank)
    for cl in cl_order:
        pr = pred_rank[cl].set_index("drug")["strength"]
        tr = true_rank[cl].set_index("drug")["strength"]
        common = [d for d in pr.index if d in tr.index]
        if len(common) < 3:
            continue
        rho = float(spearmanr(pr[common].values, tr[common].values).statistic)
        # Hit@K / NDCG: relevant = top-3 strongest TRUE drugs
        true_top = set(tr.sort_values(ascending=False).index[:3])
        scores = pr[common].values[None, :]
        relevant = np.array([[d in true_top for d in common]])
        relevance = tr[common].values[None, :]
        report["per_cell_line"][cl] = {
            "rank_spearman": rho,
            "hit_at_k": hit_at_k(scores, relevant, ks=(1, 3)),
            "ndcg_at_3": ndcg_at_k(scores, relevance, k=3),
            "n_drugs": len(common),
        }
        score_mats.append(pr[common])

    rhos = [v["rank_spearman"] for v in report["per_cell_line"].values()]
    report["mean_rank_spearman"] = float(np.mean(rhos)) if rhos else float("nan")

    # cross-cell-line consistency: Spearman of predicted strength between cell lines
    if len(score_mats) >= 2:
        cons = []
        for i in range(len(score_mats)):
            for j in range(i + 1, len(score_mats)):
                a, b = score_mats[i], score_mats[j]
                common = [d for d in a.index if d in b.index]
                if len(common) >= 3:
                    cons.append(float(spearmanr(a[common].values, b[common].values).statistic))
        report["cross_cell_line_consistency"] = float(np.mean(cons)) if cons else float("nan")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nsaved -> {args.out}")
    print("NOTE: the Mean(per-cell-line) baseline predicts the SAME logFC for every drug, "
          "so all drugs get identical strength -> all ties -> rank Spearman is UNDEFINED "
          "(it literally cannot rank drugs). CytoBridge only needs a positive, finite "
          "mean_rank_spearman to demonstrate a capability the drug-agnostic baseline lacks.")


if __name__ == "__main__":
    main()
