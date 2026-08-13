"""
eval/run_gdsc2.py
-----------------
Bulk transfer test: predict GDSC2 IC50 from CytoBridge embeddings.

For each (drug, cell-line) pair in GDSC2:
    - get cell-line baseline transcriptome (CCLE pseudobulk)
    - encode via scGPT
    - encode drug SMILES via MolFormer
    - run CytoBridge → predicted post-treatment expression
    - compute viability score = some function of (predicted - control) signature
    - correlate with measured IC50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from eval.run_internal import load_model


def _read_ic50_table(path: Path) -> pd.DataFrame:
    """Load GDSC2 dose-response data; the official download is .xlsx, accept .csv/.tsv too."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--ic50_path", type=Path,
                        default=Path("data/raw/gdsc2/GDSC2_fitted_dose_response.xlsx"),
                        help="GDSC2 dose-response file. data/download.py emits .xlsx by default; "
                             ".csv/.tsv are also accepted.")
    parser.add_argument("--ccle_emb", type=Path,
                        default=Path("data/cache/ccle_scgpt_emb.npy"),
                        help="precomputed scGPT embeddings of CCLE cell lines")
    parser.add_argument("--ccle_meta", type=Path,
                        default=Path("data/cache/ccle_cell_lines.csv"))
    parser.add_argument("--drug_emb", type=Path,
                        default=Path("data/cache/gdsc2_molformer_emb.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/gdsc2_transfer.csv"))
    args = parser.parse_args()

    model = load_model(args.ckpt)
    ic50 = _read_ic50_table(args.ic50_path)
    ccle_emb = np.load(args.ccle_emb, mmap_mode="r")
    ccle_meta = pd.read_csv(args.ccle_meta)
    drug_data = np.load(args.drug_emb)

    pred_score = []
    obs_ic50 = []
    drug_list, cell_list = [], []

    drug_ids = [str(x).lower() for x in drug_data["drug_ids"]]
    model.eval()
    for _, row in ic50.iterrows():
        cell = str(row["CELL_LINE_NAME"])
        drug = str(row["DRUG_NAME"])
        cell_hits = ccle_meta.index[ccle_meta["CELL_LINE_NAME"].astype(str) == cell].tolist()
        if not cell_hits:
            continue
        drug_key = drug.lower()
        if drug_key not in drug_ids:
            continue
        cidx = int(cell_hits[0])
        didx = drug_ids.index(drug_key)
        cell_tok = torch.from_numpy(np.asarray(ccle_emb[cidx])).float().unsqueeze(0)
        drug_tok = torch.from_numpy(np.asarray(drug_data["tokens"][didx])).float().unsqueeze(0)
        drug_mask = torch.from_numpy(np.asarray(drug_data["masks"][didx])).bool().unsqueeze(0)
        device = next(model.parameters()).device
        with torch.no_grad():
            out = model(cell_tok.to(device), drug_tok.to(device), drug_mask.to(device))
        # Larger transcriptional displacement is treated as stronger response.
        score = float(torch.log1p(out["mu"]).abs().mean().cpu())
        pred_score.append(score)
        obs_ic50.append(float(row["LN_IC50"]))
        drug_list.append(drug)
        cell_list.append(cell)

    df = pd.DataFrame({"drug": drug_list, "cell": cell_list,
                       "pred_score": pred_score, "obs_ic50": obs_ic50})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Per-cell-line summary: one Spearman per cell line so that figures/fig3
    # can read `spearman_top50` and bootstrap 95% CIs across cell lines.
    # The signed correlation is reported (we expect negative — higher
    # predicted response score → lower IC50 — but downstream panels show the
    # signed value so the direction stays visible).
    summary_rows = []
    for cell, g in df.groupby("cell"):
        if len(g) < 3:
            continue
        rho = spearmanr(g["pred_score"], g["obs_ic50"]).statistic
        if np.isnan(rho):
            continue
        summary_rows.append({"cell_line": cell, "n_pairs": int(len(g)),
                             "spearman_top50": float(rho)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out, index=False)

    detail_path = args.out.with_name(args.out.stem + "_per_pair" + args.out.suffix)
    df.to_csv(detail_path, index=False)

    rho = spearmanr(df["pred_score"], df["obs_ic50"]).statistic if len(df) >= 3 else float("nan")
    print("\n=== GDSC2 Bulk Transfer ===")
    print(f"Pooled Spearman(pred_score, observed IC50): {rho:.4f}")
    if not summary.empty:
        print(f"Per-cell-line Spearman: mean {summary['spearman_top50'].mean():.4f} "
              f"(median {summary['spearman_top50'].median():.4f}, n_cells={len(summary)})")
    print(f"saved per-cell summary -> {args.out}")
    print(f"saved per-pair detail  -> {detail_path}")
    print("(Negative correlation expected if higher score = stronger predicted response.)")


if __name__ == "__main__":
    main()
