"""
eval/run_internal.py
--------------------
Run CytoBridge on sci-Plex internal val/test split.

Outputs results/cytobridge_internal.csv per-pair Pearson/Spearman/E-distance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cytobridge.data import CytoBridgeDataset, collate_with_hard_negs
from cytobridge.model import CytoBridge, CytoBridgeConfig
from eval.aggregation import aggregate_logfc_by_pair
from eval.metrics import (
    bootstrap_ci, drug_specific_delta_spearman, paired_wilcoxon,
    per_pair_pearson, per_pair_spearman, r2_per_pair, scale_report,
)


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(ckpt_path: Path, device: str = "auto") -> CytoBridge:
    """Load CytoBridge from a Lightning or plain-state checkpoint.

    Lightning checkpoints are preferred — we read ``hyper_parameters['model_cfg']``
    saved by ``LitCytoBridge`` so the model is reconstructed with the right
    config (ablations may toggle ``use_pathway_gate`` / ``K_pathways`` etc.).
    """
    device = resolve_device(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" in state:
        sd = {k.replace("model.", ""): v for k, v in state["state_dict"].items()
              if k.startswith("model.")}
        hp = state.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg") or {}
    else:
        sd = state
        model_cfg = {}
    cfg = CytoBridgeConfig(**model_cfg) if model_cfg else CytoBridgeConfig()
    model = CytoBridge(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match CytoBridgeConfig; missing={missing}, unexpected={unexpected}"
        )
    model.eval().to(device)
    return model


@torch.no_grad()
def predict_all(model: CytoBridge, dataset: CytoBridgeDataset, device: str = "auto",
                batch_size: int = 32, num_workers: int = 4):
    """Per-cell raw arrays for downstream pseudobulk aggregation.

    Returns ``(pred_mu, true_counts, ctrl_counts, drug_ids, cell_lines)``
    where each array is per-cell raw counts/predictions. The downstream
    `aggregate_by_pair` pseudobulks within each (drug, cell_line) pair and
    *then* applies log1p, matching the ridge baseline target
    (`log1p(mean treated) - log1p(mean control)`). Applying log1p per cell
    before averaging would compute a different statistic and make the
    `Δ vs ridge` headline an artefact of the aggregation choice, not the
    model.
    """
    device = resolve_device(device)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_with_hard_negs,
        "pin_memory": device == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs.update(
            {"multiprocessing_context": "spawn", "prefetch_factor": 1}
        )
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )
    pred_mu_list, treated_list, ctrl_list = [], [], []
    drug_ids, cell_lines = [], []
    for batch in tqdm(loader, desc="predict"):
        cell = batch["cell_tokens"].to(device)
        drug = batch["drug_tokens"].to(device)
        mask = batch["drug_mask"].to(device)
        input_ctrl = batch.get("input_control_counts", batch.get("control_counts"))
        truth_ctrl = batch.get("truth_control_counts", batch.get("control_counts"))
        # Pass control to the model so the residual decoder reconstructs
        # mu = expm1(log1p(control)+delta); harmless for the plain v1 decoder.
        ctrl_dev = input_ctrl.to(device) if input_ctrl is not None else None
        out = model(cell, drug, mask, control_counts=ctrl_dev)
        treated = batch["treated_counts"].cpu().numpy()
        if truth_ctrl is None:
            ctrl_arr = np.zeros_like(treated)
        else:
            ctrl_arr = truth_ctrl.cpu().numpy()
        pred_mu_list.append(out["mu"].cpu().numpy())
        treated_list.append(treated)
        ctrl_list.append(ctrl_arr)
        drug_ids.extend(batch["drug_ids"])
        cell_lines.extend(batch["cell_lines"])

    return (
        np.concatenate(pred_mu_list),
        np.concatenate(treated_list),
        np.concatenate(ctrl_list),
        drug_ids,
        cell_lines,
    )


def aggregate_by_pair(pred_mu: np.ndarray, true_counts: np.ndarray,
                      ctrl_counts: np.ndarray,
                      drug_ids: list[str], cell_lines: list[str]
                      ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Pseudobulk by (drug, cell_line); take log1p once.

    This matches the ridge baseline's target so the headline Δ Spearman is
    a like-for-like comparison rather than a transform artefact. See
    `predict_all` docstring for the rationale.
    """
    pair_pred, pair_true, metadata = aggregate_logfc_by_pair(
        pred_mu, true_counts, ctrl_counts, drug_ids, cell_lines
    )
    return (
        pair_pred,
        pair_true,
        metadata["drug_id"].tolist(),
        metadata["context_id"].tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/processed/sciplex_accept/drug_disjoint_v2/splits/sciplex_test.parquet"))
    parser.add_argument("--cell_emb", type=Path,
                        default=Path("data/cache/sciplex_scgpt_emb.npy"))
    parser.add_argument("--drug_emb", type=Path,
                        default=Path("data/cache/sciplex_molformer_emb.npz"))
    parser.add_argument("--counts", type=Path,
                        default=Path("data/processed/sciplex_accept/drug_disjoint_v2/splits/sciplex_test_treated_counts.npy"))
    parser.add_argument("--input_control_counts", type=Path, required=True)
    parser.add_argument("--truth_control_counts", type=Path, required=True)
    parser.add_argument("--gsea", type=Path,
                        default=Path("data/processed/sciplex_accept/drug_disjoint_v2/splits/sciplex_test_pathway_gsea.npy"))
    parser.add_argument("--baseline_csv", type=Path,
                        default=Path("results/ridge_baseline.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Override output CSV path (defaults to out_dir/cytobridge_internal.csv).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    out_csv = args.out if args.out is not None else (args.out_dir / "cytobridge_internal.csv")

    print("[run_internal] loading model ...")
    model = load_model(args.ckpt)

    print("[run_internal] building dataset ...")
    ds = CytoBridgeDataset(
        manifest_path=args.manifest, cell_emb_path=args.cell_emb,
        drug_emb_path=args.drug_emb, treated_counts_path=args.counts,
        pathway_gsea_path=args.gsea,
        input_control_counts_path=args.input_control_counts,
        truth_control_counts_path=args.truth_control_counts,
        n_hard_same_drug=0, n_hard_same_cell=0,
    )

    print("[run_internal] predicting ...")
    pred_mu, true_counts, ctrl_counts, drugs, cells = predict_all(
        model, ds, batch_size=args.batch_size, num_workers=args.num_workers
    )
    preds, trues, drugs, cells = aggregate_by_pair(
        pred_mu, true_counts, ctrl_counts, drugs, cells
    )

    print("[run_internal] computing metrics ...")
    pearson = per_pair_pearson(trues, preds)
    spearman = per_pair_spearman(trues, preds)
    r2 = r2_per_pair(trues, preds)                                   # per-PAIR R2@50 (bounded)
    dsd = drug_specific_delta_spearman(trues, preds, cells)         # stringent (mean-only -> 0)

    # collapse / scale diagnostics — print BEFORE the headline so a
    # collapsed run (inter-drug pred Pearson ~0.97) is caught immediately.
    import json as _json
    diag = scale_report(preds, trues, cells)
    print("\n=== Collapse / scale diagnostics (anti-collapse gate) ===")
    print(_json.dumps(diag, indent=2))
    idp = diag["inter_drug_pearson"]
    print(f"inter-drug pred Pearson = {idp:.4f}   "
          f"({'OK (<0.7)' if idp < 0.7 else 'COLLAPSED — fix before trusting metrics'})")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "drug_id": drugs, "cell_line": cells,
        "pearson_top50": pearson, "spearman_top50": spearman,
        "r2_top50": r2, "drug_specific_delta_spearman": dsd,
    }).to_csv(out_csv, index=False)
    print(f"saved -> {out_csv}")

    p_mean, p_lo, p_hi = bootstrap_ci(pearson)
    s_mean, s_lo, s_hi = bootstrap_ci(spearman)
    print("\n=== CytoBridge Internal (sci-Plex test) ===")
    print(f"Pearson@50:  {p_mean:.4f}  [95% CI {p_lo:.4f}, {p_hi:.4f}]")
    print(f"Spearman@50: {s_mean:.4f}  [95% CI {s_lo:.4f}, {s_hi:.4f}]   "
          "(HEADLINE; compare against the Mean baseline CSV)")
    print(f"per-pair R2@50: {np.nanmean(r2):.4f}   (bounded per-pair R2; NOT the per-gene R2_all)")
    print(f"drug-specific delta Spearman: {np.nanmean(dsd):.4f}   (mean-only predictor -> ~0; our battleground)")

    if args.baseline_csv.exists():
        # primary baseline-to-beat = Mean(per-cell-line), NOT ridge/linear-adj.
        # Accept either a column named like the Mean baseline or fall back to the
        # ridge column, but label honestly. Compare per-(drug,cell) when possible.
        base = pd.read_csv(args.baseline_csv)
        bcol = next((c for c in ("mean_top50", "spearman_top50", "Spearman_DEG", "spearman")
                     if c in base.columns), None)
        dcol = "drug" if "drug" in base.columns else ("drug_id" if "drug_id" in base.columns else None)
        if bcol and dcol and "cell_line" in base.columns:
            keyed = pd.DataFrame({"drug_id": drugs, "cell_line": cells, "spearman": spearman}) \
                .merge(base[[dcol, "cell_line", bcol]].rename(columns={dcol: "drug_id", bcol: "base"}),
                       on=["drug_id", "cell_line"], how="inner")
            if len(keyed):
                stat, pval = paired_wilcoxon(keyed["spearman"].values, keyed["base"].values)
                delta = keyed["spearman"].mean() - keyed["base"].mean()
                print(f"\nvs baseline ({bcol}):  Δ Spearman = {delta:+.4f}, paired Wilcoxon p={pval:.2e}")
        print("Gate (Option E): (1) inter-drug pred Pearson < 0.7; "
              "(2) drug-specific delta Spearman > 0; "
              "(3) Spearman@50 >= the selected Mean baseline (with CI).")


if __name__ == "__main__":
    main()
