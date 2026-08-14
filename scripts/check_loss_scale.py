#!/usr/bin/env python
"""
check_loss_scale.py
-------------------
Set lam_delta SANELY before launching v2. The DEG-weighted logFC Huber is a
per-sample weighted-mean ~O(1); ZINB-NLL is a SUM over ~3000 genes ~O(1e2-1e3).
So a naive lam_delta (e.g. 0.5) makes the delta term a no-op. This script loads
a trained v1 checkpoint, takes one real batch, and reports the gradient-norm
ratio so you pick lam_delta where the two terms actually balance.

Run from the repo `code/` dir (paths default to smoke; override for real data):

    cd code
    python scripts/check_loss_scale.py --ckpt ckpts/v1_full/<best>.ckpt \
        --manifest data/processed/sciplex_accept/drug_disjoint_v2/splits/sciplex_train.parquet \
        --cell_emb data/cache/sciplex_scgpt_emb.npy \
        --drug_emb data/cache/sciplex_molformer_emb.npz \
        --treated  data/processed/sciplex_accept/drug_disjoint_v2/splits/sciplex_train_treated_counts.npy \
        --control  data/processed/sciplex_accept/drug_disjoint_v2/splits/sciplex_train_truth_control_counts.npy \
        --gsea     data/processed/sciplex_accept/drug_disjoint_v2/splits/sciplex_train_pathway_gsea.npy

Recommendation printed at the end is lam_delta for ~1:1 gradient balance; sweep
~[0.3x, 3x] of it. Re-confirm by watching train/L_recon vs train/L_delta in wandb.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `import cytobridge` work when this script is invoked by full path from the
# repo code/ dir (sys.path[0] would otherwise be the script's own dir). In the
# real env (`pip install -e .`) this is a harmless no-op.
sys.path.insert(0, os.getcwd())

import torch

from cytobridge.data import CytoBridgeDataset, collate_with_hard_negs
from cytobridge.model import CytoBridge, CytoBridgeConfig
from cytobridge.losses import zinb_nll, logfc_delta_huber


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = sum((g.detach() ** 2).sum() for g in grads if g is not None)
    return float(sq.sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=Path("data/smoke/splits/sciplex_train.csv"))
    ap.add_argument("--cell_emb", type=Path, default=Path("data/smoke/sciplex_scgpt_emb.npy"))
    ap.add_argument("--drug_emb", type=Path, default=Path("data/smoke/sciplex_molformer_emb.npz"))
    ap.add_argument("--treated", type=Path, default=Path("data/smoke/splits/sciplex_train_treated_counts.npy"))
    ap.add_argument("--control", type=Path, default=Path("data/smoke/splits/sciplex_train_control_counts.npy"))
    ap.add_argument("--gsea", type=Path, default=Path("data/smoke/splits/sciplex_train_pathway_gsea.npy"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = state.get("hyper_parameters", {}) if isinstance(state, dict) else {}
    model_cfg = hp.get("model_cfg") or {}
    cfg = CytoBridgeConfig(**model_cfg) if model_cfg else CytoBridgeConfig()
    model = CytoBridge(cfg)
    sd = state.get("state_dict", state)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = CytoBridgeDataset(
        manifest_path=args.manifest, cell_emb_path=args.cell_emb, drug_emb_path=args.drug_emb,
        treated_counts_path=args.treated, pathway_gsea_path=args.gsea,
        control_counts_path=args.control, n_hard_same_drug=0, n_hard_same_cell=0,
    )
    batch = collate_with_hard_negs([ds[i] for i in range(min(args.batch_size, len(ds)))])
    if "control_counts" not in batch:
        raise SystemExit("batch has no control_counts — pass --control with the matched control .npy")

    out = model(batch["cell_tokens"], batch["drug_tokens"], batch["drug_mask"])
    params = [p for p in model.parameters() if p.requires_grad]

    L_recon = zinb_nll(batch["treated_counts"], out["mu"], out["theta"], out["pi"]).mean()
    L_delta = logfc_delta_huber(out["mu"], batch["treated_counts"], batch["control_counts"],
                                beta=args.huber_beta)

    gr = grad_norm(L_recon, params)
    gd = grad_norm(L_delta, params)
    print("=" * 64)
    print(f"  L_recon value      = {float(L_recon):.4f}   ||grad L_recon|| = {gr:.4e}")
    print(f"  L_delta value      = {float(L_delta):.6f}   ||grad L_delta|| = {gd:.4e}")
    print("-" * 64)
    if gd < 1e-12:
        print("  ||grad L_delta|| ~ 0 (no perturbation in this batch?) — try a larger batch.")
    else:
        rec = gr / gd
        print(f"  recommended lam_delta (≈1:1 grad balance) = {rec:.1f}")
        print(f"  sweep ~[{rec*0.3:.0f}, {rec*3:.0f}]; then confirm via train/L_recon vs train/L_delta in wandb")
    print("=" * 64)


if __name__ == "__main__":
    main()
