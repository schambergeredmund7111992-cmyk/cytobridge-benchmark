#!/usr/bin/env python
"""Run a single CytoBridge forward/backward pass on the tiny smoke dataset."""
from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cytobridge.data import CytoBridgeDataset, collate_with_hard_negs  # noqa: E402
from cytobridge.losses import CytoBridgeLoss, CytoBridgeLossConfig  # noqa: E402
from cytobridge.model import CytoBridge, CytoBridgeConfig  # noqa: E402


def run_model_smoke(data_dir: Path, *, device_name: str, precision: str) -> None:
    splits = data_dir / "splits"
    manifest = splits / "sciplex_train.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} does not exist. Run scripts/create_smoke_data.py first."
        )

    dataset = CytoBridgeDataset(
        manifest_path=manifest,
        cell_emb_path=data_dir / "sciplex_scgpt_emb.npy",
        drug_emb_path=data_dir / "sciplex_molformer_emb.npz",
        treated_counts_path=splits / "sciplex_train_treated_counts.npy",
        pathway_gsea_path=splits / "sciplex_train_pathway_gsea.npy",
        control_counts_path=splits / "sciplex_train_control_counts.npy",
        n_hard_same_drug=1,
        n_hard_same_cell=1,
        seed=17,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_with_hard_negs)
    batch = next(iter(loader))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda.is_available() is false.")
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    model = CytoBridge(
        CytoBridgeConfig(
            d_cell_in=8,
            d_drug_in=6,
            d=16,
            n_layers=1,
            n_heads=2,
            K_pathways=5,
            n_genes=10,
            contrastive_dim=8,
            pathway_init_path=None,
        )
    ).to(device)
    def autocast_context():
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16"
            else nullcontext()
        )

    with autocast_context():
        out = model(batch["cell_tokens"], batch["drug_tokens"], batch["drug_mask"])
        out_pos = model(batch["cell_tokens"], batch["drug_tokens"], batch["drug_mask"])
        out["z_pos"] = out_pos["z"]

    bsz, n_hard, n_cell_tokens, d_cell = batch["hn_cell_emb"].shape
    _, _, n_drug_tokens, d_drug = batch["hn_drug_emb"].shape
    with autocast_context():
        hard_out = model(
            batch["hn_cell_emb"].reshape(bsz * n_hard, n_cell_tokens, d_cell),
            batch["hn_drug_emb"].reshape(bsz * n_hard, n_drug_tokens, d_drug),
            batch["hn_drug_mask"].reshape(bsz * n_hard, n_drug_tokens),
        )
        out["z_hard_neg"] = hard_out["z"].reshape(bsz, n_hard, -1)

        loss = CytoBridgeLoss(CytoBridgeLossConfig()).forward(out, batch)
    for name, value in {**out, **loss}.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"model smoke produced non-finite {name}")
    loss["loss"].backward()
    bad_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if bad_gradients:
        raise FloatingPointError(f"model smoke produced non-finite gradients: {bad_gradients[:5]}")
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    print(
        "[model-smoke] ok "
        f"batch={batch['cell_tokens'].shape[0]} "
        f"device={device.type} precision={precision} "
        f"loss={float(loss['loss'].detach()):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/smoke"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()
    if args.precision == "bf16" and args.device != "cuda":
        raise ValueError("bf16 smoke is registered only for CUDA.")
    run_model_smoke(args.data_dir, device_name=args.device, precision=args.precision)


if __name__ == "__main__":
    main()
