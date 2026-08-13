"""Save pred/true/meta arrays for verify_drug_specificity.py"""
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

def load_model(ckpt_path):
    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    if 'state_dict' in state:
        sd = {k.replace('model.', ''): v for k, v in state['state_dict'].items() if k.startswith('model.')}
        hp = state.get('hyper_parameters', {})
        model_cfg = hp.get('model_cfg') or {}
    else:
        sd = state
        model_cfg = {}
    cfg = CytoBridgeConfig(**model_cfg) if model_cfg else CytoBridgeConfig()
    model = CytoBridge(cfg)
    model.load_state_dict(sd, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.eval().to(device), device

@torch.no_grad()
def predict_all(model, dataset, device, batch_size=128, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_with_hard_negs,
                        pin_memory=(device == 'cuda'))
    pred_mu_list, treated_list, ctrl_list = [], [], []
    drug_ids, cell_lines = [], []
    for batch in tqdm(loader, desc='predict'):
        cell = batch['cell_tokens'].to(device)
        drug = batch['drug_tokens'].to(device)
        mask = batch['drug_mask'].to(device)
        input_ctrl = batch.get('input_control_counts', batch.get('control_counts'))
        truth_ctrl = batch.get('truth_control_counts', batch.get('control_counts'))
        ctrl_dev = input_ctrl.to(device) if input_ctrl is not None else None
        out = model(cell, drug, mask, control_counts=ctrl_dev)
        treated = batch['treated_counts'].cpu().numpy()
        ctrl_arr = truth_ctrl.cpu().numpy() if truth_ctrl is not None else np.zeros_like(treated)
        pred_mu_list.append(out['mu'].cpu().numpy())
        treated_list.append(treated)
        ctrl_list.append(ctrl_arr)
        drug_ids.extend(batch['drug_ids'])
        cell_lines.extend(batch['cell_lines'])
    return (np.concatenate(pred_mu_list), np.concatenate(treated_list),
            np.concatenate(ctrl_list), drug_ids, cell_lines)

def aggregate(pred_mu, true_counts, ctrl_counts, drug_ids, cell_lines):
    """Pseudobulk per (drug, cell_line), then log1p."""
    df = pd.DataFrame({'drug': drug_ids, 'cell': cell_lines})
    out_pred, out_true, out_drug, out_cell = [], [], [], []
    for (d, c), grp in df.groupby(['drug', 'cell']):
        idx = np.asarray(list(grp.index), dtype=int)
        mu_pb = pred_mu[idx].mean(axis=0)
        treated_pb = true_counts[idx].mean(axis=0)
        ctrl_pb = ctrl_counts[idx].mean(axis=0)
        out_pred.append(np.log1p(mu_pb) - np.log1p(ctrl_pb))
        out_true.append(np.log1p(treated_pb) - np.log1p(ctrl_pb))
        out_drug.append(d)
        out_cell.append(c)
    return np.stack(out_pred), np.stack(out_true), out_drug, out_cell

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cell_emb', type=Path, required=True)
    p.add_argument('--drug_emb', type=Path, required=True)
    p.add_argument('--counts', type=Path, required=True)
    p.add_argument('--input_control_counts', type=Path, required=True)
    p.add_argument('--truth_control_counts', type=Path, required=True)
    p.add_argument('--gsea', type=Path, required=True)
    p.add_argument('--out_pred', type=Path, required=True)
    p.add_argument('--out_true', type=Path, required=True)
    p.add_argument('--out_meta', type=Path, required=True)
    args = p.parse_args()

    model, device = load_model(args.ckpt)
    ds = CytoBridgeDataset(
        manifest_path=args.manifest,
        cell_emb_path=args.cell_emb,
        drug_emb_path=args.drug_emb,
        treated_counts_path=args.counts,
        pathway_gsea_path=args.gsea,
        input_control_counts_path=args.input_control_counts,
        truth_control_counts_path=args.truth_control_counts,
        n_hard_same_drug=0, n_hard_same_cell=0,
    )
    pred_mu, true_counts, ctrl_counts, drugs, cells = predict_all(model, ds, device)
    preds, trues, drugs, cells = aggregate(pred_mu, true_counts, ctrl_counts, drugs, cells)
    np.save(args.out_pred, preds)
    np.save(args.out_true, trues)
    pd.DataFrame({'drug': drugs, 'cell_line': cells}).to_csv(args.out_meta, index=False)
    print(f'Saved: pred={preds.shape}, true={trues.shape}, meta={len(drugs)} pairs')

if __name__ == '__main__':
    main()
