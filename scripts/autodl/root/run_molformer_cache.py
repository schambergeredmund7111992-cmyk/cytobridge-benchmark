#!/usr/bin/env python
import os, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"

from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, "/root/CytoBridge/code")
from cytobridge.encoders.molformer_wrapper import MolFormerEncoder

smiles_csv = sys.argv[1]
out_path = sys.argv[2]

print(f"Loading SMILES from {smiles_csv}")
df = pd.read_csv(smiles_csv)
print(f"Loaded {len(df)} drugs")

print("Initializing MolFormer encoder (downloading model if needed)...")
enc = MolFormerEncoder(
    model_name="ibm/MoLFormer-XL-both-10pct",
    cache_dir="/root/autodl-tmp/hf_cache",
).cuda()

print(f"Encoding {len(df)} drugs in batches of 64...")
tokens_list, masks_list = [], []
for i in range(0, len(df), 64):
    chunk = df["smiles"].iloc[i:i + 64].tolist()
    tok, mask = enc.encode(chunk)
    tokens_list.append(tok.cpu())
    masks_list.append(mask.cpu())
    done = min(i + 64, len(df))
    if done % 128 == 0 or done >= len(df):
        print(f"  progress: {done}/{len(df)}")

tokens = torch.cat(tokens_list, dim=0)
masks = torch.cat(masks_list, dim=0)
print(f"Tokens: {tokens.shape}, Masks: {masks.shape}")

out_path = Path(out_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(out_path, tokens=tokens.numpy(), masks=masks.numpy(),
         drug_ids=df["drug_id"].values)
print(f"[molformer] cached {tokens.shape} -> {out_path}")
print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
