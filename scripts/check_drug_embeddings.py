#!/usr/bin/env python
"""Step 0 — the 10-minute cheap sanity check BEFORE any architecture work.

Are the held-out test drugs' MolFormer embeddings actually distinct? If they are
near-identical (off-diagonal cosine ~> 0.97), NO decoder/pooling change can make
the model produce drug-specific predictions — the drug featurization itself is the
bottleneck and must be fixed first (full-token instead of mean-pool, add Morgan FP,
or check the cache wasn't written with one shared row).

Run:
  cd code
  python scripts/check_drug_embeddings.py \
    --molformer data/cache/sciplex_molformer_emb.npz \
    --splits_json data/processed/sciplex/splits/internal_splits.json

Reads only the MolFormer npz + split JSON; no GPU, no model, no training data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cosine_stats(emb: np.ndarray) -> dict:
    """emb: [n, D] -> off-diagonal cosine summary."""
    n = emb.shape[0]
    if n < 2:
        return {"n": n, "note": "need >=2 drugs"}
    en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    S = en @ en.T
    iu = np.triu_indices(n, 1)
    off = S[iu]
    return {
        "n": int(n),
        "cosine_mean": float(off.mean()),
        "cosine_min": float(off.min()),
        "cosine_max": float(off.max()),
        "cosine_p90": float(np.quantile(off, 0.90)),
        "frac_pairs_gt_0.97": float((off > 0.97).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molformer", type=Path,
                    default=Path("data/cache/sciplex_molformer_emb.npz"))
    ap.add_argument("--splits_json", type=Path,
                    default=Path("data/processed/sciplex/splits/internal_splits.json"))
    args = ap.parse_args()

    mf = np.load(args.molformer, allow_pickle=True)
    tokens = mf["tokens"]                      # [N, L, D] or [N, D]
    ids = [str(x) for x in mf["drug_ids"]]
    # MASKED mean over real tokens (matches DrugConditionedPool); plain mean over
    # all L tokens would mix in shared padding and make distinct drugs look similar.
    if tokens.ndim == 3:
        if "masks" in mf:
            m = mf["masks"].astype(np.float32)[..., None]       # [N, L, 1]
            emb = (tokens * m).sum(axis=1) / np.clip(m.sum(axis=1), 1.0, None)
        else:
            emb = tokens.mean(axis=1)
    else:
        emb = tokens
    id_to_row = {d: i for i, d in enumerate(ids)}

    splits = json.loads(Path(args.splits_json).read_text())
    test_drugs = [d for d in splits.get("test_drugs", []) if d in id_to_row]
    train_drugs = [d for d in (splits.get("train_drugs", []) + splits.get("val_drugs", []))
                   if d in id_to_row]

    test_emb = emb[[id_to_row[d] for d in test_drugs]] if test_drugs else np.zeros((0, emb.shape[1]))
    train_emb = emb[[id_to_row[d] for d in train_drugs]] if train_drugs else np.zeros((0, emb.shape[1]))

    print(f"embedding shape: tokens={tokens.shape}  mean-pooled={emb.shape}")
    print(f"test drugs found: {len(test_drugs)} / {len(splits.get('test_drugs', []))}")
    print(f"train+val drugs found: {len(train_drugs)}")
    print("\n--- TEST-drug pairwise cosine ---")
    ts = cosine_stats(test_emb)
    print(json.dumps(ts, indent=2))
    print("\n--- TRAIN+VAL-drug pairwise cosine (reference) ---")
    print(json.dumps(cosine_stats(train_emb), indent=2))

    verdict = "OK: test drugs are distinct -> collapse is an ARCHITECTURE problem, proceed to Step 1."
    if ts.get("n", 0) >= 2 and ts.get("cosine_mean", 0) > 0.97:
        verdict = ("DEGENERATE: test-drug embeddings are near-identical -> NO decoder fix can help. "
                   "Fix drug featurization FIRST (full-token MolFormer / + Morgan FP / check the cache).")
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
