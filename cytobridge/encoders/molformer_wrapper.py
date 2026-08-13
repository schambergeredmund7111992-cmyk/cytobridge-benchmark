"""
cytobridge.encoders.molformer_wrapper
--------------------------------------
Frozen MolFormer-XL encoder wrapper.

MolFormer paper: Ross et al. *Large-scale chemical language representations*. NMI 2022.
HuggingFace: https://huggingface.co/ibm/MoLFormer-XL-both-10pct
Trained on 1.1B SMILES from PubChem + ZINC.

Outputs per-token embeddings [B, L=128, d=768] from a frozen RoBERTa-style encoder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer


MOLFORMER_MODEL = "ibm/MoLFormer-XL-both-10pct"
MOLFORMER_REVISION = "a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MolFormerEncoder(nn.Module):
    """Frozen MolFormer-XL feature extractor."""

    def __init__(
        self,
        model_name: str = MOLFORMER_MODEL,
        revision: str = MOLFORMER_REVISION,
        max_length: int = 128,
        cache_dir: str | Path | None = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            deterministic_eval=True,
            cache_dir=cache_dir,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.max_length = max_length
        self.emb_dim = self.model.config.hidden_size  # 768 for MolFormer-XL
        self.model_name = model_name
        self.revision = revision

    @staticmethod
    def canonicalize(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)

    @torch.no_grad()
    def encode(self, smiles: list[str], device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a list of SMILES strings.
        Returns:
            tokens: [B, L, d]
            attention_mask: [B, L]  (True = real token, False = padding)
        """
        canon = [self.canonicalize(s) or s for s in smiles]
        enc = self.tokenizer(
            canon, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(device)
        out = self.model(**enc)
        return out.last_hidden_state, enc["attention_mask"].bool()


def cache_drug_embeddings(
    smiles_csv: str | Path,
    out_path: str | Path,
    model_name: str = MOLFORMER_MODEL,
    revision: str = MOLFORMER_REVISION,
    device: str | None = None,
    batch_size: int = 64,
):
    """One-shot drug embedding cache. ~1100 drugs × 128 × 768 × float32 ≈ 430MB."""
    import pandas as pd
    import numpy as np
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    smiles_csv = Path(smiles_csv)
    out_path = Path(out_path)
    provenance_path = out_path.with_suffix(out_path.suffix + ".provenance.json")
    for path in (out_path, provenance_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite MolFormer cache output: {path}")
    df = pd.read_csv(smiles_csv)
    smiles_column = "smiles" if "smiles" in df else "canonical_smiles"
    required = {"drug_id", smiles_column}
    if missing := required - set(df.columns):
        raise ValueError(f"SMILES table is missing columns: {sorted(missing)}")
    if df["drug_id"].astype(str).duplicated().any() or df[
        ["drug_id", smiles_column]
    ].isna().any().any():
        raise ValueError("SMILES table must contain unique, non-missing drug_id/smiles rows.")
    enc = MolFormerEncoder(model_name=model_name, revision=revision).to(device)
    tokens_list, masks_list = [], []
    for i in range(0, len(df), batch_size):
        chunk = df[smiles_column].iloc[i:i + batch_size].tolist()
        tok, mask = enc.encode(chunk, device=device)
        tokens_list.append(tok.cpu())
        masks_list.append(mask.cpu())
    tokens = torch.cat(tokens_list, dim=0)
    masks = torch.cat(masks_list, dim=0)
    tokens_np = tokens.numpy()
    masks_np = masks.numpy()
    all_finite = bool(np.isfinite(tokens_np).all())
    distinct_rows = len(
        {hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in tokens_np}
    )
    if not all_finite or distinct_rows != len(df):
        raise FloatingPointError("MolFormer cache is non-finite or contains duplicate drug rows.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, tokens=tokens_np, masks=masks_np, drug_ids=df["drug_id"].values)
    provenance = {
        "schema_version": 1,
        "model": model_name,
        "revision": revision,
        "max_length": enc.max_length,
        "dtype": str(tokens_np.dtype),
        "tokens_shape": list(tokens.shape),
        "masks_shape": list(masks.shape),
        "smiles_csv_sha256": _sha256_file(smiles_csv),
        "cache_sha256": _sha256_file(out_path),
        "all_finite": all_finite,
        "distinct_drug_rows": distinct_rows,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"[molformer] cached {tokens.shape} -> {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MOLFORMER_MODEL)
    parser.add_argument("--revision", default=MOLFORMER_REVISION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    cache_drug_embeddings(
        args.smiles_csv,
        args.out,
        model_name=args.model,
        revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
    )
