"""
Geneformer encoder - Plan B fallback for scGPT
"""
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)

CKPT_DIR = str(Path.home() / ".cache/geneformer")
TOKEN_DICT = str(Path.home() / ".cache/geneformer/geneformer/token_dictionary_gc104M.pkl")

class GeneformerEncoder:
    def __init__(self, ckpt_dir=CKPT_DIR, token_dict=TOKEN_DICT, device=None):
        self.device = "cpu"
        self.model = AutoModel.from_pretrained(ckpt_dir, trust_remote_code=True)
        self.model.eval().to(self.device)
        with open(token_dict, "rb") as f:
            self.gene2idx = pickle.load(f)
        logger.info(f"Geneformer loaded, vocab={len(self.gene2idx)}")

    def _tokenize(self, adata, max_len=2048):
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.array(adata.X)
        genes = list(adata.var_names)
        all_tokens = []
        for i in range(len(X)):
            expr = X[i]
            ranked = np.argsort(-expr)
            tokens = [self.gene2idx[genes[j]] for j in ranked
                      if genes[j] in self.gene2idx and expr[j] > 0][:max_len]
            all_tokens.append(tokens if tokens else [0])
        return all_tokens

    @torch.no_grad()
    def encode_anndata(self, adata, batch_size=16):
        all_tokens = self._tokenize(adata)
        all_embs = []
        for i in range(0, len(all_tokens), batch_size):
            batch = all_tokens[i:i+batch_size]
            max_len = max(len(t) for t in batch)
            input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
            attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
            for j, t in enumerate(batch):
                input_ids[j, :len(t)] = torch.tensor(t)
                attn_mask[j, :len(t)] = 1
            out = self.model(input_ids=input_ids, attention_mask=attn_mask)
            emb = out.last_hidden_state
            mask = attn_mask.unsqueeze(-1).float()
            emb = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_embs.append(emb.cpu().float().numpy())
            if i % (batch_size * 10) == 0:
                logger.info(f"  {i}/{len(all_tokens)} cells")
        return np.concatenate(all_embs, axis=0)

def cache_embeddings_to_disk(adata_path, out_path, ckpt_dir=CKPT_DIR):
    adata = sc.read_h5ad(adata_path)
    logger.info(f"Loaded {adata.shape[0]} cells")
    enc = GeneformerEncoder(ckpt_dir=ckpt_dir)
    embs = enc.encode_anndata(adata)
    np.save(out_path, embs)
    logger.info(f"Saved {embs.shape} -> {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", default=CKPT_DIR)
    args = parser.parse_args()
    cache_embeddings_to_disk(args.adata, args.out, args.ckpt)
