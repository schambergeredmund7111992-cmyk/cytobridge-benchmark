"""
cytobridge.bridge.pathway_gate
-------------------------------
Pathway gating module — Innovation #1.

K=50 learnable pathway prototypes (initialized from MSigDB Hallmark gene sets).
Each cell+drug fused token is softly routed through these prototypes.
The attention weights over prototypes ARE the pathway attribution that
gets supervised by GSEA pre-rank ground truth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathwayGate(nn.Module):
    """
    Sparse soft routing over K pathway prototypes.

    Input:  fused tokens [B, L, d]
    Output: gated tokens [B, L, d]
            attn weights [B, L, K]   (this is the pathway-attribution per token,
                                       to be averaged → [B, K] for the supervision head)
    """
    def __init__(self, d: int = 256, K: int = 50, init_pathway_emb: torch.Tensor | None = None):
        super().__init__()
        self.d = d
        self.K = K
        self.prototypes = nn.Parameter(torch.empty(K, d))
        if init_pathway_emb is not None:
            assert init_pathway_emb.shape == (K, d)
            self.prototypes.data.copy_(init_pathway_emb)
        else:
            nn.init.kaiming_uniform_(self.prototypes, a=5**0.5)

        self.q = nn.Linear(d, d, bias=False)
        self.k_proto = nn.Linear(d, d, bias=False)
        self.v_proto = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, d]
        returns:
            x_out:    [B, L, d]
            attn:     [B, L, K] softmax weights
        """
        B, L, _ = x.shape
        q = self.q(x)                               # [B, L, d]
        k_p = self.k_proto(self.prototypes)         # [K, d]
        v_p = self.v_proto(self.prototypes)         # [K, d]
        logits = q @ k_p.T * self.scale             # [B, L, K]
        attn = F.softmax(logits, dim=-1)
        gated = attn @ v_p                          # [B, L, d]
        x_out = self.out_proj(gated) + x            # residual
        return x_out, attn


def aggregate_pathway_attn(attn_per_token: torch.Tensor) -> torch.Tensor:
    """
    Reduce per-token attention [B, L, K] to per-sample attribution [B, K].
    Mean is fine; could try max or attention-pool.
    """
    return attn_per_token.mean(dim=1)
