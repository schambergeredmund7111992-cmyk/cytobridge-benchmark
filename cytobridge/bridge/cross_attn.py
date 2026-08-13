"""
cytobridge.bridge.cross_attn
----------------------------
4-layer cross-attention bridge between cell tokens (from scGPT)
and drug tokens (from MolFormer-XL), with PathwayGate inserted between layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cytobridge.bridge.pathway_gate import PathwayGate


class CrossAttnLayer(nn.Module):
    """One cross-attention layer: cell tokens attend to drug tokens, then FFN."""
    def __init__(self, d: int = 256, n_heads: int = 8, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * ffn_mult, d),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor | None = None):
        """
        q:  [B, L_c, d] cell tokens
        kv: [B, L_d, d] drug tokens
        Returns: [B, L_c, d]
        """
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(
            q_n, kv_n, kv_n,
            key_padding_mask=kv_mask,
            need_weights=False,
        )
        x = q + attn_out
        x = x + self.ffn(x)
        return x


class CytoBridgeBackbone(nn.Module):
    """
    Stacked CrossAttnLayer + PathwayGate alternations.

    cell_tok [B, L_c, d_c] + drug_tok [B, L_d, d_d]
        -> proj cell to d, proj drug to d
        -> repeat n_layers times: CrossAttn(cell, drug); PathwayGate(cell)
        -> output fused cell tokens [B, L_c, d] + pathway attn aggregated [B, K]
    """
    def __init__(
        self,
        d_cell_in: int = 512,
        d_drug_in: int = 768,
        d: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        K_pathways: int = 50,
        dropout: float = 0.1,
        init_pathway_emb: torch.Tensor | None = None,
        use_pathway_gate: bool = True,
    ):
        super().__init__()
        self.use_pathway_gate = use_pathway_gate
        self.K_pathways = K_pathways
        self.cell_proj = nn.Linear(d_cell_in, d)
        self.drug_proj = nn.Linear(d_drug_in, d)
        self.layers = nn.ModuleList([
            CrossAttnLayer(d=d, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        if use_pathway_gate:
            self.gates = nn.ModuleList([
                PathwayGate(d=d, K=K_pathways, init_pathway_emb=init_pathway_emb)
                for _ in range(n_layers)
            ])
        else:
            self.gates = nn.ModuleList()
        self.out_norm = nn.LayerNorm(d)

    def forward(self, cell_tok: torch.Tensor, drug_tok: torch.Tensor,
                drug_mask: torch.Tensor | None = None):
        c = self.cell_proj(cell_tok)
        d = self.drug_proj(drug_tok)
        attns = []
        if self.use_pathway_gate:
            for layer, gate in zip(self.layers, self.gates):
                c = layer(c, d, kv_mask=drug_mask)
                c, attn = gate(c)
                attns.append(attn)
        else:
            for layer in self.layers:
                c = layer(c, d, kv_mask=drug_mask)
        c = self.out_norm(c)
        if attns:
            attn_stack = torch.stack(attns, dim=0)            # [n_layers, B, L_c, K]
            attn_agg = attn_stack.mean(dim=(0, 2))            # [B, K]
        else:
            # Pathway gate disabled: output uniform attribution (attn_agg used by
            # PathwayReadout + ContrastiveLoss but supervision is masked out via
            # lam_pathway=0 in the matching ablation config).
            attn_agg = torch.full(
                (c.shape[0], self.K_pathways), 1.0 / max(self.K_pathways, 1),
                device=c.device, dtype=c.dtype,
            )
        return c, attn_agg
