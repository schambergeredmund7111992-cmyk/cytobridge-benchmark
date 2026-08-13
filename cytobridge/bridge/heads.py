"""
cytobridge.bridge.heads
-----------------------
Output heads on top of the cross-attention backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZNBDecoder(nn.Module):
    """
    Decode fused representation to ZINB parameters per gene.
    mu = exp(linear), theta = softplus(linear), pi = logit (we apply sigmoid in loss).
    """
    def __init__(self, d_in: int = 256, n_genes: int = 3000):
        super().__init__()
        self.fc_mu = nn.Linear(d_in, n_genes)
        self.fc_theta = nn.Parameter(torch.zeros(n_genes))   # gene-level dispersion
        self.fc_pi = nn.Linear(d_in, n_genes)

    def forward(self, z: torch.Tensor):
        """z: [B, d_in]"""
        mu = torch.exp(self.fc_mu(z).clamp(max=15))
        theta = F.softplus(self.fc_theta).expand_as(mu)
        pi = self.fc_pi(z)  # logit
        return mu, theta, pi


class ContrastiveHead(nn.Module):
    """Project to a smaller embedding space for InfoNCE."""
    def __init__(self, d_in: int = 256, d_out: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_out),
        )

    def forward(self, z: torch.Tensor):
        return self.proj(z)


class PathwayReadout(nn.Module):
    """
    Reads aggregated pathway attention [B, K] and produces logits over K pathways
    for BCE supervision against GSEA pre-rank labels.
    """
    def __init__(self, K: int = 50):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(K))
        self.scale = nn.Parameter(torch.ones(K))

    def forward(self, attn_agg: torch.Tensor):
        # Convert softmax attn into logits via inverse softmax with learnable affine
        logits = torch.log(attn_agg + 1e-8) * self.scale + self.bias
        return logits


class UncertaintyHead(nn.Module):
    """MC-Dropout / heteroscedastic uncertainty."""
    def __init__(self, d_in: int = 256, n_genes: int = 3000, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc_logvar = nn.Linear(d_in, n_genes)

    def forward(self, z: torch.Tensor):
        return self.fc_logvar(self.dropout(z))   # log variance per gene

class ResidualZINBDecoder(nn.Module):
    """Predict delta on top of known control expression.
    mu = expm1(clamp(log1p(control) + delta, min=0)), score over treated counts.
    Zero-initialised so at init delta==0 -> mu==control (no-perturbation start).
    """
    def __init__(self, d_in: int = 256, n_genes: int = 3000):
        super().__init__()
        self.fc_delta = nn.Linear(d_in, n_genes)
        nn.init.zeros_(self.fc_delta.weight)
        nn.init.zeros_(self.fc_delta.bias)
        self.fc_theta = nn.Parameter(torch.zeros(n_genes))
        self.fc_pi = nn.Linear(d_in, n_genes)

    def forward(self, z: torch.Tensor, control_counts: torch.Tensor | None = None):
        delta = self.fc_delta(z)
        if control_counts is not None:
            mu_raw = torch.log1p(control_counts.clamp_min(0)) + delta
            mu = torch.expm1(mu_raw.clamp(min=0, max=15))
        else:
            mu = torch.exp(delta.clamp(max=15))
        theta = F.softplus(self.fc_theta).expand_as(mu)
        pi = self.fc_pi(z)
        return mu, theta, pi, delta


class DrugConditionedPool(nn.Module):
    """Attention-pool cell tokens using a drug-derived query.
    z = softmax(c_fused @ q_proj(drug_pool)) * c_fused, pooled to [B, d].
    """
    def __init__(self, d: int = 256, d_drug: int = 768):
        super().__init__()
        self.q_proj = nn.Linear(d_drug, d)

    def forward(self, c_fused: torch.Tensor,
                drug_tokens: torch.Tensor,
                drug_mask: torch.Tensor | None = None):
        if drug_mask is not None:
            m = drug_mask.to(drug_tokens.dtype).unsqueeze(-1)
            drug_pool = (drug_tokens * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            drug_pool = drug_tokens.mean(1)
        w = self.q_proj.weight
        q = self.q_proj(drug_pool.to(w.dtype)).to(c_fused.dtype).unsqueeze(-1)          # [B, d, 1]
        attn = F.softmax(c_fused @ q, dim=1)              # [B, L, 1]
        z = (c_fused * attn).sum(dim=1)                   # [B, d]
        return z
