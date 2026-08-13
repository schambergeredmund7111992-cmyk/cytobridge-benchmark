"""
cytobridge.model
-----------------
Full CytoBridge model assembling encoders + bridge + heads.

Two-mode forward:
  (a) end-to-end with frozen encoders (slow, used at inference)
  (b) precomputed-embedding mode (fast, used during training):
      pass cell_tokens and drug_tokens directly, skipping encoders.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cytobridge.bridge.cross_attn import CytoBridgeBackbone
from cytobridge.bridge.heads import (
    ContrastiveHead, DrugConditionedPool, PathwayReadout, ResidualZINBDecoder,
    UncertaintyHead, ZNBDecoder,
)


@dataclass
class CytoBridgeConfig:
    d_cell_in: int = 512        # scGPT hidden
    d_drug_in: int = 768        # MolFormer-XL hidden
    d: int = 256                # bridge hidden
    n_layers: int = 4
    n_heads: int = 8
    K_pathways: int = 50
    n_genes: int = 3000
    contrastive_dim: int = 128
    dropout: float = 0.1
    use_pathway_gate: bool = True   # ablation: no_pathway_gating sets to False
    pathway_init_path: str | None = None
    # (config-gated; defaults reproduce the v1 path exactly)
    residual_decoder: bool = False  # residual decoder: predict delta on top of known control
    pool_mode: str = "mean"         # drug-query pool: "mean" | "drug_query"
    # (config-gated; default False reproduces v1 exactly)
    drug_conditioned_decoder: bool = False  # inject drug_summary DIRECTLY into the
    #   decoder input z_dec = MLP([cell_ctx, drug_ctx, drug_summary]) so the drug
    #   effect is not a small residual washed out by the cell-dominated state
    #   (the diagnosed collapse: inter-drug pred Pearson 0.97 vs true 0.13-0.42).
    # recovery (config-gated; default False reproduces the prior config exactly):
    dec_in_component_norm: bool = False  # LayerNorm EACH of [cell_ctx, drug_ctx,
    #   drug_summary] BEFORE concat in dec_in. Diagnosis: cell_ctx variance
    #   carries the frozen-scGPT cell state and numerically dominates the dec_in
    #   linear mixing even though the column norms look balanced, so the drug terms
    #   never differentiate the decoder input. Per-component norm equalizes scale so
    #   the drug signal survives. Pairs with the loss rebalance (t7_recovery).


class CytoBridge(nn.Module):
    def __init__(self, cfg: CytoBridgeConfig = CytoBridgeConfig(),
                 init_pathway_emb: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        if init_pathway_emb is None and cfg.pathway_init_path and cfg.use_pathway_gate:
            init_pathway_emb = self._load_pathway_init(
                cfg.pathway_init_path, cfg.K_pathways, cfg.d
            )
        self.backbone = CytoBridgeBackbone(
            d_cell_in=cfg.d_cell_in, d_drug_in=cfg.d_drug_in,
            d=cfg.d, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            K_pathways=cfg.K_pathways, dropout=cfg.dropout,
            init_pathway_emb=init_pathway_emb,
            use_pathway_gate=cfg.use_pathway_gate,
        )
        self.residual_decoder = cfg.residual_decoder
        if cfg.residual_decoder:
            self.zinb = ResidualZINBDecoder(d_in=cfg.d, n_genes=cfg.n_genes)
        else:
            self.zinb = ZNBDecoder(d_in=cfg.d, n_genes=cfg.n_genes)
        self.pool_mode = cfg.pool_mode
        self.pool = (DrugConditionedPool(cfg.d, cfg.d_drug_in)
                     if cfg.pool_mode == "drug_query" else None)
        # direct drug-conditioned decoder input (anti-collapse).
        self.drug_conditioned_decoder = cfg.drug_conditioned_decoder
        self.dec_in_component_norm = cfg.drug_conditioned_decoder and cfg.dec_in_component_norm
        if cfg.drug_conditioned_decoder:
            self.drug_summary_proj = nn.Sequential(
                nn.Linear(cfg.d_drug_in, cfg.d), nn.GELU(),
            )
            self.dec_in = nn.Sequential(
                nn.Linear(3 * cfg.d, cfg.d), nn.GELU(), nn.LayerNorm(cfg.d),
            )
            # per-component LayerNorm before concat (equalize cell vs drug scale).
            if self.dec_in_component_norm:
                self.dec_norm_cell = nn.LayerNorm(cfg.d)
                self.dec_norm_drugctx = nn.LayerNorm(cfg.d)
                self.dec_norm_drugemb = nn.LayerNorm(cfg.d)
        self.contrast = ContrastiveHead(d_in=cfg.d, d_out=cfg.contrastive_dim)
        self.pathway_head = PathwayReadout(K=cfg.K_pathways)
        self.uncertainty = UncertaintyHead(d_in=cfg.d, n_genes=cfg.n_genes)

    @staticmethod
    def _load_pathway_init(path: str | Path, k_pathways: int, d_model: int) -> torch.Tensor:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing pathway prototype init file: {path}. Run the pathway-prototype "
                "initialization step or set model.pathway_init_path=null to use random prototypes."
            )
        arr = np.load(path)
        expected = (k_pathways, d_model)
        if tuple(arr.shape) != expected:
            raise ValueError(f"{path} has shape {arr.shape}, expected {expected}.")
        return torch.as_tensor(arr, dtype=torch.float32)

    def forward(
        self,
        cell_tokens: torch.Tensor,           # [B, L_c, d_cell_in]
        drug_tokens: torch.Tensor,           # [B, L_d, d_drug_in] or pooled [B, d_drug_in]
        drug_mask: torch.Tensor | None = None,   # [B, L_d] True = real
        control_counts: torch.Tensor | None = None,  # [B, G] raw, for residual decoder
    ) -> dict:
        if cell_tokens.ndim != 3:
            raise ValueError(f"cell_tokens must be [B, L_c, d_cell], got {tuple(cell_tokens.shape)}")
        if drug_tokens.ndim == 2:
            drug_tokens = drug_tokens.unsqueeze(1)
            if drug_mask is None:
                drug_mask = torch.ones(
                    drug_tokens.shape[:2], dtype=torch.bool, device=drug_tokens.device
                )
        elif drug_tokens.ndim != 3:
            raise ValueError(
                f"drug_tokens must be [B, L_d, d_drug] or [B, d_drug], "
                f"got {tuple(drug_tokens.shape)}"
            )
        drug_tokens = drug_tokens.to(device=cell_tokens.device, dtype=cell_tokens.dtype)
        if drug_mask is not None:
            if drug_mask.ndim == 1:
                drug_mask = drug_mask.unsqueeze(1)
            if drug_mask.ndim != 2:
                raise ValueError(f"drug_mask must be [B, L_d], got {tuple(drug_mask.shape)}")
            if tuple(drug_mask.shape) != tuple(drug_tokens.shape[:2]):
                raise ValueError(
                    f"drug_mask shape {tuple(drug_mask.shape)} does not match "
                    f"drug token leading dims {tuple(drug_tokens.shape[:2])}"
                )
            drug_mask = drug_mask.to(device=cell_tokens.device, dtype=torch.bool)
        # cell_tokens are precomputed scGPT outputs
        c_fused, attn_agg = self.backbone(
            cell_tokens, drug_tokens,
            drug_mask=(~drug_mask if drug_mask is not None else None),
        )
        # Pool cell tokens → z. Default: mean; drug-query mode uses drug-conditioned attention.
        if self.pool is not None:
            z = self.pool(c_fused, drug_tokens, drug_mask)  # [B, d]
        else:
            z = c_fused.mean(dim=1)  # [B, d]
        # build a decoder input that carries the drug signal on a
        # STRONG, non-attenuated path. z (pooled cell state) still drives the
        # contrastive/uncertainty heads; only the decoder input gains the direct
        # drug term. When off, z_dec == z (v1 path, bit-for-bit).
        if self.drug_conditioned_decoder:
            cell_ctx = c_fused.mean(dim=1)                       # [B, d]
            drug_ctx = z                                         # drug-conditioned pool
            if drug_mask is not None:
                m = drug_mask.to(drug_tokens.dtype).unsqueeze(-1)
                drug_sum = (drug_tokens * m).sum(1) / m.sum(1).clamp_min(1.0)
            else:
                drug_sum = drug_tokens.mean(1)                   # [B, d_drug_in]
            w0 = self.drug_summary_proj[0].weight
            drug_emb = self.drug_summary_proj(drug_sum.to(w0.dtype)).to(c_fused.dtype)
            if self.dec_in_component_norm:
                cell_ctx = self.dec_norm_cell(cell_ctx)
                drug_ctx = self.dec_norm_drugctx(drug_ctx)
                drug_emb = self.dec_norm_drugemb(drug_emb)
            z_dec = self.dec_in(torch.cat([cell_ctx, drug_ctx, drug_emb], dim=-1))
        else:
            z_dec = z
        # ZINB decoder → predicted post-perturbation expression
        if self.residual_decoder:
            mu, theta, pi, delta = self.zinb(z_dec, control_counts)
        else:
            mu, theta, pi = self.zinb(z_dec)
            delta = None
        # Contrastive embedding
        z_proj = self.contrast(z)
        # Pathway logits
        pathway_logits = self.pathway_head(attn_agg)
        # Uncertainty
        log_var = self.uncertainty(z)
        return {
            "z": z_proj,
            "mu": mu, "theta": theta, "pi": pi, "delta": delta,
            "pathway_pred": pathway_logits,
            "pathway_attn": attn_agg,
            "log_var": log_var,
        }
