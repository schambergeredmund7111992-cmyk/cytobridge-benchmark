"""
cytobridge.losses
-----------------
Loss functions for CytoBridge.

This file is THE most important file in the project — getting hard-negative
InfoNCE right is what teaches the model (cell x drug) interactions and is
the only way to beat the ridge-on-pseudobulk baseline (Nat Methods 2025).

Total loss:
    L = lam_recon * L_recon          # ZINB-NLL on post-perturbation expression
      + lam_contrast * L_contrast    # InfoNCE w/ hard negatives
      + lam_pathway * L_pathway      # BCE on 50-D pathway attribution vs GSEA truth
      + lam_kl * L_kl_prior          # KL to sparse top-5 prior on pathway attn

Default weights: (1.0, 0.5, 0.3, 0.05) — tune in week 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. ZINB Negative Log-Likelihood
# ---------------------------------------------------------------------------
def zinb_nll(
    x: torch.Tensor,  # raw counts [B, G]  (use raw, not log1p)
    mu: torch.Tensor,  # predicted mean [B, G]
    theta: torch.Tensor,  # dispersion [G] or [B, G]
    pi: torch.Tensor,  # zero-inflation logit [B, G]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Zero-inflated negative binomial NLL, scvi-tools convention.
    Use mu = softplus(linear), theta = softplus, pi = sigmoid logit.

    Returns per-sample loss [B] — caller decides whether to .mean() or .sum().
    """
    # log NB
    log_theta_mu_eps = torch.log(theta + mu + eps)
    nb_case = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    # zero-inflation. With pi the zero-inflation logit:
    #     log    sigmoid(pi)  = -softplus(-pi)
    #     log(1 - sigmoid(pi)) = -softplus( pi)
    # case_zero    = log[ sigmoid(pi) + (1 - sigmoid(pi)) * NB(0; mu, theta) ]
    # case_nonzero = log[(1 - sigmoid(pi)) * NB(x; mu, theta)]
    #              = -softplus(pi) + log NB(x; mu, theta)
    softplus_neg_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps_zi = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps_zi)
    case_zero = F.softplus(pi_theta_log) - softplus_neg_pi
    case_non_zero = -F.softplus(pi) + nb_case
    mask_zero = (x < eps).float()
    res = mask_zero * case_zero + (1 - mask_zero) * case_non_zero
    return -res.sum(dim=-1)  # [B]


# ---------------------------------------------------------------------------
# 1b. Delta-aware auxiliary losses (direction + DEG-weighted Huber)
#     Operate on logFC = log1p(counts) - log1p(control). Work with ANY decoder
#     (mu from residual or plain), so they are decoder-agnostic.
# ---------------------------------------------------------------------------
def _logfc(counts: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    return torch.log1p(counts.clamp_min(0)) - torch.log1p(control.clamp_min(0))


def direction_loss(
    mu: torch.Tensor,  # predicted treated mean [B, G]
    treated: torch.Tensor,  # raw treated counts [B, G]
    control: torch.Tensor,  # raw matched control counts [B, G]
    eps: float = 1e-8,
) -> torch.Tensor:
    """Direction loss (1 - Pearson(pred_logfc, true_logfc)) per sample, scale-free.
    Directly targets the correlation/Spearman eval (MSE ignores direction)."""
    pred = _logfc(mu, control)
    true = _logfc(treated, control)
    pm = pred - pred.mean(dim=-1, keepdim=True)
    tm = true - true.mean(dim=-1, keepdim=True)
    tm_norm = tm.norm(dim=-1)
    num = (pm * tm).sum(dim=-1)
    den = pm.norm(dim=-1) * tm_norm + eps
    per = 1.0 - num / den  # [B]
    # Samples with (near-)constant true logFC have undefined Pearson; a no-change
    # sample should contribute 0, not the max penalty 1.0. Mask them out.
    valid = tm_norm > 1e-6
    per = torch.where(valid, per, torch.zeros_like(per))
    return per.mean()


def drug_specific_direction_loss(
    mu: torch.Tensor,  # predicted treated mean [B, G]
    treated: torch.Tensor,  # raw treated counts [B, G]
    control: torch.Tensor,  # raw matched control counts [B, G]
    cell_lines: list,  # [B] cell-line label per sample
    eps: float = 1e-8,
) -> torch.Tensor:
    """Anti-collapse loss. Within each cell line, subtract the
    cross-drug mean logFC (per gene) from BOTH pred and true, then 1 - Pearson of
    the residuals per sample over genes.

    Why this fights collapse: a model that outputs the SAME profile for every drug
    (the diagnosed failure, inter-drug pred Pearson 0.97) has a ~0 predicted
    residual -> its Pearson with the true residual is ~0 -> loss ~1.0 (max
    penalty), so gradient PUSHES predictions apart across drugs. The plain
    direction_loss cannot do this: it masks zero-variance to 0 and a collapsed
    model already matches the cell-line mean direction.

    Samples whose TRUE residual is ~constant (no genuine drug-specific signal)
    contribute 0 (nothing to learn). Cell lines with <2 drugs in the batch are
    skipped (the residual is undefined with one drug).
    """
    pred = _logfc(mu, control)  # [B, G]
    true = _logfc(treated, control)
    cl = list(cell_lines)
    device = mu.device
    group_losses = []
    for c in dict.fromkeys(cl):  # unique, order-stable
        idx = [i for i, x in enumerate(cl) if x == c]
        if len(idx) < 2:
            continue
        ii = torch.as_tensor(idx, device=device, dtype=torch.long)
        pr = pred.index_select(0, ii)
        tr = true.index_select(0, ii)
        pr = pr - pr.mean(0, keepdim=True)  # drug-specific residual (per gene)
        tr = tr - tr.mean(0, keepdim=True)
        pm = pr - pr.mean(-1, keepdim=True)  # center over genes for Pearson
        tm = tr - tr.mean(-1, keepdim=True)
        tnorm = tm.norm(dim=-1)
        pnorm = pm.norm(dim=-1)
        num = (pm * tm).sum(dim=-1)
        per = 1.0 - num / (pnorm * tnorm + eps)  # collapse (pnorm~0) -> ~1.0
        valid = tnorm > 1e-6  # true has drug-specific signal
        per = torch.where(valid, per, torch.zeros_like(per))
        group_losses.append(per.sum() / valid.sum().clamp_min(1))
    if not group_losses:
        return mu.new_zeros(())
    return torch.stack(group_losses).mean()


def logfc_delta_huber(
    mu: torch.Tensor,
    treated: torch.Tensor,
    control: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """DEG-weighted Huber on logFC. Per-gene residual weighted by |true_logfc|
    (per-sample normalized) so loss concentrates on genes that actually move.
    NOTE: a PLAIN unweighted Huber is degenerate (control cancels in the
    residual -> absolute log-count recon). Control enters via the weights."""
    pred = _logfc(mu, control)
    true = _logfc(treated, control)
    aw = true.abs()
    denom = aw.sum(dim=-1, keepdim=True)
    G = aw.shape[-1]
    # all-zero-true rows (no DEGs): fall back to uniform weights so a wrong delta
    # is still penalized rather than receiving exactly 0 loss.
    w = torch.where(denom > eps, aw / (denom + eps), torch.full_like(aw, 1.0 / G))
    per_gene = F.smooth_l1_loss(pred, true, beta=beta, reduction="none")
    return (w * per_gene).sum(dim=-1).mean()


def logfc_mse(
    mu: torch.Tensor,
    treated: torch.Tensor,
    control: torch.Tensor,
) -> torch.Tensor:
    """All-gene logFC MSE, summed over genes then averaged over samples."""
    pred = _logfc(mu, control)
    true = _logfc(treated, control)
    return F.mse_loss(pred, true, reduction="none").sum(dim=-1).mean()


def normalized_logfc_mse(
    mu: torch.Tensor,
    treated: torch.Tensor,
    control: torch.Tensor,
    gene_scale: torch.Tensor,
) -> torch.Tensor:
    """Per-gene variance-normalized logFC reconstruction using training scales only."""
    scale = gene_scale.to(device=mu.device, dtype=mu.dtype)
    if scale.ndim != 1 or scale.shape[0] != mu.shape[1]:
        raise ValueError(
            "gene_scale must be a [genes] vector aligned with model outputs."
        )
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("gene_scale must contain only finite positive values.")
    residual = (_logfc(mu, control) - _logfc(treated, control)) / scale.unsqueeze(0)
    return residual.square().mean()


# ---------------------------------------------------------------------------
# 2. InfoNCE with Hard Negative Mining (the secret sauce)
# ---------------------------------------------------------------------------
@dataclass
class InfoNCEConfig:
    temperature: float = 0.07
    n_hard_neg: int = 4  # per anchor: 2 same-drug-diff-cell + 2 same-cell-diff-drug
    weight_hard: float = 2.0  # weight of hard negatives vs. random in-batch


def info_nce_with_hard_negatives(
    z_anchor: torch.Tensor,  # [B, D] — fused (cell, drug) representation
    z_positive: torch.Tensor,  # [B, D] — same anchor (different view, e.g. dropout)
    z_hard_neg: Optional[torch.Tensor] = None,  # [B, n_hard, D]
    cfg: InfoNCEConfig | None = None,
) -> torch.Tensor:
    """
    InfoNCE loss with explicit hard-negative mining.

    Hard negatives are constructed by the data loader to be:
       - same drug, different cell line
       - same cell line, different drug
    These force the model to learn (cell x drug) interaction terms,
    which is the blind spot of linear baselines (ridge on pseudobulk + Morgan FP).

    z_anchor [B, D]
    z_positive [B, D] (typically z_anchor with different dropout / augmentation)
    z_hard_neg [B, n_hard, D] (per-anchor hard negatives)
    """
    cfg = cfg or InfoNCEConfig()
    B, D = z_anchor.shape
    # Normalize
    z_a = F.normalize(z_anchor, dim=-1)
    z_p = F.normalize(z_positive, dim=-1)

    # Positive logit: <z_a, z_p>
    pos_logit = (z_a * z_p).sum(dim=-1, keepdim=True) / cfg.temperature  # [B, 1]

    # In-batch random negatives (other items in batch)
    sim_inbatch = z_a @ z_p.t() / cfg.temperature  # [B, B]
    mask = torch.eye(B, device=z_a.device, dtype=torch.bool)
    sim_inbatch = sim_inbatch.masked_fill(mask, float("-inf"))

    if z_hard_neg is not None:
        z_hn = F.normalize(z_hard_neg, dim=-1)  # [B, n_hard, D]
        sim_hard = (
            torch.einsum("bd,bnd->bn", z_a, z_hn) / cfg.temperature
        )  # [B, n_hard]
        if cfg.weight_hard != 1.0:
            sim_hard = sim_hard + math.log(cfg.weight_hard)
        all_neg = torch.cat([sim_inbatch, sim_hard], dim=1)
    else:
        all_neg = sim_inbatch

    logits = torch.cat([pos_logit, all_neg], dim=1)  # [B, 1 + neg]
    labels = torch.zeros(B, dtype=torch.long, device=z_a.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# 3. Pathway Attribution BCE Loss
# ---------------------------------------------------------------------------
def pathway_bce(
    pathway_pred: torch.Tensor,  # [B, K] sigmoid logits
    pathway_gsea: torch.Tensor,  # [B, K] in [0, 1] — soft labels from GSEA pre-rank
) -> torch.Tensor:
    """
    Binary cross-entropy with logits on K=50 MSigDB Hallmark pathways.
    GSEA pre-rank scores are normalized to [0, 1] (e.g. by max-abs per drug-cell pair).
    """
    return F.binary_cross_entropy_with_logits(
        pathway_pred, pathway_gsea, reduction="mean"
    )


# ---------------------------------------------------------------------------
# 4. KL Divergence to Sparse Prior
# ---------------------------------------------------------------------------
def kl_to_sparse_prior(
    pathway_attn: torch.Tensor,  # [B, K] softmax over K=50 pathways
    top_k: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalize attention if it is too spread out across all 50 pathways.
    Prior: uniform over top-k pathways the model itself selects.
    KL(attn || top-k uniform).
    """
    # Construct top-k uniform target
    topk_vals, topk_idx = pathway_attn.topk(top_k, dim=-1)
    target = torch.zeros_like(pathway_attn)
    target.scatter_(-1, topk_idx, 1.0 / top_k)
    target = target + eps
    target = target / target.sum(dim=-1, keepdim=True)

    log_attn = torch.log(pathway_attn + eps)
    log_target = torch.log(target)
    kl = (pathway_attn * (log_attn - log_target)).sum(dim=-1)  # [B]
    return kl.mean()


# ---------------------------------------------------------------------------
# 5. Combined CytoBridge Loss
# ---------------------------------------------------------------------------
@dataclass
class CytoBridgeLossConfig:
    lam_recon: float = 1.0
    lam_contrast: float = 0.5
    lam_pathway: float = 0.3
    lam_kl: float = 0.05
    infonce: InfoNCEConfig = field(default_factory=InfoNCEConfig)
    # delta-aware aux losses (active only when lam>0 AND control present)
    lam_delta: float = 0.0  # DEG-weighted logFC Huber (set with check_loss_scale)
    lam_direction: float = 0.0  # direction loss (1 - Pearson on logFC), ~O(1)
    huber_beta: float = 1.0
    # anti-collapse (active only when lam>0 AND control+cell_lines present)
    lam_drugspec: float = 0.0  # drug-specific direction loss; penalizes collapse, ~O(1)
    lam_logfc: float = 0.0  # direct all-gene logFC supervision
    lam_norm_recon: float = 0.0  # training-scale-normalized logFC reconstruction


class CytoBridgeLoss(nn.Module):
    """Combined loss; expects model output dict with the named keys below."""

    def __init__(self, cfg: CytoBridgeLossConfig | None = None):
        super().__init__()
        self.cfg = cfg or CytoBridgeLossConfig()

    def forward(
        self,
        outputs: dict,
        batch: dict,
        *,
        include_raw_components: bool = False,
    ) -> dict:
        """
        outputs: {
            'mu': [B, G], 'theta': [B, G], 'pi': [B, G],   # ZINB params
            'z': [B, D],                                     # contrastive anchor
            'z_pos': [B, D],                                 # contrastive positive
            'z_hard_neg': Optional[B, n_hard, D],
            'pathway_pred': [B, K],                          # logits
            'pathway_attn': [B, K],                          # softmax
        }
        batch: {
            'treated_counts': [B, G] raw counts,
            'pathway_gsea': [B, K] in [0, 1],
        }
        """
        L_recon = torch.zeros((), device=outputs["mu"].device)
        if self.cfg.lam_recon > 0.0:
            L_recon = zinb_nll(
                x=batch["treated_counts"],
                mu=outputs["mu"],
                theta=outputs["theta"],
                pi=outputs["pi"],
            ).mean()

        L_contrast = info_nce_with_hard_negatives(
            z_anchor=outputs["z"],
            z_positive=outputs["z_pos"],
            z_hard_neg=outputs.get("z_hard_neg", None),
            cfg=self.cfg.infonce,
        )

        L_pathway = pathway_bce(
            pathway_pred=outputs["pathway_pred"],
            pathway_gsea=batch["pathway_gsea"],
        )

        top_k = min(5, outputs["pathway_attn"].shape[-1])
        L_kl = kl_to_sparse_prior(pathway_attn=outputs["pathway_attn"], top_k=top_k)

        # delta-aware aux losses (decoder-agnostic; need matched control)
        device = outputs["mu"].device
        L_delta = torch.zeros((), device=device)
        L_direction = torch.zeros((), device=device)
        L_drugspec = torch.zeros((), device=device)
        L_logfc = torch.zeros((), device=device)
        L_norm_recon = torch.zeros((), device=device)
        control = batch.get("truth_control_counts", batch.get("control_counts"))
        has_ctrl = control is not None
        if has_ctrl and self.cfg.lam_delta > 0.0:
            L_delta = logfc_delta_huber(
                outputs["mu"],
                batch["treated_counts"],
                control,
                beta=self.cfg.huber_beta,
            )
        if has_ctrl and self.cfg.lam_direction > 0.0:
            L_direction = direction_loss(
                outputs["mu"],
                batch["treated_counts"],
                control,
            )
        # anti-collapse, needs cell-line labels to form drug groups
        has_cl = bool(batch.get("cell_lines"))
        if has_ctrl and has_cl and self.cfg.lam_drugspec > 0.0:
            L_drugspec = drug_specific_direction_loss(
                outputs["mu"],
                batch["treated_counts"],
                control,
                batch["cell_lines"],
            )
        if has_ctrl and self.cfg.lam_logfc > 0.0:
            L_logfc = logfc_mse(outputs["mu"], batch["treated_counts"], control)
        if has_ctrl and self.cfg.lam_norm_recon > 0.0:
            if batch.get("gene_scale") is None:
                raise ValueError(
                    "lam_norm_recon > 0 requires a training-derived gene_scale."
                )
            L_norm_recon = normalized_logfc_mse(
                outputs["mu"], batch["treated_counts"], control, batch["gene_scale"]
            )

        loss = (
            self.cfg.lam_recon * L_recon
            + self.cfg.lam_contrast * L_contrast
            + self.cfg.lam_pathway * L_pathway
            + self.cfg.lam_kl * L_kl
            + self.cfg.lam_delta * L_delta
            + self.cfg.lam_direction * L_direction
            + self.cfg.lam_drugspec * L_drugspec
            + self.cfg.lam_logfc * L_logfc
            + self.cfg.lam_norm_recon * L_norm_recon
        )
        result = {
            "loss": loss,
            "L_recon": L_recon.detach(),
            "L_contrast": L_contrast.detach(),
            "L_pathway": L_pathway.detach(),
            "L_kl": L_kl.detach(),
            "L_delta": L_delta.detach(),
            "L_direction": L_direction.detach(),
            "L_drugspec": L_drugspec.detach(),
            "L_logfc": L_logfc.detach(),
            "L_norm_recon": L_norm_recon.detach(),
        }
        if include_raw_components:
            result["_raw_components"] = {
                "L_recon": L_recon,
                "L_contrast": L_contrast,
                "L_pathway": L_pathway,
                "L_kl": L_kl,
                "L_delta": L_delta,
                "L_direction": L_direction,
                "L_drugspec": L_drugspec,
                "L_logfc": L_logfc,
                "L_norm_recon": L_norm_recon,
            }
        return result
