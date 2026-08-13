"""Week5 additions — unit tests (pure-torch, no data/GPU needed).

Covers: ResidualZINBDecoder, DrugConditionedPool, direction_loss,
logfc_delta_huber, CytoBridgeLoss new keys, and full CytoBridge forward in
residual + drug_query mode. Run:  cd code && python tests/test_week5_additions.py
(also valid as a pytest module).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from eval.metrics import r2_per_pair, drug_specific_delta_spearman
from cytobridge.bridge.heads import ResidualZINBDecoder, DrugConditionedPool
from cytobridge.losses import (
    direction_loss, logfc_delta_huber, CytoBridgeLoss, CytoBridgeLossConfig,
)
from cytobridge.model import CytoBridge, CytoBridgeConfig


def test_residual_decoder_identity_and_grad():
    B, G, d = 4, 200, 32
    z = torch.randn(B, d, requires_grad=True)
    ctrl = torch.rand(B, G) * 50
    dec = ResidualZINBDecoder(d, G)
    mu, theta, pi, delta = dec(z, ctrl)
    # by-construction identity: log1p(mu) == clamp(log1p(ctrl)+delta, 0)
    log_ctrl = torch.log1p(ctrl.clamp_min(0))
    assert torch.allclose(torch.log1p(mu), (log_ctrl + delta).clamp_min(0.0), atol=1e-4)
    assert (mu >= 0).all()
    assert mu.shape == theta.shape == pi.shape == delta.shape == (B, G)
    mu.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    # control=None must not crash (hard-neg forward path)
    mu2, *_ = dec(z.detach(), None)
    assert (mu2 >= 0).all()


def test_drug_conditioned_pool():
    B, L, d, Ld, dd = 4, 7, 32, 5, 64
    c_fused = torch.randn(B, L, d, requires_grad=True)
    drug = torch.randn(B, Ld, dd)
    mask = torch.ones(B, Ld, dtype=torch.bool)
    mask[:, 3:] = False
    pool = DrugConditionedPool(d, dd)
    z = pool(c_fused, drug, mask)
    assert z.shape == (B, d)
    z.sum().backward()
    assert torch.isfinite(c_fused.grad).all()
    # no mask path
    assert pool(torch.randn(B, L, d), drug, None).shape == (B, d)


def test_direction_loss_control_sensitive_and_perfect():
    B, G = 4, 200
    mu = torch.rand(B, G) * 30 + 1
    treated = torch.rand(B, G) * 30 + 1
    ctrlA = torch.rand(B, G) * 30 + 1
    ctrlB = torch.rand(B, G) * 30 + 1
    lA = direction_loss(mu, treated, ctrlA)
    lB = direction_loss(mu, treated, ctrlB)
    assert not torch.allclose(lA, lB), "direction loss must depend on control"
    # perfect prediction (mu==treated) -> logFC identical -> loss ~ 0
    lp = direction_loss(treated.clone(), treated, ctrlA)
    assert lp.item() < 1e-4
    # grad flows
    mu2 = mu.clone().requires_grad_(True)
    direction_loss(mu2, treated, ctrlA).backward()
    assert mu2.grad is not None and torch.isfinite(mu2.grad).all()


def test_delta_huber_control_sensitive_and_zero_at_truth():
    B, G = 4, 200
    mu = torch.rand(B, G) * 30 + 1
    treated = torch.rand(B, G) * 30 + 1
    ctrlA, ctrlB = torch.rand(B, G) * 30 + 1, torch.rand(B, G) * 30 + 1
    assert not torch.allclose(logfc_delta_huber(mu, treated, ctrlA),
                              logfc_delta_huber(mu, treated, ctrlB))
    assert logfc_delta_huber(treated.clone(), treated, ctrlA).item() < 1e-6


def test_combined_loss_keys_and_gating():
    B, G, K, D = 4, 20, 5, 8
    mu = (torch.rand(B, G) * 10 + 1).requires_grad_(True)   # leaf tensor
    outputs = {
        "mu": mu,
        "theta": torch.rand(B, G) + 0.5,
        "pi": torch.randn(B, G, requires_grad=True),
        "z": torch.randn(B, D, requires_grad=True),
        "z_pos": torch.randn(B, D, requires_grad=True),
        "pathway_pred": torch.randn(B, K, requires_grad=True),
        "pathway_attn": torch.softmax(torch.randn(B, K), dim=-1),
    }
    batch = {
        "treated_counts": torch.randint(0, 20, (B, G)).float(),
        "control_counts": torch.randint(0, 20, (B, G)).float(),
        "pathway_gsea": torch.rand(B, K),
    }
    # v1 defaults: delta/direction OFF -> zero
    ld = CytoBridgeLoss(CytoBridgeLossConfig()).forward(outputs, batch)
    assert set(ld) == {"loss", "L_recon", "L_contrast", "L_pathway", "L_kl",
                       "L_delta", "L_direction", "L_drugspec"}
    assert ld["L_delta"].item() == 0.0 and ld["L_direction"].item() == 0.0
    # turn direction ON with control present -> nonzero, grad flows
    ld2 = CytoBridgeLoss(CytoBridgeLossConfig(lam_direction=1.0)).forward(outputs, batch)
    assert ld2["L_direction"].item() > 0.0
    ld2["loss"].backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    # no control in batch -> direction stays 0 (guarded)
    batch_noctrl = {k: v for k, v in batch.items() if k != "control_counts"}
    ld3 = CytoBridgeLoss(CytoBridgeLossConfig(lam_direction=1.0)).forward(
        {**outputs, "mu": outputs["mu"].detach()}, batch_noctrl)
    assert ld3["L_direction"].item() == 0.0


def test_full_model_residual_drugquery_forward():
    cfg = CytoBridgeConfig(d_cell_in=16, d_drug_in=24, d=32, n_layers=2, n_heads=4,
                           K_pathways=6, n_genes=50, contrastive_dim=8,
                           residual_decoder=True, pool_mode="drug_query")
    model = CytoBridge(cfg)
    B, Lc, Ld = 3, 5, 4
    cell = torch.randn(B, Lc, cfg.d_cell_in)
    drug = torch.randn(B, Ld, cfg.d_drug_in)
    mask = torch.ones(B, Ld, dtype=torch.bool)
    ctrl = torch.rand(B, cfg.n_genes) * 10
    out = model(cell, drug, mask, control_counts=ctrl)
    assert out["mu"].shape == (B, cfg.n_genes)
    assert (out["mu"] >= 0).all()
    assert out["delta"] is not None and out["delta"].shape == (B, cfg.n_genes)
    out["mu"].sum().backward()
    # default (v1) path still works and delta is None
    m2 = CytoBridge(CytoBridgeConfig(d_cell_in=16, d_drug_in=24, d=32, n_layers=2,
                                     n_heads=4, K_pathways=6, n_genes=50, contrastive_dim=8))
    o2 = m2(cell, drug, mask)
    assert o2["delta"] is None and o2["mu"].shape == (B, cfg.n_genes)


def test_r2_per_pair_perfect():
    rng = np.random.default_rng(0)
    true = rng.standard_normal((5, 120))
    assert np.allclose(r2_per_pair(true, true.copy()), 1.0, atol=1e-6)
    # constant (== per-pair mean) predictor -> R2 = 0 on the top-DEG subset
    const = np.stack([np.full(120, t[np.argsort(-np.abs(t))[:50]].mean()) for t in true])
    assert np.allclose(r2_per_pair(true, const), 0.0, atol=1e-6)


def test_drug_specific_delta_isolates_drug_signal():
    rng = np.random.default_rng(1)
    G = 150
    cl = ["A", "A", "A", "B", "B", "B"]
    base = {"A": rng.standard_normal(G), "B": rng.standard_normal(G)}
    drug = rng.standard_normal((6, G))
    true = np.stack([base[c] for c in cl]) + drug          # cellline mean + drug-specific
    mean_only = np.stack([base[c] for c in cl])            # predicts only cell-line mean
    sp_meanonly = drug_specific_delta_spearman(true, mean_only, cl)
    # mean-only predictor captures NO drug-specific signal -> 0.0 (not NaN)
    assert np.allclose(sp_meanonly, 0.0)
    sp_perfect = drug_specific_delta_spearman(true, true.copy(), cl)
    assert np.nanmean(sp_perfect) > 0.9
    # single-pair cell line -> true residual constant -> undefined (NaN), correctly
    sp_single = drug_specific_delta_spearman(true[:1], mean_only[:1], ["A"])
    assert np.all(np.isnan(sp_single))


def test_residual_decoder_no_overflow_large_delta():
    # Codex finding: large delta must NOT produce inf mu / NaN NLL (clamp max=15)
    from cytobridge.losses import zinb_nll
    dec = ResidualZINBDecoder(1, 3)
    with torch.no_grad():
        dec.fc_delta.weight.fill_(0.0)
        dec.fc_delta.bias.fill_(100.0)   # force a huge delta
    mu, theta, pi, _ = dec(torch.ones(1, 1), torch.zeros(1, 3))
    assert torch.isfinite(mu).all(), "mu overflowed to inf"
    assert torch.isfinite(zinb_nll(torch.ones(1, 3), mu, theta, pi)).all()
    # zero-init -> at init delta==0 -> mu==control (no perturbation start)
    dec0 = ResidualZINBDecoder(8, 5)
    ctrl = torch.rand(4, 5) * 20
    mu0, *_ = dec0(torch.randn(4, 8), ctrl)
    assert torch.allclose(mu0, ctrl, atol=1e-4), "zero-init should start at mu==control"


def test_direction_loss_zero_variance_true_is_zero():
    # constant true logFC (DMSO: treated==control) -> 0 contribution, not 1.0
    ctrl = torch.zeros(2, 8)
    treated = torch.full((2, 8), 5.0)
    mu = treated.clone().requires_grad_(True)
    loss = direction_loss(mu, treated, ctrl)
    assert abs(loss.item()) < 1e-6
    loss.backward()  # must not error
    # mixed batch: one valid + one constant-true -> finite, in [0, ~2]
    ctrl2 = torch.cat([torch.rand(1, 8) * 10, torch.zeros(1, 8)])
    treated2 = torch.cat([torch.rand(1, 8) * 10, torch.full((1, 8), 3.0)])
    assert torch.isfinite(direction_loss(torch.rand(2, 8) * 10, treated2, ctrl2))


def test_delta_huber_zero_weight_fallback_penalizes():
    # all-zero true logFC but wrong pred -> uniform fallback gives nonzero loss
    ctrl = torch.full((2, 6), 4.0)
    treated = ctrl.clone()                      # true logFC == 0 everywhere
    wrong_mu = torch.full((2, 6), 40.0)         # very wrong prediction
    assert logfc_delta_huber(wrong_mu, treated, ctrl).item() > 0.0
    assert logfc_delta_huber(treated.clone(), treated, ctrl).item() < 1e-6


def test_pool_bf16_no_dtype_crash():
    pool = DrugConditionedPool(8, 16)
    out = pool(torch.randn(2, 5, 8, dtype=torch.bfloat16),
               torch.randn(2, 3, 16, dtype=torch.bfloat16), None)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"OK: {len(fns)} Week5 unit tests passed")
