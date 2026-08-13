"""Week6 unit tests (pure torch/numpy/scipy — no lightning/hydra/data needed).

Covers:
  - T6-1 drug-conditioned decoder WIRING: varying the drug (same cell) changes
    the prediction (the anti-collapse fix actually routes drug signal to output);
    v1 path is unchanged when the flag is off.
  - T6-1 drug_specific_direction_loss SEMANTICS: collapse -> ~1.0 (penalized),
    perfect drug-specific match -> ~0.0, no-true-signal -> 0.0, <2 drugs skipped.
  - T6-3 ranking metrics (hit@k / mrr / ndcg@k) known-answer.
  - T6-1 collapse diagnostics (inter_drug_pearson, scale_report).

Run:  cd code && python tests/test_week6_additions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from cytobridge.model import CytoBridge, CytoBridgeConfig
from cytobridge.losses import drug_specific_direction_loss
from eval.metrics import (
    hit_at_k, mrr, ndcg_at_k, inter_drug_pearson, scale_report,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tiny_cfg(**kw):
    base = dict(d_cell_in=16, d_drug_in=24, d=32, n_layers=1, n_heads=4,
                K_pathways=8, n_genes=40, contrastive_dim=16)
    base.update(kw)
    return CytoBridgeConfig(**base)


def _counts_from_logfc(logfc, control):
    """raw treated counts s.t. log1p(treated)-log1p(control) == logfc."""
    return np.expm1(np.log1p(control) + logfc)


# ---------------------------------------------------------------------------
# T6-1 drug-conditioned decoder wiring
# ---------------------------------------------------------------------------
def test_drug_conditioned_decoder_routes_drug_signal():
    """Same cell, different drug -> different mu, when drug_conditioned_decoder on.
    Use the PLAIN decoder (residual zero-inits delta=0, which would mask the test)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(residual_decoder=False, pool_mode="drug_query",
                    drug_conditioned_decoder=True)
    model = CytoBridge(cfg).eval()
    B, Lc, Ld = 1, 5, 3
    cell = torch.randn(B, Lc, cfg.d_cell_in)
    drugA = torch.randn(B, Ld, cfg.d_drug_in)
    drugB = torch.randn(B, Ld, cfg.d_drug_in)
    mask = torch.ones(B, Ld, dtype=torch.bool)
    with torch.no_grad():
        muA = model(cell, drugA, mask)["mu"]
        muB = model(cell, drugB, mask)["mu"]
    assert not torch.allclose(muA, muB, atol=1e-4), \
        "drug_conditioned_decoder must make the prediction depend on the drug"


def test_flag_off_has_no_extra_modules_and_forward_ok():
    cfg_off = _tiny_cfg(residual_decoder=True, pool_mode="drug_query",
                        drug_conditioned_decoder=False)
    m_off = CytoBridge(cfg_off).eval()
    assert not hasattr(m_off, "dec_in") and not hasattr(m_off, "drug_summary_proj")
    cfg_on = _tiny_cfg(residual_decoder=True, pool_mode="drug_query",
                       drug_conditioned_decoder=True)
    m_on = CytoBridge(cfg_on).eval()
    assert hasattr(m_on, "dec_in") and hasattr(m_on, "drug_summary_proj")
    # both forwards run and produce delta (residual decoder)
    cell = torch.randn(2, 5, cfg_on.d_cell_in)
    drug = torch.randn(2, 3, cfg_on.d_drug_in)
    mask = torch.ones(2, 3, dtype=torch.bool)
    ctrl = torch.rand(2, cfg_on.n_genes) * 10
    for m in (m_off, m_on):
        out = m(cell, drug, mask, control_counts=ctrl)
        assert out["mu"].shape == (2, cfg_on.n_genes)
        assert out["delta"] is not None


def test_residual_t6_path_responds_to_drug_after_nonzero_delta():
    """Full t6 config (residual + drug_conditioned): once fc_delta is non-zero
    (i.e. after any training step), varying the drug changes delta."""
    torch.manual_seed(1)
    cfg = _tiny_cfg(residual_decoder=True, pool_mode="drug_query",
                    drug_conditioned_decoder=True)
    model = CytoBridge(cfg).eval()
    with torch.no_grad():                      # emulate "trained" delta head
        model.zinb.fc_delta.weight.normal_(0, 0.1)
        model.zinb.fc_delta.bias.normal_(0, 0.1)
    cell = torch.randn(1, 5, cfg.d_cell_in)
    drugA = torch.randn(1, 3, cfg.d_drug_in)
    drugB = torch.randn(1, 3, cfg.d_drug_in)
    mask = torch.ones(1, 3, dtype=torch.bool)
    ctrl = torch.rand(1, cfg.n_genes) * 10
    with torch.no_grad():
        dA = model(cell, drugA, mask, control_counts=ctrl)["delta"]
        dB = model(cell, drugB, mask, control_counts=ctrl)["delta"]
    assert not torch.allclose(dA, dB, atol=1e-4)


def test_t6_forward_bf16_no_crash():
    cfg = _tiny_cfg(residual_decoder=True, pool_mode="drug_query",
                    drug_conditioned_decoder=True)
    model = CytoBridge(cfg).eval()
    cell = torch.randn(2, 5, cfg.d_cell_in).bfloat16()
    drug = torch.randn(2, 3, cfg.d_drug_in).bfloat16()
    mask = torch.ones(2, 3, dtype=torch.bool)
    ctrl = (torch.rand(2, cfg.n_genes) * 10).bfloat16()
    model = model.bfloat16()
    out = model(cell, drug, mask, control_counts=ctrl)
    assert torch.isfinite(out["mu"].float()).all()


# ---------------------------------------------------------------------------
# T6-1 anti-collapse loss semantics
# ---------------------------------------------------------------------------
def test_drugspec_loss_penalizes_collapse():
    G = 30
    ctrl = np.full((2, G), 5.0, dtype=np.float64)
    # true: two distinct drug-specific signals
    Lt = np.zeros((2, G)); Lt[0, :10] = 2.0; Lt[1, 10:20] = 2.0
    # pred: IDENTICAL for both drugs (collapse)
    Lp = np.tile(np.linspace(-1, 1, G), (2, 1))
    mu = torch.tensor(_counts_from_logfc(Lp, ctrl))
    tr = torch.tensor(_counts_from_logfc(Lt, ctrl))
    cc = torch.tensor(ctrl)
    loss = drug_specific_direction_loss(mu, tr, cc, ["A", "A"]).item()
    assert loss > 0.9, f"collapsed prediction must be penalized ~1.0, got {loss}"


def test_drugspec_loss_zero_for_perfect_drug_specificity():
    G = 30
    ctrl = np.full((2, G), 5.0, dtype=np.float64)
    Lt = np.zeros((2, G)); Lt[0, :10] = 2.0; Lt[1, 10:20] = 2.0
    mu = torch.tensor(_counts_from_logfc(Lt, ctrl))   # pred == true
    tr = torch.tensor(_counts_from_logfc(Lt, ctrl))
    cc = torch.tensor(ctrl)
    loss = drug_specific_direction_loss(mu, tr, cc, ["A", "A"]).item()
    assert loss < 0.05, f"perfect drug-specific match must be ~0, got {loss}"


def test_drugspec_loss_zero_when_no_true_drug_signal():
    G = 20
    ctrl = np.full((2, G), 5.0, dtype=np.float64)
    Lt = np.tile(np.linspace(-1, 1, G), (2, 1))       # SAME true for both drugs
    Lp = np.random.default_rng(0).normal(size=(2, G))
    mu = torch.tensor(_counts_from_logfc(Lp, ctrl))
    tr = torch.tensor(_counts_from_logfc(Lt, ctrl))
    cc = torch.tensor(ctrl)
    loss = drug_specific_direction_loss(mu, tr, cc, ["A", "A"]).item()
    assert abs(loss) < 1e-6, f"no true drug-specific signal -> 0, got {loss}"


def test_drugspec_loss_skips_singleton_groups():
    G = 20
    ctrl = np.full((2, G), 5.0, dtype=np.float64)
    Lt = np.zeros((2, G)); Lt[0, :5] = 1.0; Lt[1, 5:10] = 1.0
    mu = torch.tensor(_counts_from_logfc(Lt, ctrl))
    tr = torch.tensor(_counts_from_logfc(Lt, ctrl))
    cc = torch.tensor(ctrl)
    # two different cell lines -> each group has 1 drug -> no valid group -> 0
    loss = drug_specific_direction_loss(mu, tr, cc, ["A", "B"]).item()
    assert loss == 0.0


# ---------------------------------------------------------------------------
# T6-3 ranking metrics
# ---------------------------------------------------------------------------
def test_hit_at_k_known_answer():
    scores = np.array([[0.9, 0.1, 0.5]])
    relevant = np.array([[False, False, True]])
    h = hit_at_k(scores, relevant, ks=(1, 3))
    assert h[1] == 0.0 and h[3] == 1.0


def test_mrr_known_answer():
    scores = np.array([[0.9, 0.1, 0.5]])
    relevant = np.array([[False, False, True]])
    assert abs(mrr(scores, relevant) - 0.5) < 1e-9   # relevant item ranked 2nd


def test_ndcg_known_answer():
    scores = np.array([[0.9, 0.1, 0.5]])
    relevance = np.array([[0.0, 0.0, 1.0]])
    expected = (1.0 / np.log2(3))                    # gain 1 at rank 2 (idx1), idcg=1
    assert abs(ndcg_at_k(scores, relevance, k=3) - expected) < 1e-9


# ---------------------------------------------------------------------------
# T6-1 collapse diagnostics
# ---------------------------------------------------------------------------
def test_inter_drug_pearson_detects_collapse():
    row = np.linspace(-1, 1, 25)
    collapsed = np.stack([row, row, row])             # 3 identical "drugs"
    assert abs(inter_drug_pearson(collapsed, ["A", "A", "A"]) - 1.0) < 1e-6
    rng = np.random.default_rng(0)
    distinct = rng.normal(size=(3, 25))
    assert inter_drug_pearson(distinct, ["A", "A", "A"]) < 0.9


def test_scale_report_keys_and_ratio():
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 4, size=(6, 20))             # mis-scaled (~4x)
    true = rng.normal(0, 1, size=(6, 20))
    cls = ["A", "A", "A", "B", "B", "B"]
    rep = scale_report(pred, true, cls)
    assert set(rep) >= {"pred_over_true_std_by_cell_line", "pred_logfc",
                        "true_logfc", "inter_drug_pearson"}
    assert rep["pred_over_true_std_by_cell_line"]["A"] > 2.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"OK: {len(fns)} Week6 unit tests passed")
