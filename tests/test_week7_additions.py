"""Week7 unit tests (pure torch/numpy/scipy — no lightning/hydra/data needed).

Covers:
  - V1 drug_discrimination_score (off-diagonal drug-shuffle control), the decisive
    "is it real" gate: collapse (constant-across-drugs output) -> gap ~= 0 and
    specificity_auc ~= 0.5; perfect drug-specificity -> gap large, auc == 1.0.
    This is exactly the test that exposes the Week6 pathway_gate artifact.
  - Recovery architecture: dec_in_component_norm flag is OFF by default (Week6
    path bit-for-bit, no extra modules) and, when ON, builds the per-component
    LayerNorms and the forward still runs (incl. bf16).

Run:  cd code && python tests/test_week7_additions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import pandas as pd

from cytobridge.model import CytoBridge, CytoBridgeConfig
from eval.metrics import drug_discrimination_score
from scripts.subsample_cells import select_capped_indices


def _tiny_cfg(**kw):
    base = dict(d_cell_in=16, d_drug_in=24, d=32, n_layers=1, n_heads=4,
                K_pathways=8, n_genes=40, contrastive_dim=16)
    base.update(kw)
    return CytoBridgeConfig(**base)


# ---------------------------------------------------------------------------
# V1 drug_discrimination_score — the gate
# ---------------------------------------------------------------------------
def test_discrimination_detects_collapse():
    """Output IDENTICAL across drugs (the Week6 pathway_gate failure) -> the
    per-pair on_diag looks fine but off_diag matches it -> gap ~= 0, auc ~= 0.5."""
    rng = np.random.default_rng(0)
    n_per_cl, D = 6, 30
    cell_lines = ["A"] * n_per_cl + ["B"] * n_per_cl
    true = rng.standard_normal((2 * n_per_cl, D))
    const = rng.standard_normal(D)
    pred = np.tile(const, (2 * n_per_cl, 1))          # same vector for every drug
    res = drug_discrimination_score(pred, true, cell_lines, top_k=None)
    assert abs(res["gap"]) < 0.2, res
    assert 0.25 < res["specificity_auc"] < 0.75, res


def test_discrimination_perfect_drug_specificity():
    """pred == true -> each pred matches its OWN truth best -> gap large, auc 1.0."""
    rng = np.random.default_rng(1)
    n_per_cl, D = 6, 30
    cell_lines = ["A"] * n_per_cl + ["B"] * n_per_cl
    true = rng.standard_normal((2 * n_per_cl, D))
    res = drug_discrimination_score(true.copy(), true, cell_lines, top_k=None)
    assert res["gap"] > 0.5, res
    assert res["specificity_auc"] == 1.0, res
    assert res["wilcoxon_p_on_gt_off"] < 0.01, res


def test_discrimination_keys_and_skips_singletons():
    """A cell line with a single pair contributes nothing (no off-diagonal)."""
    rng = np.random.default_rng(2)
    D = 12
    cell_lines = ["A", "A", "A", "solo"]
    true = rng.standard_normal((4, D))
    res = drug_discrimination_score(true.copy(), true, cell_lines, top_k=None)
    for k in ("on_diag_mean", "off_diag_mean", "gap", "specificity_auc", "n_pairs_scored"):
        assert k in res, (k, res)
    assert res["n_pairs_scored"] == 3, res   # only the 3 "A" pairs are scorable


# ---------------------------------------------------------------------------
# Recovery: dec_in_component_norm flag
# ---------------------------------------------------------------------------
def test_component_norm_off_has_no_extra_modules():
    cfg = _tiny_cfg(residual_decoder=True, pool_mode="drug_query",
                    drug_conditioned_decoder=True, dec_in_component_norm=False)
    model = CytoBridge(cfg)
    assert model.dec_in_component_norm is False
    assert not hasattr(model, "dec_norm_cell")


def test_component_norm_on_builds_and_forward_ok():
    torch.manual_seed(0)
    cfg = _tiny_cfg(residual_decoder=True, pool_mode="drug_query",
                    drug_conditioned_decoder=True, dec_in_component_norm=True)
    model = CytoBridge(cfg).eval()
    assert model.dec_in_component_norm is True
    assert hasattr(model, "dec_norm_cell") and hasattr(model, "dec_norm_drugemb")
    B, Lc, Ld = 2, 5, 3
    cell = torch.randn(B, Lc, cfg.d_cell_in)
    drug = torch.randn(B, Ld, cfg.d_drug_in)
    mask = torch.ones(B, Ld, dtype=torch.bool)
    control = torch.rand(B, cfg.n_genes) * 5.0
    with torch.no_grad():
        out = model(cell, drug, mask, control_counts=control)
    assert out["mu"].shape == (B, cfg.n_genes)
    assert torch.isfinite(out["mu"]).all()


def test_component_norm_bf16_no_crash():
    torch.manual_seed(0)
    cfg = _tiny_cfg(residual_decoder=True, pool_mode="drug_query",
                    drug_conditioned_decoder=True, dec_in_component_norm=True)
    model = CytoBridge(cfg).eval()
    B, Lc, Ld = 2, 5, 3
    cell = torch.randn(B, Lc, cfg.d_cell_in)
    drug = torch.randn(B, Ld, cfg.d_drug_in)
    mask = torch.ones(B, Ld, dtype=torch.bool)
    control = torch.rand(B, cfg.n_genes) * 5.0
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(cell, drug, mask, control_counts=control)
    assert torch.isfinite(out["mu"].float()).all()


# ---------------------------------------------------------------------------
# subsample_cells.select_capped_indices — tractable-training tooling
# ---------------------------------------------------------------------------
def _toy_manifest(per_pair):
    rows = []
    for d in ["A", "B", "C"]:
        for cl in ["A549", "K562", "MCF7"]:
            for _ in range(per_pair):
                rows.append({"drug_id": d, "cell_line": cl})
    return pd.DataFrame(rows).reset_index(drop=True)


def test_subsample_caps_per_pair_and_is_deterministic():
    man = _toy_manifest(per_pair=400)          # 9 pairs x 400 = 3600 rows
    keep1 = select_capped_indices(man, cap=150, seed=42)
    keep2 = select_capped_indices(man, cap=150, seed=42)
    assert np.array_equal(keep1, keep2), "must be deterministic for a fixed seed"
    sub = man.iloc[keep1]
    counts = sub.groupby(["drug_id", "cell_line"]).size()
    assert (counts <= 150).all(), counts
    assert (counts == 150).all(), "each of the 9 pairs (400 rows) should be capped to 150"
    assert list(keep1) == sorted(keep1), "indices kept in sorted order (counts alignment)"


def test_subsample_keeps_small_pairs_whole():
    man = _toy_manifest(per_pair=80)           # below cap -> keep all
    keep = select_capped_indices(man, cap=150, seed=0)
    assert len(keep) == len(man), "pairs with <=cap rows are kept whole"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"OK: {len(fns)} Week7 unit tests passed")
