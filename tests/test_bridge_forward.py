from __future__ import annotations

import numpy as np
import torch

from cytobridge.model import CytoBridge, CytoBridgeConfig


def test_cytobridge_forward_shapes_are_stable():
    cfg = CytoBridgeConfig(
        d_cell_in=12,
        d_drug_in=10,
        d=16,
        n_layers=2,
        n_heads=4,
        K_pathways=6,
        n_genes=25,
        contrastive_dim=7,
    )
    model = CytoBridge(cfg)
    batch = {
        "cell_tokens": torch.randn(3, 9, 12),
        "drug_tokens": torch.randn(3, 5, 10),
        "drug_mask": torch.tensor(
            [[True, True, True, False, False],
             [True, True, True, True, False],
             [True, True, False, False, False]]
        ),
    }

    out = model(**batch)

    assert out["z"].shape == (3, 7)
    assert out["mu"].shape == (3, 25)
    assert out["theta"].shape == (3, 25)
    assert out["pi"].shape == (3, 25)
    assert out["pathway_pred"].shape == (3, 6)
    assert out["pathway_attn"].shape == (3, 6)
    assert torch.allclose(out["pathway_attn"].sum(dim=-1), torch.ones(3), atol=1e-5)


def test_cytobridge_accepts_pooled_drug_embeddings():
    cfg = CytoBridgeConfig(
        d_cell_in=12,
        d_drug_in=10,
        d=16,
        n_layers=1,
        n_heads=4,
        K_pathways=6,
        n_genes=25,
        contrastive_dim=7,
    )
    model = CytoBridge(cfg)

    out = model(
        cell_tokens=torch.randn(3, 9, 12),
        drug_tokens=torch.randn(3, 10),
    )

    assert out["z"].shape == (3, 7)
    assert out["mu"].shape == (3, 25)


def test_pathway_init_file_populates_all_gate_prototypes(tmp_path):
    init = np.arange(6 * 16, dtype=np.float32).reshape(6, 16) / 100.0
    init_path = tmp_path / "pathway_init.npy"
    np.save(init_path, init)
    cfg = CytoBridgeConfig(
        d_cell_in=12,
        d_drug_in=10,
        d=16,
        n_layers=2,
        n_heads=4,
        K_pathways=6,
        n_genes=25,
        contrastive_dim=7,
        pathway_init_path=str(init_path),
    )

    model = CytoBridge(cfg)

    expected = torch.from_numpy(init)
    for gate in model.backbone.gates:
        assert torch.allclose(gate.prototypes.detach(), expected)
