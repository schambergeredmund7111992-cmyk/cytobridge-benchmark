from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from cytobridge.data import CytoBridgeDataModule, CytoBridgeDataset, collate_with_hard_negs


def test_dataloader_collates_hard_negative_tensors(tmp_path):
    manifest = pd.DataFrame(
        {
            "cell_idx": [0, 1, 2, 3, 4, 5],
            "control_cell_idx": [1, 1, 3, 3, 5, 5],
            "drug_id": ["A", "A", "B", "B", "C", "C"],
            "cell_line": ["X", "Y", "X", "Y", "X", "Y"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    cell_emb = np.random.default_rng(0).normal(size=(6, 4, 3)).astype("float32")
    np.save(tmp_path / "cell.npy", cell_emb)
    np.savez(
        tmp_path / "drug.npz",
        tokens=np.random.default_rng(1).normal(size=(3, 5, 2)).astype("float32"),
        masks=np.ones((3, 5), dtype=bool),
        drug_ids=np.array(["A", "B", "C"]),
    )
    np.save(tmp_path / "counts.npy", np.random.default_rng(2).poisson(size=(6, 7)).astype("float32"))
    np.save(tmp_path / "control_counts.npy", np.random.default_rng(4).poisson(size=(6, 7)).astype("float32"))
    np.save(tmp_path / "gsea.npy", np.random.default_rng(3).random(size=(6, 5)).astype("float32"))

    ds = CytoBridgeDataset(
        manifest_path,
        tmp_path / "cell.npy",
        tmp_path / "drug.npz",
        tmp_path / "counts.npy",
        tmp_path / "gsea.npy",
        control_counts_path=tmp_path / "control_counts.npy",
        n_hard_same_drug=1,
        n_hard_same_cell=1,
        seed=123,
    )
    batch = collate_with_hard_negs([ds[0], ds[1]])

    assert batch["cell_tokens"].shape == (2, 4, 3)
    assert torch.allclose(batch["cell_tokens"][0], torch.from_numpy(cell_emb[1]))
    assert batch["treated_counts"].shape == (2, 7)
    assert batch["control_counts"].shape == (2, 7)
    assert batch["hn_cell_emb"].shape == (2, 2, 4, 3)
    assert batch["hn_drug_emb"].shape == (2, 2, 5, 2)
    assert batch["hn_drug_mask"].shape == (2, 2, 5)
    assert batch["hn_drug_mask"].dtype == torch.bool


def test_dataloader_keeps_input_and_truth_control_pools_separate(tmp_path):
    manifest = pd.DataFrame(
        {
            "cell_idx": [0, 1],
            "control_cell_idx": [2, 3],
            "drug_id": ["A", "B"],
            "cell_line": ["X", "X"],
        }
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    np.save(tmp_path / "cell.npy", np.zeros((4, 2, 3), dtype="float32"))
    np.savez(
        tmp_path / "drug.npz",
        tokens=np.zeros((2, 2, 4), dtype="float32"),
        masks=np.ones((2, 2), dtype=bool),
        drug_ids=np.array(["A", "B"]),
    )
    np.save(tmp_path / "treated.npy", np.ones((2, 5), dtype="float32"))
    np.save(tmp_path / "input_ctrl.npy", np.full((2, 5), 2.0, dtype="float32"))
    np.save(tmp_path / "truth_ctrl.npy", np.full((2, 5), 7.0, dtype="float32"))
    np.save(tmp_path / "gsea.npy", np.zeros((2, 3), dtype="float32"))

    dataset = CytoBridgeDataset(
        manifest_path=tmp_path / "manifest.csv",
        cell_emb_path=tmp_path / "cell.npy",
        drug_emb_path=tmp_path / "drug.npz",
        treated_counts_path=tmp_path / "treated.npy",
        pathway_gsea_path=tmp_path / "gsea.npy",
        input_control_counts_path=tmp_path / "input_ctrl.npy",
        truth_control_counts_path=tmp_path / "truth_ctrl.npy",
        n_hard_same_drug=0,
        n_hard_same_cell=0,
    )
    batch = collate_with_hard_negs([dataset[0], dataset[1]])

    assert torch.all(batch["input_control_counts"] == 2.0)
    assert torch.all(batch["truth_control_counts"] == 7.0)
    assert torch.equal(batch["control_counts"], batch["truth_control_counts"])


def test_explicit_and_legacy_control_paths_cannot_be_mixed(tmp_path):
    manifest = pd.DataFrame(
        {
            "cell_idx": [0],
            "control_cell_idx": [1],
            "drug_id": ["A"],
            "cell_line": ["X"],
        }
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    np.save(tmp_path / "cell.npy", np.zeros((2, 2, 3), dtype="float32"))
    np.savez(
        tmp_path / "drug.npz",
        tokens=np.zeros((1, 2, 4), dtype="float32"),
        masks=np.ones((1, 2), dtype=bool),
        drug_ids=np.array(["A"]),
    )
    for name in ("treated", "control", "truth"):
        np.save(tmp_path / f"{name}.npy", np.ones((1, 5), dtype="float32"))
    np.save(tmp_path / "gsea.npy", np.zeros((1, 3), dtype="float32"))

    with pytest.raises(ValueError, match="either legacy"):
        CytoBridgeDataset(
            manifest_path=tmp_path / "manifest.csv",
            cell_emb_path=tmp_path / "cell.npy",
            drug_emb_path=tmp_path / "drug.npz",
            treated_counts_path=tmp_path / "treated.npy",
            pathway_gsea_path=tmp_path / "gsea.npy",
            control_counts_path=tmp_path / "control.npy",
            truth_control_counts_path=tmp_path / "truth.npy",
        )


def test_hard_negative_sampling_is_reproducible(tmp_path):
    manifest = pd.DataFrame(
        {
            "cell_idx": [0, 1, 2, 3, 4, 5],
            "control_cell_idx": [1, 1, 3, 3, 5, 5],
            "drug_id": ["A", "A", "B", "B", "C", "C"],
            "cell_line": ["X", "Y", "X", "Y", "X", "Y"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    np.save(tmp_path / "cell.npy", np.zeros((6, 4, 3), dtype="float32"))
    np.savez(
        tmp_path / "drug.npz",
        tokens=np.zeros((3, 5, 2), dtype="float32"),
        masks=np.ones((3, 5), dtype=bool),
        drug_ids=np.array(["A", "B", "C"]),
    )
    np.save(tmp_path / "counts.npy", np.zeros((6, 7), dtype="float32"))
    np.save(tmp_path / "gsea.npy", np.zeros((6, 5), dtype="float32"))

    kwargs = dict(
        manifest_path=manifest_path,
        cell_emb_path=tmp_path / "cell.npy",
        drug_emb_path=tmp_path / "drug.npz",
        treated_counts_path=tmp_path / "counts.npy",
        pathway_gsea_path=tmp_path / "gsea.npy",
        n_hard_same_drug=1,
        n_hard_same_cell=1,
        seed=99,
    )
    ds1 = CytoBridgeDataset(**kwargs)
    ds2 = CytoBridgeDataset(**kwargs)

    assert ds1[2]["hard_neg_indices"] == ds2[2]["hard_neg_indices"]


def test_hard_negative_fill_avoids_replacement_when_remaining_pool_is_large_enough(tmp_path):
    manifest = pd.DataFrame(
        {
            "cell_idx": [0, 1, 2, 3, 4, 5],
            "control_cell_idx": [0, 1, 2, 3, 4, 5],
            "drug_id": ["A", "A", "A", "B", "C", "D"],
            "cell_line": ["X", "Y", "Z", "Y", "Z", "W"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    np.save(tmp_path / "cell.npy", np.zeros((6, 4, 3), dtype="float32"))
    np.savez(
        tmp_path / "drug.npz",
        tokens=np.zeros((4, 5, 2), dtype="float32"),
        masks=np.ones((4, 5), dtype=bool),
        drug_ids=np.array(["A", "B", "C", "D"]),
    )
    np.save(tmp_path / "counts.npy", np.zeros((6, 7), dtype="float32"))
    np.save(tmp_path / "gsea.npy", np.zeros((6, 5), dtype="float32"))

    kwargs = dict(
        manifest_path=manifest_path,
        cell_emb_path=tmp_path / "cell.npy",
        drug_emb_path=tmp_path / "drug.npz",
        treated_counts_path=tmp_path / "counts.npy",
        pathway_gsea_path=tmp_path / "gsea.npy",
        n_hard_same_drug=3,
        n_hard_same_cell=1,
    )
    for seed in range(20):
        ds = CytoBridgeDataset(**kwargs, seed=seed)
        hard = ds._sample_hard_neg_indices(0)
        assert len(hard) == 4
        assert len(set(hard)) == 4


def test_dataset_rejects_manifest_without_control_cell_idx(tmp_path):
    """Regression: silently using cell_idx as the control input leaks
    the post-treatment embedding into every prediction. Reject manifests
    that omit control_cell_idx so callers fix the pairing upstream."""
    manifest = pd.DataFrame(
        {
            "cell_idx": [0, 1, 2, 3],
            "drug_id": ["A", "B", "A", "B"],
            "cell_line": ["X", "X", "Y", "Y"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    np.save(tmp_path / "cell.npy", np.zeros((4, 3, 2), dtype="float32"))
    np.savez(
        tmp_path / "drug.npz",
        tokens=np.zeros((2, 4, 2), dtype="float32"),
        masks=np.ones((2, 4), dtype=bool),
        drug_ids=np.array(["A", "B"]),
    )
    np.save(tmp_path / "counts.npy", np.zeros((4, 5), dtype="float32"))
    np.save(tmp_path / "gsea.npy", np.zeros((4, 3), dtype="float32"))

    with pytest.raises(ValueError, match="control_cell_idx"):
        CytoBridgeDataset(
            manifest_path=manifest_path,
            cell_emb_path=tmp_path / "cell.npy",
            drug_emb_path=tmp_path / "drug.npz",
            treated_counts_path=tmp_path / "counts.npy",
            pathway_gsea_path=tmp_path / "gsea.npy",
        )


@pytest.mark.skipif(sys.platform == "darwin", reason="Linux spawn-worker acceptance gate")
def test_spawn_workers_preserve_anchor_and_hard_negative_order(tmp_path):
    manifest = pd.DataFrame(
        {
            "cell_idx": list(range(12)),
            "control_cell_idx": list(range(12)),
            "drug_id": ["A", "B", "C"] * 4,
            "cell_line": ["X", "X", "X", "Y", "Y", "Y"] * 2,
        }
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    np.save(tmp_path / "cell.npy", np.zeros((12, 1, 3), dtype="float32"))
    np.savez(
        tmp_path / "drug.npz",
        tokens=np.zeros((3, 2, 4), dtype="float32"),
        masks=np.ones((3, 2), dtype=bool),
        drug_ids=np.array(["A", "B", "C"]),
    )
    np.save(tmp_path / "counts.npy", np.zeros((12, 5), dtype="float32"))
    np.save(tmp_path / "gsea.npy", np.zeros((12, 3), dtype="float32"))
    dataset = CytoBridgeDataset(
        manifest_path=tmp_path / "manifest.csv",
        cell_emb_path=tmp_path / "cell.npy",
        drug_emb_path=tmp_path / "drug.npz",
        treated_counts_path=tmp_path / "counts.npy",
        pathway_gsea_path=tmp_path / "gsea.npy",
        n_hard_same_drug=1,
        n_hard_same_cell=1,
        seed=17,
    )

    identities = []
    for workers in (0, 2):
        kwargs = {}
        if workers:
            kwargs["multiprocessing_context"] = "spawn"
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_with_hard_negs,
            generator=torch.Generator().manual_seed(123),
            **kwargs,
        )
        iterator = iter(loader)
        try:
            identities.append(
                [
                    (batch["anchor_indices"], batch["hard_neg_indices"])
                    for batch in iterator
                ]
            )
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
    assert identities[0] == identities[1]


def test_fit_setup_does_not_open_test_labels(tmp_path):
    manifest = pd.DataFrame(
        {
            "cell_idx": [0, 1],
            "control_cell_idx": [0, 1],
            "drug_id": ["A", "B"],
            "cell_line": ["X", "X"],
        }
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    np.save(tmp_path / "cell.npy", np.zeros((2, 1, 3), dtype="float32"))
    np.savez(
        tmp_path / "drug.npz",
        tokens=np.zeros((2, 2, 4), dtype="float32"),
        masks=np.ones((2, 2), dtype=bool),
        drug_ids=np.array(["A", "B"]),
    )
    np.save(tmp_path / "counts.npy", np.zeros((2, 5), dtype="float32"))
    np.save(tmp_path / "gsea.npy", np.zeros((2, 3), dtype="float32"))
    module = CytoBridgeDataModule(
        train_manifest=str(tmp_path / "manifest.csv"),
        val_manifest=str(tmp_path / "manifest.csv"),
        test_manifest=str(tmp_path / "forbidden-test.parquet"),
        cell_emb_path=str(tmp_path / "cell.npy"),
        drug_emb_path=str(tmp_path / "drug.npz"),
        treated_counts_path=str(tmp_path / "counts.npy"),
        pathway_gsea_path=str(tmp_path / "gsea.npy"),
        num_workers=0,
    )

    module.setup("fit")

    assert len(module.train_ds) == 2
    assert not hasattr(module, "test_ds")
