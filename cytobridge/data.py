"""
cytobridge.data
---------------
PyTorch Dataset + Lightning DataModule for CytoBridge.

Critical: implements hard-negative mining for InfoNCE loss:
    For each (cell, drug) anchor:
        - n_hard_same_drug: same drug, different cell line
        - n_hard_same_cell: same cell line, different drug
These hard negatives force the model to learn (cell × drug) interactions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:  # keep Dataset importable in lightweight test environments
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - exercised only when Lightning is absent

    class _LightningDataModule:
        def save_hyperparameters(self):
            self.hparams = type("HParams", (), {})()

    class _PLModule:
        LightningDataModule = _LightningDataModule

    pl = _PLModule()


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read manifest files without forcing a parquet dependency in lightweight environments."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    return pd.read_parquet(path)


class CytoBridgeDataset(Dataset):
    """
    One sample = one (cell, drug) treated example with:
        - cell_tokens (precomputed scGPT) [L_c, d_cell]
        - drug_tokens (precomputed MolFormer) [L_d, d_drug]
        - drug_mask [L_d] bool
        - treated_counts [n_genes] (raw, for ZINB-NLL)
        - pathway_gsea [K] in [0,1]
        - hard negatives: indices to look up via DataLoader collate
    """

    def __init__(
        self,
        manifest_path: str | Path,  # parquet: drug_id, cell_line, cell_idx, gsea_path
        cell_emb_path: str | Path,  # .npy of [n_cells, L_c, d_cell]
        drug_emb_path: str | Path,  # .npz of tokens, masks, drug_ids
        treated_counts_path: str | Path,  # .npy of [n_pairs, n_genes]
        pathway_gsea_path: str | Path,  # .npy of [n_pairs, K]
        control_counts_path: str | Path | None = None,
        input_control_counts_path: str | Path | None = None,
        truth_control_counts_path: str | Path | None = None,
        gene_scale_path: str | Path | None = None,
        n_hard_same_drug: int = 2,
        n_hard_same_cell: int = 2,
        seed: int = 42,
        randomize_drug_emb: bool = False,  # ablation: no_molformer
    ):
        self.manifest = read_manifest(manifest_path)
        # Enforce explicit control pairing. Falling back to the treated row's
        # cell_idx would leak the post-treatment cell embedding into the
        # control input and silently corrupt every (cell, drug) prediction.
        if "control_cell_idx" not in self.manifest.columns:
            raise ValueError(
                f"Manifest {manifest_path} is missing 'control_cell_idx'. "
                "Pair every treated row with an untreated control from the same "
                "cell line (see data/preprocess.py and data/build_external_split.py)."
            )
        self.cell_emb = np.load(cell_emb_path, mmap_mode="r")  # mmap for big arrays
        loaded = np.load(
            drug_emb_path, allow_pickle=True
        )  # drug_ids is an object (str) array
        self.drug_tokens = loaded["tokens"]
        self.drug_masks = loaded["masks"]
        self.drug_ids = [str(x) for x in loaded["drug_ids"]]
        self.drug_id_to_idx = {drug_id: i for i, drug_id in enumerate(self.drug_ids)}
        self.treated_counts = np.load(treated_counts_path, mmap_mode="r")
        self.pathway_gsea = np.load(pathway_gsea_path, mmap_mode="r")
        # New benchmark protocol keeps the model-input vehicle pool disjoint from
        # the truth-reference vehicle pool. `control_counts_path` remains a legacy
        # compatibility alias and may not be combined with either explicit path.
        if control_counts_path and (
            input_control_counts_path or truth_control_counts_path
        ):
            raise ValueError(
                "Use either legacy control_counts_path or the explicit input/truth control paths, "
                "not both."
            )
        if control_counts_path:
            input_control_counts_path = control_counts_path
            truth_control_counts_path = control_counts_path
        self.input_control_counts = (
            np.load(input_control_counts_path, mmap_mode="r")
            if input_control_counts_path
            else None
        )
        self.truth_control_counts = (
            np.load(truth_control_counts_path, mmap_mode="r")
            if truth_control_counts_path
            else None
        )
        self.gene_scale = np.load(gene_scale_path) if gene_scale_path else None
        if self.gene_scale is not None:
            self.gene_scale = np.asarray(self.gene_scale, dtype=np.float32)
            if (
                self.gene_scale.ndim != 1
                or self.gene_scale.shape[0] != self.treated_counts.shape[1]
            ):
                raise ValueError(
                    "gene_scale must be a one-dimensional vector aligned to treated-count genes."
                )
            if not np.isfinite(self.gene_scale).all() or (self.gene_scale <= 0).any():
                raise ValueError("gene_scale must contain only finite positive values.")
        for name, values in (
            ("input_control_counts", self.input_control_counts),
            ("truth_control_counts", self.truth_control_counts),
        ):
            if values is not None and len(values) != len(self.manifest):
                raise ValueError(
                    f"{name} has {len(values)} rows but manifest has {len(self.manifest)} rows."
                )
        self.n_hard_same_drug = n_hard_same_drug
        self.n_hard_same_cell = n_hard_same_cell
        self.n_hard = n_hard_same_drug + n_hard_same_cell
        self.seed = seed
        self.randomize_drug_emb = randomize_drug_emb
        if randomize_drug_emb:
            # Replace MolFormer outputs with deterministic random vectors so the
            # ablation can isolate the contribution of the chemical foundation
            # model. Use a fixed RNG seed to keep the ablation reproducible.
            rng = np.random.default_rng(seed + 7919)
            self.drug_tokens = rng.standard_normal(self.drug_tokens.shape).astype(
                self.drug_tokens.dtype
            )

        # Vectorized indices for hard-negative lookup.  The previous implementation
        # repeatedly called ``DataFrame.iloc`` and built an O(N) fallback pool for
        # every sample, which made worker stalls look like multiprocessing deadlocks.
        self._drug_values = self.manifest["drug_id"].astype(str).to_numpy()
        self._cell_values = self.manifest["cell_line"].astype(str).to_numpy()
        self.idx_by_drug = {
            str(key): np.asarray(value, dtype=np.int64)
            for key, value in self.manifest.groupby("drug_id", sort=False).indices.items()
        }
        self.idx_by_cell = {
            str(key): np.asarray(value, dtype=np.int64)
            for key, value in self.manifest.groupby("cell_line", sort=False).indices.items()
        }

    def __len__(self):
        return len(self.manifest)

    def _sample_hard_neg_indices(self, anchor_idx: int) -> list[int]:
        rng = np.random.default_rng(self.seed + anchor_idx * 1000003)
        anchor_drug = self._drug_values[anchor_idx]
        anchor_cell = self._cell_values[anchor_idx]
        # same drug, diff cell
        drug_pool = self.idx_by_drug.get(anchor_drug, np.empty(0, dtype=np.int64))
        cands = drug_pool[self._cell_values[drug_pool] != anchor_cell]
        sd = (
            list(
                rng.choice(
                    cands, size=min(self.n_hard_same_drug, len(cands)), replace=False
                )
            )
            if len(cands)
            else []
        )
        # same cell, diff drug
        cell_pool = self.idx_by_cell.get(anchor_cell, np.empty(0, dtype=np.int64))
        cands2 = cell_pool[self._drug_values[cell_pool] != anchor_drug]
        sc = (
            list(
                rng.choice(
                    cands2, size=min(self.n_hard_same_cell, len(cands2)), replace=False
                )
            )
            if len(cands2)
            else []
        )

        hard = sd + sc
        if len(hard) < self.n_hard:
            excluded = np.asarray([anchor_idx, *hard], dtype=np.int64)
            pool = np.setdiff1d(
                np.arange(len(self.manifest), dtype=np.int64), excluded, assume_unique=False
            )
            if len(pool):
                n_fill = self.n_hard - len(hard)
                fill = rng.choice(pool, size=n_fill, replace=len(pool) < n_fill)
                hard.extend([int(i) for i in fill])
        return [int(i) for i in hard[: self.n_hard]]

    def _get_hard_view(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load only tensors consumed by the hard-negative forward pass."""
        row = self.manifest.iloc[idx]
        cell_idx = int(row["control_cell_idx"])
        drug_lookup = self.drug_id_to_idx[str(row["drug_id"])]
        return (
            torch.from_numpy(np.array(self.cell_emb[cell_idx])).float(),
            torch.from_numpy(np.asarray(self.drug_tokens[drug_lookup])).float(),
            torch.from_numpy(np.asarray(self.drug_masks[drug_lookup])).bool(),
        )

    def _get_one(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        cell_idx = int(row["control_cell_idx"])
        treated_idx = int(row["cell_idx"])
        drug_lookup = self.drug_id_to_idx[str(row["drug_id"])]
        item = {
            "cell_tokens": torch.from_numpy(np.array(self.cell_emb[cell_idx])).float(),
            "drug_tokens": torch.from_numpy(self.drug_tokens[drug_lookup]).float(),
            "drug_mask": torch.from_numpy(self.drug_masks[drug_lookup]).bool(),
            "treated_counts": torch.from_numpy(
                np.array(self.treated_counts[idx])
            ).float(),
            "pathway_gsea": torch.from_numpy(np.array(self.pathway_gsea[idx])).float(),
            "drug_id": row["drug_id"],
            "cell_line": row["cell_line"],
            "cell_idx": cell_idx,
            "treated_cell_idx": treated_idx,
            "anchor_idx": idx,
        }
        if self.input_control_counts is not None:
            item["input_control_counts"] = torch.from_numpy(
                np.array(self.input_control_counts[idx])
            ).float()
        if self.truth_control_counts is not None:
            truth_control = torch.from_numpy(
                np.array(self.truth_control_counts[idx])
            ).float()
            item["truth_control_counts"] = truth_control
            # Compatibility alias for downstream code not yet migrated. New code
            # must use the explicit key so the two reference roles cannot drift.
            item["control_counts"] = truth_control
        if self.gene_scale is not None:
            item["gene_scale"] = torch.from_numpy(self.gene_scale.copy()).float()
        return item

    def __getitem__(self, idx: int) -> dict:
        item = self._get_one(idx)
        hard_idxs = self._sample_hard_neg_indices(idx)
        item["hard_neg_indices"] = hard_idxs
        if hard_idxs:
            hard_views = [self._get_hard_view(i) for i in hard_idxs]
            item["hn_cell_emb"] = torch.stack([view[0] for view in hard_views])
            item["hn_drug_emb"] = torch.stack([view[1] for view in hard_views])
            item["hn_drug_mask"] = torch.stack([view[2] for view in hard_views])
        return item


def collate_with_hard_negs(batch: list[dict]) -> dict:
    """
    Stack cell/drug tokens; load hard-negative cell+drug tokens for each anchor.
    NB: this uses the dataset's hard_neg_indices to fetch additional cell tokens.
    """
    out = {}
    out["cell_tokens"] = torch.stack([b["cell_tokens"] for b in batch])
    out["drug_tokens"] = torch.stack([b["drug_tokens"] for b in batch])
    out["drug_mask"] = torch.stack([b["drug_mask"] for b in batch])
    out["treated_counts"] = torch.stack([b["treated_counts"] for b in batch])
    if "control_counts" in batch[0]:
        out["control_counts"] = torch.stack([b["control_counts"] for b in batch])
    if "input_control_counts" in batch[0]:
        out["input_control_counts"] = torch.stack(
            [b["input_control_counts"] for b in batch]
        )
    if "truth_control_counts" in batch[0]:
        out["truth_control_counts"] = torch.stack(
            [b["truth_control_counts"] for b in batch]
        )
    if "gene_scale" in batch[0]:
        first = batch[0]["gene_scale"]
        if any(not torch.equal(item["gene_scale"], first) for item in batch[1:]):
            raise ValueError(
                "gene_scale must be identical for every sample in a batch."
            )
        out["gene_scale"] = first
    out["pathway_gsea"] = torch.stack([b["pathway_gsea"] for b in batch])
    out["drug_ids"] = [b["drug_id"] for b in batch]
    out["cell_lines"] = [b["cell_line"] for b in batch]
    out["anchor_indices"] = [int(b["anchor_idx"]) for b in batch]
    out["hard_neg_indices"] = [b["hard_neg_indices"] for b in batch]
    if batch and "hn_cell_emb" in batch[0]:
        out["hn_cell_emb"] = torch.stack([b["hn_cell_emb"] for b in batch])
        out["hn_drug_emb"] = torch.stack([b["hn_drug_emb"] for b in batch])
        out["hn_drug_mask"] = torch.stack([b["hn_drug_mask"] for b in batch])
    return out


class CytoBridgeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_manifest: str,
        val_manifest: str | None,
        test_manifest: str | None,
        cell_emb_path: str,
        drug_emb_path: str,
        treated_counts_path: str,
        pathway_gsea_path: str,
        val_treated_counts_path: str | None = None,
        test_treated_counts_path: str | None = None,
        val_pathway_gsea_path: str | None = None,
        test_pathway_gsea_path: str | None = None,
        control_counts_path: str | None = None,
        val_control_counts_path: str | None = None,
        test_control_counts_path: str | None = None,
        input_control_counts_path: str | None = None,
        val_input_control_counts_path: str | None = None,
        test_input_control_counts_path: str | None = None,
        truth_control_counts_path: str | None = None,
        val_truth_control_counts_path: str | None = None,
        test_truth_control_counts_path: str | None = None,
        gene_scale_path: str | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = 2,
        multiprocessing_context: Literal["spawn", "forkserver", "fork"] | None = "spawn",
        n_hard_same_drug: int = 2,
        n_hard_same_cell: int = 2,
        seed: int = 42,
        randomize_drug_emb: bool = False,
    ):
        super().__init__()
        self.train_kwargs = dict(
            cell_emb_path=cell_emb_path,
            drug_emb_path=drug_emb_path,
            treated_counts_path=treated_counts_path,
            pathway_gsea_path=pathway_gsea_path,
            control_counts_path=control_counts_path,
            input_control_counts_path=input_control_counts_path,
            truth_control_counts_path=truth_control_counts_path,
            gene_scale_path=gene_scale_path,
            n_hard_same_drug=n_hard_same_drug,
            n_hard_same_cell=n_hard_same_cell,
            seed=seed,
            randomize_drug_emb=randomize_drug_emb,
        )
        self.val_kwargs = dict(
            cell_emb_path=cell_emb_path,
            drug_emb_path=drug_emb_path,
            treated_counts_path=val_treated_counts_path or treated_counts_path,
            pathway_gsea_path=val_pathway_gsea_path or pathway_gsea_path,
            control_counts_path=val_control_counts_path or control_counts_path,
            input_control_counts_path=(
                val_input_control_counts_path or input_control_counts_path
            ),
            truth_control_counts_path=(
                val_truth_control_counts_path or truth_control_counts_path
            ),
            gene_scale_path=gene_scale_path,
            n_hard_same_drug=0,
            n_hard_same_cell=0,
            seed=seed + 1,
            randomize_drug_emb=randomize_drug_emb,
        )
        self.test_kwargs = dict(
            cell_emb_path=cell_emb_path,
            drug_emb_path=drug_emb_path,
            treated_counts_path=test_treated_counts_path
            or val_treated_counts_path
            or treated_counts_path,
            pathway_gsea_path=test_pathway_gsea_path
            or val_pathway_gsea_path
            or pathway_gsea_path,
            control_counts_path=test_control_counts_path
            or val_control_counts_path
            or control_counts_path,
            input_control_counts_path=(
                test_input_control_counts_path
                or val_input_control_counts_path
                or input_control_counts_path
            ),
            truth_control_counts_path=(
                test_truth_control_counts_path
                or val_truth_control_counts_path
                or truth_control_counts_path
            ),
            gene_scale_path=gene_scale_path,
            n_hard_same_drug=0,
            n_hard_same_cell=0,
            seed=seed + 2,
            randomize_drug_emb=randomize_drug_emb,
        )
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.test_manifest = test_manifest
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Never initialize CUDA while constructing the data module.  Forking after
        # a CUDA runtime call is a known source of worker hangs; acceptance configs
        # set this explicitly and use the safe ``spawn`` context.
        self.pin_memory = False if pin_memory is None else bool(pin_memory)
        self.persistent_workers = (
            num_workers > 0 if persistent_workers is None else bool(persistent_workers)
        )
        self.persistent_workers = bool(self.persistent_workers and num_workers > 0)
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        allowed_contexts = {None, "spawn", "forkserver", "fork"}
        if multiprocessing_context not in allowed_contexts:
            raise ValueError(
                "multiprocessing_context must be one of spawn, forkserver, fork, or null."
            )
        self.multiprocessing_context = multiprocessing_context if num_workers > 0 else None
        self.seed = int(seed)

    def _loader_kwargs(self, shuffle: bool, drop_last: bool = False) -> dict:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "collate_fn": collate_with_hard_negs,
            "pin_memory": self.pin_memory,
            "drop_last": drop_last,
            "persistent_workers": self.persistent_workers,
            "generator": torch.Generator().manual_seed(self.seed + (0 if shuffle else 1)),
            "worker_init_fn": self._seed_worker,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        if self.multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = self.multiprocessing_context
        return kwargs

    @staticmethod
    def _seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    def setup(self, stage: str | None = None):
        # Test labels must remain unopened during validation-only screening.
        if stage in (None, "fit"):
            self.train_ds = CytoBridgeDataset(self.train_manifest, **self.train_kwargs)
            if self.val_manifest is not None:
                self.val_ds = CytoBridgeDataset(self.val_manifest, **self.val_kwargs)
        if stage in ("test", "predict") and self.test_manifest is None:
            raise ValueError("test_manifest is required for test or predict setup.")
        if stage in (None, "test", "predict") and self.test_manifest is not None:
            self.test_ds = CytoBridgeDataset(self.test_manifest, **self.test_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, **self._loader_kwargs(shuffle=True, drop_last=True)
        )

    def val_dataloader(self):
        if not hasattr(self, "val_ds"):
            return None
        return DataLoader(self.val_ds, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self._loader_kwargs(shuffle=False))
