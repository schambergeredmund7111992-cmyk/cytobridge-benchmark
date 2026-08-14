"""Load the regeneration-input bundle with provenance checks.

The bundle is a directory (or zip) laid out as documented in
scripts/export_regeneration_inputs.py. `inputs_manifest.json` records the
sha256 and shape of every shipped file; every load verifies both when the
manifest is present.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Paper configuration names -> exported per-pair tag.
CONFIGS: dict[str, str] = {
    "loss-only": "t7_sub_loss_only",
    "drug-spec x1": "t7_sub_drugspec1",
    "drug-spec x3": "t7_sub_drugspec3",
    "drug-spec x5": "t7_sub_drugspec5",
    "norm-only": "t7_sub_norm_only",
    "low recon weight": "t7_sub_lamrecon01",
    "recovery baseline": "t6_sub_baseline",
}
CONFIG_ORDER = list(CONFIGS)
CELL_LINES = ("A549", "K562", "MCF7")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class BundleLoader:
    """Reads bundle files; each loader method returns None when its group is absent."""

    root: Path
    manifest: dict[str, Any] | None = None
    missing: set[str] = field(default_factory=set)
    _warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, bundle: Path) -> "BundleLoader":
        bundle = Path(bundle)
        if bundle.suffix == ".zip":
            bundle = _unzip(bundle)
        manifest_path = bundle / "inputs_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else None
        )
        return cls(root=bundle, manifest=manifest)

    def _path(self, relative: str) -> Path | None:
        path = self.root / relative
        if not path.is_file():
            self.missing.add(relative)
            return None
        return path

    def _check(self, relative: str, path: Path) -> None:
        if self.manifest is None:
            return
        entry = None
        for item in self.manifest.get("files", []):
            if item.get("file") == relative:
                entry = item
                break
        if entry is None:
            self._warnings.append(f"{relative}: not recorded in inputs_manifest.json")
            return
        want_sha = entry.get("sha256")
        if want_sha and sha256_file(path) != want_sha:
            raise ValueError(
                f"{relative}: sha256 mismatch "
                f"(got {sha256_file(path)[:12]}..., want {want_sha[:12]}...)"
            )
        want_shape = entry.get("shape")
        if want_shape and path.suffix == ".npy":
            actual = list(np.load(path).shape)
            if actual != want_shape:
                raise ValueError(
                    f"{relative}: shape {actual} != manifest {want_shape}"
                )

    def load_npy(self, relative: str, dtype: str = "float32") -> np.ndarray | None:
        path = self._path(relative)
        if path is None:
            return None
        self._check(relative, path)
        return np.load(path).astype(dtype)

    def load_csv(self, relative: str) -> pd.DataFrame | None:
        path = self._path(relative)
        if path is None:
            return None
        self._check(relative, path)
        return pd.read_csv(path)

    def load_json(self, relative: str) -> dict | None:
        path = self._path(relative)
        if path is None:
            return None
        self._check(relative, path)
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Grouped loaders
    # ------------------------------------------------------------------
    def e6e7_config(self, name: str) -> dict | None:
        tag = CONFIGS[name]
        pred = self.load_npy(f"e6e7/logfc_pred_{tag}.npy")
        true = self.load_npy(f"e6e7/logfc_true_{tag}.npy")
        meta = self.load_csv(f"e6e7/logfc_meta_{tag}.csv")
        if pred is None or true is None or meta is None:
            return None
        return {"name": name, "tag": tag, "pred": pred, "true": true, "meta": meta}

    def pooled_truth(self) -> dict | None:
        true = self.load_npy("e6e7/true_pooled.npy")
        meta = self.load_csv("e6e7/true_pooled_meta.csv")
        if true is None or meta is None:
            return None
        return {"true": true, "meta": meta}

    def mean_predictor_train(self) -> np.ndarray | None:
        return self.load_npy("e6e7/mean_predictor_train.npy")

    def logs_metrics(self, name: str) -> pd.DataFrame | None:
        return self.load_csv(f"logs/{CONFIGS[name]}/metrics.csv")

    def baseline_metrics(self, key: str) -> dict | None:
        return self.load_json(f"baselines/{key}/metrics.json")

    def oracle_inputs(self) -> dict | None:
        responses = self.load_npy("oracle/training_responses.npy")
        training = self.load_csv("oracle/training_drugs.csv")
        drugs = self.load_csv("oracle/drugs_172.csv")
        if responses is None or training is None or drugs is None:
            return None
        return {
            "responses": responses,
            "training_drugs": training,
            "drugs_172": drugs,
        }

    def replicates(self) -> dict | None:
        rep1 = self.load_npy("replicates/crossplate_rep1.npy")
        rep2 = self.load_npy("replicates/crossplate_rep2.npy")
        if rep1 is None or rep2 is None:
            return None
        return {"rep1": rep1, "rep2": rep2}

    def tahoe(self) -> dict | None:
        pred_mean = self.load_npy("tahoe/pred_mean_tahoe.npy")
        pred_ridge = self.load_npy("tahoe/pred_ridge_tahoe.npy")
        true = self.load_npy("tahoe/logfc_true_tahoe.npy")
        meta = self.load_csv("tahoe/logfc_meta_tahoe.csv")
        if None in (pred_mean, pred_ridge, true, meta):
            return None
        return {
            "pred_mean": pred_mean,
            "pred_ridge": pred_ridge,
            "true": true,
            "meta": meta,
        }


def _unzip(path: Path) -> Path:
    """Extract a bundle zip next to itself and return the extracted directory."""
    target = path.with_suffix("")
    if not target.exists():
        target.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target)
    return target
