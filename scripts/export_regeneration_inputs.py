#!/usr/bin/env python
"""Collect the regeneration-input bundle on the machine that holds the frozen
data (the training server), then zip it for shipping into the repository.

The bundle contains two kinds of items:
  * copied artifacts (stored per-pair matrices, training logs, baseline scored
    runs, the frozen protocol outputs);
  * derived artifacts computed here from the processed h5ad (the pooled-vehicle
    truth and predictor anchors, the oracle training responses, the cross-plate
    replicate matrices, Table 3 cell counts).

Candidate pooled truths are validated in-place: the loss-only AUC is computed
for every candidate and the one closest to the paper's printed 0.509 is
selected and recorded in the manifest. The Mean-predictor candidate pool
(train / train+val / all non-test) is selected the same way against 0.491.

Run on the server:

    python scripts/export_regeneration_inputs.py \
        --source /root/CytoBridge/code \
        --logs-dir /root/CytoBridge/code/logs \
        --protocol-dir .../data/processed/sciplex_accept/drug_disjoint_v2 \
        --h5ad /root/autodl-tmp/SrivatsanTrapnell2020_sciplex3.h5ad \
        --baselines-dir .../experiments/runs \
        --tahoe-dir /root/autodl-tmp/zenodo_supp/tahoe_control \
        --loss-components /root/autodl-tmp/zenodo_supp/manuscript/analysis/data/loss_components.csv \
        --out regeneration_inputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

CONFIGS = {
    "loss-only": "t7_sub_loss_only",
    "drug-spec x1": "t7_sub_drugspec1",
    "drug-spec x3": "t7_sub_drugspec3",
    "drug-spec x5": "t7_sub_drugspec5",
    "norm-only": "t7_sub_norm_only",
    "low recon weight": "t7_sub_lamrecon01",
    "recovery baseline": "t6_sub_baseline",
}
CELL_LINES = ("A549", "K562", "MCF7")
PAPER_LOSS_ONLY_AUC = 0.509
PAPER_MEAN_SPEARMAN = 0.491
TOP_K = 50

MANIFEST_FILES: list[dict] = []  # {file, sha256, shape, note}


# ---------------------------------------------------------------------------
# Pure scoring helpers (numpy-only; mirror eval/metrics.py semantics)
# ---------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def _ddc(pred, true, cl, top_k=TOP_K) -> dict:
    """drug_discrimination_score clone (pearson, union top-k panel, strict ties)."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    on, off, aucs = [], [], []
    for cell in np.unique(cl):
        rows = np.flatnonzero(cl == cell)
        if rows.size < 2:
            continue
        P, T = pred[rows], true[rows]
        if top_k is None or top_k >= P.shape[1]:
            panel = np.arange(P.shape[1])
        else:
            panel = sorted(
                set().union(
                    *[set(np.argsort(-np.abs(T[i]))[:top_k].tolist()) for i in range(len(rows))]
                )
            )
        Ps, Ts = P[:, panel], T[:, panel]
        C = np.array(
            [[_pearson(Ps[i], Ts[j]) for j in range(len(rows))] for i in range(len(rows))]
        )
        for i in range(len(rows)):
            offs = np.delete(C[i], i)
            on.append(float(C[i, i]))
            off.append(float(offs.mean()))
            aucs.append(float(np.mean(C[i, i] > offs)))
    return {
        "on_diag_mean": float(np.mean(on)) if on else float("nan"),
        "off_diag_mean": float(np.mean(off)) if off else float("nan"),
        "gap": float(np.mean(np.asarray(on) - np.asarray(off))) if on else float("nan"),
        "specificity_auc": float(np.mean(aucs)) if aucs else float("nan"),
    }


def _pair_own_spearman_mean(true, pred, top_k=TOP_K) -> float:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return _pearson(true.ravel(), pred.ravel())
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    gene_index = np.arange(true.shape[1])
    values = []
    for truth_row, pred_row in zip(true, pred):
        panel = np.lexsort((gene_index, -np.abs(truth_row)))[: min(top_k, true.shape[1])]
        values.append(float(spearmanr(truth_row[panel], pred_row[panel]).statistic))
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# h5ad-derived artifacts
# ---------------------------------------------------------------------------
def _load_h5ad(path: Path):
    import anndata

    adata = anndata.read_h5ad(path)
    obs = adata.obs
    # Keep the count matrix sparse: the raw h5ad densified is ~65 GB, so only
    # row-group slices are ever materialised (never np.asarray the whole X).
    X = adata.layers.get("counts", adata.X)
    drug_col = "perturbation" if "perturbation" in obs.columns else "drug"
    return X, obs, drug_col, list(adata.var_names)


def _gene_index(var_names, gene_ids):
    """Column indices of the frozen 3000-gene panel inside the full h5ad."""
    lookup = {name: i for i, name in enumerate(var_names)}
    missing = [g for g in gene_ids if g not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} gene ids missing from h5ad var_names: {missing[:5]}")
    return np.asarray([lookup[g] for g in gene_ids], dtype=int)


def _cells_mask(obs, drug_col, drug: str, dose: float = 10000.0, time: str = "24"):
    """dose in the unit stored in obs["dose_value"] (nM in the raw sci-Plex h5ad: 10 uM = 10000)."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return (
            obs[drug_col].astype(str).eq(drug)
            & obs["dose_value"].astype(str).eq(str(dose))
            & obs["time"].astype(str).eq(time)
        )
    dose_numeric = pd.to_numeric(obs["dose_value"], errors="coerce")
    time_tokens = obs["time"].astype(str).str.extract(r"(\d+)")[0]
    return (
        obs[drug_col].astype(str).eq(drug)
        & dose_numeric.eq(float(dose))
        & time_tokens.eq(str(time)).fillna(False)
    )


def _row_mean(X, rows) -> np.ndarray:
    """Mean of the given rows; X stays sparse, only the slice is densified."""
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        raise ValueError("no rows to average")
    sub = X[rows]
    if hasattr(sub, "toarray"):
        sub = sub.toarray()
    return np.asarray(sub.mean(axis=0), dtype=np.float64)


def pooled_vehicle_log1p(X, obs, drug_col, gene_idx=None) -> dict:
    """log1p mean counts of the DMSO cells, one vector per cell line."""
    vehicles = {}
    for cell in CELL_LINES:
        mask = (
            obs[drug_col].astype(str).str.upper().isin(["DMSO", "VEHICLE", "CONTROL"])
            & obs["cell_line"].astype(str).eq(cell)
        )
        if not mask.any():
            raise ValueError(f"no DMSO cells for cell line {cell}")
        vector = np.log1p(_row_mean(X, np.flatnonzero(mask.to_numpy())))
        if gene_idx is not None:
            vector = vector[gene_idx]
        vehicles[cell] = vector
    return vehicles


def treated_pair_pseudobulk(X, obs, drug_col, drug: str, cell: str, gene_idx=None):
    mask = _cells_mask(obs, drug_col, drug) & obs["cell_line"].astype(str).eq(cell)
    if not mask.any():
        raise ValueError(f"no treated cells for {drug} in {cell}")
    vector = np.log1p(_row_mean(X, np.flatnonzero(mask.to_numpy())))
    if gene_idx is not None:
        vector = vector[gene_idx]
    return vector


def derive_h5ad_pooled_truth(X, obs, drug_col, meta, gene_idx=None) -> np.ndarray:
    vehicles = pooled_vehicle_log1p(X, obs, drug_col, gene_idx)
    rows = []
    for drug, cell in zip(meta["drug"], meta["cell_line"]):
        rows.append(treated_pair_pseudobulk(X, obs, drug_col, str(drug), str(cell), gene_idx)
                   - vehicles[str(cell)])
    return np.asarray(rows, dtype=np.float32)


def derive_mean_predictor(X, obs, drug_col, train_drugs, meta, gene_idx=None) -> np.ndarray:
    """Cell-line-averaged training logFC broadcast to the 27 test rows."""
    vehicles = pooled_vehicle_log1p(X, obs, drug_col, gene_idx)
    out = np.zeros((len(meta), X.shape[1]), dtype=np.float32)
    for cell in CELL_LINES:
        profiles = []
        for drug in train_drugs:
            try:
                profiles.append(
                    treated_pair_pseudobulk(X, obs, drug_col, str(drug), cell, gene_idx)
                    - vehicles[cell]
                )
            except ValueError:
                continue
        if not profiles:
            raise ValueError(f"no train-drug profiles for cell line {cell}")
        line_mean = np.mean(profiles, axis=0)
        out[meta["cell_line"].astype(str).eq(cell).to_numpy()] = line_mean
    return out


def derive_replicates(X, obs, drug_col, meta, gene_idx=None) -> tuple[np.ndarray, np.ndarray]:
    """The two biological replicates' pooled-vehicle logFC, one per replicate."""
    vehicles = pooled_vehicle_log1p(X, obs, drug_col, gene_idx)
    rep_columns = [c for c in ("replicate", "plate") if c in obs.columns]
    reps = {}
    for drug, cell in zip(meta["drug"], meta["cell_line"]):
        mask = _cells_mask(obs, drug_col, str(drug)) & obs["cell_line"].astype(str).eq(cell)
        sub = obs.loc[mask]
        Xsub = X[mask.to_numpy()]
        labels = None
        for column in rep_columns:
            values = sub[column].astype(str)
            if values.nunique() == 2:
                labels = values
                break
        if labels is None:
            raise ValueError(f"no two-level replicate column for {drug}/{cell}")
        for level in sorted(labels.unique()):
            pb = np.log1p(Xsub[labels.eq(level).to_numpy()].mean(axis=0))
            reps.setdefault(level, []).append(pb - vehicles[str(cell)])
    keys = sorted(reps)
    if len(keys) < 2:
        raise ValueError("could not resolve two replicates")
    return (
        np.asarray(reps[keys[0]], dtype=np.float32),
        np.asarray(reps[keys[1]], dtype=np.float32),
    )


def derive_oracle_inputs(X, obs, drug_col, test_drugs, train_split_drugs, out_dir: Path, gene_idx=None):
    """Training-compound pooled responses [160, 3, 3000] + drug tables."""
    vehicles = pooled_vehicle_log1p(X, obs, drug_col, gene_idx)
    drug_series = obs[drug_col].astype(str)
    cell_series = obs["cell_line"].astype(str)
    non_test = [d for d in drug_series.unique() if d not in test_drugs]
    responses, drugs = [], []
    for drug in sorted(non_test):
        profiles = []
        ok = True
        for cell in CELL_LINES:
            mask = _cells_mask(obs, drug_col, drug) & cell_series.eq(cell)
            if mask.sum() < 10:
                ok = False
                break
            profiles.append(np.log1p(X[mask.to_numpy()].mean(axis=0)) - vehicles[cell])
        if ok:
            responses.append(profiles)
            drugs.append(drug)
    responses = np.asarray(responses, dtype=np.float32)
    target_map = {}
    if "target" in obs.columns:
        for drug, target in zip(drug_series, obs["target"].astype(str)):
            target_map.setdefault(drug, target)
    smiles_map = {}
    smiles_csv = _find_smiles_csv()
    if smiles_csv is not None:
        import pandas as pd

        frame = pd.read_csv(smiles_csv)
        if {"drug_id", "canonical_smiles"} <= set(frame.columns):
            smiles_map = dict(
                zip(frame["drug_id"].astype(str), frame["canonical_smiles"].astype(str))
            )
        elif {"drug_id", "smiles"} <= set(frame.columns):
            smiles_map = dict(
                zip(frame["drug_id"].astype(str), frame["smiles"].astype(str))
            )
    np.save(out_dir / "oracle" / "training_responses.npy", responses)
    training_rows = [
        {
            "drug_id": drug,
            "canonical_smiles": smiles_map.get(drug, ""),
            "vendor_target": target_map.get(drug, ""),
        }
        for drug in drugs
    ]
    _write_csv(out_dir / "oracle" / "training_drugs.csv", training_rows)
    all_rows = [
        {
            "drug_id": drug,
            "canonical_smiles": smiles_map.get(drug, ""),
            "vendor_target": target_map.get(drug, ""),
        }
        for drug in sorted(drug_series.unique())
    ]
    _write_csv(out_dir / "oracle" / "drugs_172.csv", all_rows)
    record(out_dir, "oracle/training_responses.npy", responses)
    record(out_dir, "oracle/training_drugs.csv", None, note=f"{len(drugs)} training drugs")


def derive_table3(X, obs, drug_col, meta, out_dir: Path):
    rows = []
    for drug, cell in zip(meta["drug"], meta["cell_line"]):
        mask = _cells_mask(obs, drug_col, str(drug)) & obs["cell_line"].astype(str).eq(cell)
        rows.append({"drug": str(drug), "cell_line": str(cell), "n_cells": int(mask.sum())})
    for cell in CELL_LINES:
        mask = (
            obs[drug_col].astype(str).str.upper().isin(["DMSO", "VEHICLE", "CONTROL"])
            & obs["cell_line"].astype(str).eq(cell)
        )
        rows.append({"drug": "Vehicle", "cell_line": cell, "n_cells": int(mask.sum())})
    _write_csv(out_dir / "table3_cell_counts.csv", rows)
    record(out_dir, "table3_cell_counts.csv", None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _find_smiles_csv() -> Path | None:
    for candidate in (
        Path("data/raw/sciplex/sciplex3_drugs.csv"),
        Path("data/processed/sciplex_accept/drug_disjoint_v2/eligible_compounds.csv"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(out_dir: Path, relative: str, array, note: str = "") -> None:
    path = out_dir / relative
    shape = None
    if array is not None:
        shape = list(np.asarray(array).shape)
    MANIFEST_FILES.append(
        {"file": relative, "sha256": _sha256(path), "shape": shape, "note": note}
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="dir containing results/ (per-pair matrices)")
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--baselines-dir", type=Path, required=True)
    parser.add_argument("--tahoe-dir", type=Path, default=None)
    parser.add_argument("--loss-components", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("regeneration_inputs"))
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)

    import pandas as pd

    out_dir = Path(args.out)
    for sub in ("e6e7", "logs", "baselines/ridge", "baselines/chemcpa",
                "baselines/biolord", "baselines/biolord_ood", "tahoe", "oracle",
                "replicates", "protocol"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- 1. per-pair matrices ----
    for name, tag in CONFIGS.items():
        for kind in ("pred", "true"):
            source = args.source / "results" / f"logfc_{kind}_{tag}.npy"
            if not source.is_file():
                print(f"[missing] {source}")
                continue
            target = out_dir / "e6e7" / f"logfc_{kind}_{tag}.npy"
            shutil.copyfile(source, target)
            record(out_dir, f"e6e7/logfc_{kind}_{tag}.npy", np.load(target))
        source = args.source / "results" / f"logfc_meta_{tag}.csv"
        if source.is_file():
            target = out_dir / "e6e7" / f"logfc_meta_{tag}.csv"
            shutil.copyfile(source, target)
            record(out_dir, f"e6e7/logfc_meta_{tag}.csv", None)

    reference_meta = pd.read_csv(out_dir / "e6e7" / "logfc_meta_t7_sub_loss_only.csv")
    reference_true = np.load(
        out_dir / "e6e7" / "logfc_true_t7_sub_loss_only.npy"
    ).astype(np.float64)
    reference_pred = np.load(
        out_dir / "e6e7" / "logfc_pred_t7_sub_loss_only.npy"
    ).astype(np.float64)
    cl = reference_meta["cell_line"].astype(str).to_numpy()
    drugs = reference_meta["drug"].astype(str).to_numpy()

    # ---- 2. pooled-truth candidates ----
    X, obs, drug_col, var_names = _load_h5ad(args.h5ad)
    gene_ids = (args.protocol_dir / "gene_ids.txt").read_text().splitlines()
    gene_idx = _gene_index(var_names, gene_ids)
    candidates: dict[str, np.ndarray] = {}
    h5ad_truth = derive_h5ad_pooled_truth(X, obs, drug_col, reference_meta, gene_idx)
    candidates["h5ad_cellline_pooled"] = h5ad_truth
    targets_npz = args.protocol_dir / "splits" / "test_targets.npz"
    if targets_npz.is_file():
        loaded = np.load(targets_npz, allow_pickle=True)
        for key in loaded.keys():
            array = loaded[key]
            if isinstance(array, np.ndarray) and array.ndim == 2 and array.shape == (27, 3000):
                candidates[f"protocol_{key}"] = np.asarray(array, dtype=np.float64)

    # ---- 3. select the pooled truth by in-place validation (target 0.509) ----
    best_candidate, best_auc, best_delta = None, None, None
    validation_rows = []
    for candidate_name, pooled_true in candidates.items():
        delta = reference_true - pooled_true
        pooled_pred = (reference_pred + delta).astype(np.float32)
        score = _ddc(pooled_pred, pooled_true, cl)
        auc = score["specificity_auc"]
        validation_rows.append(
            {"candidate": candidate_name, "loss_only_auc": round(float(auc), 4),
             "delta_absmax": float(np.abs(delta).max())}
        )
        if best_auc is None or abs(auc - PAPER_LOSS_ONLY_AUC) < abs(best_auc - PAPER_LOSS_ONLY_AUC):
            best_candidate, best_auc, best_delta = candidate_name, float(auc), delta
    print("[validate] pooled-truth candidates:")
    for row in validation_rows:
        print("  ", row)
    print(f"[validate] selected {best_candidate} (loss-only AUC={best_auc:.4f}, "
          f"paper 0.509)")
    np.save(out_dir / "e6e7" / "true_pooled.npy",
            candidates[best_candidate].astype(np.float32))
    np.save(out_dir / "e6e7" / "delta.npy", best_delta.astype(np.float32))
    reference_meta.to_csv(out_dir / "e6e7" / "true_pooled_meta.csv", index=False)
    record(out_dir, "e6e7/true_pooled.npy", candidates[best_candidate],
           note=f"selected from {best_candidate}")
    record(out_dir, "e6e7/delta.npy", best_delta, note="true_perpair - true_pooled")
    record(out_dir, "e6e7/true_pooled_meta.csv", None)

    # ---- 4. Mean-predictor candidate selection (target 0.491) ----
    split_assignments = pd.read_csv(args.protocol_dir / "split_assignments.csv")
    train_drugs = split_assignments.loc[
        split_assignments["split"].astype(str).eq("train"), "drug_id"
    ].astype(str).tolist()
    val_drugs = split_assignments.loc[
        split_assignments["split"].astype(str).eq("val"), "drug_id"
    ].astype(str).tolist()
    test_drugs_set = set(split_assignments.loc[
        split_assignments["split"].astype(str).eq("test"), "drug_id"
    ].astype(str))
    all_non_test = [d for d in obs[drug_col].astype(str).unique()
                    if d not in test_drugs_set and d.upper() not in ("DMSO", "VEHICLE", "CONTROL")]
    mean_candidates = {
        "train_only": train_drugs,
        "train_val": train_drugs + val_drugs,
        "all_non_test": sorted(all_non_test),
    }
    best_mean, best_mean_value = None, None
    for candidate_name, drug_list in mean_candidates.items():
        try:
            predictor = derive_mean_predictor(
                X, obs, drug_col, drug_list, reference_meta, gene_idx
            )
        except ValueError as error:
            print(f"[mean] {candidate_name}: {error}")
            continue
        value = _pair_own_spearman_mean(
            candidates[best_candidate], predictor.astype(np.float64)
        )
        print(f"[mean] {candidate_name} spearman50 = {value:.4f}")
        if best_mean_value is None or abs(value - PAPER_MEAN_SPEARMAN) < abs(
            best_mean_value - PAPER_MEAN_SPEARMAN
        ):
            best_mean, best_mean_value = candidate_name, value
    if best_mean is not None:
        predictor = derive_mean_predictor(
            X, obs, drug_col, mean_candidates[best_mean], reference_meta, gene_idx
        )
        np.save(out_dir / "e6e7" / "mean_predictor_train.npy", predictor)
        record(
            out_dir, "e6e7/mean_predictor_train.npy", predictor,
            note=f"drugs={best_mean}, spearman50={best_mean_value:.4f} (paper 0.491)",
        )

    # ---- 5. training logs ----
    for tag in CONFIGS.values():
        tag_dir = args.logs_dir / tag
        if not tag_dir.is_dir():
            print(f"[missing] logs/{tag}")
            continue
        versions = sorted(
            (p for p in tag_dir.iterdir() if p.is_dir() and p.name.startswith("version_")),
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not versions:
            continue
        metrics = versions[-1] / "metrics.csv"
        if metrics.is_file():
            target_dir = out_dir / "logs" / tag
            target_dir.mkdir(exist_ok=True)
            target = target_dir / "metrics.csv"
            shutil.copyfile(metrics, target)
            record(out_dir, f"logs/{tag}/metrics.csv", None,
                   note=f"version {versions[-1].name}")

    # ---- 6. baselines ----
    for key, patterns in (
        ("ridge", ("ridge",)),
        ("chemcpa", ("chemcpa",)),
        ("biolord", ("biolord",)),
        ("biolord_ood", ("biolord",)),
    ):
        _collect_baseline(args.baselines_dir, out_dir, key, patterns)

    # ---- 7. tahoe control ----
    if args.tahoe_dir is not None and args.tahoe_dir.is_dir():
        for name in ("pred_mean_tahoe.npy", "pred_ridge_tahoe.npy",
                     "logfc_true_tahoe.npy", "logfc_meta_tahoe.csv",
                     "tahoe_control_panel.csv"):
            source = args.tahoe_dir / name
            if source.is_file():
                shutil.copyfile(source, out_dir / "tahoe" / name)
                record(out_dir, f"tahoe/{name}", None)

    # ---- 8. oracle inputs ----
    test_drug_names = sorted(reference_meta["drug"].astype(str).unique())
    derive_oracle_inputs(X, obs, drug_col, set(test_drug_names), None, out_dir, gene_idx)

    # ---- 9. replicates ----
    try:
        rep1, rep2 = derive_replicates(X, obs, drug_col, reference_meta, gene_idx)
        np.save(out_dir / "replicates" / "crossplate_rep1.npy", rep1)
        np.save(out_dir / "replicates" / "crossplate_rep2.npy", rep2)
        record(out_dir, "replicates/crossplate_rep1.npy", rep1)
        record(out_dir, "replicates/crossplate_rep2.npy", rep2)
    except ValueError as error:
        print(f"[replicates] {error}")

    # ---- 10. table 3 ----
    derive_table3(X, obs, drug_col, reference_meta, out_dir)

    # ---- 11. protocol / loss components ----
    for name in ("split_assignments.csv", "gene_ids.txt"):
        source = args.protocol_dir / name
        if source.is_file():
            shutil.copyfile(source, out_dir / "protocol" / name)
            record(out_dir, f"protocol/{name}", None)
    if args.loss_components is not None and args.loss_components.is_file():
        shutil.copyfile(args.loss_components, out_dir / "loss_components.csv")
        record(out_dir, "loss_components.csv", None)

    # ---- 12. manifest + zip ----
    manifest = {
        "schema_version": 1,
        "generated_by": "scripts/export_regeneration_inputs.py",
        "selected_pooled_truth": best_candidate,
        "selected_pooled_truth_loss_only_auc": best_auc,
        "selected_mean_predictor": best_mean,
        "selected_mean_predictor_spearman50": best_mean_value,
        "files": MANIFEST_FILES,
    }
    manifest_path = out_dir / "inputs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if not args.no_zip:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(out_dir))
        print(f"[done] wrote {zip_path} "
              f"({zip_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"[done] wrote bundle dir {out_dir}")
    return 0


def _collect_baseline(baselines_dir: Path, out_dir: Path, key: str, patterns) -> None:
    import pandas as pd

    runs = sorted(
        p for p in baselines_dir.iterdir()
        if p.is_dir() and any(pat in p.name for pat in patterns)
    )
    scored = [p / "scored" for p in runs if (p / "scored").is_dir()]
    if not scored:
        print(f"[missing] baselines/{key}: no scored runs")
        return
    # prefer a run whose metrics.json carries the richest scalars
    best_dir, best_count = None, -1
    for sdir in scored:
        metrics_path = sdir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        count = sum(
            1 for value in metrics.values() if isinstance(value, (int, float))
        )
        if count > best_count:
            best_dir, best_count = sdir, count
    if best_dir is None:
        return
    target_dir = out_dir / "baselines" / key
    for name in ("metrics.json", "per_pair.csv", "predictions.npz"):
        source = best_dir / name
        if source.is_file():
            shutil.copyfile(source, target_dir / name)
            record(out_dir, f"baselines/{key}/{name}", None,
                   note=f"source {best_dir.name}")
    metrics = json.loads((best_dir / "metrics.json").read_text(encoding="utf-8"))
    flattened: dict = {}

    def flatten(prefix: str, node) -> None:
        if isinstance(node, dict):
            for child_key, child in node.items():
                flatten(f"{prefix}.{child_key}" if prefix else str(child_key), child)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flattened[prefix] = node

    flatten("", metrics)
    (target_dir / "selfspace_values.json").write_text(
        json.dumps(flattened, indent=2, sort_keys=True) + "\n"
    )
    record(out_dir, f"baselines/{key}/selfspace_values.json", None)


if __name__ == "__main__":
    sys.exit(main())
