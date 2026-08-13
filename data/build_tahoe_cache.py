"""
data/build_tahoe_cache.py
-------------------------
Manifest-driven cache builder for Tahoe external evaluation.

`eval/run_tahoe.py` consumes five caches per split. Three of them are indexed
*by manifest row* (treated counts, control counts, GSEA labels) and two of
them are indexed *by sliced-h5ad row* (cell embeddings) or *by drug position
in the npz* (MolFormer embeddings). If a downstream cache is generated in a
different order than the manifest expects, every prediction is silently
misaligned. This script is the one place that produces all five caches in
the correct order and writes a `<split>_cache_meta.json` with shapes,
manifest checksum and per-array SHA256 so `eval/run_tahoe.py` can assert
alignment before evaluation.

Usage:
    python data/build_tahoe_cache.py --split external_1
    python data/build_tahoe_cache.py --split external_2

Inputs (produced by data/build_external_split.py):
    data/processed/tahoe/splits/tahoe_<split>.parquet  (manifest)
    data/processed/tahoe/splits/tahoe_<split>.h5ad     (sliced AnnData;
                                                       union of treated and
                                                       paired controls)

Outputs (under --cache_dir, default `data/cache/`):
    tahoe_<split>_scgpt_emb.npy           [n_h5ad_rows, L, d_cell]
    tahoe_<split>_molformer_emb.npz       drug_ids/tokens/masks
    tahoe_<split>_treated_counts.npy      [n_manifest, n_genes]
    tahoe_<split>_control_counts.npy      [n_manifest, n_genes]
    tahoe_<split>_pathway_gsea.npy        [n_manifest, K]   (zero fallback if
                                                              --pathway_gsea is
                                                              not supplied)
    tahoe_<split>_cache_meta.json         shapes / sha256 / manifest hash
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _sha256_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _smiles_csv_path(splits_dir: Path, split: str) -> Path:
    return splits_dir / f"tahoe_{split}_smiles.csv"


def _validate_smiles_csv(path: Path, manifest: pd.DataFrame) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Tahoe SMILES CSV: {path}. Create this file with columns "
            "`drug_id,smiles` covering every `drug_id` in the split manifest, then rerun "
            "`python data/build_tahoe_cache.py --split <external_1|external_2>`. "
            "Use `--skip_embeddings` only for count/GSEA smoke-cache generation."
        )
    drug_smiles = pd.read_csv(path)
    required_cols = {"drug_id", "smiles"}
    missing_cols = required_cols - set(drug_smiles.columns)
    if missing_cols:
        raise ValueError(f"SMILES CSV {path} missing columns: {sorted(missing_cols)}")
    manifest_drugs = set(manifest["drug_id"].astype(str))
    smiles_drugs = set(drug_smiles["drug_id"].astype(str))
    missing_drugs = manifest_drugs - smiles_drugs
    if missing_drugs:
        raise ValueError(
            f"SMILES CSV {path} is missing {len(missing_drugs)} manifest drugs "
            f"(e.g. {sorted(missing_drugs)[:5]}). Add rows for those `drug_id` values "
            "before building the MolFormer cache."
        )


def build_cache(
    split: str,
    splits_dir: Path,
    cache_dir: Path,
    n_pathways: int = 50,
    pathway_gsea_path: Path | None = None,
    skip_embeddings: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = splits_dir / f"tahoe_{split}.parquet"
    h5ad_path = splits_dir / f"tahoe_{split}.h5ad"
    if not manifest_path.exists() or not h5ad_path.exists():
        raise FileNotFoundError(
            f"Expected {manifest_path} and {h5ad_path}. "
            "Run `python data/build_external_split.py --tahoe_h5ad ...` first."
        )

    manifest = pd.read_parquet(manifest_path)
    required_cols = {"cell_idx", "control_cell_idx", "drug_id", "cell_line"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest {manifest_path} missing columns: {missing}")
    smiles_path = _smiles_csv_path(splits_dir, split)
    if not skip_embeddings:
        _validate_smiles_csv(smiles_path, manifest)

    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)
    n_h5ad = adata.n_obs

    # Bounds-check before any cache write.
    treated = manifest["cell_idx"].to_numpy()
    control = manifest["control_cell_idx"].to_numpy()
    if treated.max() >= n_h5ad or control.max() >= n_h5ad:
        raise ValueError(
            f"manifest references row {max(treated.max(), control.max())} "
            f"but sliced h5ad has only {n_h5ad} rows; rebuild splits."
        )

    counts_layer = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if hasattr(counts_layer, "toarray"):
        counts_layer = counts_layer.toarray()
    counts_layer = np.asarray(counts_layer, dtype=np.float32)

    treated_counts = counts_layer[treated]
    control_counts = counts_layer[control]
    np.save(cache_dir / f"tahoe_{split}_treated_counts.npy", treated_counts)
    np.save(cache_dir / f"tahoe_{split}_control_counts.npy", control_counts)
    print(f"[tahoe-cache] treated/control counts: {treated_counts.shape}")

    if pathway_gsea_path is not None and pathway_gsea_path.exists():
        gsea = np.load(pathway_gsea_path)
        if gsea.shape[0] != len(manifest):
            raise ValueError(
                f"--pathway_gsea has {gsea.shape[0]} rows, expected "
                f"{len(manifest)} (manifest length)."
            )
    else:
        gsea = np.zeros((len(manifest), n_pathways), dtype=np.float32)
        print(
            "[tahoe-cache] no --pathway_gsea supplied; emitting zero fallback "
            f"of shape {gsea.shape} for optional external evaluation."
        )
    np.save(cache_dir / f"tahoe_{split}_pathway_gsea.npy", gsea)

    if not skip_embeddings:
        from cytobridge.encoders.scgpt_wrapper import cache_embeddings_to_disk
        from cytobridge.encoders.molformer_wrapper import cache_drug_embeddings

        cell_emb_out = cache_dir / f"tahoe_{split}_scgpt_emb.npy"
        cache_embeddings_to_disk(h5ad_path, cell_emb_out)

        # Build a SMILES table covering every manifest drug, in manifest order.
        # The MolFormer cache is keyed by drug_id, but `CytoBridgeDataset`
        # rebuilds drug_id_to_idx at runtime, so order of npz entries doesn't
        # need to match manifest — only that every drug_id is present.
        cache_drug_embeddings(
            smiles_path,
            cache_dir / f"tahoe_{split}_molformer_emb.npz",
        )

    cell_emb_path = cache_dir / f"tahoe_{split}_scgpt_emb.npy"
    drug_emb_path = cache_dir / f"tahoe_{split}_molformer_emb.npz"
    meta = {
        "split": split,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_of_file(manifest_path),
        "h5ad_path": str(h5ad_path),
        "h5ad_rows": int(n_h5ad),
        "manifest_rows": int(len(manifest)),
        "treated_counts_shape": list(treated_counts.shape),
        "treated_counts_sha256": _sha256_of_array(treated_counts),
        "control_counts_shape": list(control_counts.shape),
        "control_counts_sha256": _sha256_of_array(control_counts),
        "pathway_gsea_shape": list(gsea.shape),
    }
    if cell_emb_path.exists():
        emb = np.load(cell_emb_path, mmap_mode="r")
        meta["scgpt_emb_shape"] = list(emb.shape)
        if emb.shape[0] != n_h5ad:
            raise ValueError(
                f"scGPT embedding has {emb.shape[0]} rows, "
                f"expected {n_h5ad} (sliced h5ad rows)."
            )
    if drug_emb_path.exists():
        loaded = np.load(drug_emb_path)
        meta["molformer_emb_keys"] = list(loaded.files)
        manifest_drugs = set(manifest["drug_id"].astype(str))
        cache_drugs = set(str(x) for x in loaded["drug_ids"])
        missing_drugs = manifest_drugs - cache_drugs
        if missing_drugs:
            raise ValueError(
                f"MolFormer cache is missing {len(missing_drugs)} drugs from "
                f"the manifest (e.g. {sorted(missing_drugs)[:5]}). Add them to "
                f"{splits_dir}/tahoe_{split}_smiles.csv and rerun."
            )

    meta_path = cache_dir / f"tahoe_{split}_cache_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[tahoe-cache] wrote {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["external_1", "external_2"], required=True)
    parser.add_argument("--splits_dir", type=Path,
                        default=Path("data/processed/tahoe/splits"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--n_pathways", type=int, default=50)
    parser.add_argument("--pathway_gsea", type=Path, default=None,
                        help="Optional precomputed GSEA labels per manifest row.")
    parser.add_argument("--skip_embeddings", action="store_true",
                        help="Only build the count/GSEA caches; useful for CI smoke tests.")
    args = parser.parse_args()
    raise SystemExit(
        "LEGACY TAHOE ENTRYPOINT DISABLED: per-external-tier caches are not part of "
        "protocol 1.4. Use scripts/build_tahoe_encoder_caches.sh instead."
    )
    build_cache(
        split=args.split,
        splits_dir=args.splits_dir,
        cache_dir=args.cache_dir,
        n_pathways=args.n_pathways,
        pathway_gsea_path=args.pathway_gsea,
        skip_embeddings=args.skip_embeddings,
    )


if __name__ == "__main__":
    main()
