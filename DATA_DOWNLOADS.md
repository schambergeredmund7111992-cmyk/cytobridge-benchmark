# Data and Checkpoint Sources

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21912287.svg)](https://doi.org/10.5281/zenodo.21912287)

This repository contains only source code (~30K lines). Large binary files — datasets,
pretrained checkpoints, cached embeddings, trained model weights, and experiment results —
are distributed via Google Drive and Zenodo.

## Google Drive (recommended for large datasets)

All binary artifacts live in one shared folder:

[Google Drive shared folder](https://drive.google.com/drive/folders/1oSdZp8M1Sp39Vph9JLqrXt3PrVQJP8VF)

| File (inside the shared folder) | Contents | Size |
|---|---|---|
| `result.tar.gz` | P1/P2/P3 experiment results (104 jobs, all predictions and metrics) | 432 MB |
| `SrivatsanTrapnell2020_sciplex3.h5ad` | Raw sci-Plex 3 dataset (~500K cells, 3 cell lines, 188 drugs) | 2.4 GB |
| `tahoe_data_merged.tar.gz` | Tahoe-100M selected panel (metadata + expression) | 703 MB |
| `tahoe_from_local.h5ad` | Tahoe preprocessed AnnData (gene-aligned with sci-Plex) | 603 MB |
| `tahoe_expression_data.tar.gz` | Tahoe raw expression matrix | 173 MB |
| `checkpoints.zip` | Trained model checkpoints | 2.43 GB |

## Zenodo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21912287.svg)](https://doi.org/10.5281/zenodo.21912287)

Source code snapshot and small figure-reproduction data are archived at
[10.5281/zenodo.21912287](https://doi.org/10.5281/zenodo.21912287) (version v1.0.3;
all-version DOI: [10.5281/zenodo.21911960](https://doi.org/10.5281/zenodo.21911960)).

## Direct download of public datasets

If you prefer to download raw data directly from public sources:

| Dataset | Source | Command |
|---|---|---|
| **sci-Plex** | GEO GSE139944 / scPerturb | `python -m data.download --target sciplex --out data/raw/` |
| **MSigDB Hallmark** | GSEA/MSigDB v2024.1 | Download `h.all.v2024.1.Hs.symbols.gmt` to `data/raw/msigdb/` |
| **Tahoe-100M** | HuggingFace `tahoebio/Tahoe-100M` | `python -m data.download_tahoe_metadata --out data/raw/tahoe` |

## Setup after download

### sci-Plex

```bash
python -m data.download --target sciplex --out data/raw
python -m data.prepare_sciplex_scperturb --fetch_pubchem
python -m data.preprocess --config configs/data/sciplex.yaml

for protocol in drug_disjoint_v2 scaffold_disjoint_v2; do
  python -m data.pathway_gsea \
    --protocol-dir data/processed/sciplex_accept/$protocol \
    --gmt data/raw/msigdb/h.all.v2024.1.Hs.symbols.gmt
done
```

If you use `SrivatsanTrapnell2020_sciplex3.h5ad` from the Google Drive folder instead,
place it at `data/raw/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad` (the default `--input`);
`--fetch_pubchem` then only refreshes the SMILES table when needed.

### Tahoe

Tahoe preprocessing depends on the sci-Plex protocol output above (`gene_ids.txt`).

```bash
tar -xzf tahoe_expression_data.tar.gz -C data/raw/tahoe/
tar -xzf tahoe_data_merged.tar.gz -C data/processed/
python -m data.download_tahoe_metadata --out data/raw/tahoe
python -m data.select_tahoe_streaming \
  --obs-metadata data/raw/tahoe/metadata/obs_metadata.parquet \
  --sample-metadata data/raw/tahoe/metadata/sample_metadata.parquet \
  --drug-metadata data/raw/tahoe/metadata/drug_metadata.parquet \
  --out data/processed/tahoe_selection
python -m data.stream_tahoe_panel \
  --selected-cells data/processed/tahoe_selection/selected_cells.parquet \
  --gene-metadata data/raw/tahoe/metadata/gene_metadata.parquet \
  --out data/raw/tahoe/selected_panel.h5ad
python -m data.preprocess_tahoe \
  --selected-h5ad data/raw/tahoe/selected_panel.h5ad \
  --sciplex-gene-ids data/processed/sciplex_accept/drug_disjoint_v2/gene_ids.txt \
  --out data/processed/tahoe_accept
python -m data.pathway_gsea \
  --protocol-dir data/processed/tahoe_accept \
  --prefix tahoe \
  --gmt data/raw/msigdb/h.all.v2024.1.Hs.symbols.gmt
python -m data.combine_fit_splits \
  --protocol-dir data/processed/tahoe_accept \
  --prefix tahoe \
  --out data/processed/tahoe_accept/fit
```

### Results (regenerate paper figures)

```bash
tar -xzf result.tar.gz -C .
python manuscript/generate_fig3_collapse.py
python manuscript/generate_fig4_control.py
python manuscript/generate_fig5_mechanism.py
```

## Regenerate every table and figure

The repository ships the regeneration inputs under
`manuscript/analysis/regeneration_inputs/` (stored per-pair predictions for
the seven audited configurations, the frozen pooled-vehicle truth, baseline
scored runs, oracle and replicate matrices, training logs, and an
`inputs_manifest.json` with SHA-256 checks). No download is needed:

```bash
python scripts/regenerate_paper_numbers.py \
  --bundle manuscript/analysis/regeneration_inputs \
  --out out
python manuscript/generate_fig3_collapse.py --data out/figures_data
python manuscript/generate_fig4_control.py --data out/figures_data
python manuscript/generate_fig5_mechanism.py --data out/figures_data
```

`out/reconciliation_report.md` reconciles every computed value against the
frozen printed numbers in `manuscript/analysis/expected_values.json`. The only
optional input is the cell-level sci-Plex h5ad, needed for the Table 8
technical split-half ceiling:

```bash
python scripts/regenerate_paper_numbers.py \
  --bundle manuscript/analysis/regeneration_inputs \
  --h5ad SrivatsanTrapnell2020_sciplex3.h5ad --out out
```

The bundle was exported from the frozen data with
`scripts/export_regeneration_inputs.py` (run it on the training host to
rebuild it from scratch; see the docstring there).

## Compute environment

All experiments used a single NVIDIA RTX 5090 32GB via AutoDL. Total GPU usage: ~98 hours.

## Issues

If you encounter download issues, please open a GitHub issue or contact:
[cgx510510@gmail.com](mailto:cgx510510@gmail.com) (G. Chen).
