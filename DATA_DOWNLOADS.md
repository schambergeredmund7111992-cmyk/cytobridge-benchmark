# Data and Checkpoint Sources

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21911512.svg)](https://doi.org/10.5281/zenodo.21911512)

This repository contains only source code (~30K lines). Large binary files — datasets,
pretrained checkpoints, cached embeddings, trained model weights, and experiment results —
are distributed via Google Drive and Zenodo.

## Google Drive (recommended for large datasets)

All binary artifacts live in one shared folder:

**https://drive.google.com/drive/folders/1oSdZp8M1Sp39Vph9JLqrXt3PrVQJP8VF**

| File (inside the shared folder) | Contents | Size |
|---|---|---|
| `result.tar.gz` | P1/P2/P3 experiment results (104 jobs, all predictions and metrics) | 432 MB |
| `SrivatsanTrapnell2020_sciplex3.h5ad` | Raw sci-Plex 3 dataset (~500K cells, 3 cell lines, 188 drugs) | 2.4 GB |
| `tahoe_data_merged.tar.gz` | Tahoe-100M selected panel (metadata + expression) | 703 MB |
| `tahoe_from_local.h5ad` | Tahoe preprocessed AnnData (gene-aligned with sci-Plex) | 603 MB |
| `tahoe_expression_data.tar.gz` | Tahoe raw expression matrix | 173 MB |
| `checkpoints.zip` | Trained model checkpoints | 2.43 GB |

## Zenodo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21911512.svg)](https://doi.org/10.5281/zenodo.21911512)

Source code snapshot and small figure-reproduction data are archived at
[10.5281/zenodo.21911512](https://doi.org/10.5281/zenodo.21911512).

## Direct download of public datasets

If you prefer to download raw data directly from public sources:

| Dataset | Source | Command |
|---|---|---|
| **sci-Plex** | GEO GSE139944 / scPerturb | `python -m data.download --target sciplex --out data/raw/` |
| **MSigDB Hallmark** | GSEA/MSigDB v2024.1 | Download `h.all.v2024.1.Hs.symbols.gmt` to `data/raw/msigdb/` |
| **Tahoe-100M** | HuggingFace `tahoebio/Tahoe-100M` | `python -m data.download_tahoe_metadata --out data/raw/tahoe` |

## Setup after download

```bash
# sci-Plex
python -m data.prepare_sciplex_scperturb --h5ad SrivatsanTrapnell2020_sciplex3.h5ad --out data/raw/sciplex/
python -m data.preprocess --config configs/data/sciplex.yaml

# Tahoe
tar -xzf tahoe_expression_data.tar.gz -C data/raw/tahoe/
tar -xzf tahoe_data_merged.tar.gz -C data/processed/
python -m data.preprocess_tahoe --selected-h5ad data/raw/tahoe/selected_panel.h5ad ...

# Results (regenerate paper figures)
tar -xzf result.tar.gz -C .
python manuscript/generate_fig3_collapse.py
python manuscript/generate_fig4_control.py
python manuscript/generate_fig5_mechanism.py
```

## Compute environment

All experiments used a single NVIDIA RTX 5090 32GB via AutoDL. Total GPU usage: ~98 hours.

## Issues

If you encounter download issues, please open a GitHub issue or contact:
cgx510510@gmail.com (G. Chen).
