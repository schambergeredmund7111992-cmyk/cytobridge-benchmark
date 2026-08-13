# CytoBridge

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21911960.svg)](https://doi.org/10.5281/zenodo.21911960)

This codebase is a validation of the CytoBridge study. It tests whether
single-cell perturbation predictors preserve held-out compound identity after conditioning on
cellular context. No historical result is accepted automatically: every paper number comes
from a frozen split and a validated prediction artifact.

## What is frozen

- sci-Plex at 10 uM, 24 h, A549/K562/MCF7, with at least 50 cells per drug/context.
- Canonical-SMILES drug split and Bemis-Murcko scaffold stress split, both 70/15/15 with seed
  `20260710`.
- Disjoint input-reference and truth-reference vehicle pools within context/batch.
- Training-only HVGs, response panels, normalization scales, and validation-only selection.
- Drug-macro conditional centroid accuracy as primary; pair-own Spearman@50 as secondary.
- Five final seeds: `11, 23, 42, 67, 101`; 10,000 drug-cluster bootstrap draws.

## Repository map

- `cytobridge/`: model, losses, data loaders, and gradient audit.
- `data/`: sci-Plex/Tahoe metadata selection, split freezing, bounded streaming, and targets.
- `eval/`: metrics, statistics, baselines, artifact packaging, scoring, and multi-seed aggregation.
- `external/`: pinned official chemCPA/biolord source contract.
- `configs/`: Hydra data and training configurations.
- `release/`: result gate and gradient-audit collector.
- `tests/`: deterministic contracts for leakage, metrics, artifacts, controls, and selection.

## Environment and local gate

```bash
conda env create -f env/environment.yml
conda activate cytobridge
pip install -e '.[dev]'
bash scripts/run_student_smoke.sh
```

The acceptance gate is `pytest -q` followed by `ruff check .`. RDKit-dependent tests skip only
when RDKit is absent from the active shell.

## Build frozen sci-Plex inputs

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

Preprocessing is staged and published atomically. Existing outputs are never overwritten.

## Build frozen scGPT caches

The wrapper correction and official-oracle approval are already included. Choose an idle GPU and
run the portable builder in the background, passing the actual server paths:

```bash
nohup bash scripts/build_sciplex_scgpt_caches.sh \
  --gpu 0 \
  --ckpt /path/to/scgpt_ckpt \
  --scgpt-python /path/to/scgpt-env/bin/python \
  --core-python /path/to/core-env/bin/python \
  > ../scgpt_cache_build.log 2>&1 & disown
```

The script builds both protocol caches and validates checkpoint coverage, file/row hashes,
finite unit-normalized `[N,1,512]` output, and manifest row alignment. It refuses to overwrite
existing cache artifacts.

## Build the bounded Tahoe panel

```bash
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

This downloads metadata and streams selected expression rows; it does not materialize the full
Tahoe expression corpus. On cityu, `bash scripts/prepare_tahoe_cpu.sh` runs this immutable CPU
chain. After a fresh GPU estimate is explicitly approved, build the Tahoe scGPT/MolFormer caches
with `scripts/build_tahoe_encoder_caches.sh`. The obsolete `external_1/external_2` Tahoe scripts
are fail-closed because they do not implement the current protocol.

## Train, package, and score

Screen only on validation using `protocols/materialized_sweeps.json`. Run long jobs through
the repository's remote experiment protocol; `scripts/launch_train.sh` is a foreground command
for an already detached experiment, not a background launcher.

```bash
bash scripts/launch_train.sh accept_base seed=42
python -m eval.predict_cytobridge --help
python -m eval.package_artifact --help
bash scripts/launch_eval.sh ARTIFACT_DIR GENE_PANELS_JSON SCORED_DIR
python -m eval.aggregate_benchmark --help
```

After selection, create immutable final-fit inputs with
`python -m data.combine_fit_splits`, then use `accept_final_refit` with the selected epoch count.
That mode disables validation, early stopping, and best-checkpoint selection.

The formal path does not accept hand-entered selected values. `scripts/freeze_p1_gate.py`
collects all 25 P1 screens, replays the lexicographic epoch rule per model family through
`scripts/freeze_cytobridge_selection.py` and `scripts/freeze_external_selection.py`, and writes
the frozen selections to `experiments/selections/` plus the unlock gate
`experiments/gates/p1_selection_frozen.json`. The primary selection actually produced on the
AutoDL host is versioned here as `protocols/selections/cytobridge.json`
(CytoBridge `cytobridge_09`: `loss.lam_recon=0.1`, `loss.lam_drugspec=5.0`, selected epoch 1)
together with its ranked trials table. Final sci-Plex and Tahoe refits run
through `scripts/run_frozen_cytobridge.py`, which also consumes the frozen numerical-precision
and DataLoader reports (`supplementary/gates/`). Tahoe therefore reuses the primary
hyperparameters and epoch without retuning.

`scripts/campaign.py` materializes the 104 registered jobs. Each command is implemented by
`scripts/run_campaign_job.py`; learned jobs additionally require a confirmed fresh execution-host
preflight and exactly one exposed GPU. On RTX 5090, use `scripts/launch_autodl_campaign_phase.sh`
to run a phase through `scripts/campaign_scheduler.py` in a
detached monitored experiment. The scheduler uses all confirmed idle GPUs, never retries failed
jobs, and leaves immutable failure evidence. `scripts/freeze_p1_gate.py` unlocks P2/P3 only after
all 25 P1 screens complete, and `scripts/finalize_campaign.py` is the sole P4 paper-input path.

Reference controls come from `eval/reference_controls.py`; Ridge selects alpha on validation and
refits on train+validation. Official chemCPA/biolord outputs must pass the adapter contract in
`external/README.md`.

## Repository layout note

This repository is the `code/` tree of the executed acceptance workspace, which kept `protocols/`
and `experiments/` beside `code/`. The gate scripts write `experiments/selections/`,
`experiments/gates/`, and `experiments/campaign_manifest.json` at `../experiments` relative to
this checkout; rerun them inside the same workspace layout. The evidence actually produced on
the AutoDL host is versioned here under `protocols/selections/`, `supplementary/gates/`, and
`supplementary/campaign_manifest.json`; `handoff/STATUS.json` records the SHA-256 provenance
table of the executed profiles and frozen inputs.

## Paper and release gate

`release/result_gate.py` unlocks manuscript inputs only after both sci-Plex splits, Tahoe,
five-seed ablations, calibrated controls, paired comparisons, and gradient audits are complete.
The historical manuscript and predictions are not valid inputs to this gate.
