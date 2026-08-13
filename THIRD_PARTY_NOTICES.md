# Third-Party Notices

CytoBridge does not redistribute external datasets, pretrained checkpoints, or
commercial database dumps. Users are responsible for obtaining each resource
under its original license, citation policy, and terms of use.

## Models and Software

- PyTorch, PyTorch Lightning, Transformers, scikit-learn, Hydra, Scanpy,
  AnnData, gseapy, RDKit, Biopython, matplotlib, seaborn, and OpenAI's Python
  SDK are used through package dependencies.
- scGPT and MolFormer-compatible embeddings must be generated from separately
  obtained model weights or cached features.
- GEARS integration is optional. If the `gears` package is unavailable, the
  baseline runner records a skipped baseline instead of fabricating results.
- Open-source/OpenAI-compatible LLM providers supported by this code are Kimi,
  DeepSeek, and local Qwen/vLLM endpoints.

## Data Resources

- sci-Plex, Tahoe-100M, Replogle Perturb-seq, GDSC2, LINCS L1000, MSigDB, and
  DrugBank are referenced as external resources. Follow each provider's access
  rules and redistribution limits.
- MSigDB gene sets and DrugBank records often require registration or usage
  agreements. Keep downloaded copies outside version control.
- LINCS, Tahoe, Replogle, sci-Plex, and GDSC2 data should be cited in the
  manuscript according to their official dataset documentation.

## Reproducibility Boundary

Generated artifacts such as `.h5ad`, `.npy`, `.npz`, `.parquet`, checkpoints,
logs, and figures are intentionally ignored by `.gitignore`. Recreate them
with the documented preprocessing, training, evaluation, and figure scripts.
