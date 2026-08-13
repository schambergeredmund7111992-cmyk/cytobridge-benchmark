# Official External Baselines

chemCPA and biolord must be run from the exact official commits in
`source_manifest.json`. Do not copy their model implementations into this repository.
chemCPA is trained from scratch and must be reported as `chemCPA (from scratch)`; loading any
pretrained or previously sci-Plex-fitted weight is a protocol failure.
Build their shared, isolated runtime with `bash env/setup_external_baselines.sh`. The script
checks out the three frozen source commits, verifies clean trees, records the resolved package
set, and never downloads model weights.

On an RTX 5090, use `bash env/setup_autodl_5090_external.sh` instead. It preserves the exact
source commits and package-level model contracts but overlays `torch 2.7.1+cu128`, the minimum
registered runtime in this repository with Blackwell `sm_120` kernels. It also pins the 2024 JAX
stack used by scvi-tools 1.1.6; unconstrained current JAX releases remove an import required by
that frozen scvi version. The generated environment and checkout reports are required provenance;
do not use the CUDA 12.1 environment on a 5090.

1. Export `drug_disjoint_v2` with `data/export_external_benchmark.py`. Add
   `--with-rdkit2d` for biolord. The exporter maps train/validation/test to the official
   train/test/ood labels and excludes every truth-reference vehicle cell.
2. Select one of the six preregistered configurations on official `test` only, then refit
   the selected configuration on benchmark train+validation and predict official `ood`.
3. For every test treated row, use the exact `input_control_row_id` stored in the export.
   Save `pred_log1p`, `row_ids`, and `gene_ids` in an NPZ; predictions must remain in the
   exported `log1p(raw counts)` space.
4. Run `eval/import_external_predictions.py`. It clips negative predicted log-expression
   to zero, applies `expm1`, re-aggregates through the shared pseudobulk code, and verifies
   that reconstructed truth exactly matches the frozen targets.
5. Package and score only through `eval/package_artifact.py` and
   `eval/run_benchmark.py`. Include the official checkout commit and training log as source
   hashes.

An output that bypasses any of these gates is descriptive only and cannot enter the paper.
