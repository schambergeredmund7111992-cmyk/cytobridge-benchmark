# Prediction Artifact Contract

Every evaluated run writes one directory containing exactly the following required files:

- `predictions.npz`: `pred` and `true` arrays shaped `[pairs, genes]`, plus ordered
  `gene_ids`.
- `metadata.csv`: one row per array row with unique `pair_id`, `drug_id`, `context_id`,
  `split`, and `dataset` columns.
- `provenance.json`: schema version, dataset/split/model identifiers, seed, command, Git
  commit, and SHA-256 hashes for the split, resolved config, checkpoint, gene panel, and
  source inputs. `gene_panel_hash` identifies the ordered model gene space;
  `response_panel_hash` separately identifies the context-specific training-derived top-500
  scoring panels.

Rows and genes are positional contracts. Consumers must validate shapes, finite values,
unique pair identifiers, a single dataset/split, and file hashes before computing metrics.
Tables and figures may read only validated artifacts. Missing provenance is an error, not a
warning; non-learned controls use an explicit `checkpoint_hash: null`.
