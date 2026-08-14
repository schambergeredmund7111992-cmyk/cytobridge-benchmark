# Construction note: per-pair vehicle artifacts

Some files in this directory predate the pooled-vehicle construction used for
every number in the paper (one vehicle pseudobulk per cell line, shared by all
drugs; Section 4.5). See `../data3/README.md` for the construction background.

- `control_validation.csv` — calibration ladder on the per-pair-vehicle
  construction. Its "CytoBridge (best)" row (0.583) is the per-pair norm-only
  AUC; Fig. 4b of the paper reports the pooled-vehicle ladder, where the best
  CytoBridge configuration scores 0.54. The `chemCPA (collapsed)` and
  `biolord (collapsed)` rows referenced by
  `manuscript/analysis_scripts/build_nb2.py` are not present in this CSV.

- `config_metrics.csv` — per-configuration AUC on the per-pair-vehicle
  construction (loss-only 0.569, norm-only 0.583, ...). Table 5 of the paper
  reports the pooled-vehicle values (loss-only 0.509, norm-only 0.532, ...).
  Do not compare this CSV against Table 5.

- `onoff_*.csv` — per-anchor on- and off-diagonal similarities used for the
  gap statistics under the per-pair construction; the `gap` column of Table 5
  is the pooled-vehicle version.

- `logfc_pred_*.npy` / `logfc_true_*.npy` / `logfc_meta_*.csv` — the stored
  prediction and truth matrices (inputs to both constructions).

- Other files (inter-drug correlation matrices, case-study pairs, training
  loss components, the Tahoe control panel) do not depend on the vehicle
  construction.
