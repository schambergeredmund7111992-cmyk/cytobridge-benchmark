# Construction note: per-pair vehicle, per-anchor bootstrap (superseded analysis)

The files in this directory were produced by
`manuscript/analysis_scripts/compute_analysis2.py` and belong to the
**per-pair vehicle** construction (see `../data3/README.md`). They are not the
numbers reported in the paper.

- `bootstrap_auc.npy` / `bootstrap_meta.json` — a bootstrap that resamples the
  27 (drug, cell line) anchors individually and recomputes the AUC on the
  resampled set. Section 4.5 of the paper instead reports a **drug-clustered**
  bootstrap that resamples the nine held-out compounds with replacement,
  keeping each drug's three cell lines together, giving the 95% CI [0.37, 0.51]
  used in Fig. 4e. The interval recorded here ([0.445, 0.568], mean 0.507)
  belongs to the per-pair-vehicle point estimate (0.569) and appears nowhere
  in the paper.

- `calibration.csv` / `calibration_meta.json` — the injected-signal calibration
  curve computed on the per-pair-vehicle construction, anchored at the
  per-pair norm-only AUC 0.583. Fig. 4a of the paper uses the pooled-vehicle
  curve, where the best configuration (0.54) corresponds to an effective
  drug-signal fraction of about 0.02.
