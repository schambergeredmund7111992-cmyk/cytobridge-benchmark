# Construction note: per-pair vehicle (superseded analysis)

The two files in this directory were produced by
`manuscript/analysis_scripts/compute_robustness.py` and belong to the
**per-pair vehicle** construction, in which a separate vehicle pseudobulk is
estimated for every (drug, cell line) pair. The paper (Section 4.5) shows that
this construction credits common estimation noise to the matched pair, and
reports every paper number under the **pooled vehicle** construction instead
(one vehicle pseudobulk per cell line, shared by all drugs).

- `sensitivity_grid.csv` — the per-pair-vehicle metric x DEG-k grid. Four of
  its 56 cells (all-3000 genes x Spearman: 0.713, 0.741, 0.773, 0.778) exceed
  the 0.70 practical threshold. The paper discusses exactly this exception
  ("Spearman over all 3000 genes rising to 0.64-0.78") and shows it does not
  survive the pooled vehicle; Table 7 of the paper reports the pooled-vehicle
  grid, whose all 56 cells fall in [0.481, 0.574]. Do not compare this CSV
  against Table 7.

- `permutation_null.json` — permutation p-values on the per-pair-vehicle AUC
  (loss-only observed 0.569, p = 0.002; drug-spec x5 observed 0.537,
  p = 0.002). The paper's primary permutation endpoint is the pooled-vehicle
  loss-only AUC 0.509 with p = 0.39 (best configuration 0.542, p = 0.07;
  Section 4.5). Do not compare this file against Section 4.5.
