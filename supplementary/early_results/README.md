# Week6 Results — CytoBridge

## Key Result Files

### Agent Evaluation (DeepSeek API)
- `agent_eval.json` — All 4 metrics for IPF + GBM (6 runs, real API)

### Baselines
- `all_baselines_comparison.csv` — Mean/Linear/Linear-adj/kNN/Ridge table
- `ridge_baseline.csv` — Per-drug per-cell-line Ridge predictions
- `cytobridge_internal.csv` — CytoBridge model eval (collapsed)

### Pathway Faithfulness (Strongest Asset)
- `pathway_attribution/summary.json` — r=0.95 per-pair, 27/27 positive
- `pathway_attribution/per_pair.csv` — Per (drug, cell_line) pathway Pearson
- `pathway_attribution/meta.csv` — Sample metadata
- `pathway_attribution_no_pathway_loss/` — Ablation results (Delta=-0.82)

### Drug Ranking
- `ranking_t6.json` — Hit@K, MRR, NDCG, rank Spearman

### Paper & Progress
- `paper_skeleton.md` — KBS paper outline
- `week6_progress.md` — Weekly progress summary
- `week4_phase0_results.md` — Week4 baseline comparison report
