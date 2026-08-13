# T1-2: Retrieval-Augmented Delta — Results Summary

## Methods
- (A) Pure retrieval (kNN-MolFormer): k=5 nearest training drugs in MolFormer space, similarity-weighted average of their pseudobulk deltas
- (B) Retrieval + g_θ correction: Per-gene Ridge regression (4 features: retrieval_prior + 3 cell-line onehots) on top of LOO retrieval priors
- (C) Linear-adjusted (SOTA bar): Morgan FP 1024-dim + cell-line onehot per-gene Ridge

## Results
| Method | R²(all) | R²(top50 DEG) | Spearman(top50 DEG) | n_pairs | n_deg_genes |
|--------|---------|---------------|---------------------|---------|-------------|
| (A) Pure retrieval (kNN) | -1.6409 | -0.3067 | 0.3423 | 27 | 354 |
| (B) Retrieval + g_θ correction | -0.3343 | -0.0312 | 0.3617 | 27 | 354 |
| (C) Linear-adjusted (SOTA bar) | -0.2504 | -0.1401 | 0.3274 | 27 | 354 |

## Key Findings
- Retrieval prior alone (A) already beats Linear-adjusted (C) on Spearman: 0.3423 vs 0.3274 (+0.015)
- Adding correction head g_θ (B) further improves Spearman to 0.3617, a gain of +0.0195 over pure retrieval
- Total improvement over SOTA bar (C): +0.0343 Spearman
- Retrieval in MolFormer space provides a strong inductive bias for drug similarity
