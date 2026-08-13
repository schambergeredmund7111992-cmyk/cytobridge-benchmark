# T2-2: Lambda Sweep — val/spearman_top50 Grid Search
## lam_delta x lam_contrast (4x4=16 combinations, 1 epoch each)

## Pivot Table
| lam_delta \ lam_contrast | 0.1 | 0.5 | 1.0 | 2.0 |
|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.1450 | 0.1408 | 0.1481 | 0.1693 |
| 0.1 | 0.1326 | 0.1680 | 0.1357 | 0.1811 |
| 0.5 | 0.1476 | 0.1605 | 0.1668 | 0.1070 |
| 1.0 | 0.1310 | **0.1891** | 0.1746 | 0.1359 |

## Best Combination
- **lam_delta=1.0, lam_contrast=0.5 → val_spearman_top50 = 0.1891**

## Row/Column Averages
| lam_delta | Mean Spearman |
|-----------|---------------|
| 0.0 | 0.1508 |
| 0.1 | 0.1544 |
| 0.5 | 0.1455 |
| 1.0 | 0.1577 |

| lam_contrast | Mean Spearman |
|-------------|---------------|
| 0.1 | 0.1391 |
| 0.5 | 0.1646 |
| 1.0 | 0.1578 |
| 2.0 | 0.1483 |

## Observations
- Higher lam_delta (1.0) with moderate lam_contrast (0.5) gives the best single-epoch performance
- lam_contrast=2.0 causes degradation at higher lam_delta (≥0.5): score drops to 0.107
- Best row: lam_delta=1.0 (mean=0.1577)
- Best column: lam_contrast=0.5 (mean=0.1646)
- Delta loss improves over no-delta: best with delta (0.1891) vs best without (ld=0.0, 0.1693) = +0.0198
