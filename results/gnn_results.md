# Baseline GNN Results

**Date:** November 16, 2025

## Dataset Split
- Train: 62,829 nodes (70.0%)
- Val:   13,464 nodes (15.0%)
- Test:  13,464 nodes (15.0%)

## Fraud Distribution
- Train: 20,830 fraud / 62,829 total (33.15%)
- Val:   4,464 fraud / 13,464 total (33.16%)
- Test:  4,463 fraud / 13,464 total (33.15%)

## Training Details
- Device: cuda
- Loss: CrossEntropyLoss (balanced dataset, 10:1 ratio)
- Epochs: 100

## Test Set Evaluation Results
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.6942  |
| Precision  | 0.6446  |
| Recall     | 0.1728  |
| F1-Score   | 0.2725  |
| ROC-AUC    | 0.7115  |

Baseline GNN training complete! Use these results to compare with HMSTA model performance.
