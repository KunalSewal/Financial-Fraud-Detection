# Baseline Model Results

## Overview

This document contains the results of baseline models for financial fraud detection on the Ethereum dataset.

## Dataset Statistics

- **Total Samples**: 9,841
- **Features**: 50
- **Classes**: 2 (Normal, Fraud)
- **Fraud Ratio**: ~20%
- **Train/Val/Test Split**: 70%/10%/20%

## Models Evaluated

### 1. MLP Classifier
Simple feedforward neural network baseline.

**Architecture**:
- Input: 50 features
- Hidden layers: [256, 128, 64]
- Output: 2 classes
- Dropout: 0.3

**Results**: To be updated after training

| Metric | Value |
|--------|-------|
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1 Score | - |
| ROC-AUC | - |

### 2. GraphSAGE
Graph neural network that aggregates neighborhood information.

**Architecture**:
- Input: 50 features
- Hidden dim: 256
- Layers: 2
- Aggregator: Mean
- Dropout: 0.3

**Results**: To be updated after training

| Metric | Value |
|--------|-------|
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1 Score | - |
| ROC-AUC | - |

## Comparison

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| MLP | - | - | - | - | - | - |
| GraphSAGE | - | - | - | - | - | - |

## Key Findings

- Results pending model training
- Will compare effectiveness of graph structure vs. node features alone
- Expected: GraphSAGE should outperform MLP by leveraging graph structure

## Next Steps

1. Train baseline models
2. Implement temporal models (TGN, TGAT)
3. Compare baseline vs. temporal performance
4. Analyze false positives/negatives
5. Fine-tune hyperparameters

---

**Last Updated**: October 2025