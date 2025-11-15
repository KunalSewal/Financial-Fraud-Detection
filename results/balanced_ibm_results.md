# IBM Balanced Dataset Results

## Dataset Information
- **File**: `data/ibm/ibm_fraud_29k_nonfraud_60k.csv`
- **Total Transactions**: 89,757
- **Fraud Transactions**: 29,757 (33.15%)
- **Non-Fraud Transactions**: 60,000 (66.85%)
- **Edges Created**: 355,586 temporal connections

## Dataset Split
- **Train**: 62,829 nodes (70.0%) - 20,830 fraud (33.15%)
- **Validation**: 13,464 nodes (15.0%) - 4,464 fraud (33.16%)
- **Test**: 13,464 nodes (15.0%) - 4,463 fraud (33.15%)

---

## Model 1: GNN Baseline

**Training Configuration:**
- Device: CUDA (GPU)
- Loss Function: CrossEntropyLoss (balanced dataset)
- Epochs: 100
- Training Time: 2.68 seconds

**Test Set Results:**

| Metric | Score |
|--------|-------|
| **Accuracy** | 69.24% |
| **Precision** | 56.09% |
| **Recall** | 33.23% |
| **F1-Score** | 41.73% |
| **ROC-AUC** | 71.13% |

**Training Progress:**
- Epoch 10: Val Acc 66.84%
- Epoch 30: Val Acc 68.11%
- Epoch 50: Val Acc 68.70%
- Epoch 90: Val Acc 69.33% (Best)
- Epoch 100: Val Acc 69.05%

**Analysis:**
- Model shows moderate performance on balanced dataset
- Low recall (33.23%) indicates missing many fraud cases
- Precision (56.09%) shows moderate false positive rate
- ROC-AUC (71.13%) indicates reasonable ranking ability

---

## Model 2: TGN (Temporal Graph Networks)

*Results pending...*

---

## Model 3: Hybrid Model

*Results pending...*

---

## Comparison Summary

*To be updated after all models are trained...*
