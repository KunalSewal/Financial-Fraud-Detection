# Fraud Detection Model Evaluation Summary

## Dataset Overview
- Fraud ratio is consistent across splits (~33%)
- Models evaluated:
  - Baseline GNN
  - TGAT
  - TGN
  - Weighted Ensemble (35% TGN + 65% TGAT)
  - Voting Ensemble

## 1. Baseline GNN Results
```
Accuracy:   0.6752
Precision:  0.7099
Recall:     0.0352
F1-Score:   0.0670
ROC-AUC:    0.6910
```

## 2. TGAT Results
```
Accuracy:   0.7168
Precision:  0.6926
Recall:     0.3135
F1-Score:   0.4206
ROC-AUC:    0.6823
```

## 3. TGN Results
```
Accuracy:   0.7164
Precision:  0.7020
Recall:     0.2697
F1-Score:   0.3955
ROC-AUC:    0.6841
```

## 4. Ensemble Methods

### 4.1 Weighted Ensemble (TGN 35% + TGAT 65%)
```
Accuracy:   0.7198
Precision:  0.6944
Recall:     0.2765
F1-Score:   0.3955
ROC-AUC:    0.7478
```

### 4.2 Voting Ensemble
```
Accuracy:   0.7242
Precision:  0.7236
Recall:     0.2716
F1-Score:   0.3949
ROC-AUC:    0.6649
```
