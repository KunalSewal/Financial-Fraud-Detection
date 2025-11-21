# 🚀 Training Guide - Phase 1

## Quick Start

### 1. Train TGN on Ethereum

```powershell
# Basic training
python train_tgn_ethereum.py

# With custom hyperparameters
python train_tgn_ethereum.py --memory-dim 256 --lr 0.0005 --epochs 300

# With W&B tracking (online)
python train_tgn_ethereum.py --wandb-entity your-username

# With W&B offline mode
python train_tgn_ethereum.py --wandb-offline

# Without W&B
python train_tgn_ethereum.py --no-wandb
```

### 2. Train MPTGNN on Ethereum

```powershell
# Basic training
python train_mptgnn_ethereum.py

# With custom hyperparameters
python train_mptgnn_ethereum.py --hidden-dim 256 --num-layers 4

# With W&B tracking
python train_mptgnn_ethereum.py --wandb-entity your-username
```

---

## Training Hyperparameters

### TGN (`train_tgn_ethereum.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--memory-dim` | 128 | Memory module dimension |
| `--time-dim` | 32 | Time encoding dimension |
| `--embedding-dim` | 128 | Final embedding dimension |
| `--dropout` | 0.1 | Dropout probability |
| `--lr` | 0.001 | Learning rate |
| `--weight-decay` | 1e-5 | L2 regularization |
| `--epochs` | 200 | Maximum epochs |
| `--patience` | 30 | Early stopping patience |

### MPTGNN (`train_mptgnn_ethereum.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hidden-dim` | 128 | Hidden layer dimension |
| `--num-layers` | 3 | Number of MP layers |
| `--dropout` | 0.1 | Dropout probability |
| `--lr` | 0.001 | Learning rate |
| `--weight-decay` | 1e-5 | L2 regularization |
| `--epochs` | 200 | Maximum epochs |
| `--patience` | 30 | Early stopping patience |

---

## Expected Results

### Target Performance

**Baselines to Beat:**
- MLP: 93.99% ROC-AUC
- GraphSAGE: 91.31% ROC-AUC

**Expected TGN/MPTGNN:**
- ROC-AUC: **> 94%** (with temporal edges)
- F1-Score: **> 88%**
- Training time: ~5-10 minutes on GPU

---

## Experiment Tracking

### Using Weights & Biases

1. **Setup W&B:**
   ```powershell
   wandb login  # Enter your API key from wandb.ai
   ```

2. **Train with W&B:**
   ```powershell
   # Online mode (syncs to cloud)
   python train_tgn_ethereum.py --wandb-entity your-username
   
   # Offline mode (saves locally)
   python train_tgn_ethereum.py --wandb-offline
   ```

3. **View Results:**
   - Online: Visit https://wandb.ai/your-username/fraud-detection-phase1
   - Offline: Logs saved to `wandb/` directory

### Without W&B

```powershell
python train_tgn_ethereum.py --no-wandb
```

Results will be printed to console. Best model saved to `checkpoints/`.

---

## Output Files

### Checkpoints

Models are automatically saved to `checkpoints/`:
- `checkpoints/tgn_ethereum_best.pt` - Best TGN model
- `checkpoints/mptgnn_ethereum_best.pt` - Best MPTGNN model

### Checkpoint Structure

```python
{
    'epoch': int,                    # Best epoch number
    'model_state_dict': dict,        # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'val_auc': float,                # Best validation AUC
    'config': dict                   # Training configuration
}
```

### Loading a Checkpoint

```python
import torch
from src.models.tgn import TGN

# Load checkpoint
checkpoint = torch.load('checkpoints/tgn_ethereum_best.pt')

# Recreate model
model = TGN(
    num_nodes=checkpoint['config']['num_nodes'],
    node_dim=166,
    edge_dim=10,
    memory_dim=checkpoint['config']['memory_dim'],
    time_dim=checkpoint['config']['time_dim'],
    embedding_dim=checkpoint['config']['embedding_dim'],
    num_classes=2
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from epoch {checkpoint['epoch']} with val AUC: {checkpoint['val_auc']:.4f}")
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1: Reduce model size**
```powershell
python train_tgn_ethereum.py --memory-dim 64 --embedding-dim 64
```

**Solution 2: Use CPU**
```powershell
python train_tgn_ethereum.py --cpu
```

### Issue: Training too slow

**Check:**
1. GPU is being used: Look for "Using device: cuda" in output
2. PyG extension libraries installed (warnings are OK, but affects speed)

**Speed up:**
```powershell
# Reduce logging frequency
python train_tgn_ethereum.py --log-interval 20
```

### Issue: Low accuracy

**Try:**
1. Increase model capacity:
   ```powershell
   python train_tgn_ethereum.py --memory-dim 256 --embedding-dim 256
   ```

2. Lower learning rate:
   ```powershell
   python train_tgn_ethereum.py --lr 0.0005
   ```

3. More epochs:
   ```powershell
   python train_tgn_ethereum.py --epochs 300 --patience 50
   ```

### Issue: Overfitting

**Try:**
1. Increase dropout:
   ```powershell
   python train_tgn_ethereum.py --dropout 0.3
   ```

2. Increase weight decay:
   ```powershell
   python train_tgn_ethereum.py --weight-decay 1e-4
   ```

---

## Comparing Models

### Run Both Models

```powershell
# Train TGN
python train_tgn_ethereum.py --run-name tgn-baseline

# Train MPTGNN
python train_mptgnn_ethereum.py --run-name mptgnn-baseline
```

### Compare Results

**In W&B:**
- View both runs in same project
- Compare metrics side-by-side
- Analyze path attention weights (MPTGNN only)

**From Checkpoints:**
```python
import torch

tgn_ckpt = torch.load('checkpoints/tgn_ethereum_best.pt')
mptgnn_ckpt = torch.load('checkpoints/mptgnn_ethereum_best.pt')

print(f"TGN Val AUC: {tgn_ckpt['val_auc']:.4f} (epoch {tgn_ckpt['epoch']})")
print(f"MPTGNN Val AUC: {mptgnn_ckpt['val_auc']:.4f} (epoch {mptgnn_ckpt['epoch']})")
```

---

## Advanced Usage

### Hyperparameter Sweep (Manual)

```powershell
# Try different memory dimensions
python train_tgn_ethereum.py --memory-dim 64 --run-name mem-64
python train_tgn_ethereum.py --memory-dim 128 --run-name mem-128
python train_tgn_ethereum.py --memory-dim 256 --run-name mem-256

# Try different learning rates
python train_tgn_ethereum.py --lr 0.0001 --run-name lr-0001
python train_tgn_ethereum.py --lr 0.001 --run-name lr-001
python train_tgn_ethereum.py --lr 0.01 --run-name lr-01
```

### Multiple Seeds

```powershell
# Run with different random seeds for statistical significance
python train_tgn_ethereum.py --seed 42 --run-name seed-42
python train_tgn_ethereum.py --seed 123 --run-name seed-123
python train_tgn_ethereum.py --seed 456 --run-name seed-456
```

---

## Next Steps

After training on Ethereum:

1. **Analyze Results:**
   - Compare TGN vs MPTGNN
   - Check if you beat baselines (MLP: 93.99%, GraphSAGE: 91.31%)
   - Review W&B dashboards

2. **Scale to DGraph:**
   - 3.7M nodes (376x larger!)
   - Will need temporal batching
   - Train script coming in Phase 2

3. **Write Paper/Report:**
   - Document your approach
   - Compare temporal vs static graphs
   - Analyze path attention weights (MPTGNN)

---

## Questions?

- Check `PHASE1_README.md` for detailed documentation
- Check `QUICKREF.md` for quick commands
- Check `PROJECT_STATUS.md` for current progress

**Good luck! 🚀**
