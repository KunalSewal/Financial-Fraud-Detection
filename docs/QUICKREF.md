# 🚀 Phase 1 Quick Reference Card

## ⚡ Quick Commands

### Setup & Installation
```powershell
# Automated setup
python setup_phase1.py

# Manual W&B install
pip install wandb
wandb login

# Run tests
python test_phase1.py
```

### Test Individual Components
```powershell
# Temporal graph builder
python -c "from src.data.temporal_graph_builder import load_and_build_temporal_graph; print('✅ OK')"

# TGN model
python -c "from src.models.tgn import TGN; print('✅ OK')"

# MPTGNN model
python -c "from src.models.mptgnn import MPTGNN; print('✅ OK')"

# DGraph loader
python -c "from src.data.dgraph_loader import load_dgraph; print('✅ OK')"

# Experiment tracker
python -c "from experiments.experiment_runner import ExperimentRunner; print('✅ OK')"
```

---

## 📁 File Placement

### DGraph Data
```
data/dgraph/
├── edges.npy      # Your DGraph edges file
└── nodes.npy      # Your DGraph nodes file
```

### Processed Data (Auto-generated)
```
data/processed/
├── ethereum_processed.pt    # Cached Ethereum temporal graph
└── dgraph_processed.pt      # Cached DGraph temporal graph
```

---

## 🔑 Key Code Snippets

### 1. Build Ethereum Temporal Graph
```python
from src.data.temporal_graph_builder import load_and_build_temporal_graph

data = load_and_build_temporal_graph(
    csv_path='data/transaction_dataset.csv',
    source='ethereum'
)
```

### 2. Load DGraph
```python
from src.data.dgraph_loader import load_dgraph

data = load_dgraph('data/dgraph')
```

### 3. Create TGN Model
```python
from src.models.tgn import TGN

model = TGN(
    node_features=166,
    edge_features=10,
    hidden_dim=128,
    num_layers=2
)
```

### 4. Create MPTGNN Model
```python
from src.models.mptgnn import MPTGNN

model = MPTGNN(
    in_channels=166,
    hidden_channels=128,
    out_channels=2,
    num_layers=3
)
```

### 5. Start Experiment Tracking
```python
from experiments.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    project_name="fraud-detection-phase1"
)

runner.init_run(
    name="my-experiment",
    config={'model': 'TGN', 'lr': 0.001}
)

# Log metrics
runner.log_metrics({'auc': 0.95, 'f1': 0.88}, step=1)

# Save model
runner.log_model(model, "model.pt")
```

### 6. Temporal Batching
```python
from src.data.temporal_graph_builder import TemporalGraphBuilder

builder = TemporalGraphBuilder(data)

# Time-based batches (1-hour windows)
batches = builder.create_temporal_batches(
    batch_type='time',
    batch_size=3600  # seconds
)

# Count-based batches (1000 transactions each)
batches = builder.create_temporal_batches(
    batch_type='count',
    batch_size=1000
)
```

### 7. Graph Snapshots
```python
# Get temporal evolution
snapshots = builder.get_temporal_snapshots(
    num_snapshots=5
)

for i, snapshot in enumerate(snapshots):
    print(f"Snapshot {i}: {snapshot.num_edges} edges")
```

---

## 📊 Model Comparison Template

```python
from src.models.tgn import TGN
from src.models.mptgnn import MPTGNN
from experiments.experiment_runner import ExperimentRunner
import torch

# Initialize experiment
runner = ExperimentRunner(project_name="model-comparison")

# Models to test
models = {
    'TGN': TGN(node_features=166, edge_features=10, hidden_dim=128),
    'MPTGNN': MPTGNN(in_channels=166, hidden_channels=128, out_channels=2)
}

# Run comparison
for name, model in models.items():
    runner.init_run(name=f"{name}-baseline")
    
    # Train and evaluate...
    
    runner.log_metrics({
        'auc': auc_score,
        'f1': f1_score
    })
```

---

## 🐛 Common Issues

### Issue: Import errors
**Solution**: Activate virtual environment first
```powershell
.\venv\Scripts\Activate.ps1
```

### Issue: DGraph loader fails
**Solution**: Ensure files are in `data/dgraph/`
```powershell
# Check files exist
Test-Path data/dgraph/edges.npy
Test-Path data/dgraph/nodes.npy
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use temporal batching
```python
# Smaller batches
batches = builder.create_temporal_batches(
    batch_type='count',
    batch_size=500  # Reduced from 1000
)
```

### Issue: W&B offline
**Solution**: Use offline mode
```python
runner = ExperimentRunner(
    project_name="fraud-detection",
    offline=True  # Saves logs locally
)
```

---

## 📈 Next Steps Checklist

- [ ] Run `python setup_phase1.py`
- [ ] Place DGraph files in `data/dgraph/`
- [ ] Run `python test_phase1.py`
- [ ] Login to W&B: `wandb login`
- [ ] Build Ethereum temporal graph
- [ ] Train first TGN model
- [ ] Train first MPTGNN model
- [ ] Compare with MLP/GraphSAGE baselines
- [ ] Test DGraph integration
- [ ] Read PHASE1_README.md for deep dive

---

## 📚 Documentation Links

- **Detailed Guide**: PHASE1_README.md
- **Industrial README**: README_INDUSTRIAL.md
- **Test Suite**: test_phase1.py
- **Original README**: README.md

---

## 🆘 Need Help?

1. Check PHASE1_README.md troubleshooting section
2. Run `python test_phase1.py` to diagnose issues
3. Review inline docstrings in source files
4. Check experiment logs in W&B dashboard

---

**Status**: Phase 1 ready! 🚀
**Last Updated**: December 2024
