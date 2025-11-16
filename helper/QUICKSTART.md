# Quick Start Guide

## Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Dataset

The Ethereum fraud dataset (`transaction_dataset.csv`) is already in the `data/` folder.

Run the setup script to preprocess the data:

```bash
python setup_data.py
```

This will:
- Load and preprocess the dataset
- Create graph structure
- Split into train/val/test sets
- Save to `data/processed/ethereum_processed.pt`

### 3. Train Models

#### Train MLP Baseline
```bash
python main.py --model mlp
```

#### Train GraphSAGE
```bash
python main.py --model graphsage
```

#### Train Temporal Models
```bash
python main.py --model tgn
python main.py --model tgat
```

### 4. View Results

Results are saved in:
- **Checkpoints**: `checkpoints/` - Trained model weights
- **Results**: `results/` - Markdown reports with metrics
- **Figures**: `results/figures/` - Visualizations (confusion matrix, ROC curve, training curves)

## Project Structure

```
Financial-Fraud-Detection/
├── main.py                     # Main training script
├── setup_data.py              # Data preprocessing script
├── requirements.txt           # Python dependencies
├── configs/
│   └── config.yaml           # Experiment configuration
├── data/
│   ├── transaction_dataset.csv    # Raw data
│   ├── processed/                 # Preprocessed data
│   └── scripts/                   # Download scripts
├── src/
│   ├── data_utils.py         # Data loading & preprocessing
│   ├── models.py             # Model implementations
│   ├── train.py              # Training utilities
│   └── evaluate.py           # Evaluation & visualization
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_temporal_models.ipynb
└── results/                   # Training results
    ├── baseline_results.md
    └── figures/
```

## Available Models

### Baseline Models
1. **MLP** - Multi-layer perceptron (node features only)
2. **GraphSAGE** - Graph neural network (uses graph structure)

### Temporal Models
3. **TemporalGNN** - Simple temporal GNN with LSTM
4. **TGN** - Temporal Graph Network with memory
5. **TGAT** - Temporal Graph Attention Network

## Configuration

Edit `configs/config.yaml` to customize:
- Model architectures
- Training hyperparameters
- Data preprocessing options
- Evaluation metrics

## Common Tasks

### Change Model Hyperparameters

Edit `configs/config.yaml`:

```yaml
models:
  graphsage:
    hidden_dim: 256  # Change this
    num_layers: 3    # Change this
    dropout: 0.3     # Change this
```

### Adjust Training Settings

```yaml
training:
  num_epochs: 200
  learning_rate: 0.001
  patience: 30  # Early stopping patience
```

### Use Different Graph Construction

```yaml
graph:
  method: "knn"
  k_neighbors: 15  # Increase neighbors
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `hidden_dim` in config
- Reduce `batch_size`
- Use CPU: Set `device: "cpu"` in config

### Module Not Found
- Ensure virtual environment is activated
- Run: `pip install -r requirements.txt`

### Dataset Not Found
- Verify `data/transaction_dataset.csv` exists
- Run `python setup_data.py` first

## Next Steps

1. **Explore Data**: Open `notebooks/01_data_exploration.ipynb`
2. **Train Baselines**: Run MLP and GraphSAGE
3. **Compare Results**: Check `results/` folder
4. **Experiment**: Modify configs and retrain
5. **Implement Temporal**: Train TGN and TGAT models

## Support

For issues or questions:
- Check main README.md
- Review code comments in `src/`
- Contact team members

---

**Good luck with your experiments!** 🚀
