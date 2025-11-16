"""
Financial Fraud Detection using Temporal Graph Neural Networks
"""

__version__ = "0.1.0"

from .data_utils import (
    load_ethereum_data,
    preprocess_ethereum_data,
    create_graph_from_transactions,
    EthereumFraudDataset
)

# Legacy models from legacy_models.py
from .legacy_models import (
    MLPClassifier,
    GraphSAGE,
    TemporalGNN,
    TGAT
)

# New Phase 1 models
from .models.tgn import TGN
from .models.mptgnn import MPTGNN

from .train import (
    train_model,
    train_epoch,
    evaluate_model
)

from .evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves
)

__all__ = [
    "load_ethereum_data",
    "preprocess_ethereum_data",
    "create_graph_from_transactions",
    "EthereumFraudDataset",
    "MLPClassifier",
    "GraphSAGE",
    "TemporalGNN",
    "TGN",
    "MPTGNN",
    "TGAT",
    "train_model",
    "train_epoch",
    "evaluate_model",
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_curves"
]
