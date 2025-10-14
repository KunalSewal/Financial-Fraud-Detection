"""
Example script demonstrating the complete workflow.

This script shows how to:
1. Load and preprocess data
2. Train a model
3. Evaluate results
4. Generate visualizations
"""

import torch
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.data_utils import load_ethereum_data, preprocess_ethereum_data
from src.models import MLPClassifier, GraphSAGE
from src.train import train_model
from src.evaluate import (
    evaluate_model,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report
)


def run_example():
    """Run complete example workflow."""
    
    print("=" * 70)
    print("Financial Fraud Detection - Example Workflow")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 1. Load and preprocess data
    print("\n" + "=" * 70)
    print("Step 1: Loading and Preprocessing Data")
    print("=" * 70)
    
    data_path = "data/transaction_dataset.csv"
    data_dict = preprocess_ethereum_data(
        data_path,
        test_split=0.2,
        val_split=0.1,
        graph_method="knn",
        k_neighbors=10
    )
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Nodes: {data_dict['num_nodes']}")
    print(f"  Edges: {data_dict['num_edges']}")
    print(f"  Features: {data_dict['num_features']}")
    print(f"  Fraud ratio: {data_dict['fraud_ratio']:.2%}")
    
    # 2. Train MLP model
    print("\n" + "=" * 70)
    print("Step 2: Training MLP Baseline")
    print("=" * 70)
    
    mlp_model = MLPClassifier(
        input_dim=data_dict['num_features'],
        hidden_dims=[128, 64],
        num_classes=2,
        dropout=0.3
    ).to(device)
    
    mlp_config = {
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
        'patience': 15,
        'use_graph': False
    }
    
    mlp_history = train_model(
        model=mlp_model,
        data_dict=data_dict,
        config=mlp_config,
        device=device,
        save_path="checkpoints/mlp_example.pt"
    )
    
    # 3. Train GraphSAGE model
    print("\n" + "=" * 70)
    print("Step 3: Training GraphSAGE Model")
    print("=" * 70)
    
    sage_model = GraphSAGE(
        input_dim=data_dict['num_features'],
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    ).to(device)
    
    sage_config = {
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
        'patience': 15,
        'use_graph': True
    }
    
    sage_history = train_model(
        model=sage_model,
        data_dict=data_dict,
        config=sage_config,
        device=device,
        save_path="checkpoints/graphsage_example.pt"
    )
    
    # 4. Compare results
    print("\n" + "=" * 70)
    print("Step 4: Comparing Results")
    print("=" * 70)
    
    print("\nMLP Results:")
    print(f"  Test Accuracy: {mlp_history['test_metrics']['accuracy']:.4f}")
    print(f"  Test F1: {mlp_history['test_metrics']['f1']:.4f}")
    print(f"  Test ROC-AUC: {mlp_history['test_metrics']['roc_auc']:.4f}")
    
    print("\nGraphSAGE Results:")
    print(f"  Test Accuracy: {sage_history['test_metrics']['accuracy']:.4f}")
    print(f"  Test F1: {sage_history['test_metrics']['f1']:.4f}")
    print(f"  Test ROC-AUC: {sage_history['test_metrics']['roc_auc']:.4f}")
    
    # 5. Generate visualizations
    print("\n" + "=" * 70)
    print("Step 5: Generating Visualizations")
    print("=" * 70)
    
    # Get test predictions for GraphSAGE
    sage_model.eval()
    with torch.no_grad():
        x = data_dict["node_features"].to(device)
        edge_index = data_dict["edge_index"].to(device)
        test_mask = data_dict["test_mask"]
        
        out = sage_model(x, edge_index)
        
        y_true = data_dict["labels"][test_mask].cpu().numpy()
        y_pred = out[test_mask].argmax(dim=1).cpu().numpy()
        y_prob = out[test_mask].cpu().numpy()
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_path="results/figures/example_confusion_matrix.png"
    )
    
    # Plot ROC curve
    plot_roc_curve(
        y_true, y_prob,
        save_path="results/figures/example_roc_curve.png"
    )
    
    # Print classification report
    print_classification_report(y_true, y_pred)
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("\nCheckpoints saved to: checkpoints/")
    print("Visualizations saved to: results/figures/")
    print("\nNext steps:")
    print("  1. Review the results")
    print("  2. Experiment with hyperparameters in configs/config.yaml")
    print("  3. Try temporal models: python main.py --model tgn")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_example()
