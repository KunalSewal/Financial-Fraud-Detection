"""
Main training script for financial fraud detection models.

Usage:
    python main.py --model mlp --config configs/config.yaml
    python main.py --model graphsage --config configs/config.yaml
    python main.py --model tgn --config configs/config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from src.data_utils import load_processed_data, preprocess_ethereum_data
from src.models import get_model
from src.train import train_model
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    print_classification_report,
    save_results_to_markdown
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict, device: torch.device):
    """Prepare data for training."""
    dataset_config = config['dataset']
    
    # Check if processed data exists
    processed_path = dataset_config.get('processed_path')
    
    if processed_path and os.path.exists(processed_path):
        print(f"Loading processed data from {processed_path}...")
        data_dict = load_processed_data(processed_path)
    else:
        print("Processing raw data...")
        data_dict = preprocess_ethereum_data(
            dataset_config['data_path'],
            test_split=dataset_config['test_split'],
            val_split=dataset_config['val_split'],
            graph_method=config['graph']['method'],
            k_neighbors=config['graph']['k_neighbors']
        )
        
        # Save processed data
        if processed_path:
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            torch.save(data_dict, processed_path)
            print(f"✓ Saved processed data to {processed_path}")
    
    return data_dict


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['hardware']['device'])
    
    print("\n" + "=" * 70)
    print("Financial Fraud Detection - Model Training")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print("=" * 70 + "\n")
    
    # Prepare data
    data_dict = prepare_data(config, device)
    
    # Update model config with actual input dimension
    model_config = config['models'][args.model].copy()
    model_config['input_dim'] = data_dict['num_features']
    
    # Initialize model
    print(f"\nInitializing {args.model.upper()} model...")
    model = get_model(args.model, model_config)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training configuration
    train_config = config['training'].copy()
    train_config['use_graph'] = model_config.get('use_graph', True)
    
    # Setup checkpoint path
    checkpoint_dir = config['paths']['checkpoints']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_best.pt")
    
    # Train model
    history = train_model(
        model=model,
        data_dict=data_dict,
        config=train_config,
        device=device,
        save_path=checkpoint_path
    )
    
    # Evaluate on test set
    print("\nFinal Evaluation on Test Set...")
    test_loss, test_metrics = evaluate_model(
        model, data_dict, device, split="test",
        use_graph=train_config['use_graph']
    )
    
    # Get predictions for visualization
    model.eval()
    with torch.no_grad():
        x = data_dict["node_features"].to(device)
        test_mask = data_dict["test_mask"]
        
        if train_config['use_graph']:
            edge_index = data_dict["edge_index"].to(device)
            out = model(x, edge_index)
        else:
            out = model(x)
        
        y_true = data_dict["labels"][test_mask].cpu().numpy()
        y_pred = out[test_mask].argmax(dim=1).cpu().numpy()
        y_prob = out[test_mask].cpu().numpy()
    
    # Print classification report
    print_classification_report(y_true, y_pred)
    
    # Create results directory
    results_dir = config['paths']['results']
    figures_dir = config['paths']['figures']
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save visualizations
    if config['evaluation']['save_figures']:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            save_path=os.path.join(figures_dir, f"{args.model}_confusion_matrix.png")
        )
        
        # ROC curve
        plot_roc_curve(
            y_true, y_prob,
            save_path=os.path.join(figures_dir, f"{args.model}_roc_curve.png")
        )
        
        # Training curves
        plot_training_curves(
            history,
            save_path=os.path.join(figures_dir, f"{args.model}_training_curves.png")
        )
    
    # Save results to markdown
    results_path = os.path.join(results_dir, f"{args.model}_results.md")
    save_results_to_markdown(
        history,
        results_path,
        args.model.upper(),
        model_config
    )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Results saved to: {results_path}")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train fraud detection models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mlp", "graphsage", "temporal_gnn", "tgn", "tgat"],
        help="Model to train"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(args)
