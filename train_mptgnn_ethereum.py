"""
Train MPTGNN on Ethereum Dataset - Phase 1 Validation.

This script trains the Multi-Path Temporal GNN on the Ethereum fraud detection
dataset. Compares multi-path temporal processing with TGN approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import argparse
import os

from src.data.temporal_graph_builder import load_and_build_temporal_graph
from src.models.mptgnn import MPTGNN
from experiments.experiment_runner import ExperimentRunner


def compute_metrics(y_true, y_pred, y_prob):
    """Compute evaluation metrics."""
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    return {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_epoch(model, data, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    
    # Get data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
    timestamps = data.timestamps.to(device) if hasattr(data, 'timestamps') else None
    labels = data.y.to(device)
    train_mask = data.train_mask.to(device)
    
    # Forward pass
    logits = model(x, edge_index, edge_attr, timestamps)
    
    # Compute loss on training nodes
    loss = criterion(logits[train_mask], labels[train_mask])
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Compute training metrics
    with torch.no_grad():
        pred = logits[train_mask].argmax(dim=1)
        prob = torch.softmax(logits[train_mask], dim=1)[:, 1]
        
        y_true = labels[train_mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = prob.cpu().numpy()
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics['loss'] = loss.item()
    
    return metrics


@torch.no_grad()
def evaluate(model, data, criterion, device, split='val'):
    """Evaluate on validation or test set."""
    model.eval()
    
    # Get data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
    timestamps = data.timestamps.to(device) if hasattr(data, 'timestamps') else None
    labels = data.y.to(device)
    
    # Get mask
    if split == 'val':
        mask = data.val_mask.to(device)
    else:
        mask = data.test_mask.to(device)
    
    # Forward pass
    logits = model(x, edge_index, edge_attr, timestamps)
    
    # Compute loss
    loss = criterion(logits[mask], labels[mask])
    
    # Compute metrics
    pred = logits[mask].argmax(dim=1)
    prob = torch.softmax(logits[mask], dim=1)[:, 1]
    
    y_true = labels[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = prob.cpu().numpy()
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics['loss'] = loss.item()
    
    return metrics


@torch.no_grad()
def analyze_path_weights(model, data, device):
    """Analyze attention weights for different temporal paths."""
    model.eval()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    # Get path weights
    path_weights = model.get_path_weights(edge_index)
    
    # Compute statistics
    stats = {}
    for path_name, weights in path_weights.items():
        stats[f'{path_name}_mean'] = weights.mean().item()
        stats[f'{path_name}_std'] = weights.std().item()
    
    return stats


def train(args):
    """Main training loop."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading Ethereum Temporal Graph")
    print("="*60)
    
    data = load_and_build_temporal_graph(
        csv_path='data/transaction_dataset.csv',
        source='ethereum'
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Temporal edges: {data.num_edges:,}")
    print(f"  Node features: {data.x.shape[1]}")
    print(f"  Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    print(f"  Fraud ratio: {(data.y == 1).sum() / len(data.y) * 100:.2f}%")
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing MPTGNN Model")
    print("="*60)
    
    in_channels = data.x.shape[1]
    
    model = MPTGNN(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=2,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # Handle class imbalance
    fraud_ratio = (data.y == 1).sum().item() / len(data.y)
    class_weights = torch.tensor([1.0, 1.0 / fraud_ratio]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Initialize experiment tracking
    if not args.no_wandb:
        runner = ExperimentRunner(
            project_name="fraud-detection-phase1",
            entity=args.wandb_entity,
            offline=args.wandb_offline
        )
        
        runner.init_run(
            name=f"mptgnn-ethereum-{args.run_name}",
            config={
                'model': 'MPTGNN',
                'dataset': 'Ethereum',
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'dropout': args.dropout,
                'epochs': args.epochs,
                'seed': args.seed
            }
        )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"\nTarget: Beat MLP (93.99%) and GraphSAGE (91.31%) ROC-AUC\n")
    
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, data, optimizer, criterion, device)
        
        # Evaluate
        val_metrics = evaluate(model, data, criterion, device, split='val')
        
        # Analyze path weights
        if epoch % 20 == 0:
            path_stats = analyze_path_weights(model, data, device)
        else:
            path_stats = {}
        
        # Update learning rate
        scheduler.step(val_metrics['auc'])
        
        # Log metrics
        if not args.no_wandb:
            log_dict = {
                'train/loss': train_metrics['loss'],
                'train/auc': train_metrics['auc'],
                'train/f1': train_metrics['f1'],
                'val/loss': val_metrics['loss'],
                'val/auc': val_metrics['auc'],
                'val/f1': val_metrics['f1'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            log_dict.update(path_stats)
            runner.log_metrics(log_dict, step=epoch)
        
        # Print progress
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['auc']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc']:.4f} F1: {val_metrics['f1']:.4f}")
            
            if path_stats:
                print(f"  Path weights: Short={path_stats.get('short_mean', 0):.3f} "
                      f"Medium={path_stats.get('medium_mean', 0):.3f} "
                      f"Long={path_stats.get('long_mean', 0):.3f}")
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/mptgnn_ethereum_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': vars(args)
            }, checkpoint_path)
            
            if not args.no_wandb:
                runner.log_model(checkpoint_path, f"mptgnn_ethereum_epoch_{epoch}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, data, criterion, device, split='test')
    final_path_stats = analyze_path_weights(model, data, device)
    
    print(f"\n📊 Test Results:")
    print(f"  ROC-AUC:   {test_metrics['auc']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    
    print(f"\n🔍 Path Importance:")
    print(f"  Short-term:  {final_path_stats.get('short_mean', 0):.4f} ± {final_path_stats.get('short_std', 0):.4f}")
    print(f"  Medium-term: {final_path_stats.get('medium_mean', 0):.4f} ± {final_path_stats.get('medium_std', 0):.4f}")
    print(f"  Long-term:   {final_path_stats.get('long_mean', 0):.4f} ± {final_path_stats.get('long_std', 0):.4f}")
    
    print(f"\n🎯 Baselines to Beat:")
    print(f"  MLP:       93.99% ROC-AUC")
    print(f"  GraphSAGE: 91.31% ROC-AUC")
    
    if test_metrics['auc'] > 0.9399:
        print(f"\n🎉 SUCCESS! Beat MLP baseline!")
    if test_metrics['auc'] > 0.9131:
        print(f"🎉 SUCCESS! Beat GraphSAGE baseline!")
    
    # Log final results
    if not args.no_wandb:
        final_results = {
            'test/auc': test_metrics['auc'],
            'test/f1': test_metrics['f1'],
            'test/precision': test_metrics['precision'],
            'test/recall': test_metrics['recall'],
            'best_epoch': best_epoch,
            'best_val_auc': best_val_auc
        }
        final_results.update(final_path_stats)
        runner.log_final_results(final_results)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MPTGNN on Ethereum')
    
    # Model hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of MP layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
    # Experiment tracking
    parser.add_argument('--run-name', type=str, default='baseline', help='Run name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--wandb-offline', action='store_true', help='Use W&B offline mode')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    
    args = parser.parse_args()
    
    train(args)
