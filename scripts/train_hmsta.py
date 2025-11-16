"""
Training script for HMSTA (Hybrid Multi-Scale Temporal Attention).

This is our NOVEL architecture that combines:
- TGN (temporal memory)
- MPTGNN (multi-path processing)  
- Anomaly-aware attention

Goal: Demonstrate superior performance over baselines and individual components.
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import time
from pathlib import Path

from src.models.hmsta import create_hmsta_model, HMSTA
from src.data.temporal_graph_builder import load_and_build_temporal_graph


def train_epoch(model, data, optimizer, device):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    timestamps = data.edge_time.to(device) if hasattr(data, 'edge_time') else torch.zeros(edge_index.size(1)).to(device)
    
    # Forward pass
    output = model(x, edge_index, edge_attr, timestamps)
    logits = output['logits']
    
    # Check for NaN
    if torch.isnan(logits).any():
        print("⚠️  Warning: NaN detected in logits, skipping batch")
        return float('nan')
    
    # Compute loss on training nodes with class weighting
    train_mask = data.train_mask
    y_train = data.y[train_mask].to(device)
    
    # Calculate class weights to handle imbalance
    fraud_count = (y_train == 1).sum().item()
    normal_count = (y_train == 0).sum().item()
    total = fraud_count + normal_count
    
    # Weight inversely proportional to class frequency
    weight_fraud = total / (2.0 * fraud_count) if fraud_count > 0 else 1.0
    weight_normal = total / (2.0 * normal_count) if normal_count > 0 else 1.0
    class_weights = torch.tensor([weight_normal, weight_fraud], device=device)
    
    loss = F.cross_entropy(logits[train_mask], y_train, weight=class_weights)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping to prevent instability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device, return_predictions=False):
    """Evaluate model performance."""
    model.eval()
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    timestamps = data.edge_time.to(device) if hasattr(data, 'edge_time') else torch.zeros(edge_index.size(1)).to(device)
    
    # Forward pass
    output = model.predict(x, edge_index, edge_attr, timestamps, return_explanation=False)
    
    logits = output['logits'][mask]
    probs = output['probs'][mask]
    preds = output['predictions'][mask]
    labels = data.y[mask].cpu().numpy()
    
    # Compute metrics
    probs_cpu = probs[:, 1].cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    
    # Check for NaN
    if np.isnan(probs_cpu).any() or np.isnan(preds_cpu).any():
        print("⚠️  Warning: NaN detected in predictions")
        return {
            'accuracy': 0.0,
            'roc_auc': 0.5,
            'pr_auc': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    metrics = {
        'accuracy': (preds_cpu == labels).mean(),
        'roc_auc': roc_auc_score(labels, probs_cpu),
        'pr_auc': average_precision_score(labels, probs_cpu),
        'f1': f1_score(labels, preds_cpu),
        'precision': precision_score(labels, preds_cpu, zero_division=0),
        'recall': recall_score(labels, preds_cpu, zero_division=0)
    }
    
    if return_predictions:
        return metrics, preds_cpu, probs_cpu
    
    return metrics


def train_hmsta_ethereum(
    hidden_dim=128,
    num_epochs=100,
    learning_rate=0.0001,  # Lower learning rate for stability
    patience=30,  # More patience before early stopping
    device=None
):
    """
    Train HMSTA on Ethereum dataset.
    
    This demonstrates our novel architecture on the smaller dataset first.
    """
    print("=" * 80)
    print("Training HMSTA on Ethereum Dataset")
    print("=" * 80)
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Using device: {device}")
    
    # Load Ethereum dataset
    print("\n📦 Loading Ethereum dataset...")
    graph_dict = load_and_build_temporal_graph('ethereum', 'data/transaction_dataset.csv')
    
    # Extract data from dictionary
    num_nodes = graph_dict['num_nodes']
    edge_index = graph_dict['edge_index']
    edge_time = graph_dict['edge_time']
    edge_attr = graph_dict['edge_attr']
    node_to_id = graph_dict['node_to_id']
    
    # Load node features and labels from original CSV
    import pandas as pd
    df = pd.read_csv('data/transaction_dataset.csv')
    
    # Create node features (initialize with zeros)
    # Using feature dimension from edge_attr as proxy for now
    node_feature_dim = 166  # Standard Ethereum feature dimension
    node_features = torch.zeros((num_nodes, node_feature_dim), dtype=torch.float32)
    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    
    # For each node, aggregate features from their transactions
    for addr in df['Address'].unique():
        if addr in node_to_id:
            node_id = node_to_id[addr]
            addr_data = df[df['Address'] == addr]
            
            # Set label (use the last known label for this address)
            node_labels[node_id] = int(addr_data['FLAG'].iloc[-1])
            
            # Create simple node features from transaction statistics
            # (In production, you'd use more sophisticated features)
            features = []
            for col in df.columns:
                if col not in ['Index', 'Address', 'FLAG'] and pd.api.types.is_numeric_dtype(df[col]):
                    val = addr_data[col].mean()
                    # Handle NaN and Inf values
                    if pd.isna(val) or np.isinf(val):
                        val = 0.0
                    features.append(val)
            
            # Pad or truncate to match node_feature_dim
            features = features[:node_feature_dim]
            if len(features) < node_feature_dim:
                features.extend([0.0] * (node_feature_dim - len(features)))
            
            node_features[node_id] = torch.tensor(features, dtype=torch.float32)
    
    # Normalize features to prevent NaN issues
    # Replace any remaining NaN/Inf
    node_features = torch.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize to [0, 1] range for stability
    feat_min = node_features.min(dim=0, keepdim=True)[0]
    feat_max = node_features.max(dim=0, keepdim=True)[0]
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0  # Avoid division by zero
    node_features = (node_features - feat_min) / feat_range
    
    # Final safety check
    node_features = torch.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Create PyG Data object
    from torch_geometric.data import Data
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=node_labels
    )
    # Store edge_time separately
    data.edge_time = edge_time
    
    # Create train/val/test splits
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   • Nodes: {data.num_nodes:,}")
    print(f"   • Edges: {data.num_edges:,}")
    print(f"   • Node features: {data.num_node_features}")
    print(f"   • Edge features: {data.edge_attr.size(1) if data.edge_attr is not None else 0}")
    print(f"   • Fraud nodes: {data.y.sum().item():,} ({data.y.sum().item() / data.num_nodes * 100:.2f}%)")
    print(f"   • Train/Val/Test: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")
    
    # Create HMSTA model
    print("\n🏗️  Creating HMSTA model...")
    model = create_hmsta_model(
        dataset_name='ethereum',
        node_features=data.num_node_features,
        edge_features=data.edge_attr.size(1) if data.edge_attr is not None else 4,
        hidden_dim=hidden_dim,
        num_nodes=data.num_nodes
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    # Training loop
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Hidden dim: {hidden_dim}")
    print(f"   • Early stopping patience: {patience}")
    print()
    
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    train_start = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, data, optimizer, device)
        
        # Evaluate
        train_metrics = evaluate(model, data, data.train_mask, device)
        val_metrics = evaluate(model, data, data.val_mask, device)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train AUC: {train_metrics['roc_auc']:.4f} | "
                  f"Val AUC: {val_metrics['roc_auc']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint_path = Path('checkpoints/hmsta_ethereum_best.pt')
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'val_metrics': val_metrics
            }, checkpoint_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - train_start
    
    # Load best model for final evaluation
    print(f"\n📥 Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load('checkpoints/hmsta_ethereum_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    train_metrics = evaluate(model, data, data.train_mask, device)
    val_metrics = evaluate(model, data, data.val_mask, device)
    test_metrics, test_preds, test_probs = evaluate(model, data, data.test_mask, device, return_predictions=True)
    
    print(f"\n{'Metric':<15} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 50)
    for metric in ['accuracy', 'roc_auc', 'pr_auc', 'f1', 'precision', 'recall']:
        print(f"{metric.upper():<15} "
              f"{train_metrics[metric]:>9.4f} "
              f"{val_metrics[metric]:>9.4f} "
              f"{test_metrics[metric]:>9.4f}")
    
    print(f"\n⏱️  Training completed in {total_time:.2f} seconds")
    print(f"   • Best epoch: {best_epoch}")
    print(f"   • Best val AUC: {best_val_auc:.4f}")
    print(f"   • Test AUC: {test_metrics['roc_auc']:.4f}")
    
    # Compare with baselines
    print("\n" + "=" * 80)
    print("📈 Comparison with Baselines")
    print("=" * 80)
    print(f"\n{'Model':<20} {'ROC-AUC':<12} {'F1 Score':<12} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'MLP (Baseline)':<20} {0.9399:<12.4f} {0.8650:<12.4f} {'Baseline':<15}")
    print(f"{'GraphSAGE':<20} {0.9131:<12.4f} {0.8482:<12.4f} {'-2.8%':<15}")
    print(f"{'HMSTA (Ours)':<20} {test_metrics['roc_auc']:<12.4f} {test_metrics['f1']:<12.4f} ", end='')
    
    improvement = (test_metrics['roc_auc'] - 0.9399) / 0.9399 * 100
    if improvement > 0:
        print(f"+{improvement:.1f}%")
    else:
        print(f"{improvement:.1f}%")
    
    print("\n✅ Training complete! Model saved to checkpoints/hmsta_ethereum_best.pt")
    
    return model, test_metrics


def train_hmsta_dgraph(
    hidden_dim=128,
    num_epochs=50,
    learning_rate=0.001,
    patience=10,
    device=None
):
    """
    Train HMSTA on DGraph dataset (3.7M nodes).
    
    This demonstrates scalability of our novel architecture.
    """
    print("=" * 80)
    print("Training HMSTA on DGraph Dataset (3.7M nodes)")
    print("=" * 80)
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Using device: {device}")
    
    # Load DGraph dataset
    print("\n📦 Loading DGraph dataset...")
    from src.data.dgraph_loader_npz import load_dgraph
    data = load_dgraph()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   • Nodes: {data.num_nodes:,}")
    print(f"   • Edges: {data.num_edges:,}")
    print(f"   • Node features: {data.num_node_features}")
    print(f"   • Edge features: {data.edge_attr.size(1) if data.edge_attr is not None else 0}")
    print(f"   • Fraud nodes: {data.y.sum().item():,} ({data.y.sum().item() / data.num_nodes * 100:.4f}%)")
    print(f"   • Class imbalance: {data.y.sum().item() / data.num_nodes:.4f} fraud rate")
    
    # Create HMSTA model
    print("\n🏗️  Creating HMSTA model...")
    model = create_hmsta_model(
        dataset_name='dgraph',
        node_features=data.num_node_features,
        edge_features=data.edge_attr.size(1) if data.edge_attr is not None else 4,
        hidden_dim=hidden_dim,
        num_nodes=data.num_nodes
    )
    model = model.to(device)
    
    # Optimizer with class weighting
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    # Class weights for imbalanced dataset
    fraud_weight = (data.y == 0).sum().float() / (data.y == 1).sum().float()
    class_weights = torch.tensor([1.0, fraud_weight.item()]).to(device)
    print(f"\n⚖️  Class weights: [1.0, {fraud_weight.item():.2f}] (due to imbalance)")
    
    # Training loop
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Hidden dim: {hidden_dim}")
    print(f"   • Early stopping patience: {patience}")
    print()
    
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    train_start = time.time()
    
    # Note: For 3.7M nodes, you may need mini-batch training
    print("⚠️  Note: For full DGraph training, consider using mini-batch training")
    print("   This demo uses full-batch on available memory\n")
    
    for epoch in range(1, min(num_epochs, 20) + 1):  # Limit to 20 epochs for demo
        epoch_start = time.time()
        
        # Train
        model.train()
        optimizer.zero_grad()
        
        # Sample subset for demo (remove for full training)
        sample_nodes = min(10000, data.num_nodes)
        node_indices = torch.randperm(data.num_nodes)[:sample_nodes]
        
        x = data.x[node_indices].to(device)
        edge_mask = (edge_index[0] < sample_nodes) & (edge_index[1] < sample_nodes)
        edge_index_sample = data.edge_index[:, edge_mask].to(device)
        edge_attr_sample = data.edge_attr[edge_mask].to(device) if data.edge_attr is not None else None
        timestamps_sample = torch.zeros(edge_index_sample.size(1)).to(device)
        
        output = model(x, edge_index_sample, edge_attr_sample, timestamps_sample)
        
        train_mask_sample = data.train_mask[node_indices]
        y_sample = data.y[node_indices].to(device)
        
        loss = F.cross_entropy(
            output['logits'][train_mask_sample],
            y_sample[train_mask_sample],
            weight=class_weights
        )
        
        loss.backward()
        optimizer.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Time: {epoch_time:.2f}s")
    
    total_time = time.time() - train_start
    
    print(f"\n⏱️  Training completed in {total_time:.2f} seconds")
    print(f"\n✅ Demo training complete!")
    print(f"   • For production, implement mini-batch training with NeighborLoader")
    print(f"   • Model saved to checkpoints/hmsta_dgraph_demo.pt")
    
    # Save demo model
    checkpoint_path = Path('checkpoints/hmsta_dgraph_demo.pt')
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    return model


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 80)
    print("HMSTA Training - Novel Architecture")
    print("=" * 80)
    
    # Train on Ethereum first (smaller dataset)
    print("\n🎯 Training on Ethereum dataset (validation)...")
    model_eth, metrics_eth = train_hmsta_ethereum(
        hidden_dim=128,
        num_epochs=100,
        learning_rate=0.001
    )
    
    print("\n" + "=" * 80)
    print("✅ Ethereum Training Complete!")
    print("=" * 80)
    print(f"\n🎯 HMSTA Test Performance:")
    print(f"   • ROC-AUC: {metrics_eth['roc_auc']:.4f}")
    print(f"   • F1 Score: {metrics_eth['f1']:.4f}")
    print(f"   • Precision: {metrics_eth['precision']:.4f}")
    print(f"   • Recall: {metrics_eth['recall']:.4f}")
    
    # Optionally train on DGraph
    if len(sys.argv) > 1 and sys.argv[1] == '--dgraph':
        print("\n" + "=" * 80)
        print("\n🎯 Training on DGraph dataset (scalability test)...")
        model_dgraph = train_hmsta_dgraph(
            hidden_dim=128,
            num_epochs=20,
            learning_rate=0.0005
        )
    
    print("\n" + "=" * 80)
    print("🎉 All training complete!")
    print("=" * 80)
    print("\n📝 Next Steps:")
    print("   1. Compare HMSTA with TGN and MPTGNN individually")
    print("   2. Analyze attention weights for explainability")
    print("   3. Extract discovered fraud patterns")
    print("   4. Prepare presentation with results")
    print()
