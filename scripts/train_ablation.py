"""
Ablation Study Training Script for HMSTA v2

Trains all 6 versions of HMSTA incrementally to prove component contributions.
"""

import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.models.hmsta_v2 import create_hmsta_model, VERSION_DESCRIPTIONS


def load_ethereum_data(data_path='data/transaction_dataset.csv'):
    """Load and preprocess Ethereum dataset"""
    print("📦 Loading Ethereum dataset...")
    df = pd.read_csv(data_path)
    
    # Get unique addresses
    unique_addresses = df['Address'].unique()
    address_to_id = {addr: i for i, addr in enumerate(unique_addresses)}
    
    print(f"  Found {len(unique_addresses)} unique addresses")
    
    # Extract features per address
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'FLAG']
    
    node_features = []
    labels = []
    
    for addr in unique_addresses:
        addr_data = df[df['Address'] == addr]
        
        # Extract features
        features = []
        for col in numeric_columns:
            val = addr_data[col].mean()
            if pd.isna(val) or np.isinf(val):
                val = 0.0
            features.append(val)
        
        node_features.append(features)
        labels.append(addr_data['FLAG'].iloc[0])
    
    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Normalize features to [0, 1]
    node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
    feat_min = node_features.min(dim=0)[0]
    feat_max = node_features.max(dim=0)[0]
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    node_features = (node_features - feat_min) / feat_range
    
    print(f"  Features shape: {node_features.shape}")
    print(f"  Labels: Fraud={labels.sum().item()}, Normal={(labels == 0).sum().item()}")
    
    # Create temporal graph (KNN based on transaction patterns)
    from sklearn.neighbors import kneighbors_graph
    
    A = kneighbors_graph(node_features.numpy(), n_neighbors=10, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    
    # Create timestamps (use feature-based temporal proxy)
    timestamps = torch.rand(edge_index.size(1))  # Placeholder timestamps
    
    print(f"  Created {edge_index.size(1)} temporal edges")
    
    # Create train/val/test splits (60/20/20)
    num_nodes = len(unique_addresses)
    indices = torch.randperm(num_nodes)
    
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # Add edge attributes (simple features)
    edge_attr = torch.ones(edge_index.size(1), 1)
    data.edge_attr = edge_attr
    data.timestamps = timestamps
    
    return data


def compute_class_weights(labels):
    """Compute class weights for imbalanced data"""
    total = len(labels)
    fraud_count = (labels == 1).sum().item()
    normal_count = (labels == 0).sum().item()
    
    weight_fraud = total / (2.0 * fraud_count)
    weight_normal = total / (2.0 * normal_count)
    
    return torch.tensor([weight_normal, weight_fraud])


def train_epoch(model, data, optimizer, class_weights, device):
    """Train for one epoch"""
    model.train()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
    # Use edge_timestamps if available (for scatter operations), otherwise node timestamps
    timestamps = data.edge_timestamps.to(device) if hasattr(data, 'edge_timestamps') else (data.timestamps.to(device) if hasattr(data, 'timestamps') else None)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(x, edge_index, edge_attr, timestamps)
    
    # Compute loss on training set
    loss = F.cross_entropy(
        logits[train_mask], 
        y[train_mask], 
        weight=class_weights.to(device)
    )
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Compute training metrics
    with torch.no_grad():
        probs = F.softmax(logits[train_mask], dim=1)[:, 1]
        y_train = y[train_mask].cpu().numpy()
        probs_train = probs.cpu().numpy()
        
        train_auc = roc_auc_score(y_train, probs_train)
    
    return loss.item(), train_auc


@torch.no_grad()
def evaluate(model, data, mask, device):
    """Evaluate model on given mask"""
    model.eval()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
    # Use edge_timestamps if available (for scatter operations), otherwise node timestamps
    timestamps = data.edge_timestamps.to(device) if hasattr(data, 'edge_timestamps') else (data.timestamps.to(device) if hasattr(data, 'timestamps') else None)
    y = data.y.to(device)
    
    # Forward pass
    logits = model(x, edge_index, edge_attr, timestamps)
    
    # Get predictions for masked nodes
    probs = F.softmax(logits[mask], dim=1)[:, 1]
    preds = logits[mask].argmax(dim=1)
    y_true = y[mask].cpu().numpy()
    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    # Compute metrics
    auc = roc_auc_score(y_true, probs_np)
    f1 = f1_score(y_true, preds_np)
    precision = precision_score(y_true, preds_np, zero_division=0)
    recall = recall_score(y_true, preds_np, zero_division=0)
    
    return {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_model(version, data, hidden_dim=128, epochs=100, patience=30, lr=0.0001, device='cuda'):
    """Train a specific HMSTA version"""
    print(f"\n{'='*80}")
    print(f"Training HMSTA v{version}: {VERSION_DESCRIPTIONS[version]}")
    print(f"{'='*80}")
    
    # Create model
    model = create_hmsta_model(
        version=version,
        node_features=data.x.size(1),
        hidden_dim=hidden_dim,
        dropout=0.5
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {num_params:,}")
    
    # Reset memory for versions with temporal memory
    if version >= 3:
        model.reset_memory(data.x.size(0), device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    class_weights = compute_class_weights(data.y[data.train_mask])
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        loss, train_auc = train_epoch(model, data, optimizer, class_weights, device)
        
        # Validate
        val_metrics = evaluate(model, data, data.val_mask, device)
        val_auc = val_metrics['auc']
        
        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'checkpoints/hmsta_v{version}_best.pt')
        else:
            patience_counter += 1
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️  Early stopping at epoch {epoch} (best: {best_epoch})")
            break
    
    training_time = time.time() - start_time
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f'checkpoints/hmsta_v{version}_best.pt'))
    
    # Reset memory before final evaluation
    if version >= 3:
        model.reset_memory(data.x.size(0), device)
    
    test_metrics = evaluate(model, data, data.test_mask, device)
    
    print(f"\n✅ Training complete!")
    print(f"📊 Test Results:")
    print(f"   AUC:       {test_metrics['auc']:.4f}")
    print(f"   F1:        {test_metrics['f1']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"⏱️  Training time: {training_time:.1f}s")
    
    return {
        'version': version,
        'description': VERSION_DESCRIPTIONS[version],
        'num_params': num_params,
        'training_time': training_time,
        'best_epoch': best_epoch,
        **test_metrics
    }


def run_ablation_study(versions=None, data_path='data/transaction_dataset.csv'):
    """
    Run complete ablation study.
    
    Args:
        versions: List of versions to train (default: all [0,1,2,3,4,5])
        data_path: Path to dataset
    """
    if versions is None:
        versions = [0, 1, 2, 3, 4, 5]
    
    print("="*80)
    print("HMSTA ABLATION STUDY")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Load data
    data = load_ethereum_data(data_path)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Nodes: {data.x.size(0):,}")
    print(f"   Edges: {data.edge_index.size(1):,}")
    print(f"   Features: {data.x.size(1)}")
    print(f"   Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train each version
    results = []
    for version in versions:
        result = train_model(
            version=version,
            data=data,
            hidden_dim=128,
            epochs=100,
            patience=30,
            lr=0.0001,
            device=device
        )
        results.append(result)
    
    # Print comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"\n{'Version':<8} {'Description':<35} {'AUC':<8} {'F1':<8} {'Params':<10} {'Time(s)':<10}")
    print("-" * 90)
    
    baseline_auc = results[0]['auc'] if results else 0
    
    for result in results:
        improvement = ((result['auc'] - baseline_auc) / baseline_auc * 100) if baseline_auc > 0 else 0
        print(f"v{result['version']:<7} {result['description']:<35} "
              f"{result['auc']:.4f}   {result['f1']:.4f}   "
              f"{result['num_params']:<10,} {result['training_time']:<10.1f}")
        if result['version'] > 0:
            print(f"{'':>45} (+{improvement:+.2f}%)")
    
    # Print component contributions
    print("\n" + "="*80)
    print("COMPONENT CONTRIBUTIONS")
    print("="*80)
    
    for i in range(1, len(results)):
        prev_auc = results[i-1]['auc']
        curr_auc = results[i]['auc']
        improvement = (curr_auc - prev_auc) / prev_auc * 100
        component = VERSION_DESCRIPTIONS[i].replace('+ ', '')
        print(f"{component:<30} → {improvement:+.2f}% AUC improvement")
    
    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/ablation_study_results.csv', index=False)
    print(f"\n💾 Results saved to results/ablation_study_results.csv")
    
    return results


if __name__ == '__main__':
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run ablation study on all versions
    results = run_ablation_study()
    
    print("\n✅ Ablation study complete!")
