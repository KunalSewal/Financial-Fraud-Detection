"""
Test: Does using real temporal features improve v2-v5 performance?

This script replaces random timestamps with actual temporal features
to see if temporal components work better with real data.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

def load_ethereum_with_real_temporal():
    """Load data using REAL temporal features instead of random timestamps"""
    print("🧪 Testing with REAL temporal features from dataset")
    print("=" * 80)
    
    df = pd.read_csv('data/transaction_dataset.csv')
    
    # Get unique addresses
    unique_addresses = df['Address'].unique()
    address_to_id = {addr: i for i, addr in enumerate(unique_addresses)}
    
    print(f"📊 Dataset: {len(unique_addresses)} addresses")
    
    # Extract features per address
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'FLAG']
    
    # Identify temporal feature columns
    temporal_cols = [
        'Avg min between sent tnx',
        'Avg min between received tnx',
        'Time Diff between first and last (Mins)',
        'min value received',
        'max value received'
    ]
    
    available_temporal = [col for col in temporal_cols if col in df.columns]
    print(f"📅 Found temporal features: {available_temporal}")
    
    node_features = []
    labels = []
    node_temporal_scores = []  # ← NEW: Real temporal information per node
    
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
        
        # Compute temporal score from actual features
        temporal_score = 0.0
        if 'Avg min between sent tnx' in addr_data.columns:
            temporal_score += addr_data['Avg min between sent tnx'].mean()
        if 'Time Diff between first and last (Mins)' in addr_data.columns:
            temporal_score += addr_data['Time Diff between first and last (Mins)'].mean()
        
        # Normalize
        node_temporal_scores.append(temporal_score if not np.isnan(temporal_score) else 0.0)
    
    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    node_temporal_scores = torch.tensor(node_temporal_scores, dtype=torch.float32)
    
    # Normalize features
    node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
    feat_min = node_features.min(dim=0)[0]
    feat_max = node_features.max(dim=0)[0]
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    node_features = (node_features - feat_min) / feat_range
    
    # Normalize temporal scores
    if node_temporal_scores.max() > node_temporal_scores.min():
        node_temporal_scores = (node_temporal_scores - node_temporal_scores.min()) / \
                               (node_temporal_scores.max() - node_temporal_scores.min() + 1e-8)
    
    print(f"✅ Features shape: {node_features.shape}")
    print(f"✅ Temporal scores range: [{node_temporal_scores.min():.4f}, {node_temporal_scores.max():.4f}]")
    
    # Create KNN graph
    A = kneighbors_graph(node_features.numpy(), n_neighbors=10, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    
    # Create REAL timestamps for each edge based on node temporal scores
    # For edge (u, v), use average of source and target temporal scores
    source_times = node_temporal_scores[edge_index[0]]
    target_times = node_temporal_scores[edge_index[1]]
    edge_timestamps = (source_times + target_times) / 2.0
    
    # Add small noise to break ties
    edge_timestamps += torch.rand_like(edge_timestamps) * 0.01
    
    print(f"✅ Created {edge_index.size(1)} edges")
    print(f"✅ Edge timestamps range: [{edge_timestamps.min():.4f}, {edge_timestamps.max():.4f}]")
    
    # Compare with random timestamps
    random_timestamps = torch.rand(edge_index.size(1))
    
    print(f"\n📊 Temporal Signal Analysis:")
    print(f"   Real timestamps std: {edge_timestamps.std():.4f}")
    print(f"   Random timestamps std: {random_timestamps.std():.4f}")
    
    # Check correlation with fraud
    fraud_mask = labels == 1
    fraud_temporal_mean = node_temporal_scores[fraud_mask].mean()
    normal_temporal_mean = node_temporal_scores[~fraud_mask].mean()
    
    print(f"\n🎯 Fraud vs Normal temporal patterns:")
    print(f"   Fraud addresses temporal score: {fraud_temporal_mean:.4f}")
    print(f"   Normal addresses temporal score: {normal_temporal_mean:.4f}")
    print(f"   Difference: {abs(fraud_temporal_mean - normal_temporal_mean):.4f}")
    
    if abs(fraud_temporal_mean - normal_temporal_mean) > 0.1:
        print(f"   ✅ Significant difference! Temporal features are informative!")
    else:
        print(f"   ⚠️  Small difference. Temporal signal may be weak.")
    
    # Create train/val/test splits
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
    
    # Create PyG Data objects
    data_real = Data(
        x=node_features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    data_real.edge_attr = torch.ones(edge_index.size(1), 1)
    data_real.timestamps = edge_timestamps  # ← REAL timestamps
    
    data_random = Data(
        x=node_features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    data_random.edge_attr = torch.ones(edge_index.size(1), 1)
    data_random.timestamps = random_timestamps  # ← RANDOM timestamps
    
    return data_real, data_random


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("EXPERIMENT: Real vs Random Temporal Features")
    print("=" * 80)
    
    data_real, data_random = load_ethereum_with_real_temporal()
    
    print("\n" + "=" * 80)
    print("📋 Summary:")
    print("=" * 80)
    print(f"✅ Created two identical datasets:")
    print(f"   1. data_real: Uses actual temporal features from transactions")
    print(f"   2. data_random: Uses random timestamps (current approach)")
    print(f"\n💡 Next step: Train v2-v5 on BOTH datasets and compare!")
    print(f"   Expected: v2-v5 should perform BETTER with real temporal data")
    print(f"   If true → Architecture is good, dataset was the issue!")
