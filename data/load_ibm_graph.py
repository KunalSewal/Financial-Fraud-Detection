"""
IBM Transaction Graph Loader

Builds a temporal transaction graph from IBM credit card data.
This provides REAL temporal sequences for v3-v5 models!

Graph Structure:
- Nodes: Users (with features: avg transaction stats, fraud label)
- Edges: User → User (temporal transaction patterns)
- Edge timestamps: Real transaction times
- Edge features: Amount, MCC, time-of-day, etc.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from datetime import datetime
import os

def load_ibm_transaction_graph(
    csv_path='data/ibm/card_transaction.v1.csv',
    sample_size=None,  # None = use all data (24M rows!)
    min_transactions_per_user=10,  # Filter users with too few transactions
    fraud_label_strategy='user_level'  # 'user_level' or 'transaction_level'
):
    """
    Load IBM transaction data and build temporal graph.
    
    Args:
        csv_path: Path to IBM CSV file
        sample_size: Number of transactions to sample (None = all)
        min_transactions_per_user: Minimum transactions to include user
        fraud_label_strategy: How to assign fraud labels
            'user_level': User is fraud if ANY transaction is fraud
            'transaction_level': Each transaction labeled individually
    
    Returns:
        PyG Data object with temporal graph
    """
    
    print("="*80)
    print("Loading IBM Transaction Graph")
    print("="*80)
    
    # Load data
    print(f"\n📦 Loading data from {csv_path}...")
    if sample_size:
        print(f"   Sampling {sample_size:,} transactions...")
        df = pd.read_csv(csv_path, nrows=sample_size)
    else:
        print(f"   Loading ALL transactions (this will take ~1 minute)...")
        df = pd.read_csv(csv_path)
    
    print(f"✅ Loaded {len(df):,} transactions")
    
    # Clean data
    print(f"\n🧹 Cleaning data...")
    
    # Convert fraud label to binary
    df['fraud'] = (df['Is Fraud?'] == 'Yes').astype(int)
    
    # Convert amount to numeric
    df['amount'] = pd.to_numeric(
        df['Amount'].astype(str).str.replace('$', '').str.replace(',', ''),
        errors='coerce'
    ).fillna(0)
    
    # Create datetime from Year, Month, Day, Time
    df['datetime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + 
        df['Month'].astype(str).str.zfill(2) + '-' + 
        df['Day'].astype(str).str.zfill(2) + ' ' + 
        df['Time'].astype(str),
        errors='coerce'
    )
    
    # Extract temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['timestamp_unix'] = df['datetime'].astype(np.int64) // 10**9  # Unix timestamp
    
    # Normalize timestamps to [0, 1] for model
    min_time = df['timestamp_unix'].min()
    max_time = df['timestamp_unix'].max()
    df['timestamp_norm'] = (df['timestamp_unix'] - min_time) / (max_time - min_time + 1e-8)
    
    print(f"   Fraud transactions: {df['fraud'].sum():,} ({df['fraud'].mean()*100:.2f}%)")
    print(f"   Temporal range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Filter users by transaction count
    print(f"\n👥 Processing users...")
    user_txn_counts = df['User'].value_counts()
    valid_users = user_txn_counts[user_txn_counts >= min_transactions_per_user].index
    df = df[df['User'].isin(valid_users)]
    
    print(f"   Total users: {df['User'].nunique():,}")
    print(f"   (Filtered to users with ≥{min_transactions_per_user} transactions)")
    
    # Sort by timestamp for temporal ordering
    df = df.sort_values('timestamp_unix').reset_index(drop=True)
    
    # Create user-level features and labels
    print(f"\n🔨 Building graph structure...")
    
    unique_users = df['User'].unique()
    user_to_id = {user: idx for idx, user in enumerate(unique_users)}
    
    # Extract user-level features
    user_features_list = []
    user_labels = []
    
    for user in unique_users:
        user_df = df[df['User'] == user]
        
        # Aggregate features per user
        features = [
            user_df['amount'].mean(),
            user_df['amount'].std(),
            user_df['amount'].min(),
            user_df['amount'].max(),
            len(user_df),  # Number of transactions
            user_df['hour'].mean(),
            user_df['day_of_week'].mean(),
            user_df['MCC'].nunique(),  # Diversity of merchants
            (user_df['Use Chip'] == 'Chip Transaction').mean(),  # Chip usage rate
            user_df['Merchant State'].nunique(),  # Geographic diversity
        ]
        
        user_features_list.append(features)
        
        # Label: user is fraud if ANY transaction is fraud
        if fraud_label_strategy == 'user_level':
            user_labels.append(1 if user_df['fraud'].sum() > 0 else 0)
        else:
            # For transaction-level, use majority vote
            user_labels.append(1 if user_df['fraud'].mean() > 0.5 else 0)
    
    # Convert to tensors
    node_features = torch.tensor(user_features_list, dtype=torch.float32)
    labels = torch.tensor(user_labels, dtype=torch.long)
    
    # Normalize features
    node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
    feat_mean = node_features.mean(dim=0, keepdim=True)
    feat_std = node_features.std(dim=0, keepdim=True) + 1e-8
    node_features = (node_features - feat_mean) / feat_std
    
    print(f"   Node features shape: {node_features.shape}")
    print(f"   Fraud users: {labels.sum().item()} ({labels.float().mean()*100:.2f}%)")
    
    # Build edges from temporal transaction sequences
    # Strategy: Connect users who transacted at similar times/places
    print(f"\n🔗 Creating temporal edges...")
    
    edges = []
    edge_timestamps = []
    edge_features = []
    
    # Group transactions by time windows (e.g., same day)
    df['date'] = df['datetime'].dt.date
    
    # For each day, connect users who had transactions
    for date, group in df.groupby('date'):
        users_in_day = group['User'].unique()
        if len(users_in_day) < 2:
            continue
        
        # Create edges between users active on same day
        for i, user1 in enumerate(users_in_day):
            for user2 in users_in_day[i+1:]:
                if len(edges) < 100000:  # Limit edges to avoid memory issues
                    user1_id = user_to_id[user1]
                    user2_id = user_to_id[user2]
                    
                    # Bidirectional edges
                    edges.append([user1_id, user2_id])
                    edges.append([user2_id, user1_id])
                    
                    # Edge timestamp: average of their transaction times that day
                    user1_time = group[group['User'] == user1]['timestamp_norm'].mean()
                    user2_time = group[group['User'] == user2]['timestamp_norm'].mean()
                    avg_time = (user1_time + user2_time) / 2
                    
                    edge_timestamps.append(avg_time)
                    edge_timestamps.append(avg_time)
                    
                    # Edge features: transaction similarity
                    user1_amt = group[group['User'] == user1]['amount'].mean()
                    user2_amt = group[group['User'] == user2]['amount'].mean()
                    amt_diff = abs(user1_amt - user2_amt)
                    
                    edge_features.append([avg_time, amt_diff, 1.0])  # [time, amount_diff, weight]
                    edge_features.append([avg_time, amt_diff, 1.0])
    
    # Convert edges to tensor
    if len(edges) == 0:
        print("   ⚠️ No edges created! Using KNN fallback...")
        # Fallback: KNN graph
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(node_features.numpy(), n_neighbors=10, mode='connectivity')
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        edge_timestamps = torch.rand(edge_index.size(1))  # Random timestamps as fallback
        edge_features = torch.ones(edge_index.size(1), 3)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_timestamps = torch.tensor(edge_timestamps, dtype=torch.float32)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
    
    print(f"   Created {edge_index.size(1):,} temporal edges")
    print(f"   Edge timestamps range: [{edge_timestamps.min():.4f}, {edge_timestamps.max():.4f}]")
    
    # Create train/val/test splits (60/20/20)
    num_nodes = len(unique_users)
    indices = torch.randperm(num_nodes)
    
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create node timestamps (average transaction time per user)
    node_timestamps = torch.zeros(num_nodes, dtype=torch.float32)
    for idx, user in enumerate(unique_users):
        user_times = df[df['User'] == user]['timestamp_norm']
        node_timestamps[idx] = user_times.mean()
    
    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        timestamps=node_timestamps,  # ← Node timestamps for v2+
        edge_timestamps=edge_timestamps  # ← Edge timestamps for future use
    )
    
    # Print summary
    print(f"\n" + "="*80)
    print("📊 Graph Statistics")
    print("="*80)
    print(f"Nodes: {data.x.size(0):,}")
    print(f"Edges: {data.edge_index.size(1):,}")
    print(f"Node features: {data.x.size(1)}")
    print(f"Edge features: {data.edge_attr.size(1)}")
    print(f"Fraud labels: {data.y.sum().item()} ({data.y.float().mean()*100:.2f}%)")
    print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    print(f"\n✅ Temporal graph ready!")
    print(f"   This graph has REAL temporal sequences for v3-v5!")
    print("="*80)
    
    return data


if __name__ == '__main__':
    # Test with sample
    print("\n🧪 Testing graph loader with 500K transaction sample...")
    data = load_ibm_transaction_graph(
        sample_size=500000,
        min_transactions_per_user=20
    )
    
    print(f"\n✅ Graph created successfully!")
    print(f"\n💡 Next: Run ablation study on this temporal graph!")
    print(f"   Expected: v3-v5 should now work properly with real temporal data")
