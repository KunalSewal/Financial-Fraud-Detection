"""
Build TRANSACTION-LEVEL graph instead of user-level aggregation.

This creates a graph where:
- Nodes = Individual transactions
- Edges = Temporal connections between transactions (same user, chronological)
- Features = Transaction-level (amount, time, MCC, etc.)
- Labels = Is this transaction fraud?

This is what temporal GNNs are DESIGNED for!
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def load_ibm_transaction_level_graph(
    csv_path='data/ibm/card_transaction.v1.csv',
    sample_size=50000,  # Smaller for memory
    max_time_gap_days=7  # Connect transactions within N days
):
    print("\n" + "="*80)
    print("Loading IBM TRANSACTION-LEVEL Graph")
    print("="*80)
    
    print(f"\n📦 Loading data from {csv_path}...")
    print(f"   Sampling {sample_size:,} transactions...")
    
    # Load data
    df = pd.read_csv(csv_path, nrows=sample_size)
    print(f"✅ Loaded {len(df):,} transactions")
    
    print(f"\n🧹 Cleaning data...")
    # Parse fraud label
    df['fraud'] = (df['Is Fraud?'] == 'Yes').astype(int)
    fraud_count = df['fraud'].sum()
    print(f"   Fraud transactions: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
    
    # Parse amount
    df['amount'] = pd.to_numeric(df['Amount'].str.replace('$', '').str.replace(',', ''), errors='coerce')
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' +
        df['Month'].astype(str).str.zfill(2) + '-' +
        df['Day'].astype(str).str.zfill(2) + ' ' +
        df['Time'],
        errors='coerce'
    )
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['amount', 'datetime', 'User'])
    print(f"   After cleaning: {len(df):,} transactions")
    
    # Sort by user and time (CRITICAL for temporal edges)
    df = df.sort_values(['User', 'datetime']).reset_index(drop=True)
    
    # Create temporal features (normalize)
    df['timestamp'] = df['datetime'].astype(int) // 10**9  # Unix timestamp
    min_time, max_time = df['timestamp'].min(), df['timestamp'].max()
    df['timestamp_norm'] = (df['timestamp'] - min_time) / (max_time - min_time)
    
    df['hour'] = df['datetime'].dt.hour / 23.0
    df['day_of_week'] = df['datetime'].dt.dayofweek / 6.0
    df['day_of_month'] = df['datetime'].dt.day / 31.0
    
    print(f"   Temporal range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Create transaction-level features
    features_list = []
    for idx, row in df.iterrows():
        features = [
            row['amount'] / 1000.0,  # Normalize amount
            row['timestamp_norm'],
            row['hour'],
            row['day_of_week'],
            row['day_of_month'],
            1.0 if row['Use Chip'] == 'Chip Transaction' else 0.0,
            hash(str(row['MCC'])) % 100 / 100.0,  # Merchant category hash
        ]
        features_list.append(features)
    
    node_features = torch.tensor(features_list, dtype=torch.float32)
    
    # Standardize features
    scaler = StandardScaler()
    node_features = torch.tensor(
        scaler.fit_transform(node_features.numpy()),
        dtype=torch.float32
    )
    
    print(f"\n🔨 Building transaction graph...")
    print(f"   Node features shape: {node_features.shape}")
    
    # Create labels (transaction-level fraud)
    labels = torch.tensor(df['fraud'].values, dtype=torch.long)
    print(f"   Fraud labels: {labels.sum().item()} ({labels.float().mean()*100:.2f}%)")
    
    # Build temporal edges: connect consecutive transactions by same user
    print(f"\n🔗 Creating temporal edges...")
    edges = []
    edge_timestamps = []
    
    max_time_gap_seconds = max_time_gap_days * 24 * 3600
    
    for user in df['User'].unique():
        user_txns = df[df['User'] == user].index.tolist()
        user_times = df.loc[user_txns, 'timestamp'].values
        
        # Connect each transaction to next transactions within time window
        for i in range(len(user_txns)):
            for j in range(i+1, len(user_txns)):
                time_diff = user_times[j] - user_times[i]
                
                if time_diff > max_time_gap_seconds:
                    break  # Too far apart, stop looking
                
                # Directed edge: i -> j (chronological)
                edges.append([user_txns[i], user_txns[j]])
                edge_timestamps.append(df.loc[user_txns[j], 'timestamp_norm'])
    
    if len(edges) == 0:
        print("   ⚠️  No temporal edges created!")
        # Fallback: connect all transactions chronologically
        for i in range(len(df) - 1):
            edges.append([i, i+1])
            edge_timestamps.append(df.loc[i+1, 'timestamp_norm'])
    
    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_timestamps = torch.tensor(edge_timestamps, dtype=torch.float32)
    
    # Create edge attributes (simple: [timestamp, weight, edge_type])
    edge_attr = torch.ones(edge_index.size(1), 3, dtype=torch.float32)
    edge_attr[:, 0] = edge_timestamps  # Timestamp
    edge_attr[:, 1] = 1.0  # Weight
    edge_attr[:, 2] = 1.0  # Edge type (all temporal connections)
    
    print(f"   Created {edge_index.size(1):,} temporal edges")
    print(f"   Edge timestamps range: [{edge_timestamps.min():.4f}, {edge_timestamps.max():.4f}]")
    
    # Create train/val/test splits (stratified to ensure fraud in all splits)
    num_nodes = len(df)
    
    # Get fraud and normal indices
    fraud_indices = torch.where(labels == 1)[0]
    normal_indices = torch.where(labels == 0)[0]
    
    # Shuffle each group
    fraud_perm = fraud_indices[torch.randperm(len(fraud_indices))]
    normal_perm = normal_indices[torch.randperm(len(normal_indices))]
    
    # Split each group 60/20/20
    fraud_train_size = int(0.6 * len(fraud_perm))
    fraud_val_size = int(0.2 * len(fraud_perm))
    
    normal_train_size = int(0.6 * len(normal_perm))
    normal_val_size = int(0.2 * len(normal_perm))
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Assign fraud samples
    train_mask[fraud_perm[:fraud_train_size]] = True
    val_mask[fraud_perm[fraud_train_size:fraud_train_size + fraud_val_size]] = True
    test_mask[fraud_perm[fraud_train_size + fraud_val_size:]] = True
    
    # Assign normal samples
    train_mask[normal_perm[:normal_train_size]] = True
    val_mask[normal_perm[normal_train_size:normal_train_size + normal_val_size]] = True
    test_mask[normal_perm[normal_train_size + normal_val_size:]] = True
    
    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        timestamps=edge_timestamps  # Edge timestamps for temporal models
    )
    
    # Print summary
    print(f"\n" + "="*80)
    print("📊 Transaction Graph Statistics")
    print("="*80)
    print(f"Nodes (transactions): {data.x.size(0):,}")
    print(f"Edges (temporal connections): {data.edge_index.size(1):,}")
    print(f"Node features: {data.x.size(1)}")
    print(f"Fraud rate: {data.y.float().mean()*100:.2f}%")
    print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    print(f"   Train fraud: {data.y[train_mask].float().mean()*100:.1f}%")
    print(f"   Val fraud: {data.y[val_mask].float().mean()*100:.1f}%")
    print(f"   Test fraud: {data.y[test_mask].float().mean()*100:.1f}%")
    print(f"\n✅ TRANSACTION-level graph ready!")
    print(f"   Now temporal models can learn sequential fraud patterns!")
    print("="*80)
    
    return data


if __name__ == '__main__':
    # Test
    data = load_ibm_transaction_level_graph(sample_size=10000)
    print(f"\n✅ Graph loaded successfully!")
