"""
DGraph Dataset Loader for .npz format.

Handles loading and preprocessing the DGraph dataset from dgraphfin.npz.
DGraph: Large-scale financial transaction network (NeurIPS 2022).

Dataset Structure:
- x: Node features [3.7M nodes, 17 features]
- y: Node labels (fraud/legitimate)
- edge_index: Edge list [4.3M edges, 2]
- edge_type: Edge types
- edge_timestamp: Transaction timestamps
- train_mask, valid_mask, test_mask: Pre-defined splits
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler


class DGraphDataset:
    """
    DGraph financial transaction dataset loader for .npz format.
    
    This is the official NeurIPS 2022 DGraph dataset format.
    """
    
    def __init__(self, data_path: str = "data/dgraphfin.npz"):
        """
        Args:
            data_path: Path to dgraphfin.npz file
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_raw_data(self) -> Dict[str, np.ndarray]:
        """
        Load raw .npz file.
        
        Returns:
            Dictionary with numpy arrays
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"DGraph file not found: {self.data_path}")
        
        print(f"📂 Loading DGraph from {self.data_path}...")
        
        # Load npz file
        data = np.load(self.data_path)
        
        # Extract arrays
        raw_data = {
            'x': data['x'],
            'y': data['y'],
            'edge_index': data['edge_index'],
            'edge_type': data['edge_type'],
            'edge_timestamp': data['edge_timestamp'],
            'train_mask': data['train_mask'],
            'valid_mask': data['valid_mask'],
            'test_mask': data['test_mask']
        }
        
        print(f"  Nodes: {raw_data['x'].shape[0]:,}")
        print(f"  Node features: {raw_data['x'].shape[1]}")
        print(f"  Edges: {raw_data['edge_index'].shape[0]:,}")
        print(f"  Edge types: {len(np.unique(raw_data['edge_type']))}")
        print(f"  Labels: {len(np.unique(raw_data['y']))} classes")
        
        self.raw_data = raw_data
        return raw_data
    
    def analyze_structure(self):
        """
        Analyze and print dataset statistics.
        """
        if self.raw_data is None:
            self.load_raw_data()
        
        data = self.raw_data
        
        print("\n🔍 Dataset Analysis:")
        
        # Node analysis
        print("\n  Nodes:")
        print(f"    Total nodes: {data['x'].shape[0]:,}")
        print(f"    Feature dimensions: {data['x'].shape[1]}")
        print(f"    Features dtype: {data['x'].dtype}")
        print(f"    Feature range: [{data['x'].min():.2f}, {data['x'].max():.2f}]")
        
        # Label analysis
        print("\n  Labels:")
        unique_labels, label_counts = np.unique(data['y'], return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            print(f"    Class {label}: {count:,} ({count/len(data['y'])*100:.2f}%)")
        
        # Edge analysis
        print("\n  Edges:")
        print(f"    Total edges: {data['edge_index'].shape[0]:,}")
        print(f"    Edge format: {data['edge_index'].shape}")
        
        # Edge types
        unique_types, type_counts = np.unique(data['edge_type'], return_counts=True)
        print(f"\n  Edge Types:")
        for edge_type, count in zip(unique_types, type_counts):
            print(f"    Type {edge_type}: {count:,} ({count/len(data['edge_type'])*100:.2f}%)")
        
        # Timestamps
        print(f"\n  Timestamps:")
        print(f"    Range: {data['edge_timestamp'].min()} to {data['edge_timestamp'].max()}")
        print(f"    Unique timestamps: {len(np.unique(data['edge_timestamp'])):,}")
        
        # Splits
        print(f"\n  Pre-defined Splits:")
        print(f"    Train nodes: {len(data['train_mask']):,}")
        print(f"    Valid nodes: {len(data['valid_mask']):,}")
        print(f"    Test nodes: {len(data['test_mask']):,}")
        
        print()
    
    def process_data(self, normalize: bool = True) -> Dict:
        """
        Process raw data into PyTorch-ready format.
        
        Args:
            normalize: Whether to normalize node features
            
        Returns:
            Dictionary with processed tensors
        """
        if self.raw_data is None:
            self.load_raw_data()
        
        print("⚙️  Processing DGraph data...")
        
        data = self.raw_data
        
        # ===== Process Node Features =====
        node_features = data['x'].astype(np.float32)
        
        if normalize:
            print("  Normalizing node features...")
            scaler = StandardScaler()
            node_features = scaler.fit_transform(node_features)
        
        # ===== Process Labels =====
        labels = data['y'].astype(np.int64)
        
        # ===== Process Edge Index =====
        # Format: [num_edges, 2] -> need [2, num_edges]
        edge_index = data['edge_index'].T  # Transpose to [2, num_edges]
        
        # ===== Process Edge Features =====
        # Combine edge type and timestamp as features
        edge_type = data['edge_type'].astype(np.int64)
        edge_timestamp = data['edge_timestamp'].astype(np.float32)
        
        # Create edge features: [edge_type (one-hot), normalized_timestamp]
        # Edge types are 1-indexed (1 to max_type), convert to 0-indexed
        num_edge_types = len(np.unique(edge_type))
        edge_type_zero_indexed = edge_type - edge_type.min()
        edge_type_onehot = np.eye(num_edge_types)[edge_type_zero_indexed]
        
        # Normalize timestamps to [0, 1]
        time_min = edge_timestamp.min()
        time_max = edge_timestamp.max()
        edge_timestamp_norm = (edge_timestamp - time_min) / (time_max - time_min + 1e-8)
        
        edge_features = np.concatenate([
            edge_type_onehot,
            edge_timestamp_norm.reshape(-1, 1)
        ], axis=1).astype(np.float32)
        
        # ===== Process Masks =====
        # Create boolean masks for all nodes
        num_nodes = node_features.shape[0]
        train_mask = np.zeros(num_nodes, dtype=bool)
        valid_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        train_mask[data['train_mask']] = True
        valid_mask[data['valid_mask']] = True
        test_mask[data['test_mask']] = True
        
        print(f"  Processed {num_nodes:,} nodes, {edge_index.shape[1]:,} edges")
        print(f"  Node features: {node_features.shape}")
        print(f"  Edge features: {edge_features.shape} (type one-hot + timestamp)")
        print(f"  Train/Val/Test: {train_mask.sum():,} / {valid_mask.sum():,} / {test_mask.sum():,}")
        
        # ===== Convert to PyTorch =====
        processed_data = {
            'num_nodes': num_nodes,
            'num_edges': edge_index.shape[1],
            'x': torch.from_numpy(node_features).float(),
            'edge_index': torch.from_numpy(edge_index).long(),
            'edge_attr': torch.from_numpy(edge_features).float(),
            'edge_type': torch.from_numpy(edge_type).long(),
            'timestamps': torch.from_numpy(edge_timestamp).float(),
            'y': torch.from_numpy(labels).long(),
            'train_mask': torch.from_numpy(train_mask),
            'val_mask': torch.from_numpy(valid_mask),
            'test_mask': torch.from_numpy(test_mask)
        }
        
        self.processed_data = processed_data
        return processed_data
    
    def get_temporal_batches(self, batch_size: int = 1000):
        """
        Create temporal batches of edges for streaming/mini-batch training.
        
        Args:
            batch_size: Number of edges per batch
            
        Yields:
            Batches of edge indices and features
        """
        if self.processed_data is None:
            self.process_data()
        
        data = self.processed_data
        num_edges = data['num_edges']
        
        # Sort by timestamp
        timestamps = data['timestamps']
        time_order = torch.argsort(timestamps)
        
        # Create batches
        for start_idx in range(0, num_edges, batch_size):
            end_idx = min(start_idx + batch_size, num_edges)
            batch_indices = time_order[start_idx:end_idx]
            
            batch = {
                'edge_index': data['edge_index'][:, batch_indices],
                'edge_attr': data['edge_attr'][batch_indices],
                'timestamps': data['timestamps'][batch_indices],
                'batch_idx': (start_idx, end_idx)
            }
            
            yield batch
    
    def save_processed(self, output_path: str):
        """
        Save processed data to disk.
        
        Args:
            output_path: Path to save .pt file
        """
        if self.processed_data is None:
            self.process_data()
        
        torch.save(self.processed_data, output_path)
        print(f"💾 Saved processed data to {output_path}")


def load_dgraph(
    data_path: str = "data/dgraphfin.npz",
    use_cached: bool = True,
    cache_path: Optional[str] = None,
    normalize: bool = True
) -> Dict:
    """
    Load DGraph dataset with caching.
    
    Args:
        data_path: Path to dgraphfin.npz
        use_cached: Whether to use cached processed data
        cache_path: Path to cached .pt file
        normalize: Whether to normalize features
        
    Returns:
        Processed data dictionary
    """
    if cache_path is None:
        cache_path = Path(data_path).parent / 'processed' / 'dgraph_processed.pt'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try loading cached data
    if use_cached and Path(cache_path).exists():
        print(f"📦 Loading cached DGraph from {cache_path}...")
        data = torch.load(cache_path)
        print(f"  Loaded {data['num_nodes']:,} nodes, {data['num_edges']:,} edges")
        return data
    
    # Process from scratch
    print(f"\n{'='*70}")
    print("Loading DGraph Dataset (NeurIPS 2022)")
    print(f"{'='*70}")
    
    dataset = DGraphDataset(data_path)
    dataset.analyze_structure()
    data = dataset.process_data(normalize=normalize)
    
    # Cache for future use
    dataset.save_processed(cache_path)
    
    print(f"\n{'='*70}")
    print("✅ DGraph loaded successfully!")
    print(f"{'='*70}\n")
    
    return data


if __name__ == '__main__':
    # Test loading
    data = load_dgraph(data_path='data/dgraphfin.npz', use_cached=False)
    
    print("\nQuick Test:")
    print(f"  Node features shape: {data['x'].shape}")
    print(f"  Edge index shape: {data['edge_index'].shape}")
    print(f"  Labels shape: {data['y'].shape}")
    print(f"  Train nodes: {data['train_mask'].sum()}")
    print(f"  Class distribution: {torch.bincount(data['y'])}")
