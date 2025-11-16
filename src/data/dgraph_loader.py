"""
DGraph Dataset Loader (Legacy - for separate .npy files).

⚠️ NOTE: For the official DGraph dataset (dgraphfin.npz), use dgraph_loader_npz.py instead!

This loader is for custom .npy file format with separate edges.npy and nodes.npy.
DGraph: Large-scale financial transaction network (NeurIPS 2022).

For the official DGraph format, use:
    from src.data.dgraph_loader_npz import load_dgraph
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DGraphDataset:
    """
    DGraph financial transaction dataset loader.
    
    Dataset structure:
    - edges.npy: Transaction edges [num_edges, ...] (source, target, timestamp, features)
    - nodes.npy: Node features or labels [num_nodes, ...]
    """
    
    def __init__(
        self,
        data_dir: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Directory containing edges.npy and nodes.npy
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        self.edges_data = None
        self.nodes_data = None
        self.processed_data = None
        
    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw .npy files.
        
        Returns:
            (edges_array, nodes_array)
        """
        edges_path = self.data_dir / 'edges.npy'
        nodes_path = self.data_dir / 'nodes.npy'
        
        if not edges_path.exists():
            raise FileNotFoundError(f"edges.npy not found in {self.data_dir}")
        if not nodes_path.exists():
            raise FileNotFoundError(f"nodes.npy not found in {self.data_dir}")
        
        print(f"📂 Loading DGraph from {self.data_dir}...")
        
        # Load data
        edges = np.load(edges_path, allow_pickle=True)
        nodes = np.load(nodes_path, allow_pickle=True)
        
        print(f"  Edges shape: {edges.shape}")
        print(f"  Nodes shape: {nodes.shape}")
        print(f"  Edges dtype: {edges.dtype}")
        print(f"  Nodes dtype: {nodes.dtype}")
        
        self.edges_data = edges
        self.nodes_data = nodes
        
        return edges, nodes
    
    def analyze_structure(self):
        """
        Analyze and print dataset structure.
        """
        if self.edges_data is None or self.nodes_data is None:
            self.load_raw_data()
        
        print("\n🔍 Dataset Analysis:")
        
        # Analyze edges
        print("\n  Edges:")
        if self.edges_data.ndim == 2:
            print(f"    Shape: {self.edges_data.shape}")
            print(f"    Columns: {self.edges_data.shape[1]}")
            
            # Check if first 3 columns are source, target, timestamp
            if self.edges_data.shape[1] >= 3:
                print(f"    Unique sources: {len(np.unique(self.edges_data[:, 0]))}")
                print(f"    Unique targets: {len(np.unique(self.edges_data[:, 1]))}")
                print(f"    Time range: {self.edges_data[:, 2].min()} to {self.edges_data[:, 2].max()}")
                
                if self.edges_data.shape[1] > 3:
                    print(f"    Edge features: {self.edges_data.shape[1] - 3} dimensions")
        
        # Analyze nodes
        print("\n  Nodes:")
        if self.nodes_data.ndim == 1:
            print(f"    1D array (likely labels): {self.nodes_data.shape}")
            print(f"    Unique values: {np.unique(self.nodes_data)}")
            print(f"    Value counts:")
            unique, counts = np.unique(self.nodes_data, return_counts=True)
            for val, count in zip(unique, counts):
                print(f"      {val}: {count} ({count/len(self.nodes_data)*100:.2f}%)")
        elif self.nodes_data.ndim == 2:
            print(f"    2D array (features): {self.nodes_data.shape}")
            print(f"    Feature dimensions: {self.nodes_data.shape[1]}")
        
        print()
    
    def process_data(self) -> Dict:
        """
        Process raw data into PyTorch-ready format.
        
        Returns:
            Dictionary with processed tensors
        """
        if self.edges_data is None or self.nodes_data is None:
            self.load_raw_data()
        
        print("⚙️  Processing DGraph data...")
        
        # ===== Process Edges =====
        if self.edges_data.ndim == 2 and self.edges_data.shape[1] >= 3:
            # Extract components
            edge_src = self.edges_data[:, 0].astype(np.int64)
            edge_dst = self.edges_data[:, 1].astype(np.int64)
            edge_time = self.edges_data[:, 2].astype(np.float32)
            
            # Edge features (if available)
            if self.edges_data.shape[1] > 3:
                edge_features = self.edges_data[:, 3:].astype(np.float32)
            else:
                edge_features = np.ones((len(edge_src), 1), dtype=np.float32)
        else:
            raise ValueError(f"Unexpected edges format: {self.edges_data.shape}")
        
        # ===== Build Node Mapping =====
        # Get all unique nodes
        unique_nodes = np.unique(np.concatenate([edge_src, edge_dst]))
        num_nodes = len(unique_nodes)
        
        print(f"  Found {num_nodes:,} unique nodes")
        
        # Create mapping: original_id -> new_id (0 to num_nodes-1)
        node_to_id = {int(orig_id): new_id for new_id, orig_id in enumerate(unique_nodes)}
        
        # Remap edges to contiguous node IDs
        edge_src_mapped = np.array([node_to_id[int(n)] for n in edge_src])
        edge_dst_mapped = np.array([node_to_id[int(n)] for n in edge_dst])
        
        # ===== Process Node Features/Labels =====
        if self.nodes_data.ndim == 1:
            # 1D array - likely labels
            # Map labels to new node IDs
            node_labels = np.zeros(num_nodes, dtype=np.int64)
            
            for orig_id, label in enumerate(self.nodes_data):
                if orig_id in node_to_id:
                    new_id = node_to_id[orig_id]
                    node_labels[new_id] = int(label)
            
            # Create dummy features (we'll use degree or aggregate from edges)
            node_features = self._create_node_features_from_edges(
                edge_src_mapped, edge_dst_mapped, edge_features, num_nodes
            )
            
        elif self.nodes_data.ndim == 2:
            # 2D array - node features
            # Map features to new node IDs
            feature_dim = self.nodes_data.shape[1]
            node_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
            
            for orig_id, features in enumerate(self.nodes_data):
                if orig_id in node_to_id:
                    new_id = node_to_id[orig_id]
                    node_features[new_id] = features.astype(np.float32)
            
            # Check if last column is labels
            # (heuristic: if last column has few unique values, it might be labels)
            unique_last_col = len(np.unique(node_features[:, -1]))
            if unique_last_col < 10:
                print(f"  Detected potential label column (column {feature_dim-1})")
                node_labels = node_features[:, -1].astype(np.int64)
                node_features = node_features[:, :-1]
            else:
                # No labels detected - create dummy labels
                node_labels = np.zeros(num_nodes, dtype=np.int64)
        else:
            raise ValueError(f"Unexpected nodes format: {self.nodes_data.shape}")
        
        # ===== Normalize Features =====
        if node_features.shape[1] > 0:
            scaler = StandardScaler()
            node_features = scaler.fit_transform(node_features)
        
        print(f"  Node features: {node_features.shape}")
        print(f"  Edge features: {edge_features.shape}")
        print(f"  Labels: {node_labels.shape}, unique: {np.unique(node_labels)}")
        
        # ===== Sort by Time =====
        time_order = np.argsort(edge_time)
        edge_src_mapped = edge_src_mapped[time_order]
        edge_dst_mapped = edge_dst_mapped[time_order]
        edge_time = edge_time[time_order]
        edge_features = edge_features[time_order]
        
        # ===== Create Train/Val/Test Splits =====
        # Temporal split: early edges for train, later for val/test
        num_edges = len(edge_time)
        train_end = int(num_edges * (1 - self.val_ratio - self.test_ratio))
        val_end = int(num_edges * (1 - self.test_ratio))
        
        # Node splits based on when they first appear
        node_first_appear = np.zeros(num_nodes, dtype=np.int64)
        for i, (src, dst) in enumerate(zip(edge_src_mapped, edge_dst_mapped)):
            if node_first_appear[src] == 0:
                node_first_appear[src] = i
            if node_first_appear[dst] == 0:
                node_first_appear[dst] = i
        
        train_mask = node_first_appear < train_end
        val_mask = (node_first_appear >= train_end) & (node_first_appear < val_end)
        test_mask = node_first_appear >= val_end
        
        print(f"\n  Temporal splits:")
        print(f"    Train nodes: {train_mask.sum():,} ({train_mask.sum()/num_nodes*100:.1f}%)")
        print(f"    Val nodes: {val_mask.sum():,} ({val_mask.sum()/num_nodes*100:.1f}%)")
        print(f"    Test nodes: {test_mask.sum():,} ({test_mask.sum()/num_nodes*100:.1f}%)")
        
        # ===== Convert to PyTorch =====
        processed_data = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'node_features': torch.from_numpy(node_features).float(),
            'edge_index': torch.from_numpy(
                np.stack([edge_src_mapped, edge_dst_mapped], axis=0)
            ).long(),
            'edge_time': torch.from_numpy(edge_time).float(),
            'edge_attr': torch.from_numpy(edge_features).float(),
            'labels': torch.from_numpy(node_labels).long(),
            'train_mask': torch.from_numpy(train_mask),
            'val_mask': torch.from_numpy(val_mask),
            'test_mask': torch.from_numpy(test_mask),
            'node_to_id': node_to_id
        }
        
        self.processed_data = processed_data
        return processed_data
    
    def _create_node_features_from_edges(
        self,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_features: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """
        Create node features from edge statistics.
        
        Features:
        - In-degree, out-degree
        - Total in-weight, out-weight
        - Average in-weight, out-weight
        """
        # Initialize features
        in_degree = np.zeros(num_nodes)
        out_degree = np.zeros(num_nodes)
        in_weight = np.zeros(num_nodes)
        out_weight = np.zeros(num_nodes)
        
        # Compute degrees and weights
        for src, dst, feat in zip(edge_src, edge_dst, edge_features):
            out_degree[src] += 1
            in_degree[dst] += 1
            out_weight[src] += feat.sum()
            in_weight[dst] += feat.sum()
        
        # Average weights
        avg_in_weight = np.divide(in_weight, in_degree, where=in_degree > 0)
        avg_out_weight = np.divide(out_weight, out_degree, where=out_degree > 0)
        
        # Combine features
        node_features = np.stack([
            in_degree,
            out_degree,
            in_weight,
            out_weight,
            avg_in_weight,
            avg_out_weight
        ], axis=1).astype(np.float32)
        
        return node_features
    
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
    data_dir: str,
    use_cached: bool = True,
    cache_path: Optional[str] = None
) -> Dict:
    """
    Load DGraph dataset with caching.
    
    ⚠️ NOTE: This function expects edges.npy and nodes.npy files.
    For the official DGraph format (dgraphfin.npz), use:
        from src.data.dgraph_loader_npz import load_dgraph
    
    Args:
        data_dir: Directory with edges.npy and nodes.npy
        use_cached: Whether to use cached processed data
        cache_path: Path to cached .pt file
        
    Returns:
        Processed data dictionary
    """
    # Check if using official DGraph format
    dgraph_npz = Path(data_dir).parent / 'dgraphfin.npz'
    if dgraph_npz.exists() and not (Path(data_dir) / 'edges.npy').exists():
        print("⚠️  Detected dgraphfin.npz - redirecting to dgraph_loader_npz...")
        print("    For better performance, use:")
        print("    from src.data.dgraph_loader_npz import load_dgraph")
        try:
            from .dgraph_loader_npz import load_dgraph as load_dgraph_npz
            return load_dgraph_npz(str(dgraph_npz), use_cached=use_cached)
        except ImportError:
            print("    Could not import dgraph_loader_npz, falling back to manual load")
    
    if cache_path is None:
        cache_path = Path(data_dir) / 'dgraph_processed.pt'
    
    # Try loading cached data
    if use_cached and Path(cache_path).exists():
        print(f"📦 Loading cached DGraph from {cache_path}...")
        data = torch.load(cache_path)
        print(f"  Loaded {data['num_nodes']:,} nodes, {data['num_edges']:,} edges")
        return data
    
    # Process from scratch
    dataset = DGraphDataset(data_dir)
    dataset.analyze_structure()
    data = dataset.process_data()
    
    # Cache for future use
    dataset.save_processed(cache_path)
    
    return data
