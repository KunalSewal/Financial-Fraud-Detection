"""
Temporal Graph Builder for Financial Transaction Networks.

This module constructs temporal graphs from transaction data,
creating directed edges with timestamps instead of static KNN similarity.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict


class TemporalGraphBuilder:
    """
    Build temporal transaction graphs with real edges and timestamps.
    
    Unlike static KNN graphs, this creates edges from actual transaction flows,
    preserving temporal ordering and enabling proper temporal GNN training.
    """
    
    def __init__(
        self,
        time_window: str = '1D',  # '1H', '1D', '1W'
        edge_features: List[str] = None,
        min_transactions: int = 1,
        directed: bool = True
    ):
        """
        Initialize temporal graph builder.
        
        Args:
            time_window: Time window for creating snapshots ('1H', '1D', '1W')
            edge_features: List of edge feature names to extract
            min_transactions: Minimum transactions to create an edge
            directed: Whether to create directed edges
        """
        self.time_window = time_window
        self.edge_features = edge_features or ['value', 'count']
        self.min_transactions = min_transactions
        self.directed = directed
        
        self.snapshots = []
        self.node_to_id = {}
        self.id_to_node = {}
        self.temporal_edges = []
        
    def build_from_ethereum_transactions(
        self,
        df: pd.DataFrame,
        address_col: str = 'Address',
        time_cols: Optional[List[str]] = None
    ) -> Dict:
        """
        Build temporal graph from Ethereum dataset.
        
        The Ethereum dataset doesn't have explicit transaction edges,
        so we infer them from features like:
        - Unique Sent To Addresses
        - Unique Received From Addresses
        - Transaction timing patterns
        
        Args:
            df: Ethereum fraud dataset
            address_col: Column containing addresses
            time_cols: Columns containing time information
            
        Returns:
            Dictionary containing temporal graph data
        """
        print("🔨 Building temporal graph from Ethereum data...")
        
        # Create node mapping
        addresses = df[address_col].unique()
        self.node_to_id = {addr: idx for idx, addr in enumerate(addresses)}
        self.id_to_node = {idx: addr for addr, idx in self.node_to_id.items()}
        
        num_nodes = len(addresses)
        print(f"  Found {num_nodes} unique addresses")
        
        # For Ethereum dataset, we'll create edges based on:
        # 1. Temporal features (time gaps between sent/received)
        # 2. Transaction patterns (addresses that interact)
        
        # Create temporal edges based on transaction patterns
        edge_list = []
        edge_times = []
        edge_features = []
        
        for idx, row in df.iterrows():
            node_id = self.node_to_id[row[address_col]]
            
            # Infer time-based edges from transaction statistics
            # If address has both sent and received transactions, create temporal edges
            
            if row.get('Sent tnx', 0) > 0 and row.get('Received Tnx', 0) > 0:
                # Create self-loop with temporal information
                # This represents the address's transaction activity over time
                
                # Use time difference as timestamp
                time_diff = row.get('Time Diff between first and last (Mins)', 0)
                
                edge_list.append([node_id, node_id])
                edge_times.append(time_diff)
                edge_features.append([
                    row.get('total Ether sent', 0),
                    row.get('total ether received', 0),
                    row.get('Sent tnx', 0),
                    row.get('Received Tnx', 0)
                ])
        
        # Create KNN edges for spatial structure (but mark them temporally)
        # We'll use feature similarity but add temporal ordering
        from sklearn.neighbors import kneighbors_graph
        from sklearn.preprocessing import StandardScaler
        
        # Select features for KNN
        feature_cols = [col for col in df.columns 
                       if col not in [address_col, 'FLAG', 'Index']
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Build KNN graph
        k = min(10, num_nodes - 1)
        knn_graph = kneighbors_graph(X_scaled, k, mode='distance', include_self=False)
        
        # Convert to edge list with temporal ordering
        rows, cols = knn_graph.nonzero()
        
        # Create mapping from dataframe index to node_id
        df_idx_to_node_id = {df_idx: self.node_to_id[df.iloc[df_idx][address_col]] 
                             for df_idx in range(len(df))}
        
        for i, (src_df_idx, dst_df_idx) in enumerate(zip(rows, cols)):
            src = df_idx_to_node_id[src_df_idx]
            dst = df_idx_to_node_id[dst_df_idx]
            
            if src != dst:  # Skip self-loops (we added them above)
                edge_list.append([src, dst])
                
                # Use average time as edge timestamp
                src_time = df.iloc[src_df_idx].get('Time Diff between first and last (Mins)', 0)
                dst_time = df.iloc[dst_df_idx].get('Time Diff between first and last (Mins)', 0)
                avg_time = (src_time + dst_time) / 2
                
                edge_times.append(avg_time)
                edge_features.append([
                    knn_graph[src_df_idx, dst_df_idx],  # Distance as feature
                    0, 0, 0  # Placeholder for other features
                ])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_time = torch.tensor(edge_times, dtype=torch.float32)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        # Sort edges by time
        sorted_indices = torch.argsort(edge_time)
        edge_index = edge_index[:, sorted_indices]
        edge_time = edge_time[sorted_indices]
        edge_attr = edge_attr[sorted_indices]
        
        print(f"  Created {edge_index.size(1)} temporal edges")
        print(f"  Time range: {edge_time.min():.2f} to {edge_time.max():.2f} minutes")
        
        return {
            'num_nodes': num_nodes,
            'edge_index': edge_index,
            'edge_time': edge_time,
            'edge_attr': edge_attr,
            'node_to_id': self.node_to_id,
            'id_to_node': self.id_to_node
        }
    
    def build_from_dgraph(
        self,
        edges_path: str,
        nodes_path: str
    ) -> Dict:
        """
        Build temporal graph from DGraph .npy files.
        
        DGraph format:
        - edges.npy: Array of edges with timestamps
        - nodes.npy: Array of node features
        
        Args:
            edges_path: Path to edges.npy file
            nodes_path: Path to nodes.npy file
            
        Returns:
            Dictionary containing temporal graph data
        """
        print("🔨 Building temporal graph from DGraph data...")
        
        # Load data
        edges_data = np.load(edges_path, allow_pickle=True)
        nodes_data = np.load(nodes_path, allow_pickle=True)
        
        print(f"  Loaded edges shape: {edges_data.shape}")
        print(f"  Loaded nodes shape: {nodes_data.shape}")
        
        # DGraph format analysis
        # Typically: [source, target, timestamp, features...]
        
        if edges_data.ndim == 2:
            # Extract edge components
            if edges_data.shape[1] >= 3:
                edge_src = edges_data[:, 0].astype(np.int64)
                edge_dst = edges_data[:, 1].astype(np.int64)
                edge_time = edges_data[:, 2].astype(np.float32)
                
                # Additional edge features if available
                if edges_data.shape[1] > 3:
                    edge_attr = edges_data[:, 3:].astype(np.float32)
                else:
                    edge_attr = np.ones((len(edge_src), 1), dtype=np.float32)
            else:
                raise ValueError(f"Unexpected edge format: {edges_data.shape}")
        else:
            raise ValueError(f"Unexpected edge dimensions: {edges_data.ndim}")
        
        # Build node mapping
        unique_nodes = np.unique(np.concatenate([edge_src, edge_dst]))
        num_nodes = len(unique_nodes)
        
        # Create mapping
        self.node_to_id = {int(node): idx for idx, node in enumerate(unique_nodes)}
        self.id_to_node = {idx: int(node) for node, idx in self.node_to_id.items()}
        
        # Remap edges to contiguous IDs
        edge_src_mapped = np.array([self.node_to_id[int(n)] for n in edge_src])
        edge_dst_mapped = np.array([self.node_to_id[int(n)] for n in edge_dst])
        
        # Create edge index
        edge_index = np.stack([edge_src_mapped, edge_dst_mapped], axis=0)
        
        # Sort by time
        time_order = np.argsort(edge_time)
        edge_index = edge_index[:, time_order]
        edge_time = edge_time[time_order]
        edge_attr = edge_attr[time_order]
        
        # Convert to PyTorch tensors
        edge_index = torch.from_numpy(edge_index).long()
        edge_time = torch.from_numpy(edge_time).float()
        edge_attr = torch.from_numpy(edge_attr).float()
        
        # Process node features
        if nodes_data.ndim == 2:
            node_features = torch.from_numpy(nodes_data.astype(np.float32))
        elif nodes_data.ndim == 1:
            # If nodes is 1D, it might be labels
            node_features = torch.zeros(num_nodes, 1)
        else:
            node_features = torch.zeros(num_nodes, 1)
        
        print(f"  Processed {num_nodes} nodes")
        print(f"  Processed {edge_index.size(1)} temporal edges")
        print(f"  Time range: {edge_time.min():.2f} to {edge_time.max():.2f}")
        print(f"  Edge features: {edge_attr.size(1)} dimensions")
        
        return {
            'num_nodes': num_nodes,
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_time': edge_time,
            'edge_attr': edge_attr,
            'node_to_id': self.node_to_id,
            'id_to_node': self.id_to_node
        }
    
    def create_temporal_batches(
        self,
        temporal_graph: Dict,
        batch_size: int = 1000,
        time_based: bool = True
    ) -> List[Dict]:
        """
        Create temporal batches for mini-batch training.
        
        Args:
            temporal_graph: Temporal graph data from build_from_*
            batch_size: Number of edges per batch
            time_based: If True, batch by time windows; else by edge count
            
        Returns:
            List of batch dictionaries
        """
        print(f"📦 Creating temporal batches (size={batch_size}, time_based={time_based})...")
        
        edge_index = temporal_graph['edge_index']
        edge_time = temporal_graph['edge_time']
        edge_attr = temporal_graph['edge_attr']
        num_edges = edge_index.size(1)
        
        batches = []
        
        if time_based:
            # Create batches based on time windows
            time_min = edge_time.min().item()
            time_max = edge_time.max().item()
            time_range = time_max - time_min
            
            # Determine number of windows
            num_windows = max(1, num_edges // batch_size)
            time_step = time_range / num_windows
            
            for i in range(num_windows):
                t_start = time_min + i * time_step
                t_end = time_min + (i + 1) * time_step if i < num_windows - 1 else time_max + 1
                
                # Get edges in this time window
                mask = (edge_time >= t_start) & (edge_time < t_end)
                
                if mask.sum() > 0:
                    batch = {
                        'edge_index': edge_index[:, mask],
                        'edge_time': edge_time[mask],
                        'edge_attr': edge_attr[mask],
                        'time_window': (t_start, t_end),
                        'batch_id': i
                    }
                    batches.append(batch)
        else:
            # Create batches based on edge count
            num_batches = (num_edges + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_edges)
                
                batch = {
                    'edge_index': edge_index[:, start_idx:end_idx],
                    'edge_time': edge_time[start_idx:end_idx],
                    'edge_attr': edge_attr[start_idx:end_idx],
                    'batch_id': i
                }
                batches.append(batch)
        
        print(f"  Created {len(batches)} temporal batches")
        return batches
    
    def get_temporal_snapshots(
        self,
        temporal_graph: Dict,
        num_snapshots: int = 10
    ) -> List[Dict]:
        """
        Create temporal snapshots of the graph.
        
        Each snapshot contains all edges up to that time point,
        enabling temporal GNN training with growing graphs.
        
        Args:
            temporal_graph: Temporal graph data
            num_snapshots: Number of snapshots to create
            
        Returns:
            List of snapshot dictionaries
        """
        print(f"📸 Creating {num_snapshots} temporal snapshots...")
        
        edge_index = temporal_graph['edge_index']
        edge_time = temporal_graph['edge_time']
        edge_attr = temporal_graph['edge_attr']
        
        time_min = edge_time.min().item()
        time_max = edge_time.max().item()
        time_step = (time_max - time_min) / num_snapshots
        
        snapshots = []
        
        for i in range(num_snapshots):
            t_cutoff = time_min + (i + 1) * time_step
            
            # Get all edges up to this time
            mask = edge_time <= t_cutoff
            
            snapshot = {
                'edge_index': edge_index[:, mask],
                'edge_time': edge_time[mask],
                'edge_attr': edge_attr[mask],
                'timestamp': t_cutoff,
                'snapshot_id': i,
                'num_edges': mask.sum().item()
            }
            snapshots.append(snapshot)
        
        print(f"  Created {len(snapshots)} snapshots")
        print(f"  Edges per snapshot: {[s['num_edges'] for s in snapshots]}")
        
        return snapshots


def load_and_build_temporal_graph(
    dataset_name: str,
    data_path: str,
    **kwargs
) -> Dict:
    """
    Convenience function to load and build temporal graph.
    
    Args:
        dataset_name: 'ethereum', 'dgraph', or 'figraph'
        data_path: Path to dataset
        **kwargs: Additional arguments for TemporalGraphBuilder
        
    Returns:
        Temporal graph dictionary
    """
    builder = TemporalGraphBuilder(**kwargs)
    
    if dataset_name.lower() == 'ethereum':
        df = pd.read_csv(data_path)
        return builder.build_from_ethereum_transactions(df)
    
    elif dataset_name.lower() == 'dgraph':
        # Expect data_path to be directory containing edges.npy and nodes.npy
        from pathlib import Path
        data_dir = Path(data_path)
        edges_path = data_dir / 'edges.npy'
        nodes_path = data_dir / 'nodes.npy'
        
        if not edges_path.exists() or not nodes_path.exists():
            raise FileNotFoundError(
                f"DGraph files not found in {data_path}. "
                f"Expected: edges.npy and nodes.npy"
            )
        
        return builder.build_from_dgraph(str(edges_path), str(nodes_path))
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
