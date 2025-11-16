"""
Data loading and preprocessing utilities for financial fraud detection.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import networkx as nx


class EthereumFraudDataset(Dataset):
    """PyTorch Dataset for Ethereum fraud detection."""
    
    def __init__(self, data_dict: Dict, split: str = "train"):
        """
        Initialize dataset.
        
        Args:
            data_dict: Dictionary containing processed data
            split: One of "train", "val", "test"
        """
        self.data_dict = data_dict
        self.split = split
        
        # Get mask for this split
        mask_key = f"{split}_mask"
        if mask_key not in data_dict:
            raise ValueError(f"Mask {mask_key} not found in data_dict")
        
        self.mask = data_dict[mask_key]
        self.indices = torch.where(self.mask)[0]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        node_idx = self.indices[idx]
        return {
            "node_idx": node_idx,
            "features": self.data_dict["node_features"][node_idx],
            "label": self.data_dict["labels"][node_idx]
        }


def load_ethereum_data(data_path: str) -> pd.DataFrame:
    """
    Load Ethereum fraud detection dataset.
    
    Args:
        data_path: Path to transaction_dataset.csv
        
    Returns:
        DataFrame containing the dataset
    """
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load CSV
    df = pd.read_csv(data_path)
    
    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns or df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]
    
    print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"  Fraud cases: {df['FLAG'].sum()} ({df['FLAG'].mean():.2%})")
    
    return df


def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocess and select features for modeling.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    # Select numerical features (excluding FLAG, Index, Address)
    exclude_cols = ["FLAG", "Index", "Address"]
    
    # Also exclude string/categorical columns
    # These are token name columns that contain text
    categorical_patterns = [
        "token type", "token name", "Address", "ERC20 most sent token type",
        "ERC20_most_rec_token_type", "ERC20 most rec token type",
        "ERC20 uniq sent token name", "ERC20 uniq rec token name"
    ]
    
    # Get all columns
    feature_cols = []
    for col in df.columns:
        # Skip if in exclude list
        if col in exclude_cols:
            continue
        # Skip if matches categorical patterns
        if any(pattern.lower() in col.lower() for pattern in categorical_patterns):
            continue
        # Only keep if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    print(f"Selected {len(feature_cols)} numerical features (excluded {len(df.columns) - len(feature_cols) - 1} non-numerical columns)")
    
    # Handle missing values
    df_features = df[feature_cols].fillna(0)
    
    # Replace infinity with large numbers
    df_features = df_features.replace([np.inf, -np.inf], 0)
    
    # Convert to numpy
    features = df_features.values.astype(np.float32)
    
    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    print(f"✓ Preprocessed {features.shape[1]} features")
    
    return features, feature_cols


def create_graph_from_transactions(
    df: pd.DataFrame,
    method: str = "knn",
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create graph structure from transaction data.
    
    Args:
        df: Transaction dataframe
        method: Graph construction method ("knn" or "threshold")
        k: Number of neighbors for KNN
        
    Returns:
        Tuple of (edge_index, edge_weights)
    """
    from sklearn.neighbors import kneighbors_graph
    
    print(f"Creating graph using {method} method...")
    
    # Get features for graph construction
    features, _ = preprocess_features(df)
    
    if method == "knn":
        # Create KNN graph
        connectivity = kneighbors_graph(
            features,
            n_neighbors=k,
            mode="connectivity",
            include_self=False
        )
        
        # Convert to edge list
        edge_index = np.array(connectivity.nonzero())
        edge_weights = connectivity.data
        
    else:
        raise ValueError(f"Unknown graph construction method: {method}")
    
    print(f"✓ Created graph with {edge_index.shape[1]} edges")
    
    return edge_index, edge_weights


def preprocess_ethereum_data(
    data_path: str,
    test_split: float = 0.2,
    val_split: float = 0.1,
    graph_method: str = "knn",
    k_neighbors: int = 10
) -> Dict:
    """
    Complete preprocessing pipeline for Ethereum dataset.
    
    Args:
        data_path: Path to data directory or CSV file
        test_split: Fraction for test set
        val_split: Fraction for validation set
        graph_method: Method for graph construction
        k_neighbors: Number of neighbors for KNN graph
        
    Returns:
        Dictionary containing processed data and metadata
    """
    # Load data
    if os.path.isdir(data_path):
        csv_path = os.path.join(data_path, "transaction_dataset.csv")
    else:
        csv_path = data_path
    
    df = load_ethereum_data(csv_path)
    
    # Preprocess features
    features, feature_names = preprocess_features(df)
    labels = df["FLAG"].values
    
    # Create graph structure
    edge_index, edge_weights = create_graph_from_transactions(
        df, method=graph_method, k=k_neighbors
    )
    
    # Split data
    n_samples = len(df)
    indices = np.arange(n_samples)
    
    # Train/test split
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        stratify=labels,
        random_state=42
    )
    
    # Train/val split
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_split / (1 - test_split),
        stratify=labels[train_val_idx],
        random_state=42
    )
    
    # Create masks
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Convert to tensors
    node_features = torch.FloatTensor(features)
    labels_tensor = torch.LongTensor(labels)
    edge_index_tensor = torch.LongTensor(edge_index)
    edge_weights_tensor = torch.FloatTensor(edge_weights)
    
    train_mask_tensor = torch.BoolTensor(train_mask)
    val_mask_tensor = torch.BoolTensor(val_mask)
    test_mask_tensor = torch.BoolTensor(test_mask)
    
    # Compile data dictionary
    data_dict = {
        "node_features": node_features,
        "labels": labels_tensor,
        "edge_index": edge_index_tensor,
        "edge_weights": edge_weights_tensor,
        "train_mask": train_mask_tensor,
        "val_mask": val_mask_tensor,
        "test_mask": test_mask_tensor,
        "num_nodes": n_samples,
        "num_edges": edge_index.shape[1],
        "num_features": features.shape[1],
        "num_classes": 2,
        "num_fraud": int(labels.sum()),
        "fraud_ratio": float(labels.mean()),
        "feature_names": feature_names
    }
    
    print("\n" + "=" * 50)
    print("Data preprocessing complete!")
    print("=" * 50)
    print(f"Total nodes: {data_dict['num_nodes']}")
    print(f"Total edges: {data_dict['num_edges']}")
    print(f"Features: {data_dict['num_features']}")
    print(f"Fraud cases: {data_dict['num_fraud']} ({data_dict['fraud_ratio']:.2%})")
    print(f"\nTrain: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")
    
    return data_dict


def preprocess_dgraph_data(
    data_path: str,
    test_split: float = 0.2,
    val_split: float = 0.1
) -> Dict:
    """
    Preprocess DGraph dataset (placeholder for future implementation).
    
    Args:
        data_path: Path to DGraph data directory
        test_split: Fraction for test set
        val_split: Fraction for validation set
        
    Returns:
        Dictionary containing processed data
    """
    print("DGraph preprocessing not yet implemented.")
    print("This dataset requires:")
    print("1. Download from https://dgraph.xinye.com/")
    print("2. Custom preprocessing for temporal edges")
    print("3. Handling of 3M+ nodes and 4M+ edges")
    
    raise NotImplementedError("DGraph preprocessing coming soon")


def load_processed_data(data_path: str) -> Dict:
    """
    Load preprocessed data from disk.
    
    Args:
        data_path: Path to processed data file (.pt or .pkl)
        
    Returns:
        Dictionary containing processed data
    """
    if data_path.endswith(".pt"):
        return torch.load(data_path)
    elif data_path.endswith(".pkl"):
        import pickle
        with open(data_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown file format: {data_path}")


def save_processed_data(data_dict: Dict, output_path: str):
    """
    Save preprocessed data to disk.
    
    Args:
        data_dict: Dictionary containing processed data
        output_path: Path to save file (.pt or .pkl)
    """
    if output_path.endswith(".pt"):
        torch.save(data_dict, output_path)
    elif output_path.endswith(".pkl"):
        import pickle
        with open(output_path, "wb") as f:
            pickle.dump(data_dict, f)
    else:
        raise ValueError(f"Unknown file format: {output_path}")
    
    print(f"✓ Saved processed data to {output_path}")

