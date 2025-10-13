"""
Preprocessing script for financial fraud datasets.
Converts raw data into PyTorch Geometric format.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_utils import (
    preprocess_ethereum_data,
    preprocess_dgraph_data,
    create_graph_from_transactions
)


def save_processed_data(data_dict, output_dir, dataset_name):
    """Save processed data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    with open(output_path / f"{dataset_name}_processed.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    
    # Save PyTorch tensors
    torch.save(data_dict, output_path / f"{dataset_name}_processed.pt")
    
    print(f"✓ Saved processed data to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess financial fraud datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ethereum", "dgraph", "figraph"],
        help="Dataset to preprocess"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing raw data"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Test set split ratio"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation set split ratio"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Preprocessing {args.dataset.upper()} Dataset")
    print("=" * 70)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_path}")
        sys.exit(1)
    
    # Preprocess based on dataset type
    if args.dataset == "ethereum":
        print("\nLoading Ethereum dataset...")
        data_dict = preprocess_ethereum_data(
            str(input_path),
            test_split=args.test_split,
            val_split=args.val_split
        )
    elif args.dataset == "dgraph":
        print("\nLoading DGraph dataset...")
        data_dict = preprocess_dgraph_data(
            str(input_path),
            test_split=args.test_split,
            val_split=args.val_split
        )
    elif args.dataset == "figraph":
        print("\nFiGraph preprocessing not yet implemented.")
        print("Dataset will be available after WWW 2025 publication.")
        sys.exit(1)
    
    # Save processed data
    print("\nSaving processed data...")
    save_processed_data(data_dict, args.output, args.dataset)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"\nDataset: {args.dataset}")
    print(f"Number of nodes: {data_dict['num_nodes']}")
    print(f"Number of edges: {data_dict['num_edges']}")
    print(f"Number of features: {data_dict['num_features']}")
    print(f"Number of fraud cases: {data_dict['num_fraud']}")
    print(f"Fraud ratio: {data_dict['fraud_ratio']:.2%}")
    print(f"\nTrain samples: {len(data_dict['train_mask'].nonzero()[0])}")
    print(f"Val samples: {len(data_dict['val_mask'].nonzero()[0])}")
    print(f"Test samples: {len(data_dict['test_mask'].nonzero()[0])}")
    

if __name__ == "__main__":
    main()
