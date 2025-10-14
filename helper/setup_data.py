"""
Quick setup and data preprocessing script.

This script prepares the Ethereum fraud dataset for training.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_utils import preprocess_ethereum_data
import torch


def main():
    print("=" * 70)
    print("Financial Fraud Detection - Data Setup")
    print("=" * 70)
    
    # Check if dataset exists
    data_path = "data/transaction_dataset.csv"
    
    if not os.path.exists(data_path):
        print("\nERROR: Dataset not found!")
        print(f"Expected location: {data_path}")
        print("\nPlease download the Ethereum dataset:")
        print("1. Visit: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset")
        print("2. Download transaction_dataset.csv")
        print(f"3. Place it in {data_path}")
        print("\nOr use the download script:")
        print("  python data/scripts/download_ethereum.py")
        sys.exit(1)
    
    print(f"\n✓ Found dataset: {data_path}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    print("This may take a few minutes...")
    
    data_dict = preprocess_ethereum_data(
        data_path,
        test_split=0.2,
        val_split=0.1,
        graph_method="knn",
        k_neighbors=10
    )
    
    # Save processed data
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "ethereum_processed.pt")
    torch.save(data_dict, output_path)
    
    print(f"\n✓ Saved processed data to: {output_path}")
    
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print("\nYou can now train models using:")
    print("  python main.py --model mlp")
    print("  python main.py --model graphsage")
    print("  python main.py --model tgn")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
