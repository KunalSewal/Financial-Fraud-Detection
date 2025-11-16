"""
Script to download DGraph dataset (NeurIPS 2022).

Note: DGraph requires registration and manual download from:
https://dgraph.xinye.com/
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="DGraph Dataset Download Instructions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../raw/dgraph",
        help="Output directory for dataset"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DGraph Dataset (NeurIPS 2022) - Manual Download Instructions")
    print("=" * 70)
    print("\nDGraph is a large-scale dataset that requires manual download.")
    print("\nSteps to download:")
    print("1. Visit: https://dgraph.xinye.com/")
    print("2. Register for an account")
    print("3. Request dataset access")
    print("4. Download the dataset files")
    print(f"5. Extract files to: {output_path.absolute()}")
    print("\nExpected files:")
    print("  - edges.csv (transaction edges with timestamps)")
    print("  - nodes.csv (node features and labels)")
    print("  - metadata.json (dataset metadata)")
    print("\nDataset size: ~2-3 GB")
    print("Processing time: May take several hours due to size")
    print("=" * 70)
    

if __name__ == "__main__":
    main()
