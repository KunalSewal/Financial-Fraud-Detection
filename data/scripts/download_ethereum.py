"""
Script to download Ethereum Fraud Detection dataset from Kaggle.
"""

import os
import sys
import argparse
from pathlib import Path


def download_kaggle_dataset(dataset_name: str, output_dir: str):
    """
    Download dataset from Kaggle using kaggle API.
    
    Args:
        dataset_name: Kaggle dataset identifier (owner/dataset-name)
        output_dir: Directory to save downloaded files
    """
    try:
        import kaggle
        print(f"Downloading {dataset_name}...")
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True
        )
        print(f"✓ Dataset downloaded successfully to {output_dir}")
    except ImportError:
        print("ERROR: Kaggle package not installed.")
        print("Install it using: pip install kaggle")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed kaggle: pip install kaggle")
        print("2. Set up Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("3. Accepted the dataset terms on Kaggle website")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Ethereum Fraud Detection dataset from Kaggle"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../raw/ethereum",
        help="Output directory for downloaded data"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Kaggle dataset identifier
    dataset_name = "vagifa/ethereum-frauddetection-dataset"
    
    print("=" * 60)
    print("Ethereum Fraud Detection Dataset Downloader")
    print("=" * 60)
    
    download_kaggle_dataset(dataset_name, str(output_path))
    
    print("\n" + "=" * 60)
    print("Download completed!")
    print("=" * 60)
    print(f"\nFiles saved to: {output_path.absolute()}")
    
    # List downloaded files
    files = list(output_path.glob("*"))
    if files:
        print("\nDownloaded files:")
        for f in files:
            print(f"  - {f.name}")
    

if __name__ == "__main__":
    main()
