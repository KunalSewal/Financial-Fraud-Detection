"""
Test script to verify project setup and imports.

Run this to check if all dependencies are installed correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    
    print("Testing imports...")
    print("=" * 60)
    
    tests = {
        "NumPy": "import numpy",
        "Pandas": "import pandas",
        "PyTorch": "import torch",
        "Scikit-learn": "from sklearn.metrics import accuracy_score",
        "NetworkX": "import networkx",
        "Matplotlib": "import matplotlib.pyplot",
        "Seaborn": "import seaborn",
        "YAML": "import yaml",
        "TQDM": "from tqdm import tqdm",
    }
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests.items():
        try:
            exec(import_stmt)
            print(f"✓ {name:20s} OK")
            passed += 1
        except ImportError as e:
            print(f"✗ {name:20s} FAILED - {str(e)}")
            failed += 1
    
    print("=" * 60)
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n⚠️  Some packages are missing!")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All core dependencies are installed!")
        return True


def test_pytorch_geometric():
    """Test PyTorch Geometric installation."""
    
    print("\n" + "=" * 60)
    print("Testing PyTorch Geometric...")
    print("=" * 60)
    
    try:
        import torch_geometric
        from torch_geometric.nn import SAGEConv, GATConv
        print("✓ PyTorch Geometric OK")
        return True
    except ImportError as e:
        print(f"✗ PyTorch Geometric FAILED - {str(e)}")
        print("\nInstallation instructions:")
        print("1. Visit: https://pytorch-geometric.readthedocs.io/")
        print("2. Follow installation guide for your system")
        return False


def test_project_structure():
    """Test if project structure is correct."""
    
    print("\n" + "=" * 60)
    print("Testing project structure...")
    print("=" * 60)
    
    required_paths = [
        "src/__init__.py",
        "src/data_utils.py",
        "src/models.py",
        "src/train.py",
        "src/evaluate.py",
        "configs/config.yaml",
        "data/README.md",
        "main.py",
        "setup_data.py",
        "requirements.txt"
    ]
    
    passed = 0
    failed = 0
    
    for path in required_paths:
        if Path(path).exists():
            print(f"✓ {path:30s} exists")
            passed += 1
        else:
            print(f"✗ {path:30s} MISSING")
            failed += 1
    
    print("=" * 60)
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n⚠️  Some files are missing!")
        return False
    else:
        print("\n✓ Project structure is complete!")
        return True


def test_dataset():
    """Test if dataset is available."""
    
    print("\n" + "=" * 60)
    print("Testing dataset...")
    print("=" * 60)
    
    data_path = Path("data/transaction_dataset.csv")
    
    if data_path.exists():
        print(f"✓ Dataset found: {data_path}")
        
        # Try to load it
        try:
            import pandas as pd
            df = pd.read_csv(data_path, nrows=5)
            print(f"✓ Dataset can be loaded")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Sample shape: {df.shape}")
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {str(e)}")
            return False
    else:
        print(f"✗ Dataset not found: {data_path}")
        print("\nThe Ethereum dataset should be at:")
        print(f"  {data_path.absolute()}")
        return False


def test_src_imports():
    """Test if src modules can be imported."""
    
    print("\n" + "=" * 60)
    print("Testing src modules...")
    print("=" * 60)
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent))
    
    tests = [
        ("data_utils", "from src import data_utils"),
        ("models", "from src import models"),
        ("train", "from src import train"),
        ("evaluate", "from src import evaluate"),
    ]
    
    passed = 0
    failed = 0
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✓ src.{name:15s} OK")
            passed += 1
        except Exception as e:
            print(f"✗ src.{name:15s} FAILED - {str(e)}")
            failed += 1
    
    print("=" * 60)
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n⚠️  Some modules have errors!")
        print("This might be due to missing dependencies.")
        return False
    else:
        print("\n✓ All src modules can be imported!")
        return True


def main():
    """Run all tests."""
    
    print("\n" + "=" * 70)
    print("Financial Fraud Detection - Setup Verification")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Dependencies", test_imports()))
    results.append(("PyTorch Geometric", test_pytorch_geometric()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Dataset", test_dataset()))
    results.append(("Src Modules", test_src_imports()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:25s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All tests passed! Your project is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python setup_data.py")
        print("  2. Run: python main.py --model mlp")
        print("  3. Check results in results/ folder")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Verify dataset location: data/transaction_dataset.csv")
        print("  - Check PyTorch Geometric installation")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
