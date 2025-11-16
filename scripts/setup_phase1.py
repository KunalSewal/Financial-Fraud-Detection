#!/usr/bin/env python
"""
Quick Setup Script for Phase 1.

Installs dependencies and verifies setup.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and print status."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✅ {description} - SUCCESS")
    else:
        print(f"\n❌ {description} - FAILED")
        return False
    
    return True

def main():
    """Run setup."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          PHASE 1 SETUP - Industrial Scale TGNN          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    
    if not in_venv:
        print("⚠️  WARNING: Not in virtual environment!")
        print("   Recommended: Run '.\\venv\\Scripts\\Activate.ps1' first")
        print()
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting. Please activate virtual environment and try again.")
            return False
    
    # Install W&B
    if not run_command(
        "pip install wandb",
        "Installing Weights & Biases"
    ):
        return False
    
    # Upgrade requirements (optional)
    print("\n" + "="*60)
    print("📦 Upgrading Core Dependencies (Optional)")
    print("="*60)
    response = input("Upgrade torch, pandas, scikit-learn? (y/N): ")
    
    if response.lower() == 'y':
        run_command(
            "pip install --upgrade torch pandas scikit-learn numpy",
            "Upgrading core dependencies"
        )
    
    # Create directory structure
    print("\n" + "="*60)
    print("📁 Creating Directory Structure")
    print("="*60)
    
    directories = [
        'data/dgraph',
        'data/processed',
        'checkpoints',
        'results/figures',
        'experiments/configs',
        'api',
        'tests'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")
    
    # Test imports
    print("\n" + "="*60)
    print("🧪 Testing Phase 1 Components")
    print("="*60)
    
    if not run_command(
        "python test_phase1.py",
        "Running component tests"
    ):
        print("\n⚠️  Some tests failed, but this is OK if you haven't placed data files yet")
    
    # Summary
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    
    print("""
    Next steps:
    
    1. Login to Weights & Biases (optional):
       > wandb login
       
    2. Place your DGraph data:
       > Copy edges.npy and nodes.npy to data/dgraph/
       
    3. Test temporal graph builder:
       > python -c "from src.data.temporal_graph_builder import load_and_build_temporal_graph; print('OK')"
       
    4. Test DGraph loader (after placing files):
       > python -c "from src.data.dgraph_loader import load_dgraph; load_dgraph('data/dgraph')"
       
    5. Read PHASE1_README.md for detailed usage examples
    
    6. Start experimenting! 🚀
    """)
    
    return True

if __name__ == '__main__':
    success = main()
    
    if not success:
        print("\n❌ Setup failed. Please check errors above.")
        sys.exit(1)
    else:
        print("\n🎉 All set! Happy coding!")
        sys.exit(0)
