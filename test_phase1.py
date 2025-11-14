"""
Phase 1 Component Tests.

Run this to verify all Phase 1 implementations work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all new modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from src.data.temporal_graph_builder import TemporalGraphBuilder, load_and_build_temporal_graph
        print("  ✅ temporal_graph_builder")
    except Exception as e:
        print(f"  ❌ temporal_graph_builder: {e}")
        return False
    
    try:
        from src.data.dgraph_loader import DGraphDataset, load_dgraph
        print("  ✅ dgraph_loader")
    except Exception as e:
        print(f"  ❌ dgraph_loader: {e}")
        return False
    
    try:
        from src.models.tgn import TGN, TimeEncoder, MemoryModule, MessageFunction
        print("  ✅ TGN model")
    except Exception as e:
        print(f"  ❌ TGN model: {e}")
        return False
    
    try:
        from src.models.mptgnn import MPTGNN, MultiPathConvolution, TemporalAttention
        print("  ✅ MPTGNN model")
    except Exception as e:
        print(f"  ❌ MPTGNN model: {e}")
        return False
    
    try:
        from experiments.experiment_runner import ExperimentRunner
        print("  ✅ experiment_runner")
    except Exception as e:
        print(f"  ❌ experiment_runner: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_temporal_graph_builder():
    """Test temporal graph builder on Ethereum data."""
    print("🔍 Testing Temporal Graph Builder...")
    
    try:
        from src.data.temporal_graph_builder import load_and_build_temporal_graph
        import pandas as pd
        
        # Check if Ethereum data exists
        data_path = Path('data/transaction_dataset.csv')
        if not data_path.exists():
            print(f"  ⚠️  Ethereum data not found at {data_path}")
            print("     Skipping test (not critical)")
            return True
        
        # Load small sample for testing
        df = pd.read_csv(data_path)
        df_sample = df.head(100)  # Just first 100 rows
        df_sample.to_csv('temp_sample.csv', index=False)
        
        # Build temporal graph
        graph = load_and_build_temporal_graph(
            'ethereum',
            'temp_sample.csv'
        )
        
        # Verify structure
        assert 'num_nodes' in graph
        assert 'edge_index' in graph
        assert 'edge_time' in graph
        assert 'edge_attr' in graph
        
        print(f"  ✅ Built graph: {graph['num_nodes']} nodes, {graph['edge_index'].size(1)} edges")
        
        # Cleanup
        import os
        os.remove('temp_sample.csv')
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dgraph_loader():
    """Test DGraph loader if data exists."""
    print("🔍 Testing DGraph Loader...")
    
    try:
        from src.data.dgraph_loader import DGraphDataset
        import numpy as np
        
        # Check if DGraph data exists
        dgraph_dir = Path('data/dgraph')
        edges_path = dgraph_dir / 'edges.npy'
        nodes_path = dgraph_dir / 'nodes.npy'
        
        if not edges_path.exists() or not nodes_path.exists():
            print(f"  ⚠️  DGraph data not found in {dgraph_dir}")
            print("     Place edges.npy and nodes.npy there to test")
            print("     Skipping test (not critical)")
            return True
        
        # Create loader
        loader = DGraphDataset(str(dgraph_dir))
        
        # Analyze structure
        loader.analyze_structure()
        
        # Process data
        data = loader.process_data()
        
        # Verify
        assert 'num_nodes' in data
        assert 'edge_index' in data
        assert 'node_features' in data
        
        print(f"  ✅ Loaded DGraph: {data['num_nodes']:,} nodes, {data['num_edges']:,} edges")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tgn_model():
    """Test TGN model instantiation and forward pass."""
    print("🔍 Testing TGN Model...")
    
    try:
        import torch
        from src.models.tgn import TGN
        
        # Create dummy data
        num_nodes = 100
        num_edges = 500
        node_dim = 10
        edge_dim = 4
        
        node_features = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_features = torch.randn(num_edges, edge_dim)
        edge_times = torch.arange(num_edges, dtype=torch.float32)
        
        # Initialize model
        model = TGN(
            num_nodes=num_nodes,
            node_dim=node_dim,
            edge_dim=edge_dim,
            memory_dim=32,
            time_dim=16,
            num_classes=2
        )
        
        # Forward pass
        logits = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_times=edge_times
        )
        
        # Verify output
        assert logits.shape == (num_nodes, 2)
        
        print(f"  ✅ TGN forward pass successful: {logits.shape}")
        
        # Test memory update
        model.reset()
        print(f"  ✅ Memory reset successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mptgnn_model():
    """Test MPTGNN model instantiation and forward pass."""
    print("🔍 Testing MPTGNN Model...")
    
    try:
        import torch
        from src.models.mptgnn import MPTGNN
        
        # Create dummy data
        num_nodes = 100
        num_edges = 500
        node_dim = 10
        edge_dim = 4
        
        node_features = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_features = torch.randn(num_edges, edge_dim)
        edge_times = torch.arange(num_edges, dtype=torch.float32)
        
        # Initialize model
        model = MPTGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=32,
            num_layers=2,
            num_paths=3,
            num_classes=2
        )
        
        # Forward pass
        logits = model(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            edge_time=edge_times
        )
        
        # Verify output
        assert logits.shape == (num_nodes, 2)
        
        print(f"  ✅ MPTGNN forward pass successful: {logits.shape}")
        
        # Test attention weights
        attention_weights = model.get_path_weights(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            edge_time=edge_times
        )
        
        assert len(attention_weights) == 2  # num_layers
        
        print(f"  ✅ Attention weights extraction successful: {len(attention_weights)} layers")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_runner():
    """Test experiment runner (offline mode)."""
    print("🔍 Testing Experiment Runner...")
    
    try:
        from experiments.experiment_runner import ExperimentRunner
        
        # Create runner in offline mode (no W&B connection needed)
        runner = ExperimentRunner(offline=True)
        
        # Initialize run
        config = {
            'model': 'test',
            'learning_rate': 0.001
        }
        
        runner.init_run(config, name="test_run")
        
        # Log metrics
        runner.log_metrics({
            'train_loss': 0.5,
            'val_f1': 0.75
        }, step=1)
        
        # Finish
        runner.finish()
        
        print(f"  ✅ Experiment runner (offline mode) working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("PHASE 1 COMPONENT TESTS")
    print("=" * 60)
    print()
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    print()
    
    results['temporal_graph'] = test_temporal_graph_builder()
    print()
    
    results['dgraph'] = test_dgraph_loader()
    print()
    
    results['tgn'] = test_tgn_model()
    print()
    
    results['mptgnn'] = test_mptgnn_model()
    print()
    
    results['experiment'] = test_experiment_runner()
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print()
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 1 implementation is ready!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
