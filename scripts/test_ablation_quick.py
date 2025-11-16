"""Quick test of HMSTA v2 models"""

import torch
import torch.nn.functional as F
from src.models.hmsta_v2 import create_hmsta_model, VERSION_DESCRIPTIONS

# Create small synthetic data
num_nodes = 100
num_edges = 500
node_features = 166

x = torch.randn(num_nodes, node_features)
edge_index = torch.randint(0, num_nodes, (2, num_edges))
edge_attr = torch.ones(num_edges, 1)
timestamps = torch.rand(num_edges)
labels = torch.randint(0, 2, (num_nodes,))

print("Testing HMSTA v2 models on synthetic data")
print("=" * 60)

for version in range(6):
    print(f"\nTesting v{version}: {VERSION_DESCRIPTIONS[version]}")
    
    try:
        # Create model
        model = create_hmsta_model(version, node_features, hidden_dim=64)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
        
        # Reset memory for v3+
        if version >= 3:
            model.reset_memory(num_nodes, torch.device('cpu'))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index, edge_attr, timestamps)
        
        # Check output shape
        assert logits.shape == (num_nodes, 2), f"Wrong output shape: {logits.shape}"
        
        # Check for NaN
        assert not torch.isnan(logits).any(), "NaN in output!"
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        print(f"  Output shape: {logits.shape} ✓")
        print(f"  Loss: {loss.item():.4f} ✓")
        print(f"  ✅ PASS")
        
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All models tested!")
