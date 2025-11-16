"""
Quick test to verify HMSTA architecture works.
Run this before training to catch any issues.
"""

import torch
from src.models.hmsta import create_hmsta_model, HMSTA

print("=" * 60)
print("Testing HMSTA Architecture")
print("=" * 60)

# Test 1: Model creation
print("\n✅ Test 1: Creating HMSTA model...")
try:
    model = create_hmsta_model(
        dataset_name='test',
        node_features=166,
        edge_features=4,
        hidden_dim=128,
        num_nodes=1000
    )
    print("   ✅ Model created successfully!")
    print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 2: Forward pass
print("\n✅ Test 2: Testing forward pass...")
try:
    # Create dummy data
    x = torch.randn(100, 166)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 4)
    timestamps = torch.linspace(0, 100, 500)
    
    output = model(x, edge_index, edge_attr, timestamps)
    
    print(f"   ✅ Forward pass successful!")
    print(f"   • Output logits shape: {output['logits'].shape}")
    print(f"   • Embeddings shape: {output['embeddings'].shape}")
    print(f"   • Path weights shape: {output['path_weights'].shape}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 3: Prediction with explanation
print("\n✅ Test 3: Testing prediction with explanation...")
try:
    pred = model.explain_prediction(0, x, edge_index, edge_attr, timestamps)
    
    print(f"   ✅ Explanation generated!")
    print(f"   • Prediction: {pred['prediction']}")
    print(f"   • Confidence: {pred['confidence']:.2%}")
    print(f"   • Top reason: {pred['top_reasons'][0]['category']}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 4: Check components
print("\n✅ Test 4: Verifying architecture components...")
try:
    from src.models.hmsta import AnomalyAttention, TemporalExplainer, PathAggregator
    
    anomaly_attn = AnomalyAttention(128, num_heads=4)
    explainer = TemporalExplainer(128)
    path_agg = PathAggregator(128, num_paths=3)
    
    print("   ✅ All components work!")
    print("   • AnomalyAttention: ✓")
    print("   • TemporalExplainer: ✓")
    print("   • PathAggregator: ✓")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nHMSTA architecture is ready to use.")
print("\nNext steps:")
print("  1. Run: python train_hmsta.py")
print("  2. Run: python compare_models.py")
print("  3. Present results!")
print()
