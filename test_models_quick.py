"""Quick test to verify all HMSTA versions can forward pass without errors"""
import torch
from src.models.hmsta_v2 import (
    HMSTA_v0_Baseline,
    HMSTA_v1_Graph,
    HMSTA_v2_Temporal,
    HMSTA_v3_Memory,
    HMSTA_v4_MultiPath,
    HMSTA_v5_Full
)
from data.load_ibm_graph import load_ibm_transaction_graph

print("🔍 Loading IBM graph (small sample)...")
data = load_ibm_transaction_graph(
    csv_path='data/ibm/card_transaction.v1.csv',
    sample_size=100000,
    min_transactions_per_user=20
)

print(f"\n📊 Data loaded:")
print(f"   Nodes: {data.x.size(0)}")
print(f"   Edges: {data.edge_index.size(1)}")
print(f"   Node timestamps: {data.timestamps.size()}")
print(f"   Edge timestamps: {data.edge_timestamps.size()}")

node_features = data.x.size(1)
hidden_dim = 64

models = [
    ("v0 (Baseline)", HMSTA_v0_Baseline(node_features, hidden_dim)),
    ("v1 (Graph)", HMSTA_v1_Graph(node_features, hidden_dim)),
    ("v2 (Temporal)", HMSTA_v2_Temporal(node_features, hidden_dim)),
    ("v3 (Memory)", HMSTA_v3_Memory(node_features, hidden_dim)),
    ("v4 (MultiPath)", HMSTA_v4_MultiPath(node_features, hidden_dim)),
    ("v5 (Full)", HMSTA_v5_Full(node_features, hidden_dim)),
]

print(f"\n🧪 Testing all versions with edge_timestamps...")
for name, model in models:
    try:
        model.eval()
        with torch.no_grad():
            # Test with edge_timestamps (what training code will use)
            logits = model(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.edge_timestamps  # ← Using edge timestamps
            )
        print(f"   ✅ {name}: Output shape {logits.shape}")
    except Exception as e:
        print(f"   ❌ {name}: {str(e)[:100]}")

print(f"\n🧪 Testing all versions with node timestamps...")
for name, model in models:
    try:
        model.eval()
        with torch.no_grad():
            # Test with node timestamps
            logits = model(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.timestamps  # ← Using node timestamps
            )
        print(f"   ✅ {name}: Output shape {logits.shape}")
    except Exception as e:
        print(f"   ❌ {name}: {str(e)[:100]}")

print(f"\n✅ All tests complete!")
