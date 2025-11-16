"""Debug IBM graph structure to understand why complex models fail"""
import torch
from data.load_ibm_graph import load_ibm_transaction_graph

print("=" * 80)
print("DEBUGGING IBM GRAPH STRUCTURE")
print("=" * 80)

# Load graph
data = load_ibm_transaction_graph(
    csv_path='data/ibm/card_transaction.v1.csv',
    sample_size=1000000,
    min_transactions_per_user=50
)

print(f"\n📊 Basic Stats:")
print(f"   Nodes: {data.x.size(0)}")
print(f"   Edges: {data.edge_index.size(1)}")
print(f"   Features per node: {data.x.size(1)}")
print(f"   Fraud rate: {data.y.float().mean()*100:.2f}%")

print(f"\n🔗 Graph Connectivity:")
# Check node degrees
src, dst = data.edge_index
in_degrees = torch.bincount(dst, minlength=data.x.size(0))
out_degrees = torch.bincount(src, minlength=data.x.size(0))
total_degrees = in_degrees + out_degrees

print(f"   Avg degree: {total_degrees.float().mean():.1f}")
print(f"   Min degree: {total_degrees.min()}")
print(f"   Max degree: {total_degrees.max()}")
print(f"   Nodes with degree 0: {(total_degrees == 0).sum()}")
print(f"   Nodes with degree > 1000: {(total_degrees > 1000).sum()}")

print(f"\n⏰ Temporal Distribution:")
print(f"   Edge timestamps: min={data.edge_timestamps.min():.4f}, max={data.edge_timestamps.max():.4f}")
print(f"   Node timestamps: min={data.timestamps.min():.4f}, max={data.timestamps.max():.4f}")

# Check temporal spread
temporal_range = data.edge_timestamps.max() - data.edge_timestamps.min()
print(f"   Temporal range: {temporal_range:.4f}")

# Check how many unique timestamp values
unique_edge_times = torch.unique(data.edge_timestamps)
print(f"   Unique edge timestamps: {len(unique_edge_times)}")

print(f"\n🎯 Fraud vs Normal:")
fraud_mask = data.y == 1
normal_mask = data.y == 0

print(f"   Fraud nodes: {fraud_mask.sum()}")
print(f"   Normal nodes: {normal_mask.sum()}")

# Check if fraud nodes have different temporal patterns
if fraud_mask.sum() > 0 and normal_mask.sum() > 0:
    fraud_times = data.timestamps[fraud_mask]
    normal_times = data.timestamps[normal_mask]
    
    print(f"   Fraud avg time: {fraud_times.mean():.4f}")
    print(f"   Normal avg time: {normal_times.mean():.4f}")
    print(f"   Time difference: {abs(fraud_times.mean() - normal_times.mean()):.4f}")

# Check train/val/test split
print(f"\n✂️ Data Splits:")
print(f"   Train: {data.train_mask.sum()} ({data.y[data.train_mask].float().mean()*100:.1f}% fraud)")
print(f"   Val: {data.val_mask.sum()} ({data.y[data.val_mask].float().mean()*100:.1f}% fraud)")
print(f"   Test: {data.test_mask.sum()} ({data.y[data.test_mask].float().mean()*100:.1f}% fraud)")

print(f"\n🔍 Feature Analysis:")
print(f"   Feature matrix shape: {data.x.shape}")
print(f"   Feature means: {data.x.mean(dim=0)}")
print(f"   Feature stds: {data.x.std(dim=0)}")

# Check if features are already perfect for classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

X_train = data.x[data.train_mask].numpy()
y_train = data.y[data.train_mask].numpy()
X_test = data.x[data.test_mask].numpy()
y_test = data.y[data.test_mask].numpy()

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
f1 = f1_score(y_test, y_pred)

print(f"\n🎯 Simple Logistic Regression (features only):")
print(f"   Test F1: {f1:.4f}")
print(f"   → If this is high, features alone are enough!")

print(f"\n💡 KEY INSIGHTS:")
if data.x.size(0) < 100:
    print(f"   ⚠️  VERY SMALL GRAPH ({data.x.size(0)} nodes)")
    print(f"   → GNN models need more nodes to learn patterns")
    print(f"   → Consider using min_transactions_per_user=10 instead of 50")

if total_degrees.max() > 10000:
    print(f"   ⚠️  Some nodes are super-connected (degree {total_degrees.max()})")
    print(f"   → This creates hub nodes that dominate message passing")

if len(unique_edge_times) < 100:
    print(f"   ⚠️  Very few unique timestamps ({len(unique_edge_times)})")
    print(f"   → Temporal models can't learn fine-grained patterns")

if f1 > 0.9:
    print(f"   ⚠️  Features alone achieve {f1:.1%} F1!")
    print(f"   → Graph/temporal components may not add much value")
    print(f"   → This is similar to Ethereum: features are too good!")

print("=" * 80)
