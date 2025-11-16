"""Debug script to check feature quality."""

import torch
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/transaction_dataset.csv')

print("=" * 60)
print("DATA ANALYSIS")
print("=" * 60)

print(f"\n✅ Dataset shape: {df.shape}")
print(f"✅ Unique addresses: {len(df['Address'].unique())}")
print(f"✅ Fraud count: {df['FLAG'].sum()} ({df['FLAG'].mean()*100:.2f}%)")

# Check numeric columns
numeric_cols = [c for c in df.columns if c not in ['Index', 'Address', 'FLAG'] and pd.api.types.is_numeric_dtype(df[c])]
print(f"\n✅ Numeric feature columns: {len(numeric_cols)}")

# Check for NaN
print(f"\n📊 NaN values per column:")
for col in numeric_cols[:10]:  # First 10 columns
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        print(f"   {col}: {nan_count} ({nan_count/len(df)*100:.1f}%)")

# Check value ranges
print(f"\n📊 Sample statistics (first 5 columns):")
for col in numeric_cols[:5]:
    vals = df[col].dropna()
    if len(vals) > 0:
        print(f"   {col}:")
        print(f"      Min: {vals.min():.4f}, Max: {vals.max():.4f}, Mean: {vals.mean():.4f}")

# Test feature extraction for one address
test_addr = df['Address'].iloc[0]
addr_data = df[df['Address'] == test_addr]

features = []
for col in numeric_cols:
    val = addr_data[col].mean()
    if pd.isna(val) or np.isinf(val):
        val = 0.0
    features.append(val)

print(f"\n✅ Features extracted for one address: {len(features)}")
print(f"   Non-zero features: {sum(1 for f in features if f != 0.0)}")
print(f"   Sample values: {features[:10]}")

# Check if features are all zeros
if all(f == 0.0 for f in features):
    print("\n⚠️  WARNING: All features are ZERO!")
else:
    print(f"\n✅ Features have values (range: {min(features):.4f} to {max(features):.4f})")
