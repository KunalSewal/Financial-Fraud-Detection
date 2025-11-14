"""
Analyze IBM card transactions dataset in detail
"""

import pandas as pd
import numpy as np

print("="*80)
print("IBM Card Transactions Dataset - Detailed Analysis")
print("="*80)

csv_path = 'data/ibm/card_transaction.v1.csv'

print("\n📊 Loading dataset (this may take a moment with 24M rows)...")
# Load sample first to understand structure
df_sample = pd.read_csv(csv_path, nrows=100000)

print(f"\n✅ Loaded 100K sample transactions")
print(f"   Columns: {list(df_sample.columns)}")

# Check fraud distribution
print(f"\n🎯 Fraud Distribution (in sample):")
fraud_values = df_sample['Is Fraud?'].value_counts()
print(fraud_values)

if 'Yes' in fraud_values.index or 'No' in fraud_values.index:
    fraud_count = len(df_sample[df_sample['Is Fraud?'] == 'Yes'])
    normal_count = len(df_sample[df_sample['Is Fraud?'] == 'No'])
    fraud_pct = (fraud_count / len(df_sample)) * 100
    print(f"\n   Fraud: {fraud_count:,} ({fraud_pct:.2f}%)")
    print(f"   Normal: {normal_count:,} ({100-fraud_pct:.2f}%)")

# Check temporal range
print(f"\n📅 Temporal Information:")
print(f"   Years: {df_sample['Year'].min()} - {df_sample['Year'].max()}")
print(f"   Months: {df_sample['Month'].unique()}")
print(f"   Sample times: {df_sample['Time'].head()}")

# Check unique users and cards
print(f"\n👥 Users and Cards:")
print(f"   Unique Users (in sample): {df_sample['User'].nunique():,}")
print(f"   Unique Cards (in sample): {df_sample['Card'].nunique():,}")
print(f"   Transactions per user (avg): {len(df_sample) / df_sample['User'].nunique():.1f}")

# Check amounts
print(f"\n💰 Transaction Amounts:")
# Convert Amount to numeric (might have $ signs)
df_sample['Amount_numeric'] = pd.to_numeric(df_sample['Amount'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
print(f"   Min: ${df_sample['Amount_numeric'].min():.2f}")
print(f"   Max: ${df_sample['Amount_numeric'].max():.2f}")
print(f"   Mean: ${df_sample['Amount_numeric'].mean():.2f}")
print(f"   Median: ${df_sample['Amount_numeric'].median():.2f}")

# Check for missing values
print(f"\n🔍 Missing Values:")
missing = df_sample.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("   No missing values!")

# Suggest graph construction approach
print(f"\n" + "="*80)
print("💡 How We'll Build the Transaction Graph:")
print("="*80)
print("""
Option 1: User-Merchant Graph (RECOMMENDED)
    Nodes: Users + Merchants
    Edges: User → Merchant (transaction)
    Timestamps: Transaction time
    Edge features: Amount, MCC, etc.
    Labels: Fraudulent users

Option 2: Card-Merchant Graph
    Nodes: Cards + Merchants  
    Edges: Card → Merchant
    Similar to Option 1

Option 3: User Similarity Graph
    Nodes: Users only
    Edges: Users with similar transaction patterns
    Temporal: Sequence of transactions per user
    Labels: Fraudulent users
    
Recommended: Option 1 (most natural for fraud detection)
""")

# Quick fraud pattern check
print(f"\n🔬 Quick Fraud Pattern Analysis:")
fraud_df = df_sample[df_sample['Is Fraud?'] == 'Yes']
normal_df = df_sample[df_sample['Is Fraud?'] == 'No']

if len(fraud_df) > 0:
    print(f"\n   Fraud transactions:")
    fraud_df['Amount_numeric'] = pd.to_numeric(fraud_df['Amount'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    normal_df['Amount_numeric'] = pd.to_numeric(normal_df['Amount'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    
    print(f"   - Average amount: ${fraud_df['Amount_numeric'].mean():.2f}")
    print(f"   - Median amount: ${fraud_df['Amount_numeric'].median():.2f}")
    print(f"   - Top 5 MCCs: {fraud_df['MCC'].value_counts().head()}")
    
    print(f"\n   Normal transactions:")
    print(f"   - Average amount: ${normal_df['Amount_numeric'].mean():.2f}")
    print(f"   - Median amount: ${normal_df['Amount_numeric'].median():.2f}")

print(f"\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)
print("\nNext: I'll create a proper graph loader for this dataset")
print("Expected: v3-v5 (Memory, Multi-Path, Attention) should work well!")
