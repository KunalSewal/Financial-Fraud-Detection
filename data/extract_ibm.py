"""
Extract and explore IBM transactions dataset
"""

import tarfile
import os
import pandas as pd

print("="*80)
print("Extracting IBM Transactions Dataset")
print("="*80)

# Extract the .tgz file
tgz_path = 'data/transactions.tgz'
extract_to = 'data/ibm/'

print(f"\n📦 Extracting {tgz_path}...")

# Create extraction directory
os.makedirs(extract_to, exist_ok=True)

# Extract
with tarfile.open(tgz_path, 'r:gz') as tar:
    tar.extractall(path=extract_to)
    print(f"✅ Extracted to {extract_to}")
    
    # List extracted files
    members = tar.getmembers()
    print(f"\n📁 Extracted files ({len(members)} total):")
    for member in members[:20]:  # Show first 20
        print(f"   - {member.name}")
    if len(members) > 20:
        print(f"   ... and {len(members) - 20} more files")

# Find CSV files
print(f"\n🔍 Looking for CSV files...")
csv_files = []
for root, dirs, files in os.walk(extract_to):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

if csv_files:
    print(f"\n✅ Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"\n📄 {csv_file}")
        
        # Load and inspect first CSV
        df = pd.read_csv(csv_file, nrows=5)
        
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n   First few rows:")
        print(df.head())
        
        # Check for key columns
        print(f"\n   🔍 Checking for transaction-level data:")
        key_columns = {
            'timestamp': ['timestamp', 'time', 'date', 'datetime', 'Timestamp'],
            'sender': ['sender', 'from', 'source', 'From Account', 'nameOrig'],
            'receiver': ['receiver', 'to', 'destination', 'To Account', 'nameDest'],
            'amount': ['amount', 'value', 'Amount'],
            'label': ['label', 'isFraud', 'isfraud', 'fraud', 'FLAG']
        }
        
        found_columns = {}
        for key, possible_names in key_columns.items():
            for col in df.columns:
                if col in possible_names or col.lower() in [n.lower() for n in possible_names]:
                    found_columns[key] = col
                    print(f"   ✅ {key.upper()}: '{col}'")
                    break
            if key not in found_columns:
                print(f"   ❌ {key.upper()}: Not found")
        
        # Load full dataset stats
        print(f"\n   📊 Loading full dataset statistics...")
        df_full = pd.read_csv(csv_file)
        print(f"   Total rows: {len(df_full):,}")
        print(f"   Total columns: {len(df_full.columns)}")
        
        if 'label' in found_columns or 'isFraud' in df_full.columns:
            label_col = found_columns.get('label', 'isFraud')
            fraud_count = df_full[label_col].sum()
            normal_count = len(df_full) - fraud_count
            fraud_pct = (fraud_count / len(df_full)) * 100
            print(f"   Fraud transactions: {fraud_count:,} ({fraud_pct:.2f}%)")
            print(f"   Normal transactions: {normal_count:,}")
        
        print(f"\n   ✅ This looks like {'TRANSACTION-LEVEL' if 'timestamp' in found_columns else 'AGGREGATED'} data!")

else:
    print("❌ No CSV files found. Checking for other formats...")
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            print(f"   - {os.path.join(root, file)}")

print("\n" + "="*80)
print("Extraction complete! Review the output above.")
print("="*80)
