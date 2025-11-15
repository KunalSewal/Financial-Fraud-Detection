import pandas as pd

# File paths
input_path = 'data/ibm/card_transaction.v1.csv'
output_path = 'data/ibm/ibm_fraud_29k_nonfraud_60k.csv'

# Read in chunks to avoid memory issues
fraud_rows = []
nonfraud_rows = []
nonfraud_needed = 60000

for chunk in pd.read_csv(input_path, chunksize=100000):
    fraud_chunk = chunk[chunk['Is Fraud?'] == 'Yes']
    nonfraud_chunk = chunk[chunk['Is Fraud?'] == 'No']
    fraud_rows.append(fraud_chunk)
    if len(nonfraud_rows) < nonfraud_needed:
        nonfraud_rows.append(nonfraud_chunk)

# Concatenate all frauds
fraud_df = pd.concat(fraud_rows, ignore_index=True)
# Concatenate all non-frauds, then sample 60k
nonfraud_df = pd.concat(nonfraud_rows, ignore_index=True)
nonfraud_sample = nonfraud_df.sample(n=nonfraud_needed, random_state=42)

# Combine and shuffle
final_df = pd.concat([fraud_df, nonfraud_sample], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
final_df.to_csv(output_path, index=False)
print(f"Saved new dataset with {len(fraud_df)} fraud and {len(nonfraud_sample)} non-fraud to {output_path}")
