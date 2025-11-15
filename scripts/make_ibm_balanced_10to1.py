import pandas as pd
import numpy as np

RAW_PATH = 'data/IBM_dataset/credit_card_transactions-ibm_v2.csv'
OUT_PATH = 'data/IBM_dataset/ibm_balanced_10to1.csv'
FRAUD_COL = 'Is Fraud?'
RATIO = 10  # 10:1 non-fraud:fraud

print('Loading fraud rows...')
df_iter = pd.read_csv(RAW_PATH, usecols=None, dtype=str, chunksize=500_000, low_memory=False)
fraud_chunks = []
nonfraud_chunks = []
for chunk in df_iter:
    fraud_mask = chunk[FRAUD_COL] == 'Yes'
    fraud_chunks.append(chunk[fraud_mask])
    nonfraud_chunks.append(chunk[~fraud_mask])

df_fraud = pd.concat(fraud_chunks, ignore_index=True)
df_nonfraud = pd.concat(nonfraud_chunks, ignore_index=True)
print(f"Fraud: {len(df_fraud):,} | Non-fraud: {len(df_nonfraud):,}")

n_fraud = len(df_fraud)
n_nonfraud = n_fraud * RATIO

print(f"Sampling {n_nonfraud:,} non-fraud (10x)...")
df_nonfraud_sampled = df_nonfraud.sample(n=n_nonfraud, random_state=42)

print("Combining and shuffling...")
df_balanced = pd.concat([df_fraud, df_nonfraud_sampled], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Saving to {OUT_PATH} ...")
df_balanced.to_csv(OUT_PATH, index=False)
print("Done! Balanced dataset shape:", df_balanced.shape)
