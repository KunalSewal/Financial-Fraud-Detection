import pandas as pd
p='data/IBM_dataset/credit_card_transactions-ibm_v2.csv'
chunksize=10**6
fraud_total=0
rows=0
for i,chunk in enumerate(pd.read_csv(p, usecols=['Is Fraud?'], chunksize=chunksize, low_memory=False)):
    # Column uses 'Yes'/'No' strings; map to 1/0 safely
    col = chunk['Is Fraud?']
    if col.dtype == object:
        vals = col.map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    else:
        vals = col.astype(int)
    fraud_total += vals.sum()
    rows += len(chunk)
    if (i+1) % 5 == 0:
        print(f'Processed {(i+1)*chunksize:,} rows...')

print('\nDone.')
print(f'Total rows: {rows:,}')
print(f'Total fraud: {fraud_total:,}')
print(f'Non-fraud: {rows - fraud_total:,}')
print(f'Fraud rate: {fraud_total/rows*100:.4f}%')
print(f'Imbalance ratio: {(rows - fraud_total)/fraud_total:.0f}:1')
