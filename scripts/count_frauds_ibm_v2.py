import pandas as pd

fraud_count = 0
total = 0
for chunk in pd.read_csv('data/IBM_dataset/credit_card_transactions-ibm_v2.csv', usecols=['Is Fraud?'], chunksize=100000):
    fraud_count += (chunk['Is Fraud?'] == 'Yes').sum()
    total += len(chunk)
print(f"Total frauds: {fraud_count}")
print(f"Total transactions: {total}")
print(f"Fraud ratio: {fraud_count/total:.4%}")
