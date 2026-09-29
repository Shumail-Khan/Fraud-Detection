"""
Create a stratified sample from the large PaySim CSV dataset.
Extracts a balanced distribution of fraud and non-fraud transactions
so developers can run tests, pipelines, and demos immediately without
downloading the 500MB dataset.
"""

import os
import sys
import pandas as pd


def create_sample(
    input_path: str,
    output_path: str,
    n_normal: int = 15000,
    n_fraud: int = 5000,
    random_state: int = 42,
) -> None:
    if not os.path.exists(input_path):
        print(f"[!] Input file '{input_path}' not found. Cannot generate sample.")
        return

    print(f"Reading dataset from {input_path} in chunks to conserve memory...")
    chunks = []
    chunksize = 200_000
    
    fraud_records = []
    normal_records = []
    
    total_fraud_collected = 0
    total_normal_collected = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        frauds = chunk[chunk["isFraud"] == 1]
        normals = chunk[chunk["isFraud"] == 0]

        if total_fraud_collected < n_fraud and not frauds.empty:
            needed_f = n_fraud - total_fraud_collected
            sample_f = frauds.sample(min(len(frauds), needed_f), random_state=random_state)
            fraud_records.append(sample_f)
            total_fraud_collected += len(sample_f)

        if total_normal_collected < n_normal and not normals.empty:
            needed_n = n_normal - total_normal_collected
            sample_n = normals.sample(min(len(normals), needed_n), random_state=random_state)
            normal_records.append(sample_n)
            total_normal_collected += len(sample_n)

        if total_fraud_collected >= n_fraud and total_normal_collected >= n_normal:
            break

    if not fraud_records and not normal_records:
        print("[!] No records extracted.")
        return

    sample_df = pd.concat(fraud_records + normal_records, ignore_index=True)
    sample_df = sample_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df.to_csv(output_path, index=False)
    print(f"[+] Successfully created sample dataset with {len(sample_df)} rows at: {output_path}")
    print(f"    - Fraud transactions: {sample_df['isFraud'].sum()}")
    print(f"    - Non-fraud transactions: {len(sample_df) - sample_df['isFraud'].sum()}")
    print(f"    - File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "PS_20174392719_1491204439457_log.csv"
    if not os.path.exists(raw_path):
        raw_path = os.path.join("data", "raw", "PS_20174392719_1491204439457_log.csv")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", "sample", "sample_transactions.csv")
    create_sample(raw_path, out_path)
