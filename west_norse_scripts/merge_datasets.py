import pandas as pd
import os

parquet_files = [
    'datasets/alexander_saga_ner_dataset.parquet',
    'datasets/am162b0_ner_dataset.parquet',
    'datasets/am162bk_ner_dataset.parquet',
    'datasets/drauma_jons_ner_dataset.parquet',
    'datasets/islendingabok_ner_dataset.parquet',
    'datasets/laknisbok_ner_dataset.parquet',
    'datasets/olaf_ner_dataset.parquet'
]

def merge_parquet_files(input_files, output_file):
    """
    Merge multiple parquet files into one large parquet file.

    Args:
        input_files: List of input parquet filenames
        output_file: Output filename for merged dataset
    """
    # Initialize an empty list to store DataFrames
    dfs = []

    # Read each parquet file and append to the list
    for file in input_files:
        print(f"Reading {file}...")
        df = pd.read_parquet(file)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {file}")

    # Concatenate all DataFrames
    print("Merging datasets...")
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged DataFrame to a new parquet file
    print(f"Saving merged dataset to {output_file}...")
    merged_df.to_parquet(output_file)

    print(f"Merging complete. Final dataset has {len(merged_df)} rows.")

# Example usage
output_filename = 'merged_ner_dataset.parquet'
merge_parquet_files(parquet_files, output_filename)
