import pandas as pd

# Load the original CSV file
original_file = 'dlr_data/dlr-results.csv'
df_original = pd.read_csv(original_file)

# Check the range of idx values
print(f"Minimum idx: {df_original['idx'].min()}")
print(f"Maximum idx: {df_original['idx'].max()}")
print(f"Total rows: {len(df_original)}")

# List all unique idx values
unique_idx = df_original['idx'].unique()
print(f"\nUnique idx values: {unique_idx}")
