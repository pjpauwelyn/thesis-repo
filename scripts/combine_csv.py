import pandas as pd

# Load the two CSV files
file1 = 'results_to_be_processed/refined-context_results_v2_1-35.csv'
file2 = 'results_to_be_processed/refined-context_results_v2_36-70.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine the DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_file = 'results_to_be_processed/refined-context_results_v2_combined.csv'
combined_df.to_csv(output_file, index=False)

print(f"Combined CSV saved to {output_file}")
print(f"Total rows: {len(combined_df)}")
print(f"Question IDs range: {combined_df['question_id'].min()} to {combined_df['question_id'].max()}")
