import pandas as pd

# Load the two CSV files
file1 = 'results_to_be_processed/refined-context_results_v2_1-35.csv'
file2 = 'results_to_be_processed/refined-context_results_v2_36-70.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Update question IDs in the second file to start from 36
df2['question_id'] = range(36, 36 + len(df2))

# Combine the DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_file = 'results_to_be_processed/refined-context_results_v2_combined.csv'
combined_df.to_csv(output_file, index=False)

print(f"Combined CSV saved to {output_file}")
print(f"Total rows: {len(combined_df)}")
print(f"Question IDs range: {combined_df['question_id'].min()} to {combined_df['question_id'].max()}")

# Check for any invalid scores
invalid_scores = combined_df[(combined_df['overall_score'] == 1.0) | (combined_df['overall_score'] == -1)]
if len(invalid_scores) > 0:
    print(f"Warning: Found {len(invalid_scores)} rows with invalid scores (1.0 or -1).")
else:
    print("All scores are valid (no 1.0 or -1).")
