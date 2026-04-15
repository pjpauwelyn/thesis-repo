import pandas as pd

# Load the CSV file
csv_file = 'results_final/validation_results/dlr_evaluation_results_70_unique_questions.csv'
df = pd.read_csv(csv_file)

# Sort by question_id
sorted_df = df.sort_values(by='question_id', ascending=True)

# Save the sorted CSV file
output_file = 'results_final/validation_results/dlr_evaluation_results_70_unique_questions_sorted.csv'
sorted_df.to_csv(output_file, index=False)

print(f"Sorted CSV saved to {output_file}")
print(f"Total rows: {len(sorted_df)}")
print(f"Question IDs range: {sorted_df['question_id'].min()} to {sorted_df['question_id'].max()}")
