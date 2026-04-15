import pandas as pd

# Load the DLR evaluation results CSV file
dlr_eval_file = 'results_to_be_processed/dlr_evaluation_results.csv'
df = pd.read_csv(dlr_eval_file)

# Filter for two-step pipeline results (assuming pipeline names contain 'two_step' or similar)
two_step_df = df[df['pipeline'].str.contains('two_step', case=False, na=False)]

print(f"Total two-step questions: {len(two_step_df)}")
print(f"Unique question IDs: {two_step_df['question_id'].nunique()}")

# Display the unique question IDs
print("\nUnique Question IDs:")
print(two_step_df['question_id'].unique())

# Calculate mean scores per criteria for two-step questions
mean_scores = two_step_df[['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']].mean()
print("\nMean Scores per Criteria (Two-Step Questions):")
print(mean_scores)

# Save the filtered results to a new CSV file
output_file = 'results_to_be_processed/dlr_evaluation_results_two_step.csv'
two_step_df.to_csv(output_file, index=False)
print(f"\nFiltered results saved to {output_file}")
