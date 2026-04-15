import pandas as pd

# Load the deduplicated two-step results CSV file
deduplicated_file = 'results_to_be_processed/dlr_evaluation_results_two_step_deduplicated.csv'
df = pd.read_csv(deduplicated_file)

# Filter for the first 70 questions
filtered_df = df.head(70)

print(f"Total rows after filtering: {len(filtered_df)}")
print(f"Unique question IDs: {filtered_df['question_id'].nunique()}")

# Verify that all question IDs are unique
if filtered_df['question_id'].nunique() == len(filtered_df):
    print("All question IDs are unique.")
else:
    print("Warning: Duplicate question IDs exist.")

# Calculate mean scores per criteria for the filtered questions
mean_scores = filtered_df[['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']].mean()
print("\nMean Scores per Criteria (70 Two-Step Questions):")
print(mean_scores)

# Save the filtered results to a new CSV file
output_file = 'results_to_be_processed/dlr_evaluation_results_70_questions.csv'
filtered_df.to_csv(output_file, index=False)
print(f"\nFiltered results saved to {output_file}")
