import pandas as pd

# Load the filtered two-step results CSV file
two_step_file = 'results_to_be_processed/dlr_evaluation_results_two_step.csv'
df = pd.read_csv(two_step_file)

# Deduplicate by keeping the first occurrence of each question ID
deduplicated_df = df.drop_duplicates(subset=['question_id'], keep='first')

print(f"Total rows after deduplication: {len(deduplicated_df)}")
print(f"Unique question IDs: {deduplicated_df['question_id'].nunique()}")

# Verify that all question IDs are unique
if deduplicated_df['question_id'].nunique() == len(deduplicated_df):
    print("All question IDs are unique.")
else:
    print("Warning: Duplicate question IDs still exist.")

# Calculate mean scores per criteria for deduplicated two-step questions
mean_scores = deduplicated_df[['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']].mean()
print("\nMean Scores per Criteria (Deduplicated Two-Step Questions):")
print(mean_scores)

# Save the deduplicated results to a new CSV file
output_file = 'results_to_be_processed/dlr_evaluation_results_two_step_deduplicated.csv'
deduplicated_df.to_csv(output_file, index=False)
print(f"\nDeduplicated results saved to {output_file}")
