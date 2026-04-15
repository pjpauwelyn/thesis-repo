import pandas as pd

# Load the DLR evaluation results CSV file
dlr_eval_file = 'results_to_be_processed/dlr_evaluation_results.csv'
df = pd.read_csv(dlr_eval_file)

# Deduplicate by keeping the first occurrence of each question
deduplicated_df = df.drop_duplicates(subset=['question'], keep='first')

print(f"Total rows after deduplication: {len(deduplicated_df)}")
print(f"Unique questions: {deduplicated_df['question'].nunique()}")

# Verify that all questions are unique
if deduplicated_df['question'].nunique() == len(deduplicated_df):
    print("All questions are unique.")
else:
    print("Warning: Duplicate questions still exist.")

# Calculate mean scores per criteria for the deduplicated questions
mean_scores = deduplicated_df[['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']].mean()
print("\nMean Scores per Criteria (70 Unique Questions):")
print(mean_scores)

# Save the deduplicated results to a new CSV file
output_file = 'results_to_be_processed/dlr_evaluation_results_70_unique_questions.csv'
deduplicated_df.to_csv(output_file, index=False)
print(f"\nDeduplicated results saved to {output_file}")
