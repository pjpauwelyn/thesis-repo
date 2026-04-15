import pandas as pd

# Load the DLR evaluation results CSV file
dlr_eval_file = 'results_to_be_processed/dlr_evaluation_results.csv'
df = pd.read_csv(dlr_eval_file)

# Display basic information
print("DLR Evaluation Results Analysis:")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values)

# Calculate mean scores per criteria
mean_scores = df[['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']].mean()
print("\nMean Scores per Criteria:")
print(mean_scores)

# Check for any invalid scores
invalid_scores = df[(df['overall_score'] == 1.0) | (df['overall_score'] == -1)]
if len(invalid_scores) > 0:
    print(f"\nWarning: Found {len(invalid_scores)} rows with invalid scores (1.0 or -1).")
else:
    print("\nAll scores are valid (no 1.0 or -1).")

# Display the distribution of overall scores
print("\nDistribution of Overall Scores:")
print(df['overall_score'].value_counts().sort_index())

# Check the distribution of experiment types
print("\nDistribution of Experiment Types:")
print(df['experiment_type'].value_counts())
