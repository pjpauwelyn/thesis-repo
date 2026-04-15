import pandas as pd

# Load the combined CSV file
combined_file = 'results_to_be_processed/refined-context_results_v2_combined.csv'
df = pd.read_csv(combined_file)

# Calculate mean scores per criteria
mean_scores = df[['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']].mean()

print("Mean Scores per Criteria:")
print(mean_scores)

# Check for any invalid scores
invalid_scores = df[(df['overall_score'] == 1.0) | (df['overall_score'] == -1)]
if len(invalid_scores) > 0:
    print(f"\nWarning: Found {len(invalid_scores)} rows with invalid scores (1.0 or -1).")
else:
    print("\nAll scores are valid (no 1.0 or -1).")

# Check for any validation anomalies
validation_anomalies = df[df['validation_anomaly'] == True]
if len(validation_anomalies) > 0:
    print(f"\nWarning: Found {len(validation_anomalies)} rows with validation anomalies.")
else:
    print("\nNo validation anomalies detected.")

# Display the distribution of overall scores
print("\nDistribution of Overall Scores:")
print(df['overall_score'].value_counts().sort_index())
