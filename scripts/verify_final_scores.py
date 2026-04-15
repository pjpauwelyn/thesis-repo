import pandas as pd

# Load the final CSV file
final_file = 'results_to_be_processed/refined-context_results_v2_final.csv'
df = pd.read_csv(final_file)

# Calculate mean scores per criteria
mean_scores = df[['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']].mean()

print("Mean Scores per Criteria:")
print(mean_scores)

# Check for any invalid scores
invalid_scores = df[(df['overall_score'] == 1.0) | (df['overall_score'] == -1)]
if len(invalid_scores) > 0:
    print(f"\nWarning: Found {len(invalid_scores)} rows with invalid scores (1.0 or -1).")
    print("Rows with invalid scores:")
    for idx, row in invalid_scores.iterrows():
        print(f"  Row {idx + 1} (Question ID {row['question_id']}): overall_score={row['overall_score']}")
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

# Verify that all overall_scores match the mean of criteria scores
mismatches = []
for idx, row in df.iterrows():
    criteria_scores = [
        row['factuality'],
        row['relevance'],
        row['groundedness'],
        row['helpfulness'],
        row['depth']
    ]
    calculated_mean = sum(criteria_scores) / len(criteria_scores)
    
    if abs(row['overall_score'] - calculated_mean) > 0.01:  # Allow for minor floating-point differences
        mismatches.append((idx, row['question_id'], calculated_mean, row['overall_score']))

if mismatches:
    print("\nWarning: Found mismatches between overall_score and criteria mean:")
    for idx, qid, calculated_mean, overall_score in mismatches:
        print(f"  Row {idx + 1} (Question ID {qid}): calculated_mean={calculated_mean:.2f}, overall_score={overall_score:.2f}")
else:
    print("\nAll overall_scores match the mean of criteria scores.")
