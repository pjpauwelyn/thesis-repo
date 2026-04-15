import pandas as pd

# Load the combined CSV file
combined_file = 'results_to_be_processed/refined-context_results_v2_combined.csv'
df = pd.read_csv(combined_file)

# Correct the overall_score for each row based on criteria scores
for idx, row in df.iterrows():
    criteria_scores = [
        row['factuality'],
        row['relevance'],
        row['groundedness'],
        row['helpfulness'],
        row['depth']
    ]
    
    # Calculate the correct overall_score as the mean of criteria scores
    valid_scores = [score for score in criteria_scores if score != -1]
    if valid_scores:
        corrected_overall_score = sum(valid_scores) / len(valid_scores)
    else:
        corrected_overall_score = -1.0  # All criteria scores are -1
    
    df.at[idx, 'overall_score'] = corrected_overall_score

# Save the corrected CSV file
output_file = 'results_to_be_processed/refined-context_results_v2_corrected.csv'
df.to_csv(output_file, index=False)

print(f"Corrected CSV saved to {output_file}")

# Recalculate mean scores per criteria
mean_scores = df[['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']].mean()
print("\nCorrected Mean Scores per Criteria:")
print(mean_scores)

# Check for any invalid scores
invalid_scores = df[(df['overall_score'] == 1.0) | (df['overall_score'] == -1)]
if len(invalid_scores) > 0:
    print(f"\nWarning: Found {len(invalid_scores)} rows with invalid scores (1.0 or -1).")
else:
    print("\nAll scores are valid (no 1.0 or -1).")
