import pandas as pd

# Load the combined CSV file
combined_file = 'results_to_be_processed/refined-context_results_v2_combined.csv'
df = pd.read_csv(combined_file)

# Check if overall_score matches the mean of criteria scores
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
        print(f"Row {idx + 1} (Question ID {row['question_id']}):")
        print(f"  Criteria mean: {calculated_mean:.2f}")
        print(f"  Overall score: {row['overall_score']:.2f}")
        print(f"  Criteria scores: {criteria_scores}")
        print()
