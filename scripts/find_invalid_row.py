import pandas as pd

# Load the corrected CSV file
corrected_file = 'results_to_be_processed/refined-context_results_v2_corrected.csv'
df = pd.read_csv(corrected_file)

# Find the row with invalid scores
invalid_rows = df[(df['overall_score'] == 1.0) | (df['overall_score'] == -1)]

if len(invalid_rows) > 0:
    print("Rows with invalid scores:")
    for idx, row in invalid_rows.iterrows():
        print(f"Row {idx + 1} (Question ID {row['question_id']}):")
        print(f"  Criteria scores: factuality={row['factuality']}, relevance={row['relevance']}, groundedness={row['groundedness']}, helpfulness={row['helpfulness']}, depth={row['depth']}")
        print(f"  Overall score: {row['overall_score']}")
        print()
else:
    print("No rows with invalid scores found.")
