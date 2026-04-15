import pandas as pd

# Load the CSV file
refined_df = pd.read_csv('results_to_be_processed/refined-context_results.csv')

# Delete the last 5 rows (question_id 71-75)
refined_df = refined_df.iloc[:-5]

# Save the updated CSV file
refined_df.to_csv('results_to_be_processed/refined-context_results.csv', index=False)

print("Last 5 rows (question_id 71-75) have been deleted.")
