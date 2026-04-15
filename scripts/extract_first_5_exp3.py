import pandas as pd

# Load the CSV file
refined_df = pd.read_csv('results_to_be_processed/refined-context_results.csv')

# Extract the first 5 rows (question_id 71-75)
first_5_df = refined_df.head(5)

# Save the first 5 rows to a new CSV file
first_5_df.to_csv('results_final/validation_results/first_5_exp3.csv', index=False)

# Remove the first 5 rows from the original CSV file
refined_df = refined_df.iloc[5:]

# Save the updated CSV file
refined_df.to_csv('results_to_be_processed/refined-context_results.csv', index=False)

print("First 5 rows (question_id 71-75) have been saved to first_5_exp3.csv.")
print("First 5 rows have been removed from refined-context_results.csv.")
