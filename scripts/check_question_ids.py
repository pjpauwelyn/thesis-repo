import pandas as pd

# Load the two CSV files
file1 = 'results_to_be_processed/refined-context_results_v2_1-35.csv'
file2 = 'results_to_be_processed/refined-context_results_v2_36-70.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print(f"Question IDs in file1: {df1['question_id'].min()} to {df1['question_id'].max()}")
print(f"Question IDs in file2: {df2['question_id'].min()} to {df2['question_id'].max()}")
