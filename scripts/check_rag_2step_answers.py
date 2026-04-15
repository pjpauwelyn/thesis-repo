import pandas as pd

# Load the CSV file
df = pd.read_csv('dlr_data/DARES25_EarthObsertvation_QA_RAG_results_v1.csv')

# Filter for the rag_2step pipeline
rag_2step_df = df[df['pipeline'] == 'rag_2step']

# Get unique question_ids
unique_question_ids = rag_2step_df['question_id'].unique()

print(f"Total rows for rag_2step: {len(rag_2step_df)}")
print(f"Unique question IDs for rag_2step: {len(unique_question_ids)}")
print(f"Question IDs: {sorted(unique_question_ids)}")

# Check if there are exactly 70 unique question IDs
if len(unique_question_ids) == 70:
    print("✅ There are exactly 70 unique question IDs for rag_2step.")
else:
    print(f"❌ There are {len(unique_question_ids)} unique question IDs for rag_2step, not 70.")
