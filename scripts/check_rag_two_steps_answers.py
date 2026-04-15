import pandas as pd

# Load the CSV file
df = pd.read_csv('dlr_data/DARES25_EarthObsertvation_QA_RAG_results_v1.csv')

# Check if the rag_two_steps_answer column contains valid answers
rag_two_steps_answers = df['rag_two_steps_answer'].dropna()

print(f"Total rows in CSV: {len(df)}")
print(f"Rows with rag_two_steps_answer: {len(rag_two_steps_answers)}")

# Get unique question_ids
unique_question_ids = df['question_id'].unique()

print(f"Unique question IDs: {len(unique_question_ids)}")
print(f"Question IDs: {sorted(unique_question_ids)}")

# Check if there are exactly 70 unique question IDs
if len(unique_question_ids) == 70:
    print("✅ There are exactly 70 unique question IDs.")
else:
    print(f"❌ There are {len(unique_question_ids)} unique question IDs, not 70.")
