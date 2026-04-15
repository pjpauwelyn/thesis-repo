import pandas as pd

# Load the CSV file
df = pd.read_csv('dlr_data/DARES25_EarthObsertvation_QA_RAG_results_v1.csv')

# Check if the question column contains unique questions
unique_questions = df['question'].dropna().unique()

print(f"Total unique questions: {len(unique_questions)}")

# Check if there are exactly 70 unique questions
if len(unique_questions) == 70:
    print("✅ There are exactly 70 unique questions.")
else:
    print(f"❌ There are {len(unique_questions)} unique questions, not 70.")
