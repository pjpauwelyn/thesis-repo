import pandas as pd

# Load the DLR evaluation results CSV file
dlr_eval_file = 'results_to_be_processed/dlr_evaluation_results.csv'
df = pd.read_csv(dlr_eval_file)

# Check for duplicate questions by comparing the 'question' column
duplicate_questions = df[df.duplicated(subset=['question'], keep=False)]

print(f"Total duplicate questions: {len(duplicate_questions)}")
print(f"Unique questions: {df['question'].nunique()}")

# Display the duplicate questions
if not duplicate_questions.empty:
    print("\nDuplicate Questions:")
    for question in duplicate_questions['question'].unique():
        print(f"- {question[:100]}...")

# Check which column varies between duplicates
if not duplicate_questions.empty:
    print("\nColumns that vary between duplicates:")
    for column in df.columns:
        if column != 'question':
            # Check if the column values differ for any duplicate question
            for question in duplicate_questions['question'].unique():
                subset = df[df['question'] == question]
                if subset[column].nunique() > 1:
                    print(f"- {column}")
                    break
