import pandas as pd

# Load the original CSV file to extract Question 40
original_file = 'dlr_data/dlr-results.csv'
df_original = pd.read_csv(original_file)

# Extract the row where idx is 40 (Question ID 40)
question_40 = df_original[df_original['idx'] == 40].iloc[0]

print("Question 40 extracted:")
print(f"Index: {question_40['idx']}")
print(f"Question: {question_40['question'][:100]}...")

# Save Question 40 to a temporary CSV file for reprocessing
temp_csv = 'temp_question_40.csv'
question_40_df = pd.DataFrame([question_40])
question_40_df.to_csv(temp_csv, index=False)

print(f"\nQuestion 40 saved to {temp_csv} for reprocessing.")
