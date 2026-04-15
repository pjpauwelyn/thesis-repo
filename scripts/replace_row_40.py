import pandas as pd

# Load the corrected CSV file
corrected_file = 'results_to_be_processed/refined-context_results_v2_corrected.csv'
df_corrected = pd.read_csv(corrected_file)

# Load the re-evaluated CSV file
re_evaluated_file = 'results_to_be_processed/refined-context_results_v2_re-evaluated.csv'
df_re_evaluated = pd.read_csv(re_evaluated_file)

# Check if the re-evaluated file contains Question ID 40
if 40 in df_re_evaluated['question_id'].values:
    # Get the row with Question ID 40 from the re-evaluated file
    row_40_re_evaluated = df_re_evaluated[df_re_evaluated['question_id'] == 40].iloc[0]
    
    # Replace the row with Question ID 40 in the corrected file
    df_corrected.loc[df_corrected['question_id'] == 40] = row_40_re_evaluated
    
    # Save the updated CSV file
    output_file = 'results_to_be_processed/refined-context_results_v2_final.csv'
    df_corrected.to_csv(output_file, index=False)
    
    print(f"Row 40 (Question ID 40) has been replaced and saved to {output_file}")
    
    # Verify the update
    updated_row = df_corrected[df_corrected['question_id'] == 40].iloc[0]
    print(f"\nUpdated scores for Question ID 40:")
    print(f"  Criteria scores: factuality={updated_row['factuality']}, relevance={updated_row['relevance']}, groundedness={updated_row['groundedness']}, helpfulness={updated_row['helpfulness']}, depth={updated_row['depth']}")
    print(f"  Overall score: {updated_row['overall_score']}")
else:
    print("Question ID 40 not found in the re-evaluated file.")
