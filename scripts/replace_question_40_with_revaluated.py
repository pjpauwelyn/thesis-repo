import pandas as pd

# Load the corrected CSV file
corrected_file = 'results_to_be_processed/refined-context_results_v2_corrected.csv'
df_corrected = pd.read_csv(corrected_file)

# Load the re-evaluated CSV file
re_evaluated_file = 'results_to_be_processed/refined-context_results_v2_re-evaluated.csv'
df_re_evaluated = pd.read_csv(re_evaluated_file)

# Get the re-evaluated data (Question ID 1)
re_evaluated_row = df_re_evaluated.iloc[0]

# Replace the scores for Question ID 40 in the corrected file
if 40 in df_corrected['question_id'].values:
    # Update the scores for Question ID 40
    df_corrected.loc[df_corrected['question_id'] == 40, ['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']] = (
        re_evaluated_row['factuality'],
        re_evaluated_row['relevance'],
        re_evaluated_row['groundedness'],
        re_evaluated_row['helpfulness'],
        re_evaluated_row['depth'],
        re_evaluated_row['overall_score']
    )
    
    # Save the updated CSV file
    output_file = 'results_to_be_processed/refined-context_results_v2_final.csv'
    df_corrected.to_csv(output_file, index=False)
    
    print(f"Question ID 40 has been updated with re-evaluated scores and saved to {output_file}")
    
    # Verify the update
    updated_row = df_corrected[df_corrected['question_id'] == 40].iloc[0]
    print(f"\nUpdated scores for Question ID 40:")
    print(f"  Criteria scores: factuality={updated_row['factuality']}, relevance={updated_row['relevance']}, groundedness={updated_row['groundedness']}, helpfulness={updated_row['helpfulness']}, depth={updated_row['depth']}")
    print(f"  Overall score: {updated_row['overall_score']}")
else:
    print("Question ID 40 not found in the corrected file.")
