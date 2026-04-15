#!/usr/bin/env python3

import pandas as pd

# Load the full results
input_file = 'results_to_be_processed/dlr_evaluation_results.csv'
output_file = 'results_to_be_processed/filtered_results_1_70.csv'

try:
    df = pd.read_csv(input_file)
    
    # Filter to keep only questions 1-70
    questions_1_70 = df[df['question_id'] <= 70].copy()
    
    # Verify we have exactly 70 unique questions
    unique_questions = questions_1_70['question_id'].nunique()
    total_rows = len(questions_1_70)
    
    print(f"Original results: {len(df)} evaluations")
    print(f"Filtered results: {total_rows} evaluations")
    print(f"Unique questions 1-70: {unique_questions}")
    
    # Check if we have all questions 1-70
    expected_ids = set(range(1, 71))
    actual_ids = set(questions_1_70['question_id'].unique())
    missing_ids = expected_ids - actual_ids
    
    if missing_ids:
        print(f"WARNING: Missing question IDs: {sorted(missing_ids)}")
    else:
        print("SUCCESS: All questions 1-70 are present")
    
    # Save the filtered results
    questions_1_70.to_csv(output_file, index=False)
    print(f"Saved filtered results to: {output_file}")
    
    # Show summary statistics
    print("\n=== FILTERED RESULTS SUMMARY ===")
    print(f"Total evaluations: {len(questions_1_70)}")
    print(f"Pipeline: {questions_1_70['pipeline'].unique()}")
    print("\nScore distribution:")
    print(questions_1_70['overall_score'].value_counts().sort_index())
    
    print("\nAverage scores:")
    for criterion in ['overall_score', 'factuality', 'relevance', 'groundedness', 'helpfulness', 'depth']:
        print(f"{criterion}: {questions_1_70[criterion].mean():.2f}/5.0")
        
except Exception as e:
    print(f"Error: {e}")
