#!/usr/bin/env python3

import pandas as pd

print("=== REFINED CONTEXT RESULTS ANALYSIS ===")
print("Loading refined-context_results_v2.csv...")

try:
    df = pd.read_csv('results_to_be_processed/refined-context_results_v2.csv')
    
    print(f"✓ File loaded successfully")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Check for error messages in answers
    error_mask = df['answer'].str.contains('Error: could not generate', na=False)
    error_count = error_mask.sum()
    
    if error_count > 0:
        print(f"✗ FOUND {error_count} answers with generation errors")
        print("Question IDs with errors:", df[error_mask]['question_id'].tolist())
    else:
        print("✓ No generation errors found in answers")
    
    # Check score validity
    invalid_scores = df[df['overall_score'].isna() | (df['overall_score'] < 0)]
    invalid_count = len(invalid_scores)
    
    if invalid_count > 0:
        print(f"✗ FOUND {invalid_count} evaluations with invalid scores")
        print("Question IDs with invalid scores:", invalid_scores['question_id'].tolist())
    else:
        print("✓ All scores are valid")
    
    # Score distribution
    print(f"\n=== SCORE DISTRIBUTION ===")
    if len(df) > 0:
        score_counts = df['overall_score'].value_counts().sort_index()
        for score, count in score_counts.items():
            print(f"Score {score}: {count} evaluations")
        
        # Calculate means
        print(f"\n=== MEAN SCORES ===")
        for criterion in ['overall_score', 'factuality', 'relevance', 'groundedness', 'helpfulness', 'depth']:
            if criterion in df.columns:
                mean_val = df[criterion].mean()
                print(f"{criterion}: {mean_val:.3f}/5.0")
    
    # Check pipeline distribution
    if 'pipeline' in df.columns:
        print(f"\n=== PIPELINE DISTRIBUTION ===")
        print(df['pipeline'].value_counts())
    
    # Summary
    valid_answers = len(df) - error_count
    print(f"\n=== SUMMARY ===")
    print(f"Valid answers: {valid_answers}/{len(df)}")
    print(f"Complete evaluations: {len(df) - invalid_count}/{len(df)}")
    
    if error_count == 0 and invalid_count == 0:
        print("✓ All answers are generated and all scores are valid")
    else:
        print("✗ Some answers or scores need attention")
        
except FileNotFoundError:
    print("✗ Error: refined-context_results_v2.csv not found")
except Exception as e:
    print(f"✗ Error during analysis: {str(e)}")
