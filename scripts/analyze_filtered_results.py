#!/usr/bin/env python3

import pandas as pd
import numpy as np

print("=== FILTERED RESULTS ANALYSIS ===")
print("Loading filtered results...")

try:
    # Load the filtered results
    df = pd.read_csv('results_to_be_processed/filtered_results_1_70.csv')
    
    # Basic verification
    total_rows = len(df)
    unique_questions = df['question_id'].nunique()
    question_range = f"{df['question_id'].min()}-{df['question_id'].max()}"
    
    print(f"✓ Total rows: {total_rows}")
    print(f"✓ Unique questions: {unique_questions}")
    print(f"✓ Question ID range: {question_range}")
    
    # Check for missing question IDs
    expected_ids = set(range(1, 71))
    actual_ids = set(df['question_id'].unique())
    missing_ids = expected_ids - actual_ids
    
    if missing_ids:
        print(f"✗ MISSING QUESTION IDs: {sorted(missing_ids)}")
        print("The filtered file does not contain all questions 1-70")
    else:
        print("✓ All questions 1-70 are present")
    
    # Check for invalid scores
    invalid_mask = (df['factuality'] == -1) | (df['factuality'].isna())
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        print(f"✗ INVALID SCORES FOUND: {invalid_count} evaluations")
        print("Question IDs with invalid scores:", df[invalid_mask]['question_id'].tolist())
    else:
        print("✓ All answers have valid scores (no -1 or NaN values)")
    
    # Calculate exact means
    print("\n=== EXACT MEAN SCORES ===")
    means = {}
    for criterion in ['overall_score', 'factuality', 'relevance', 'groundedness', 'helpfulness', 'depth']:
        means[criterion] = float(f"{df[criterion].mean():.3f}")
        print(f"{criterion}: {means[criterion]}/5.0")
    
    # Score distribution
    print("\n=== SCORE DISTRIBUTION ===")
    score_counts = df['overall_score'].value_counts().sort_index()
    for score, count in score_counts.items():
        percentage = (count / len(df)) * 100
        print(f"Score {score}: {count} evaluations ({percentage:.1f}%)")
    
    # Quality categories
    perfect = (df['overall_score'] == 5.0).sum()
    high_quality = ((df['overall_score'] >= 4.8) & (df['overall_score'] <= 5.0)).sum()
    good_quality = ((df['overall_score'] >= 4.0) & (df['overall_score'] < 4.8)).sum()
    low_quality = (df['overall_score'] < 4.0).sum()
    
    print(f"\n=== QUALITY BREAKDOWN ===")
    print(f"Perfect scores (5.0): {perfect} ({perfect/len(df)*100:.1f}%)")
    print(f"High quality (4.8-5.0): {high_quality} ({high_quality/len(df)*100:.1f}%)")
    print(f"Good quality (4.0-4.8): {good_quality} ({good_quality/len(df)*100:.1f}%)")
    print(f"Low quality (<4.0): {low_quality} ({low_quality/len(df)*100:.1f}%)")
    
    # Criteria performance
    print(f"\n=== CRITERIA PERFORMANCE ===")
    for criterion in ['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth']:
        perf = df[criterion].value_counts(normalize=True).sort_index()
        print(f"\n{criterion}:")
        for score, pct in perf.items():
            if score >= 0:  # Skip -1 if present
                print(f"  Score {score}: {pct:.1%}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Overall quality: {'Excellent' if means['overall_score'] >= 4.8 else 'Good'}")
    print(f"Strongest criterion: {'Relevance' if means['relevance'] == max(means.values()) else 'Helpfulness'}")
    print(f"Area for improvement: {'Groundedness' if means['groundedness'] == min(means.values()) else 'Depth'}")
    
    # Save summary to file
    with open('results_analysis_summary.txt', 'w') as f:
        f.write("=== FILTERED RESULTS ANALYSIS SUMMARY ===\n\n")
        f.write(f"Total evaluations: {total_rows}\n")
        f.write(f"Unique questions: {unique_questions}\n")
        f.write(f"Question ID range: {question_range}\n\n")
        
        f.write("EXACT MEAN SCORES:\n")
        for criterion, mean_val in means.items():
            f.write(f"{criterion}: {mean_val}/5.0\n")
        
        f.write(f"\nOverall mean score: {means['overall_score']:.3f}/5.0\n")
        f.write(f"Quality rating: {'Excellent' if means['overall_score'] >= 4.8 else 'Good'}\n")
    
    print(f"\n✓ Analysis complete! Summary saved to results_analysis_summary.txt")
    
except FileNotFoundError:
    print("✗ Error: filtered_results_1_70.csv not found")
    print("Please create the filtered file first using the filter_results.py script")
    
except Exception as e:
    print(f"✗ Error during analysis: {str(e)}")
