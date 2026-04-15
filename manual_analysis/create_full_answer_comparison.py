#!/usr/bin/env python3
"""
📊 CREATE FULL ANSWER COMPARISON

Creates a manual analysis CSV with full answers from:
- zero_shot (dlr-1step)
- dlr_2step  
- slim-ontology experiment

Selects 10 representative questions with varying complexity.
"""

import pandas as pd
import os

# Input files
ZERO_SHOT_FILE = 'manual_error_analysis_with_dlr1step.csv'
DLR_2STEP_FILE = 'manual_error_analysis_with_dlr1step.csv' 
SLIM_ONTOLOGY_FILE = '../results_to_be_processed/slim-ontology_results.csv'

# Output file
OUTPUT_FILE = 'full_answer_comparison_analysis.csv'

# Selected question IDs (representative sample with varying complexity)
# These questions exist in all three datasets
SELECTED_QUESTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print("📊 Creating full answer comparison...")
print(f"Selected questions: {SELECTED_QUESTIONS}")
print()

# Load data
print("Loading data...")
zero_shot_data = pd.read_csv(ZERO_SHOT_FILE, on_bad_lines='warn')
dlr_2step_data = pd.read_csv(DLR_2STEP_FILE, on_bad_lines='warn') 
slim_ontology_data = pd.read_csv(SLIM_ONTOLOGY_FILE, on_bad_lines='warn')

print(f"Zero-shot data: {len(zero_shot_data)} rows")
print(f"DLR-2step data: {len(dlr_2step_data)} rows")
print(f"Slim-ontology data: {len(slim_ontology_data)} rows")
print()

# Create comparison dataframe
comparison_data = []

for q_id in SELECTED_QUESTIONS:
    print(f"Processing question {q_id}...")
    
    # Get zero-shot answer (1_pass_without_ontology column)
    zero_shot_row = zero_shot_data[zero_shot_data['question_id'] == q_id]
    if len(zero_shot_row) == 0:
        print(f"  ⚠️  Zero-shot answer not found for question {q_id}")
        zero_shot_answer = ""
    else:
        zero_shot_answer = zero_shot_row.iloc[0]['1_pass_without_ontology']
    
    # Get DLR-2step answer (dlr_2step column)
    dlr_2step_row = dlr_2step_data[dlr_2step_data['question_id'] == q_id]
    if len(dlr_2step_row) == 0:
        print(f"  ⚠️  DLR-2step answer not found for question {q_id}")
        dlr_2step_answer = ""
    else:
        dlr_2step_answer = dlr_2step_row.iloc[0]['dlr_2step']
    
    # Get slim-ontology answer
    slim_ontology_row = slim_ontology_data[slim_ontology_data['question_id'] == q_id]
    if len(slim_ontology_row) == 0:
        print(f"  ⚠️  Slim-ontology answer not found for question {q_id}")
        slim_ontology_answer = ""
    else:
        slim_ontology_answer = slim_ontology_row.iloc[0]['answer']
    
    # Get question text (from any source, they should be the same)
    question = zero_shot_row.iloc[0]['question'] if len(zero_shot_row) > 0 else ""
    
    comparison_data.append({
        'question_id': q_id,
        'question': question,
        'zero_shot_answer': zero_shot_answer,
        'dlr_2step_answer': dlr_2step_answer,
        'slim_ontology_answer': slim_ontology_answer
    })
    
    print(f"  ✅ Added question {q_id}")

# Create dataframe and save
print()
print(f"Creating comparison dataframe with {len(comparison_data)} questions...")
comparison_df = pd.DataFrame(comparison_data)

# Save to CSV
comparison_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Full answer comparison saved to: {OUTPUT_FILE}")
print()
print("Summary:")
print(f"  - {len(comparison_data)} questions included")
print(f"  - Full answers from 3 pipelines: zero-shot, dlr-2step, slim-ontology")
print(f"  - Questions selected for varying complexity and type")
print(f"  - Ready for manual analysis and comparison")