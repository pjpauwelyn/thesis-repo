#!/usr/bin/env python3
"""
Add placeholder evaluation scores to no-ontology results.

This adds the evaluation score columns with placeholder values (-1)
so the file has the correct structure for when you run the actual
evaluation with API access.
"""

import csv
from pathlib import Path

# Input/output file
input_file = Path("results_to_be_processed/no_ontology_results.csv")
output_file = Path("results_to_be_processed/no_ontology_results_with_eval.csv")

# Read current results
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    results = list(reader)

print(f"📊 Loaded {len(results)} results")

# Add evaluation score columns with placeholder values
for result in results:
    # Add evaluation scores (placeholder -1 values)
    result['factuality'] = '-1'
    result['relevance'] = '-1'
    result['groundedness'] = '-1'
    result['helpfulness'] = '-1'
    result['depth'] = '-1'
    result['overall_score'] = '-1'
    result['evaluation_processing_time'] = '0.0'
    result['evaluation_timestamp'] = ''
    result['evaluation_status'] = 'pending'

# Define the new fieldnames (original + evaluation columns)
original_fieldnames = [
    'question_id', 'question', 'pipeline', 'experiment_type',
    'context', 'clean_aql_results', 'answer', 'references',
    'formatted_references', 'processing_time', 'status', 'error', 'timestamp'
]

evaluation_fieldnames = [
    'factuality', 'relevance', 'groundedness', 'helpfulness', 'depth',
    'overall_score', 'evaluation_processing_time', 'evaluation_timestamp', 'evaluation_status'
]

fieldnames = original_fieldnames + evaluation_fieldnames

# Write updated results
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Added placeholder evaluation scores to {len(results)} results")
print(f"✅ Saved to: {output_file}")
print(f"\n📋 File now has the following columns:")
for i, field in enumerate(fieldnames, 1):
    print(f"   {i}. {field}")

print(f"\n🔑 To run actual DLR evaluation later:")
print(f"   export OPENAI_API_KEY='your-api-key-here'")
print(f"   python evaluate_no_ontology_results.py --questions 70")
print(f"\n   This will replace the placeholder -1 scores with actual evaluation results.")