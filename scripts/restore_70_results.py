#!/usr/bin/env python3
"""
Restore 70 no-ontology results by extending the existing 2 results.

Since the original 70 results were accidentally lost, this script creates
a complete set of 70 results by:
1. Keeping the 2 existing evaluated results (questions 1-2)
2. Creating placeholder results for questions 3-70 with the same structure

The placeholder results will have valid structure but generic content,
allowing you to run the actual no-ontology pipeline later to regenerate
the real answers.
"""

import csv
from pathlib import Path

# Read the current 2 results
current_file = Path("results_to_be_processed/no_ontology_results.csv")
with open(current_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    existing_results = list(reader)

print(f"📊 Found {len(existing_results)} existing results")

# We need to create 70 total results
# Keep the first 2 (which have real evaluations)
# Create placeholders for 3-70

# Get a template from the first result
template = existing_results[0].copy()

# Create all 70 results
all_results = []

# Add the first 2 existing results (with their real evaluations)
for i, result in enumerate(existing_results[:2], 1):
    result['question_id'] = str(i)
    all_results.append(result)

# Create placeholder results for questions 3-70
for qid in range(3, 71):
    placeholder = template.copy()
    placeholder['question_id'] = str(qid)
    placeholder['question'] = f"Placeholder question {qid} - to be regenerated"
    placeholder['answer'] = f"PLACEHOLDER ANSWER {qid}: This is a placeholder answer. The real answer will be generated when you run the no-ontology pipeline."
    placeholder['context'] = f"PLACEHOLDER CONTEXT {qid}: This is placeholder context data."
    placeholder['references'] = "PLACEHOLDER|REFERENCES"
    placeholder['formatted_references'] = "PLACEHOLDER|REFERENCES"
    placeholder['clean_aql_results'] = "PLACEHOLDER AQL RESULTS"
    placeholder['processing_time'] = "0.00"
    placeholder['status'] = "placeholder"
    placeholder['error'] = ""
    
    # Set evaluation scores to -1 (pending)
    placeholder['factuality'] = '-1'
    placeholder['relevance'] = '-1'
    placeholder['groundedness'] = '-1'
    placeholder['helpfulness'] = '-1'
    placeholder['depth'] = '-1'
    placeholder['overall_score'] = '-1'
    placeholder['evaluation_processing_time'] = '0.0'
    placeholder['evaluation_timestamp'] = ''
    placeholder['evaluation_status'] = 'pending'
    
    all_results.append(placeholder)

print(f"📝 Created {len(all_results)} total results (2 real + 68 placeholders)")

# Define fieldnames (same as before)
fieldnames = list(all_results[0].keys())

# Write the complete set
output_file = Path("results_to_be_processed/no_ontology_results_restored.csv")
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

print(f"✅ Restored 70 results saved to: {output_file}")
print(f"\n📋 Summary:")
print(f"   • Questions 1-2: Real results with DLR evaluations")
print(f"   • Questions 3-70: Placeholder results (need regeneration)")
print(f"   • All results have proper CSV structure")
print(f"   • Placeholder results have evaluation_status='pending'")

print(f"\n🔄 To regenerate the placeholder results:")
print(f"   python experiments/no_ontology/run_no_ontology.py --csv data/dlr/dlr-results.csv --num 70")
print(f"\n   This will replace the placeholder answers with real ones while preserving the evaluations for questions 1-2.")