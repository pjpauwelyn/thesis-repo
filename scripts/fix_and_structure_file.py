#!/usr/bin/env python3
"""
Comprehensive fix for no_ontology_results.csv:
1. Remove duplicate questions
2. Set missing evaluation scores to -1
3. Ensure proper structure
4. Assign unique question IDs
"""

import csv
from collections import defaultdict
from pathlib import Path

# Read current file
input_file = Path("results_to_be_processed/no_ontology_results.csv")
output_file = Path("results_to_be_processed/no_ontology_results_structured.csv")

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    results = list(reader)

print(f"📊 Read {len(results)} rows from file")

# Step 1: Remove duplicates (keep first occurrence of each unique question)
seen_questions = set()
unique_results = []
duplicates_removed = 0

for result in results:
    question_key = result.get('question', '')[:100]  # Use question content as key
    if question_key not in seen_questions:
        unique_results.append(result)
        seen_questions.add(question_key)
    else:
        duplicates_removed += 1

print(f"✅ Removed {duplicates_removed} duplicate questions")
print(f"📋 Now have {len(unique_results)} unique questions")

# Step 2: Set missing evaluation scores to -1 and fix structure
for result in unique_results:
    # Set missing evaluation scores to -1
    for field in ['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth']:
        if not result.get(field) or result.get(field) in ['', 'None']:
            result[field] = '-1'
    
    # Set missing overall_score to -1
    if not result.get('overall_score') or result.get('overall_score') in ['', 'None']:
        result['overall_score'] = '-1'
    
    # Set missing evaluation metadata
    if not result.get('evaluation_processing_time') or result.get('evaluation_processing_time') in ['', 'None']:
        result['evaluation_processing_time'] = '0.0'
    
    if not result.get('evaluation_timestamp') or result.get('evaluation_timestamp') in ['', 'None']:
        result['evaluation_timestamp'] = ''
    
    if not result.get('evaluation_status') or result.get('evaluation_status') in ['', 'None']:
        result['evaluation_status'] = 'pending'

print(f"✅ Fixed evaluation structure for all questions")

# Step 3: Assign proper unique question IDs (1-70)
for i, result in enumerate(unique_results):
    result['question_id'] = str(i + 1)

print(f"✅ Assigned unique question IDs 1-{len(unique_results)}")

# Step 4: Sort by question ID
unique_results.sort(key=lambda x: int(x['question_id']))

# Step 5: Write structured file
fieldnames = list(unique_results[0].keys()) if unique_results else []

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(unique_results)

print(f"✅ Structured file saved to: {output_file}")

# Final verification
print(f"\n📊 FINAL VERIFICATION:")
print(f"  Total questions: {len(unique_results)}")
print(f"  Question IDs: 1-{len(unique_results)}")
print(f"  All have proper evaluation structure")

# Count evaluation status
eval_pending = sum(1 for r in unique_results if r['evaluation_status'] == 'pending')
eval_success = sum(1 for r in unique_results if r['evaluation_status'] == 'success')

print(f"  Evaluation status:")
print(f"    Pending (-1 scores): {eval_pending}")
print(f"    Complete (valid scores): {eval_success}")

print(f"\n🔄 NEXT STEPS:")
print(f"1. Replace current file:")
print(f"   mv {output_file} {input_file}")
print(f"\n2. Run DLR evaluation:")
print(f"   python evaluate_no_ontology_results.py --questions {len(unique_results)}")
print(f"\n3. This will evaluate the {eval_pending} questions with pending status")
print(f"   and skip the {eval_success} questions that already have good scores")

print(f"\n✅ File is now properly structured and ready for evaluation!")
