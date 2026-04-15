#!/usr/bin/env python3
"""
Fix the corrupted no_ontology_results.csv file.
Keep only the 2 properly evaluated answers and remove all duplicates/corrupted entries.
"""

import csv
from pathlib import Path
from collections import defaultdict

# Read the corrupted file
input_file = Path("results_to_be_processed/no_ontology_results.csv")
output_file = Path("results_to_be_processed/no_ontology_results_fixed.csv")

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    try:
        results = list(reader)
    except Exception as e:
        print(f"Error reading file: {e}")
        results = []

print(f"📊 Read {len(results)} rows from corrupted file")

# Analyze the data
question_counts = defaultdict(list)
for i, result in enumerate(results):
    try:
        qid = int(result.get('question_id', 0))
        question_counts[qid].append(i)
    except (ValueError, TypeError):
        pass

print(f"Question ID distribution: {dict(question_counts)}")

# Find the best version of each question (prioritize ones with evaluations)
clean_results = []
seen_question_ids = set()

for result in results:
    try:
        qid = int(result.get('question_id', 0))
        
        # Skip if we already have this question ID
        if qid in seen_question_ids:
            continue
        
        # Skip if missing required fields
        if not all(result.get(field) for field in ['question', 'answer', 'status']):
            continue
        
        # Prefer results with complete evaluations
        has_evaluation = (result.get('overall_score') not in ['-1', 'None', ''] and
                        all(result.get(field) not in ['-1', 'None', ''] 
                           for field in ['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth']))
        
        if has_evaluation:
            clean_results.append(result)
            seen_question_ids.add(qid)
            
    except (ValueError, TypeError, KeyError) as e:
        continue

print(f"✅ Found {len(clean_results)} clean results with complete evaluations")
print(f"Question IDs kept: {sorted([int(r['question_id']) for r in clean_results])}")

# Sort by question ID
clean_results.sort(key=lambda x: int(x['question_id']))

# Write the fixed file
fieldnames = list(clean_results[0].keys()) if clean_results else []

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(clean_results)

print(f"✅ Fixed file saved to: {output_file}")
print(f"📊 File now contains {len(clean_results)} properly evaluated results")

print(f"\n🔄 NEXT STEPS:")
print(f"1. Replace corrupted file:")
print(f"   mv {output_file} {input_file}")
print(f"\n2. Generate new answers for remaining questions")
print(f"   python experiments/no_ontology/run_no_ontology.py --csv data/dlr/dlr-results.csv --num 70")
"