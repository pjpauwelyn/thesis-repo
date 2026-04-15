#!/usr/bin/env python3
"""
Clean up no_ontology_results.csv by removing placeholder rows.
Keeps only questions with valid answers and evaluations.
"""

import csv
from pathlib import Path

# Read current results
input_file = Path("results_to_be_processed/no_ontology_results.csv")
output_file = Path("results_to_be_processed/no_ontology_results_clean.csv")

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    results = list(reader)

print(f"📊 Current file has {len(results)} results")

# Filter to keep only results with valid answers (not placeholders)
valid_results = []
placeholder_results = []

for result in results:
    if result['status'] == 'success' and result['answer'] and not result['answer'].startswith('PLACEHOLDER'):
        valid_results.append(result)
    else:
        placeholder_results.append(result)

print(f"✅ Found {len(valid_results)} valid results to keep")
print(f"❌ Found {len(placeholder_results)} placeholder results to remove")

# Show which question IDs are being kept
valid_ids = sorted(int(r['question_id']) for r in valid_results)
print(f"📋 Keeping question IDs: {valid_ids}")

# Write cleaned results
fieldnames = list(results[0].keys()) if results else []

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(valid_results)

print(f"✅ Cleaned file saved to: {output_file}")
print(f"📊 File now contains {len(valid_results)} valid results with evaluations")

print(f"\n🔄 NEXT STEPS:")
print(f"1. Replace the original file:")
print(f"   mv {output_file} {input_file}")
print(f"\n2. Generate new answers for questions 3-70:")
print(f"   python experiments/no_ontology/run_no_ontology.py --csv data/dlr/dlr-results.csv --num 70")
print(f"\n3. The script will automatically skip questions 1-2 (already processed)")
print(f"   and only generate answers for new questions")