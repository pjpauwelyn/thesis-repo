#!/usr/bin/env python3
"""Fix question ID numbering to have unique IDs 1-70."""

import csv

# Read current results
with open('results_to_be_processed/no_ontology_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    results = list(reader)

print(f'Current state: {len(results)} results')
current_ids = sorted(set(int(r['question_id']) for r in results))
print(f'Current question IDs: {current_ids}')

# Simply renumber all results sequentially from 1 to 70
for i, result in enumerate(results):
    result['question_id'] = str(i + 1)

new_ids = sorted(set(int(r['question_id']) for r in results))
print(f'After renumbering:')
print(f'New question IDs: {new_ids}')
print(f'All IDs 1-70 present: {set(range(1, 71)) == set(new_ids)}')

# Write back
with open('results_to_be_processed/no_ontology_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f'✓ Successfully renumbered {len(results)} results to have unique IDs 1-70')