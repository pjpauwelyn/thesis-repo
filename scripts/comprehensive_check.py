#!/usr/bin/env python3
"""Comprehensive check of the no_ontology_results.csv file."""

import csv
from collections import Counter

print('🔍 COMPREHENSIVE FILE CHECK')
print('=' * 40)

with open('results_to_be_processed/no_ontology_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    results = list(reader)

print(f'Total rows: {len(results)}')

# Check 1: Unique question IDs
question_ids = [int(r['question_id']) for r in results]
unique_ids = set(question_ids)
print(f'\\n✅ Question ID Check:')
print(f'  Total IDs: {len(question_ids)}')
print(f'  Unique IDs: {len(unique_ids)}')
print(f'  All unique: {len(question_ids) == len(unique_ids)}')
print(f'  ID range: {min(question_ids)}-{max(question_ids)}')
print(f'  All IDs present: {set(question_ids) == set(range(1, 71))}')

# Check 2: All questions have answers
questions_with_answers = sum(1 for r in results if r['answer'] and len(r['answer'].strip()) > 10)
questions_without_answers = sum(1 for r in results if not r['answer'] or len(r['answer'].strip()) <= 10)

print(f'\\n✅ Answer Check:')
print(f'  Questions with answers: {questions_with_answers}')
print(f'  Questions without answers: {questions_without_answers}')
print(f'  All have answers: {questions_without_answers == 0}')

# Check 3: No errors
questions_with_errors = sum(1 for r in results if r['status'] == 'error')
questions_with_success = sum(1 for r in results if r['status'] == 'success')

print(f'\\n✅ Error Check:')
print(f'  Questions with success status: {questions_with_success}')
print(f'  Questions with error status: {questions_with_errors}')
print(f'  All successful: {questions_with_errors == 0}')

# Check 4: Answer quality (minimum length)
short_answers = []
for r in results:
    if len(r['answer'].strip()) < 50:  # Minimum reasonable answer length
        short_answers.append(int(r['question_id']))

print(f'\\n✅ Answer Quality Check:')
print(f'  Short answers (<50 chars): {len(short_answers)}')
if short_answers:
    print(f'  Question IDs with short answers: {short_answers}')

# Check 5: Context presence
questions_with_context = sum(1 for r in results if r['context'] and len(r['context'].strip()) > 10)
questions_without_context = sum(1 for r in results if not r['context'] or len(r['context'].strip()) <= 10)

print(f'\\n✅ Context Check:')
print(f'  Questions with context: {questions_with_context}')
print(f'  Questions without context: {questions_without_context}')

# Check 6: Evaluation structure
eval_issues = []
for r in results:
    # Check if evaluation fields exist and are properly formatted
    for field in ['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']:
        value = r.get(field, '')
        if value not in ['-1', '1', '2', '3', '4', '5', '1.0', '2.0', '3.0', '4.0', '5.0']:
            eval_issues.append(f"Q{r['question_id']}: {field}={value}")

print(f'\\n✅ Evaluation Structure Check:')
print(f'  Evaluation field issues: {len(eval_issues)}')
if eval_issues[:3]:  # Show first 3 issues
    for issue in eval_issues[:3]:
        print(f'    {issue}')
    if len(eval_issues) > 3:
        print(f'    ... and {len(eval_issues) - 3} more')

# Final summary
print(f'\\n🎯 FINAL SUMMARY:')
all_good = (
    len(question_ids) == 70 and
    len(unique_ids) == 70 and
    questions_without_answers == 0 and
    questions_with_errors == 0 and
    len(short_answers) == 0 and
    len(eval_issues) == 0
)

if all_good:
    print(f'✅ PERFECT: All 70 questions are unique, have answers, no errors, and proper structure!')
    print(f'🎉 File is ready for DLR evaluation!')
else:
    print(f'⚠️  Issues found - see details above')
    
print(f'\\n📊 Detailed breakdown:')
print(f'  • Total questions: {len(results)}')
print(f'  • Unique question IDs: {len(unique_ids)}/70')
print(f'  • Questions with answers: {questions_with_answers}/70')
print(f'  • Questions with success status: {questions_with_success}/70')
print(f'  • Questions with context: {questions_with_context}/70')
print(f'  • Evaluation issues: {len(eval_issues)}')