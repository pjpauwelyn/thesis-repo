#!/usr/bin/env python3
"""Test script to verify that the modified run_no_ontology.py skips existing successful results."""

import csv
import os
import tempfile
import shutil
from pathlib import Path

# Create a temporary directory for testing
test_dir = tempfile.mkdtemp()
print(f"Test directory: {test_dir}")

try:
    # Create a simple input CSV with 5 questions
    input_csv = os.path.join(test_dir, "input.csv")
    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question", "aql_results", "structured_context"])
        for i in range(1, 6):
            writer.writerow([i, f"Question {i}", "", "Context for question {i}"])
    
    # Create an output CSV with 2 successful results (questions 1 and 2)
    output_csv = os.path.join(test_dir, "output.csv")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "question", "pipeline", "experiment_type",
            "context", "clean_aql_results", "answer", "references",
            "formatted_references", "processing_time", "status", "error", "timestamp"
        ])
        writer.writeheader()
        for i in range(1, 3):  # Questions 1 and 2
            writer.writerow({
                "question_id": i,
                "question": f"Question {i}",
                "pipeline": "no_ontology",
                "experiment_type": "no_ontology",
                "context": f"Context for question {i}",
                "clean_aql_results": "",
                "answer": f"Answer {i}",
                "references": "",
                "formatted_references": "",
                "processing_time": "1.00",
                "status": "success",
                "error": "",
                "timestamp": "2023-01-01T00:00:00"
            })
    
    print(f"Created input CSV with 5 questions")
    print(f"Created output CSV with 2 successful results (questions 1-2)")
    
    # Run the modified script
    print(f"\nRunning the modified script...")
    os.system(f"cd {test_dir} && python /Users/pjpauwelyn/THESIS_2025/THESIS-REP/clean_repo/experiments/no_ontology/run_no_ontology.py --csv input.csv --num 5 --output-csv output.csv")
    
    # Check the results
    print(f"\nChecking results...")
    with open(output_csv, "r", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    
    print(f"Total results in output CSV: {len(results)}")
    successful_results = [r for r in results if r["status"] == "success"]
    print(f"Successful results: {len(successful_results)}")
    
    # Verify that questions 1 and 2 were not reprocessed
    question_ids = [int(r["question_id"]) for r in results]
    print(f"Question IDs in output: {sorted(question_ids)}")
    
    # The script should have processed questions 3, 4, and 5
    expected_new_questions = [3, 4, 5]
    new_questions_processed = [qid for qid in question_ids if qid in expected_new_questions]
    
    if len(new_questions_processed) == 3:
        print("✓ SUCCESS: Script correctly skipped existing successful results!")
        print(f"✓ Processed new questions: {new_questions_processed}")
    else:
        print("✗ FAILURE: Script did not skip existing results correctly")
        print(f"Expected to process questions {expected_new_questions}, but processed: {new_questions_processed}")

finally:
    # Clean up
    shutil.rmtree(test_dir)
    print(f"\nCleaned up test directory")