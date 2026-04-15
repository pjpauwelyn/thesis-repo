#!/usr/bin/env python3
"""
🔍 VALIDATION EVALUATION SCRIPT

Independent script for re-evaluating refined-context and dlr-results answers.
Ensures clean, correct, and complete evaluation results with proper re-evaluation logic.
"""

import os
import sys
import csv
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse
import pandas as pd

# Add project directory to path
project_dir = os.path.abspath('.')
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, 'core'))

from evaluation.dlr_evaluation import DLREvaluator

class ValidationEvaluator:
    def __init__(self):
        # Initialize evaluator with placeholders for failed evaluations
        self.evaluator = DLREvaluator(use_placeholders=True)
        
        # Output directory for validation results
        self.output_dir = Path("results_final/validation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
     
    def _initialize_output_csv(self, pipeline_name: str) -> Path:
        """Initialize output CSV with proper headers for validation results"""
        fieldnames = [
            'question_id', 'question', 'pipeline', 'context', 'answer', 'references',
            'factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score',
            'processing_time', 'timestamp', 'evaluation_status'
        ]
        
        output_file = self.output_dir / f"{pipeline_name}_validation_results.csv"
        
        # Create file if it doesn't exist
        if not output_file.exists():
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            print(f"✅ Created new {output_file} with headers")
        else:
            print(f"✅ Using existing {output_file}")
        
        return output_file
    
    def _load_refined_context_results(self, num_questions: int = None) -> List[Dict]:
        """Load refined-context results from CSV"""
        input_file = Path("results_to_be_processed/refined-context_results.csv")
        
        if not input_file.exists():
            print(f"⚠️  File not found: {input_file}")
            return []
        
        df = pd.read_csv(input_file)
        results = df.to_dict('records')
        
        return results[:num_questions] if num_questions else results
    
    def _load_dlr_rag_results(self, num_questions: int = None) -> List[Dict]:
        """Load dlr-results with rag_two_steps_answer from CSV"""
        input_file = Path("dlr_data/DARES25_EarthObsertvation_QA_RAG_results_v1.csv")
        
        if not input_file.exists():
            print(f"⚠️  File not found: {input_file}")
            return []
        
        df = pd.read_csv(input_file)
        
        # Create results with question_id based on index (since question_id is NaN)
        results = []
        for idx, row in df.iterrows():
            result = {
                'question_id': idx + 1,  # Assign sequential question_id
                'question': row['question'],
                'pipeline': 'dlr_2step',
                'context': row.get('context', ''),
                'answer': row['rag_two_steps_answer'],
                'references': row.get('references', ''),
                'rag_two_steps_answer': row['rag_two_steps_answer']  # Keep original column
            }
            results.append(result)
        
        return results[:num_questions] if num_questions else results
    
    def _check_if_needs_evaluation(self, result: Dict, output_file: Path) -> bool:
        """Check if a result needs evaluation based on existing evaluation file"""
        # If output file doesn't exist, all results need evaluation
        if not output_file.exists():
            return True
        
        # Check if this question already has a valid evaluation
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row['question_id'] == str(result['question_id']) and 
                    row['pipeline'] == result['pipeline']):
                    # If overall_score is -1 or 1.0, needs re-evaluation
                    if float(row.get('overall_score', -1)) in [-1, 1.0]:
                        return True
                    
                    # Check if any criteria score is -1 or 1.0 (indicating failure)
                    criteria_scores = [
                        float(row.get('factuality', -1)),
                        float(row.get('relevance', -1)),
                        float(row.get('groundedness', -1)),
                        float(row.get('helpfulness', -1)),
                        float(row.get('depth', -1))
                    ]
                    
                    # If any criteria score is -1 or 1.0, needs re-evaluation
                    if any(score in [-1, 1.0] for score in criteria_scores):
                        return True
                    else:
                        return False
        
        # If not found in evaluation file, needs evaluation
        return True
    
    def _evaluate_result(self, result: Dict) -> Dict:
        """Evaluate a single result using DLR evaluator"""
        import time
        
        question = result['question']
        answer = result['answer']
        context = result.get('context', '')
        
        print(f"🔍 Evaluating question {result['question_id']}: {question[:50]}...")
        
        start_time = time.time()
        evaluation = self.evaluator.evaluate_answer(question, answer, context)
        processing_time = time.time() - start_time
        
        # Extract scores
        criteria_scores = evaluation.get('criteria_scores', {})
        
        return {
            'question_id': result['question_id'],
            'question': question,
            'pipeline': result.get('pipeline', 'unknown'),
            'context': context,
            'answer': answer,
            'references': result.get('references', ''),
            'factuality': criteria_scores.get('factuality', -1),
            'relevance': criteria_scores.get('relevance', -1),
            'groundedness': criteria_scores.get('groundedness', -1),
            'helpfulness': criteria_scores.get('helpfulness', -1),
            'depth': criteria_scores.get('depth', -1),
            'overall_score': evaluation.get('overall_score', -1),
            'processing_time': round(processing_time, 2),
            'timestamp': datetime.now().isoformat(),
            'evaluation_status': 'success' if evaluation.get('overall_score', -1) != -1 else 'failed'
        }
    
    def _update_existing_evaluation(self, evaluated_result: Dict, output_file: Path):
        """Update existing evaluation in CSV file (for re-evaluation)"""
        # Read all existing rows
        existing_rows = []
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        
        # Remove existing row for this question_id + pipeline
        question_id = str(evaluated_result['question_id'])
        pipeline = evaluated_result['pipeline']
        
        existing_rows = [row for row in existing_rows 
                        if not (row['question_id'] == question_id and 
                               row['pipeline'] == pipeline)]
        
        # Add updated evaluation
        existing_rows.append(evaluated_result)
        
        # Sort all rows by question_id (numeric sort)
        existing_rows.sort(key=lambda x: int(x['question_id']))
        
        # Write all data back
        fieldnames = [
            'question_id', 'question', 'pipeline', 'context', 'answer', 'references',
            'factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score',
            'processing_time', 'timestamp', 'evaluation_status'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
        
        print(f"  ✅ Updated evaluation for question {question_id}")
    
    def _append_new_evaluation(self, evaluated_result: Dict, output_file: Path):
        """Append new evaluation to CSV file and ensure sorted order"""
        fieldnames = [
            'question_id', 'question', 'pipeline', 'context', 'answer', 'references',
            'factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score',
            'processing_time', 'timestamp', 'evaluation_status'
        ]
        
        # Read existing rows
        existing_rows = []
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        
        # Add new evaluation
        existing_rows.append(evaluated_result)
        
        # Sort all rows by question_id (numeric sort)
        existing_rows.sort(key=lambda x: int(x['question_id']))
        
        # Write all rows back in sorted order
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
        
        print(f"  ✅ Saved evaluation for question {evaluated_result['question_id']}")
    
    def evaluate_refined_context(self, num_questions: int = None, force_revaluate: bool = False):
        """Evaluate refined-context results and save to output file"""
        print(f"\n🚀 Evaluating refined-context results...")
        
        # Initialize output CSV
        output_file = self._initialize_output_csv("refined-context")
        
        # Load results
        results = self._load_refined_context_results(num_questions)
        if not results:
            print(f"⚠️  No results found in refined-context_results.csv")
            return
        
        # Filter results to only those that need evaluation
        results_to_evaluate = []
        skipped_count = 0
        for result in results:
            if force_revaluate or self._check_if_needs_evaluation(result, output_file):
                results_to_evaluate.append(result)
            else:
                skipped_count += 1
        
        print(f"📊 Found {len(results_to_evaluate)} results to evaluate (skipped {skipped_count} already evaluated)")
        
        # Evaluate each result that needs it
        evaluated_results = []
        for result in results_to_evaluate:
            evaluated_result = self._evaluate_result(result)
            evaluated_results.append(evaluated_result)
            
            # Check if this is a re-evaluation (existing row exists)
            needs_update = False
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if (row['question_id'] == str(evaluated_result['question_id']) and 
                            row['pipeline'] == evaluated_result['pipeline']):
                            needs_update = True
                            break
            
            # Save appropriately
            if needs_update:
                self._update_existing_evaluation(evaluated_result, output_file)
            else:
                self._append_new_evaluation(evaluated_result, output_file)
            
            score = evaluated_result['overall_score']
            status = "Score: " + str(score) if score != -1 else "Failed (-1)"
            print(f"    {status}")
        
        print(f"✅ Completed refined-context evaluation")
        return evaluated_results
    
    def evaluate_dlr_rag_results(self, num_questions: int = None, force_revaluate: bool = False):
        """Evaluate dlr-results with rag_two_steps_answer and save to output file"""
        print(f"\n🚀 Evaluating dlr-results (rag_two_steps_answer)...")
        
        # Initialize output CSV
        output_file = self._initialize_output_csv("dlr_2step")
        
        # Load results
        results = self._load_dlr_rag_results(num_questions)
        if not results:
            print(f"⚠️  No results found in dlr-results.csv")
            return
        
        # Filter results to only those that need evaluation
        results_to_evaluate = []
        skipped_count = 0
        for result in results:
            if force_revaluate or self._check_if_needs_evaluation(result, output_file):
                results_to_evaluate.append(result)
            else:
                skipped_count += 1
        
        print(f"📊 Found {len(results_to_evaluate)} results to evaluate (skipped {skipped_count} already evaluated)")
        
        # Evaluate each result that needs it
        evaluated_results = []
        for result in results_to_evaluate:
            evaluated_result = self._evaluate_result(result)
            evaluated_results.append(evaluated_result)
            
            # Check if this is a re-evaluation (existing row exists)
            needs_update = False
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if (row['question_id'] == str(evaluated_result['question_id']) and 
                            row['pipeline'] == evaluated_result['pipeline']):
                            needs_update = True
                            break
            
            # Save appropriately
            if needs_update:
                self._update_existing_evaluation(evaluated_result, output_file)
            else:
                self._append_new_evaluation(evaluated_result, output_file)
            
            score = evaluated_result['overall_score']
            status = "Score: " + str(score) if score != -1 else "Failed (-1)"
            print(f"    {status}")
        
        print(f"✅ Completed dlr-results evaluation")
        return evaluated_results

def main():
    parser = argparse.ArgumentParser(description="Validation Evaluation Script")
    parser.add_argument('--pipeline', type=str, required=True, choices=['refined-context', 'dlr-2step'],
                       help='Pipeline to evaluate: refined-context or dlr-2step')
    parser.add_argument('--questions', type=int, default=70,
                       help='Number of questions to evaluate (default: 70)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-evaluation of all questions, even if already evaluated')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - only evaluate first 2 questions and show detailed output')
    
    args = parser.parse_args()
    
    evaluator = ValidationEvaluator()
    
    if args.test:
        print("🧪 Running in test mode - evaluating first 2 questions")
        args.questions = 2
    
    print(f"📊 Evaluating {args.questions} questions from {args.pipeline}")
    if args.force:
        print("🔄 Force re-evaluation enabled - will evaluate all specified questions")
    
    # Run evaluation based on pipeline
    if args.pipeline == 'refined-context':
        results = evaluator.evaluate_refined_context(args.questions, args.force)
    elif args.pipeline == 'dlr-2step':
        results = evaluator.evaluate_dlr_rag_results(args.questions, args.force)
    
    print(f"\n✅ Validation evaluation complete!")
    print(f"📁 Results saved to: {evaluator.output_dir}")
    print(f"📊 Evaluated {len(results)} questions")
    
    # Show summary
    if results:
        successful = sum(1 for r in results if r['overall_score'] != -1)
        failed = len(results) - successful
        print(f"📈 Summary: {successful} successful, {failed} failed evaluations")

if __name__ == "__main__":
    main()
