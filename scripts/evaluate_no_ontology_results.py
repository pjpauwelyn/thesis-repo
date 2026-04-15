#!/usr/bin/env python3
"""
DLR Evaluation Script for No-Ontology Results

Evaluates the no-ontology pipeline results using DLR criteria and adds
the evaluation scores to the same CSV file.
"""

import os
import sys
import csv
import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import argparse

# Add project directory to path
project_dir = os.path.abspath('.')
sys.path.insert(0, project_dir)

from evaluation.dlr_evaluator import DLREvaluator
from dotenv import load_dotenv

class NoOntologyEvaluator:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize evaluator with placeholders for failed evaluations
        self.evaluator = DLREvaluator(use_placeholders=True)
        
        # Input and output file
        self.input_output_file = Path("results_to_be_processed/no_ontology_results.csv")
        
    def _load_results(self) -> List[Dict]:
        """Load no-ontology results from CSV"""
        if not self.input_output_file.exists():
            print(f"⚠️  File not found: {self.input_output_file}")
            return []
        
        with open(self.input_output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        
        return results
    
    def _check_if_needs_evaluation(self, result: Dict) -> bool:
        """Check if a result needs evaluation based on existing scores"""
        # Check each criteria score
        criteria_fields = ['factuality', 'relevance', 'groundedness', 'helpfulness', 'depth', 'overall_score']
        
        for field in criteria_fields:
            score = result.get(field, '-1')
            
            # Needs evaluation if:
            # - Score is -1 (never evaluated)
            # - Score is 1.0 (low quality, needs re-evaluation)
            # - Score is empty/missing (never evaluated)
            # - Score is not a valid number
            if score in ['-1', '1.0', '']:
                return True
            
            # Check if it's a valid number between 2-5 (good scores that don't need re-evaluation)
            try:
                score_float = float(score)
                if score_float < 2.0 or score_float > 5.0:
                    return True
            except (ValueError, TypeError):
                return True
        
        # If all scores are valid and in range 2-5, no need to re-evaluate
        return False
    
    def _evaluate_result(self, result: Dict) -> Dict:
        """Evaluate a single result using DLR evaluator"""
        import time
        
        question = result['question']
        answer = result['answer']
        context = result.get('context', '')
        
        print(f"🔍 Evaluating question {result['question_id']}: {question[:60]}...")
        
        start_time = time.time()
        evaluation = self.evaluator.evaluate_answer(question, answer, context)
        processing_time = time.time() - start_time
        
        # Extract scores
        criteria_scores = evaluation.get('criteria_scores', {})
        
        # Create evaluation result (only evaluation scores, not full result)
        eval_result = {
            'factuality': criteria_scores.get('factuality', -1),
            'relevance': criteria_scores.get('relevance', -1),
            'groundedness': criteria_scores.get('groundedness', -1),
            'helpfulness': criteria_scores.get('helpfulness', -1),
            'depth': criteria_scores.get('depth', -1),
            'overall_score': evaluation.get('overall_score', -1),
            'evaluation_processing_time': round(processing_time, 2),
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_status': 'success' if evaluation.get('overall_score', -1) != -1 else 'failed'
        }
        
        return eval_result
    
    def _update_result_with_evaluation(self, original_result: Dict, eval_result: Dict) -> Dict:
        """Update original result with evaluation scores"""
        # Add evaluation scores to the original result
        updated_result = original_result.copy()
        updated_result.update({
            'factuality': eval_result['factuality'],
            'relevance': eval_result['relevance'],
            'groundedness': eval_result['groundedness'],
            'helpfulness': eval_result['helpfulness'],
            'depth': eval_result['depth'],
            'overall_score': eval_result['overall_score'],
            'evaluation_processing_time': eval_result['evaluation_processing_time'],
            'evaluation_timestamp': eval_result['evaluation_timestamp'],
            'evaluation_status': eval_result['evaluation_status']
        })
        return updated_result
    
    def _write_results(self, results: List[Dict]):
        """Write updated results back to CSV"""
        # Get fieldnames from first result (preserve original structure)
        fieldnames = list(results[0].keys()) if results else [
            'question_id', 'question', 'pipeline', 'experiment_type',
            'context', 'clean_aql_results', 'answer', 'references',
            'formatted_references', 'processing_time', 'status', 'error', 'timestamp',
            'factuality', 'relevance', 'groundedness', 'helpfulness', 'depth',
            'overall_score', 'evaluation_processing_time', 'evaluation_timestamp', 'evaluation_status'
        ]
        
        # Sort results by question_id
        results.sort(key=lambda x: int(x['question_id']))
        
        with open(self.input_output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    def evaluate_results(self, num_questions: int = None, force_revaluate: bool = False):
        """Evaluate no-ontology results and update the CSV file"""
        print(f"\n🚀 Evaluating no-ontology results...")
        
        # Load results
        results = self._load_results()
        if not results:
            print(f"⚠️  No results found in {self.input_output_file}")
            return []
        
        # Limit to specified number of questions
        if num_questions:
            results = results[:num_questions]
        
        print(f"📊 Loaded {len(results)} results")
        
        # Filter results to only those that need evaluation
        results_to_evaluate = []
        skipped_count = 0
        for result in results:
            if force_revaluate or self._check_if_needs_evaluation(result):
                results_to_evaluate.append(result)
            else:
                skipped_count += 1
        
        print(f"📋 Will evaluate {len(results_to_evaluate)} results")
        print(f"⏭️  Skipping {skipped_count} results (already evaluated)")
        
        # Evaluate each result
        evaluated_results = []
        for i, result in enumerate(results_to_evaluate, 1):
            print(f"\n[{i}/{len(results_to_evaluate)}] ", end="")
            
            eval_result = self._evaluate_result(result)
            updated_result = self._update_result_with_evaluation(result, eval_result)
            evaluated_results.append(updated_result)
            
            score = eval_result['overall_score']
            status = f"Score: {score}" if score != -1 else "Failed (-1)"
            print(f"  {status}")
        
        # Merge evaluated results back with skipped results
        final_results = []
        eval_results_dict = {r['question_id']: r for r in evaluated_results}
        
        for result in results:
            if result['question_id'] in eval_results_dict:
                final_results.append(eval_results_dict[result['question_id']])
            else:
                final_results.append(result)
        
        # Write updated results back to file
        self._write_results(final_results)
        
        print(f"\n✅ Completed no-ontology evaluation")
        print(f"✅ Updated {self.input_output_file}")
        
        return evaluated_results

def main():
    parser = argparse.ArgumentParser(description="DLR Evaluation for No-Ontology Results")
    parser.add_argument('--questions', type=int, default=70,
                       help='Number of questions to evaluate (default: 70)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-evaluation of all questions, even if already evaluated')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - only evaluate first 2 questions')
    
    args = parser.parse_args()
    
    evaluator = NoOntologyEvaluator()
    
    if args.test:
        print("🧪 Running in test mode - evaluating first 2 questions")
        args.questions = 2
    
    print(f"📊 Evaluating {args.questions} questions from no-ontology pipeline")
    if args.force:
        print("🔄 Force re-evaluation enabled")
    
    # Run evaluation
    results = evaluator.evaluate_results(args.questions, args.force)
    
    print(f"\n✅ DLR evaluation complete!")
    print(f"📊 Evaluated {len(results)} questions")
    
    # Show summary
    if results:
        successful = sum(1 for r in results if r['overall_score'] != -1)
        failed = len(results) - successful
        print(f"📈 Summary: {successful} successful, {failed} failed evaluations")

if __name__ == "__main__":
    main()