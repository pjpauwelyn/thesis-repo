#!/usr/bin/env python3
"""
Test DeepEval evaluation on refined context results.

This script evaluates the two refined context results using DeepEval metrics.
"""

import csv
import json
from pathlib import Path
from evaluation.deepeval_evaluator import DeepEvalEvaluator, DEEPEVAL_AVAILABLE

def load_refined_results(csv_path: str) -> list:
    """Load refined context results from CSV."""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['pipeline'] == '1_pass_with_ontology_refined' and row['question']:
                results.append({
                    'question_id': row['question_id'],
                    'question': row['question'],
                    'answer': row['answer'],
                    'context': row['context'],
                    'references': row['references']
                })
    return results

def test_deepeval_on_results():
    """Test DeepEval evaluation on refined context results."""
    if not DEEPEVAL_AVAILABLE:
        print("❌ DeepEval not available. Please install deepeval package.")
        return

    # Load the refined context results
    results_path = Path("results_to_be_processed/refined-context_results.csv")
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return

    print("✅ DeepEval available. Loading refined context results...")
    results = load_refined_results(str(results_path))
    
    if not results:
        print("❌ No refined context results found. Run pipeline first.")
        return

    print(f"✅ Found {len(results)} refined context results to evaluate")
    
    # Initialize DeepEval evaluator
    evaluator = DeepEvalEvaluator()
    
    # Evaluate each result
    evaluations = []
    for i, result in enumerate(results, 1):
        print(f"\n🔍 Evaluating result {i}/{len(results)}")
        print(f"Question: {result['question'][:80]}...")
        
        try:
            evaluation = evaluator.evaluate(
                question=result['question'],
                answer=result['answer'],
                context=result['context']
            )
            
            evaluation['question_id'] = result['question_id']
            evaluation['question_preview'] = result['question'][:100]
            evaluations.append(evaluation)
            
            print(f"✅ Evaluation completed:")
            print(f"   Overall Score: {evaluation.get('overall_score', 'N/A')}")
            print(f"   Relevancy: {evaluation.get('answer_relevancy', 'N/A')}")
            print(f"   Faithfulness: {evaluation.get('faithfulness', 'N/A')}")
            print(f"   Contextual Relevancy: {evaluation.get('contextual_relevancy', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Evaluation failed for result {i}: {e}")
            evaluation = {
                'question_id': result['question_id'],
                'question_preview': result['question'][:100],
                'error': str(e)
            }
            evaluations.append(evaluation)

    # Save evaluation results
    output_path = Path("results/deepeval_evaluations/refined_context_deepeval.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'question_id', 'question_preview',
            'overall_score', 'answer_relevancy', 'faithfulness',
            'contextual_relevancy', 'hallucination', 'coherence',
            'toxicity', 'bias', 'error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for eval_result in evaluations:
            row = {
                'question_id': eval_result.get('question_id', ''),
                'question_preview': eval_result.get('question_preview', '')[:100],
                'overall_score': eval_result.get('overall_score', ''),
                'answer_relevancy': eval_result.get('answer_relevancy', ''),
                'faithfulness': eval_result.get('faithfulness', ''),
                'contextual_relevancy': eval_result.get('contextual_relevancy', ''),
                'hallucination': eval_result.get('hallucination', ''),
                'coherence': eval_result.get('coherence', ''),
                'toxicity': eval_result.get('toxicity', ''),
                'bias': eval_result.get('bias', ''),
                'error': eval_result.get('error', '')
            }
            writer.writerow(row)

    print(f"\n✅ DeepEval evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"- Total results evaluated: {len(evaluations)}")
    print(f"- Successful evaluations: {sum(1 for e in evaluations if 'error' not in e)}")
    print(f"- Failed evaluations: {sum(1 for e in evaluations if 'error' in e)}")

if __name__ == "__main__":
    test_deepeval_on_results()