#!/usr/bin/env python3
"""
Targeted regeneration of placeholder questions (3-70).

This script runs the no-ontology pipeline only for questions that have
placeholder status, skipping the already-processed questions 1-2.
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

# Add project directory to path
project_dir = os.path.abspath('.')
sys.path.insert(0, project_dir)

from core.utils.helpers import get_llm_model, DEFAULT_MODEL
from core.utils.aql_parser import parse_aql_results
from core.agents.generation_agent import GenerationAgent, Answer
from experiments.no_ontology.refinement_agent_no_ontology import RefinementAgentNoOntology

def load_existing_results(existing_csv: str) -> dict:
    """Load existing results and identify which questions need regeneration."""
    with open(existing_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    # Identify questions that need regeneration (status == 'placeholder')
    needs_regeneration = {}
    for result in results:
        if result['status'] == 'placeholder':
            needs_regeneration[int(result['question_id'])] = result
    
    print(f"📊 Found {len(results)} existing results")
    print(f"⏳ Questions needing regeneration: {len(needs_regeneration)}")
    print(f"✅ Questions to preserve: {len(results) - len(needs_regeneration)}")
    
    return results, needs_regeneration

def load_input_data(input_csv: str, question_ids_to_process: set) -> list:
    """Load only the input rows that need processing."""
    import csv
    csv.field_size_limit(int(1e8))
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_data = list(reader)
    
    # Filter to only include rows whose question_id is in our target set
    # Since input has question_id=0 for all rows, we need to map by index
    # Question 1 = index 0, Question 2 = index 1, etc.
    data_to_process = []
    for idx, row in enumerate(all_data):
        if (idx + 1) in question_ids_to_process:  # Convert 0-based index to 1-based question_id
            data_to_process.append(row)
    
    print(f"📋 Loaded {len(data_to_process)} input rows for regeneration")
    return data_to_process

def process_question(row: dict, idx: int, ref_llm, gen_llm, refinement_agent, generation_agent) -> dict:
    """Process a single question through the no-ontology pipeline."""
    import time
    
    qid = idx + 1  # 1-based question ID
    question = row.get("question", "")
    aql_results = row.get("aql_results", "")
    structured_context = row.get("structured_context", "")
    
    print(f"\n[{qid}] Processing: {question[:60]}...")
    
    result = {
        'question_id': str(qid),
        'question': question,
        'pipeline': 'no_ontology',
        'experiment_type': 'no_ontology'
    }
    
    start_time = time.time()
    
    try:
        # Parse AQL
        clean_aql = parse_aql_results(aql_results) if aql_results else ""
        result['clean_aql_results'] = clean_aql
        
        # Refinement – no ontology
        refined = refinement_agent.process_context(
            question=question,
            structured_context=structured_context,
            ontology=None,
            include_ontology=False,
            aql_results_str=clean_aql or None,
        )
        final_ctx = refined.enriched_context or structured_context or ""
        result['context'] = final_ctx
        
        # Generation – no ontology
        answer_obj = generation_agent.generate(
            question=question,
            text_context=final_ctx,
            ontology=None,
        )
        
        result['answer'] = answer_obj.answer
        result['references'] = " | ".join(answer_obj.references)
        result['formatted_references'] = " | ".join(answer_obj.formatted_references)
        result['status'] = "success"
        result['error'] = ""
        
        print(f"  ✅ Success ({time.time()-start_time:.1f}s)")
        
    except Exception as exc:
        result['status'] = "error"
        result['error'] = str(exc)[:200]
        result['answer'] = f"ERROR: {str(exc)[:100]}"
        result['references'] = ""
        result['formatted_references'] = ""
        result['context'] = structured_context or ""
        result['clean_aql_results'] = ""
        print(f"  ❌ Error: {exc}")
    
    result['processing_time'] = f"{time.time()-start_time:.2f}"
    result['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Set evaluation scores to -1 (pending)
    result['factuality'] = '-1'
    result['relevance'] = '-1'
    result['groundedness'] = '-1'
    result['helpfulness'] = '-1'
    result['depth'] = '-1'
    result['overall_score'] = '-1'
    result['evaluation_processing_time'] = '0.0'
    result['evaluation_timestamp'] = ''
    result['evaluation_status'] = 'pending'
    
    return result

def merge_results(existing_results: list, new_results: list) -> list:
    """Merge new results with existing ones, preserving evaluations for completed questions."""
    
    # Create lookup for new results
    new_results_dict = {int(r['question_id']): r for r in new_results}
    
    # Merge: use new results for regenerated questions, keep existing for others
    final_results = []
    for existing in existing_results:
        qid = int(existing['question_id'])
        if qid in new_results_dict:
            # Use the new regenerated result
            final_results.append(new_results_dict[qid])
        else:
            # Keep the existing result (preserves evaluations for questions 1-2)
            final_results.append(existing)
    
    # Sort by question_id
    final_results.sort(key=lambda x: int(x['question_id']))
    
    return final_results

def write_results(output_csv: str, results: list):
    """Write results to CSV file."""
    fieldnames = [
        'question_id', 'question', 'pipeline', 'experiment_type',
        'context', 'clean_aql_results', 'answer', 'references',
        'formatted_references', 'processing_time', 'status', 'error', 'timestamp',
        'factuality', 'relevance', 'groundedness', 'helpfulness', 'depth',
        'overall_score', 'evaluation_processing_time', 'evaluation_timestamp', 'evaluation_status'
    ]
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description="Regenerate placeholder questions in no-ontology results")
    parser.add_argument('--input-csv', default='data/dlr/dlr-results.csv',
                       help='Input CSV with questions')
    parser.add_argument('--existing-csv', default='results_to_be_processed/no_ontology_results.csv',
                       help='Existing results CSV')
    parser.add_argument('--output-csv', default='results_to_be_processed/no_ontology_results.csv',
                       help='Output CSV')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help='LLM model to use')
    
    args = parser.parse_args()
    
    print("🔄 Targeted Regeneration of Placeholder Questions")
    print("=" * 50)
    
    # Load existing results and identify what needs regeneration
    existing_results, needs_regeneration = load_existing_results(args.existing_csv)
    
    if not needs_regeneration:
        print("✅ No questions need regeneration - all are complete!")
        return
    
    # Load input data only for questions that need processing
    question_ids_to_process = set(needs_regeneration.keys())
    data_to_process = load_input_data(args.input_csv, question_ids_to_process)
    
    if not data_to_process:
        print("⚠️  No input data found for questions needing regeneration")
        return
    
    # Initialize agents
    print("🤖 Initializing LLM agents...")
    ref_llm = get_llm_model(model=args.model, temperature=0.1)
    gen_llm = get_llm_model(model=args.model, temperature=0.2)
    
    refinement_agent = RefinementAgentNoOntology(ref_llm, prompt_dir="prompts/refinement")
    generation_agent = GenerationAgent(gen_llm, prompt_dir="prompts/generation")
    
    # Override generation template with no-ontology variant
    no_ont_prompt_path = os.path.join(
        os.path.dirname(__file__), "experiments/no_ontology/generation_prompt_no_ontology.txt"
    )
    with open(no_ont_prompt_path, "r", encoding="utf-8") as f:
        generation_agent.refinement_template = f.read()
    
    print("🚀 Processing questions...")
    
    # Process each question
    new_results = []
    for idx, row in enumerate(data_to_process):
        result = process_question(row, idx, ref_llm, gen_llm, refinement_agent, generation_agent)
        new_results.append(result)
    
    # Merge with existing results
    final_results = merge_results(existing_results, new_results)
    
    # Write final results
    write_results(args.output_csv, final_results)
    
    # Summary
    regenerated_count = len(new_results)
    preserved_count = len(existing_results) - regenerated_count
    
    print(f"\n✅ Regeneration Complete!")
    print(f"📊 Regenerated {regenerated_count} questions")
    print(f"💾 Preserved {preserved_count} existing questions (with evaluations)")
    print(f"📁 Saved to: {args.output_csv}")
    
    # Show which questions were regenerated
    regenerated_ids = sorted(question_ids_to_process)
    print(f"🔢 Regenerated question IDs: {regenerated_ids}")

if __name__ == "__main__":
    main()