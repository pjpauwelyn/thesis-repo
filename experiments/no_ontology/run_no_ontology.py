#!/usr/bin/env python3
"""experiment a: run the full pipeline without any ontology.

changes compared to the standard pipeline:
  - ontology construction is skipped entirely (no llm call)
  - refinement uses refinement_agent_no_ontology which loads a prompt
    that never mentions "ontology" and selects aql results based on
    question relevance alone
  - the generation prompt omits ontology wording (uses
    generation_prompt_no_ontology.txt)
  - results are written to a dedicated csv so they don't mix with
    ontology-based runs

usage:
    python experiments/no_ontology/run_no_ontology.py \\
        --csv data.csv --num 5 --output-csv results_no_ontology.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

csv.field_size_limit(int(1e8))

# ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.utils.helpers import get_llm_model, DEFAULT_MODEL
from core.utils.aql_parser import parse_aql_results
from core.agents.generation_agent import GenerationAgent, Answer
from experiments.no_ontology.refinement_agent_no_ontology import RefinementAgentNoOntology

logger = logging.getLogger("no_ontology_experiment")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)


@dataclass
class ExperimentResult:
    question_id: int
    question: str
    answer: str = ""
    references: str = ""
    formatted_references: str = ""
    context: str = ""
    clean_aql: str = ""
    processing_time: float = 0.0
    status: str = "success"
    error: str = ""


def run_experiment(
    input_csv: str,
    output_csv: str,
    num_questions: Optional[int] = None,
    model_name: str = DEFAULT_MODEL,
    refinement_temp: float = 0.1,
    generation_temp: float = 0.2,
) -> List[ExperimentResult]:
    """run the no-ontology pipeline over a csv of questions."""
    # load data
    with open(input_csv, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    if num_questions:
        data = data[:num_questions]

    # Check if output CSV exists and read existing successful results
    existing_results = []
    if os.path.exists(output_csv):
        try:
            with open(output_csv, "r", encoding="utf-8") as f:
                existing_results = list(csv.DictReader(f))
        except Exception as e:
            logger.warning(f"Could not read existing results: {e}")

    # Determine how many questions have already been successfully processed
    processed_indices = set()
    for result in existing_results:
        if result.get("status") == "success":
            # The question_id in output is idx+1, so we need to map back to the original index
            output_qid = int(result.get("question_id", 0))
            if output_qid > 0:  # Only consider positive question IDs (generated ones)
                processed_indices.add(output_qid - 1)  # Convert back to 0-based index

    # Filter data to only include questions not yet successfully processed
    if processed_indices:
        logger.info(f"Found {len(processed_indices)} already processed questions, skipping them")
        original_count = len(data)
        # Filter out rows whose index (when processed) would match processed question IDs
        data = [row for idx, row in enumerate(data) if idx not in processed_indices]
        logger.info(f"Processing {len(data)} new questions (was {original_count} total)")

    # init agents
    ref_llm = get_llm_model(model=model_name, temperature=refinement_temp)
    gen_llm = get_llm_model(model=model_name, temperature=generation_temp)

    refinement_agent = RefinementAgentNoOntology(ref_llm, prompt_dir="prompts/refinement")
    generation_agent = GenerationAgent(gen_llm, prompt_dir="prompts/generation")

    # changed: override the generation template with the no-ontology variant
    no_ont_prompt_path = os.path.join(
        os.path.dirname(__file__), "generation_prompt_no_ontology.txt"
    )
    with open(no_ont_prompt_path, "r", encoding="utf-8") as f:
        generation_agent.refinement_template = f.read()

    results: List[ExperimentResult] = []

    for idx, row in enumerate(data):
        qid = int(row.get("question_id") or (idx + 1))
        question = row.get("question", "")
        aql_results = row.get("aql_results", "")
        structured_context = row.get("structured_context", "")

        logger.info(f"\n[{idx+1}/{len(data)}] {question[:70]}...")
        r = ExperimentResult(question_id=qid, question=question)
        start = time.time()

        try:
            # parse aql
            clean_aql = parse_aql_results(aql_results) if aql_results else ""
            r.clean_aql = clean_aql

            # changed: no ontology construction at all
            # refinement – ontology=None, include_ontology=False
            refined = refinement_agent.process_context(
                question=question,
                structured_context=structured_context,
                ontology=None,
                include_ontology=False,
                aql_results_str=clean_aql or None,
            )
            final_ctx = refined.enriched_context or structured_context or ""
            r.context = final_ctx

            # generation – ontology=None
            answer_obj = generation_agent.generate(
                question=question,
                text_context=final_ctx,
                ontology=None,
            )

            r.answer = answer_obj.answer
            r.references = " | ".join(answer_obj.references)
            r.formatted_references = " | ".join(answer_obj.formatted_references)
            r.status = "success"
            logger.info(f"  ok ({time.time()-start:.1f}s)")

        except Exception as exc:
            r.status = "error"
            r.error = str(exc)[:200]
            r.answer = f"ERROR: {str(exc)[:100]}"
            logger.error(f"  failed: {exc}")
            traceback.print_exc()

        finally:
            r.processing_time = time.time() - start

        results.append(r)

    # write csv
    _write_csv(output_csv, results)
    logger.info(f"\nwrote {len(results)} results to {output_csv}")
    return results


def _write_csv(path: str, results: List[ExperimentResult]):
    fields = [
        "question_id", "question", "pipeline", "experiment_type",
        "context", "clean_aql_results", "answer", "references",
        "formatted_references", "processing_time", "status", "error", "timestamp",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and has content
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    
    with open(path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        for r in results:
            w.writerow({
                "question_id": r.question_id,
                "question": r.question,
                "pipeline": "no_ontology",
                "experiment_type": "no_ontology",
                "context": r.context,
                "clean_aql_results": r.clean_aql,
                "answer": r.answer,
                "references": r.references,
                "formatted_references": r.formatted_references,
                "processing_time": f"{r.processing_time:.2f}",
                "status": r.status,
                "error": r.error,
                "timestamp": datetime.now().isoformat(),
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment A: no-ontology pipeline")
    parser.add_argument("--csv", required=True, help="input csv")
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--output-csv", default="results_to_be_processed/no_ontology_results.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--refinement-temp", type=float, default=0.1)
    parser.add_argument("--generation-temp", type=float, default=0.2)
    args = parser.parse_args()

    run_experiment(
        input_csv=args.csv,
        output_csv=args.output_csv,
        num_questions=args.num,
        model_name=args.model,
        refinement_temp=args.refinement_temp,
        generation_temp=args.generation_temp,
    )
