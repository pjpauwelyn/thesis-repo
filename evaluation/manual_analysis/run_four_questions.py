"""run_four_questions.py
Runs the four manual-analysis questions through the fully refined pipeline
(1_pass_with_ontology_refined) and writes the results to:

    evaluation/manual_analysis/four_q_results.csv

Columns: question_id, question, answer

Per-question AQL context is read from JSON files in the same directory:
    evaluation/manual_analysis/q1.json
    evaluation/manual_analysis/q2.json
    evaluation/manual_analysis/q3.json
    evaluation/manual_analysis/q4.json

Each JSON file must be a list of document objects that contain at least:
    title, abstract, uri

This mirrors the format produced by parse_aql_results() from the DLR CSV,
so the refinement agent receives identical input in both cases.

Usage:
    python3 evaluation/manual_analysis/run_four_questions.py
    python3 evaluation/manual_analysis/run_four_questions.py --model mistral-large-latest
    python3 evaluation/manual_analysis/run_four_questions.py --overwrite
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What satellite sensors can be used to monitor vegetation health?",
    "How do optical and radar sensors compare for forest monitoring?",
    "How does GEDI lidar compare to Sentinel-1 SAR in boreal forests?",
    (
        "How does the accuracy of above-ground biomass estimation compare "
        "between GEDI lidar data and Sentinel-1 SAR backscatter in boreal forests?"
    ),
]

# Directory that holds both this script and the q1..q4 JSON files
MANUAL_ANALYSIS_DIR = Path("evaluation/manual_analysis")
OUTPUT_PATH         = MANUAL_ANALYSIS_DIR / "four_q_results.csv"
PIPELINE_TYPE       = "1_pass_with_ontology_refined"

# ---------------------------------------------------------------------------
# JSON → compact AQL string (same output format as parse_aql_results())
# ---------------------------------------------------------------------------

def load_json_as_aql_results(qid: int) -> Optional[str]:
    """Load evaluation/manual_analysis/q{qid}.json and return a compact JSON
    string containing only title / abstract / uri per document.

    This produces exactly the same string format that parse_aql_results()
    produces from the DLR CSV, so the refinement agent sees identical input.

    Returns None (with a warning) if the file is missing or malformed.
    """
    json_path = MANUAL_ANALYSIS_DIR / f"q{qid}.json"

    if not json_path.exists():
        print(f"    ⚠  {json_path} not found — running without AQL context")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"    ⚠  {json_path} is not valid JSON ({exc}) — running without AQL context")
        return None

    if not isinstance(data, list):
        print(f"    ⚠  {json_path} top-level is not a list — running without AQL context")
        return None

    # Extract only the three fields the pipeline needs (same as parse_aql_results)
    clean_docs = [
        {
            "title":    doc.get("title", ""),
            "abstract": doc.get("abstract", ""),
            "uri":      doc.get("uri", ""),
        }
        for doc in data
        if isinstance(doc, dict)
    ]

    if not clean_docs:
        print(f"    ⚠  {json_path} produced no valid documents — running without AQL context")
        return None

    compact = json.dumps(clean_docs, separators=(",", ":"), ensure_ascii=False)
    print(f"    ✓  Loaded {len(clean_docs)} docs from {json_path.name}")
    return compact


# ---------------------------------------------------------------------------
# CSV output + validation
# ---------------------------------------------------------------------------

def _write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓  Results written → {path}  ({len(rows)} rows)")


def _validate(path: Path) -> None:
    issues = []
    with open(path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if not row.get("question", "").strip():
                issues.append(f"row {i}: empty question")
            a = row.get("answer", "").strip()
            if not a:
                issues.append(f"row {i}: empty answer")
            elif a.startswith("ERROR") or a.startswith("Error:"):
                issues.append(f"row {i}: pipeline error — {a[:80]}")
    if issues:
        print("\n⚠  Validation issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✓  All rows validated successfully.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the four manual-analysis questions through the refined pipeline."
    )
    parser.add_argument(
        "--model", default=None,
        help="Mistral model name (default: DEFAULT_MODEL from helpers)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite the output CSV if it already exists",
    )
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.overwrite:
        print(f"Output already exists: {OUTPUT_PATH}")
        print("Use --overwrite to re-run.")
        sys.exit(0)

    # lazy imports — keeps the module parseable without the full env
    from core.utils.helpers import get_llm_model, DEFAULT_MODEL
    from core.agents.ontology_agent import OntologyConstructionAgent
    from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
    from core.agents.generation_agent import GenerationAgent

    model_name = args.model or DEFAULT_MODEL
    print(f"Model    : {model_name}")
    print(f"Pipeline : {PIPELINE_TYPE}")
    print(f"Output   : {OUTPUT_PATH}")
    print(f"JSON dir : {MANUAL_ANALYSIS_DIR}\n")

    llm     = get_llm_model(model=model_name, temperature=0.2)
    llm_ref = get_llm_model(model=model_name, temperature=0.1)

    ontology_agent   = OntologyConstructionAgent(llm_ref, prompt_dir="prompts/ontology")
    refinement_agent = RefinementAgent1PassRefined(llm_ref, prompt_dir="prompts/refinement")
    generation_agent = GenerationAgent(llm, prompt_dir="prompts/generation")

    rows = []
    for qid, question in enumerate(QUESTIONS, start=1):
        print(f"[{qid}/{len(QUESTIONS)}] {question[:80]}")
        t0 = time.time()

        try:
            # --- load per-question AQL context from JSON file ---
            aql_results_str = load_json_as_aql_results(qid)

            # --- 1. build ontology ---
            print("    → building ontology...", end=" ", flush=True)
            ontology = ontology_agent.process(question, include_relationships=False)
            if ontology and not ontology.should_use_ontology:
                ontology = None
            print("done" if ontology else "skipped")

            # --- 2. refine context (with AQL docs as input) ---
            print("    → refining context...", end=" ", flush=True)
            refined = refinement_agent.process_context(
                question=question,
                structured_context="",
                ontology=ontology,
                include_ontology=True,
                aql_results_str=aql_results_str,  # compact JSON from q{qid}.json
                context_filter="full",
            )
            final_context = refined.enriched_context or ""
            print("done")

            # --- 3. generate answer ---
            print("    → generating answer...", end=" ", flush=True)
            answer_obj = generation_agent.generate(
                question=question,
                text_context=final_context,
                ontology=ontology,
            )
            print(f"done ({time.time() - t0:.1f}s)")

            rows.append({
                "question_id": qid,
                "question":    question,
                "answer":      answer_obj.answer,
            })

        except Exception as exc:
            print(f"    ✗ ERROR: {exc}")
            rows.append({
                "question_id": qid,
                "question":    question,
                "answer":      f"ERROR: {exc}",
            })

    _write_csv(rows, OUTPUT_PATH)
    _validate(OUTPUT_PATH)

    failed = sum(1 for r in rows if str(r["answer"]).startswith("ERROR"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
