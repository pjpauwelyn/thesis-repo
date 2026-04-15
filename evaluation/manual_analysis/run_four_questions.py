"""run_four_questions.py
Runs the four manual-analysis questions through the fully refined pipeline
(1_pass_with_ontology_refined) and writes the results to:

    evaluation/manual_analysis/four_q_results.csv

Columns: question_id, question, answer

Usage:
    python3 evaluation/manual_analysis/run_four_questions.py
    python3 evaluation/manual_analysis/run_four_questions.py --model mistral-large-latest
    python3 evaluation/manual_analysis/run_four_questions.py --overwrite
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# The four questions for manual error analysis
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

OUTPUT_PATH = Path("evaluation/manual_analysis/four_q_results.csv")
PIPELINE_TYPE = "1_pass_with_ontology_refined"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question", "answer"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n Results written → {path}  ({len(rows)} rows)")


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

    # lazy imports so the module is parseable without the full env
    from core.utils.helpers import get_llm_model, DEFAULT_MODEL
    from core.agents.ontology_agent import OntologyConstructionAgent
    from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
    from core.agents.generation_agent import GenerationAgent

    model_name = args.model or DEFAULT_MODEL
    print(f"Model : {model_name}")
    print(f"Pipeline: {PIPELINE_TYPE}")
    print(f"Output : {OUTPUT_PATH}\n")

    llm = get_llm_model(model=model_name, temperature=0.2)
    llm_ref = get_llm_model(model=model_name, temperature=0.1)

    ontology_agent   = OntologyConstructionAgent(llm_ref, prompt_dir="prompts/ontology")
    refinement_agent = RefinementAgent1PassRefined(llm_ref, prompt_dir="prompts/refinement")
    generation_agent = GenerationAgent(llm, prompt_dir="prompts/generation")

    rows = []
    for qid, question in enumerate(QUESTIONS, start=1):
        print(f"[{qid}/{len(QUESTIONS)}] {question[:80]}")
        t0 = time.time()

        try:
            # 1. build ontology
            print("    → building ontology...", end=" ", flush=True)
            ontology = ontology_agent.process(question, include_relationships=False)
            if ontology and not ontology.should_use_ontology:
                ontology = None
            print(f"{'done' if ontology else 'skipped'}")

            # 2. refine context
            # no pre-fetched AQL results for custom questions – the refinement
            # agent uses the ontology + question to construct the best context it can.
            print("    → refining context...", end=" ", flush=True)
            refined = refinement_agent.process_context(
                question=question,
                structured_context="",
                ontology=ontology,
                include_ontology=True,
                aql_results_str=None,
                context_filter="full",
            )
            final_context = refined.enriched_context or ""
            print("done")

            # 3. generate answer
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
