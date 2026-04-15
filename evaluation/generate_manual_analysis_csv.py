"""generate_manual_analysis_csv.py
Runs the fully refined pipeline (1_pass_with_ontology_refined) over the DLR
question set and writes a clean CSV for manual error analysis.

Output columns:
    question_id  – original ID from the DLR dataset
    question     – the question text
    answer       – the pipeline answer

Usage:
    # all questions
    python3 evaluation/generate_manual_analysis_csv.py

    # subset (first N questions)
    python3 evaluation/generate_manual_analysis_csv.py --num 20

    # specific question indices (0-based)
    python3 evaluation/generate_manual_analysis_csv.py --indices 0 4 9

    # different model
    python3 evaluation/generate_manual_analysis_csv.py --model mistral-large-latest

    # custom input / output paths
    python3 evaluation/generate_manual_analysis_csv.py \\
        --input  data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv \\
        --output evaluation/my_output.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT  = "data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv"
DEFAULT_OUTPUT = "evaluation/manual_error_analysis.csv"
PIPELINE_TYPE  = "1_pass_with_ontology_refined"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_clean_csv(results, output_path: str) -> None:
    """Write question_id / question / answer to a clean CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question", "answer"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "question_id": r.original_question_id,
                "question":    r.question,
                "answer":      r.answer,
            })

    print(f"\nManual analysis CSV written → {out}  ({len(results)} rows)")


def _validate_csv(output_path: str) -> None:
    """Basic sanity check: all rows must have non-empty question and answer."""
    issues = []
    with open(output_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if not row.get("question", "").strip():
                issues.append(f"row {i}: missing question")
            if not row.get("answer", "").strip():
                issues.append(f"row {i}: missing answer")
            if row.get("answer", "").startswith("ERROR"):
                issues.append(f"row {i}: pipeline error – {row['answer'][:80]}")

    if issues:
        print("\n⚠  Validation issues found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✓  Validation passed – all rows have non-empty question and answer")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manual error analysis CSV using the refined pipeline."
    )
    parser.add_argument(
        "--input",  default=DEFAULT_INPUT,
        help=f"Input CSV with DLR questions (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--num", type=int, default=None,
        help="Maximum number of questions to process",
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", default=None,
        help="Specific 0-based row indices to process",
    )
    parser.add_argument(
        "--model", default=None,
        help="Mistral model name to use (default: whatever DEFAULT_MODEL is in helpers)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run questions that were already processed",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-question progress output",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # lazy import so the script can be parsed without the full env set up
    # ------------------------------------------------------------------
    from core.main import PipelineOrchestrator
    from core.utils.helpers import DEFAULT_MODEL

    model = args.model or DEFAULT_MODEL

    orchestrator = PipelineOrchestrator(
        pipeline_type=PIPELINE_TYPE,
        model_name=model,
        overwrite=args.overwrite,
    )

    results, _raw_path, stats = orchestrator.run(
        input_csv=args.input,
        num_questions=args.num,
        # write full results to a temp path; we'll produce our own clean CSV
        output_csv="results_to_be_processed/refined-context_results.csv",
        verbose=not args.quiet,
        indices=args.indices,
    )

    if not results:
        print("No results produced – check your input CSV and pipeline configuration.")
        sys.exit(1)

    # write the clean manual-analysis CSV
    _write_clean_csv(results, args.output)

    # validate
    _validate_csv(args.output)

    sys.exit(0 if stats.failed == 0 else 1)


if __name__ == "__main__":
    main()
