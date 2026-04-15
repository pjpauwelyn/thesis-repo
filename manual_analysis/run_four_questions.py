"""run_four_questions.py

Runs four questions through the fully refined pipeline
(1_pass_with_ontology_refined) and writes a clean CSV to:

    manual_analysis/four_q_results.csv

Output columns: question_id, question, answer

── AQL SOURCE MODES ──────────────────────────────────────────────────────────

  --source json   (default)
      Loads per-question AQL context from JSON files in manual_analysis/:
          q1 → satellite_sensors.json
          q2 → optical_radar_sensors.json
          q3 → gedi.json
          q4 → accuracy.json
      Each file must be a JSON array of {title, abstract, uri, ...} objects.
      Extra fields are silently ignored.

  --source csv
      Load questions + AQL context from a CSV file (--csv <path>).
      Expected columns: question_id, question, aql_results
      aql_results must be the raw Python-repr string as stored in the DLR CSV.
      Use --num N to limit to the first N rows, or --indices 0 2 3 to pick rows.

  --source kg
      Placeholder for future live Knowledge Graph integration.
      Implement fetch_from_kg() below to connect to your ArangoDB/KG client.
      Must return a compact JSON string: '[{"title":"...","abstract":"...","uri":"..."}]'

── USAGE ────────────────────────────────────────────────────────────────────────

  PYTHONPATH=. python3 manual_analysis/run_four_questions.py
  PYTHONPATH=. python3 manual_analysis/run_four_questions.py --overwrite
  PYTHONPATH=. python3 manual_analysis/run_four_questions.py --model mistral-large-latest
  PYTHONPATH=. python3 manual_analysis/run_four_questions.py \\
      --source csv --csv data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv --num 4
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_ANALYSIS_DIR = Path("manual_analysis")
OUTPUT_PATH         = MANUAL_ANALYSIS_DIR / "four_q_results.csv"
PIPELINE_TYPE       = "1_pass_with_ontology_refined"

# Questions and their corresponding JSON filenames (--source json)
QUESTIONS: List[Tuple[str, str]] = [
    (
        "What satellite sensors can be used to monitor vegetation health?",
        "satellite_sensors.json",
    ),
    (
        "How do optical and radar sensors compare for forest monitoring?",
        "optical_radar_sensors.json",
    ),
    (
        "How does GEDI lidar compare to Sentinel-1 SAR in boreal forests?",
        "gedi.json",
    ),
    (
        "How does the accuracy of above-ground biomass estimation compare "
        "between GEDI lidar data and Sentinel-1 SAR backscatter in boreal forests?",
        "accuracy.json",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# AQL SOURCE: JSON files
# ─────────────────────────────────────────────────────────────────────────────

def load_from_json(filename: str, quiet: bool = False) -> Optional[str]:
    """Load a JSON file from manual_analysis/ and return a compact JSON string
    of [{title, abstract, uri}, ...] — the same format that parse_aql_results()
    produces from the DLR CSV.
    """
    json_path = MANUAL_ANALYSIS_DIR / filename

    if not json_path.exists():
        if not quiet:
            print(f"    ⚠  {json_path} not found — running without AQL context")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        if not quiet:
            print(f"    ⚠  {json_path} invalid JSON ({exc}) — running without AQL context")
        return None

    if not isinstance(data, list):
        if not quiet:
            print(f"    ⚠  {json_path} top-level is not a list — running without AQL context")
        return None

    clean = [
        {
            "title":    d.get("title", ""),
            "abstract": d.get("abstract", ""),
            "uri":      d.get("uri", ""),
        }
        for d in data if isinstance(d, dict)
    ]

    if not clean:
        if not quiet:
            print(f"    ⚠  {json_path} has no valid documents — running without AQL context")
        return None

    if not quiet:
        print(f"    ✓  {len(clean)} docs loaded from {filename}")
    return json.dumps(clean, separators=(",", ":"), ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# AQL SOURCE: CSV (DLR format)
# ─────────────────────────────────────────────────────────────────────────────

def load_from_csv(
    csv_path: str,
    num: Optional[int] = None,
    indices: Optional[List[int]] = None,
) -> List[Tuple[int, str, str]]:
    """Return [(question_id, question, raw_aql_results_str), ...] from a CSV
    with columns: question_id, question, aql_results.
    The raw aql_results string is passed as-is; parse_aql_results() handles
    ast.literal_eval internally inside the refinement agent.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if num is not None and len(rows) >= num:
                break
            if indices is not None and i not in indices:
                continue
            qid      = int(row.get("question_id") or (i + 1))
            question = row.get("question", "").strip()
            aql_raw  = row.get("aql_results", "").strip()
            if question:
                rows.append((qid, question, aql_raw))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# AQL SOURCE: live Knowledge Graph (future hook)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_from_kg(question: str, question_id: int) -> Optional[str]:
    """Fetch AQL results for a question from the live Knowledge Graph.

    TODO: implement once the KG client is available.

    Must return a compact JSON string:
        '[{"title":"...","abstract":"...","uri":"..."}]'
    or None if no results found.

    Skeleton:
        from core.kg.client import KGClient
        client = KGClient()
        docs = client.query(question)
        clean = [{"title": d["title"], "abstract": d["abstract"], "uri": d["uri"]}
                 for d in docs]
        return json.dumps(clean, separators=(",", ":"))
    """
    raise NotImplementedError(
        "fetch_from_kg() is not yet implemented. "
        "Connect your KG client here and return a compact JSON string."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV output + validation
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Single-question pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_question(
    qid: int,
    question: str,
    aql_results_str: Optional[str],
    ontology_agent,
    refinement_agent,
    generation_agent,
    quiet: bool = False,
) -> dict:
    t0 = time.time()
    try:
        # 1. ontology
        if not quiet:
            print("    → building ontology...", end=" ", flush=True)
        ontology = ontology_agent.process(question, include_relationships=False)
        if ontology and not ontology.should_use_ontology:
            ontology = None
        if not quiet:
            print("done" if ontology else "skipped")

        # 2. refinement
        if not quiet:
            print("    → refining context...", end=" ", flush=True)
        refined = refinement_agent.process_context(
            question=question,
            structured_context="",
            ontology=ontology,
            include_ontology=True,
            aql_results_str=aql_results_str,
            context_filter="full",
        )
        final_context = refined.enriched_context or ""
        if not quiet:
            print("done")

        # 3. generation
        if not quiet:
            print("    → generating answer...", end=" ", flush=True)
        answer_obj = generation_agent.generate(
            question=question,
            text_context=final_context,
            ontology=ontology,
        )
        if not quiet:
            print(f"done ({time.time() - t0:.1f}s)")

        return {"question_id": qid, "question": question, "answer": answer_obj.answer}

    except Exception as exc:
        if not quiet:
            print(f"    ✗ ERROR: {exc}")
        return {"question_id": qid, "question": question, "answer": f"ERROR: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the four manual-analysis questions through the refined pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", choices=["json", "csv", "kg"], default="json",
        help="AQL context source: json (default), csv, or kg",
    )
    parser.add_argument(
        "--csv", dest="csv_path", default=None,
        help="Input CSV path (required when --source csv)",
    )
    parser.add_argument(
        "--num", type=int, default=None,
        help="Max rows to process (csv mode only)",
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", default=None,
        help="0-based row indices to process (csv mode only)",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help=f"Output CSV path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model", default=None,
        help="Mistral model name (default: DEFAULT_MODEL from helpers)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run and overwrite output if it already exists",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step progress output",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists: {output_path}")
        print("Use --overwrite to re-run.")
        sys.exit(0)

    if args.source == "csv" and not args.csv_path:
        parser.error("--source csv requires --csv <path>")

    from core.utils.helpers import get_llm_model, DEFAULT_MODEL
    from core.agents.ontology_agent import OntologyConstructionAgent
    from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
    from core.agents.generation_agent import GenerationAgent

    model_name = args.model or DEFAULT_MODEL

    if not args.quiet:
        print(f"Model    : {model_name}")
        print(f"Pipeline : {PIPELINE_TYPE}")
        print(f"Source   : {args.source}")
        print(f"JSON dir : {MANUAL_ANALYSIS_DIR}")
        print(f"Output   : {output_path}\n")

    llm     = get_llm_model(model=model_name, temperature=0.2)
    llm_ref = get_llm_model(model=model_name, temperature=0.1)

    ontology_agent   = OntologyConstructionAgent(llm_ref, prompt_dir="prompts/ontology")
    refinement_agent = RefinementAgent1PassRefined(llm_ref, prompt_dir="prompts/refinement")
    generation_agent = GenerationAgent(llm, prompt_dir="prompts/generation")

    # Build work list: [(question_id, question, aql_results_str | None)]
    work: List[Tuple[int, str, Optional[str]]] = []

    if args.source == "csv":
        for qid, question, aql in load_from_csv(args.csv_path, args.num, args.indices):
            work.append((qid, question, aql))

    elif args.source == "kg":
        for qid, (question, _) in enumerate(QUESTIONS, start=1):
            aql = fetch_from_kg(question, qid)
            work.append((qid, question, aql))

    else:  # json (default)
        for qid, (question, filename) in enumerate(QUESTIONS, start=1):
            aql = load_from_json(filename, quiet=args.quiet)
            work.append((qid, question, aql))

    results = []
    for i, (qid, question, aql_results_str) in enumerate(work):
        if not args.quiet:
            print(f"[{i + 1}/{len(work)}] {question[:80]}")
        row = run_question(
            qid, question, aql_results_str,
            ontology_agent, refinement_agent, generation_agent,
            quiet=args.quiet,
        )
        results.append(row)

    _write_csv(results, output_path)
    _validate(output_path)

    failed = sum(1 for r in results if str(r["answer"]).startswith("ERROR"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
