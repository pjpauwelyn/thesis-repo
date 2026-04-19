"""run_fulltext_pipeline.py

Generate the manual-analysis CSV for the full-indexing variant of the refined
RAG pipeline.

For every question it:
    1. parses the AQL results  (title/abstract/uri per document)
    2. builds full-text excerpts for each OA document using FullTextIndexer
       (ontology-weighted top-K chunks)
    3. runs Ontology → Fulltext-Refinement → Generation agents with Mistral
    4. writes (question_id, question, answer) to the output CSV so it can be
       consumed by evaluation/run_judge.py exactly like Exp 1 / Exp 2.

This file does NOT modify any existing pipeline file; it wires the existing
agents together via a small orchestrator-lite and a new refinement agent
subclass that accepts per-question excerpts.

Usage:
    python3 evaluation/run_fulltext_pipeline.py \
        --model mistral-small-latest \
        --output evaluation/results/exp3_small_fulltext_results.csv

    python3 evaluation/run_fulltext_pipeline.py \
        --model mistral-large-latest \
        --output evaluation/results/exp4_large_fulltext_results.csv

    # prebuild the PDF / text cache once before running:
    python3 -m core.utils.fulltext_indexer prebuild \
        --input data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

csv.field_size_limit(int(1e8))

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv"
DEFAULT_CACHE = "cache/fulltext"

PER_DOC_BUDGET_DEFAULT = 6000
GLOBAL_BUDGET_DEFAULT = 90000
TOP_K_DEFAULT = 10


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> logging.Logger:
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # silence noisy libs
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("run_fulltext_pipeline")


def _load_questions(input_csv: str, num: Optional[int], indices: Optional[List[int]]) -> List[Dict[str, Any]]:
    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if indices:
        rows = [r for i, r in enumerate(rows) if i in set(indices)]
    if num:
        rows = rows[:num]
    return rows


def _write_row(out_path: Path, row: Dict[str, str]) -> None:
    """incremental writer: append to CSV, creating header if file is new."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "question", "answer"])
        if new_file:
            w.writeheader()
        w.writerow(row)


def _load_already_done(out_path: Path) -> Dict[str, str]:
    if not out_path.exists():
        return {}
    done: Dict[str, str] = {}
    with open(out_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row.get("question_id", "").strip()
            ans = row.get("answer", "").strip()
            if qid and ans and not ans.startswith("ERROR"):
                done[qid] = ans
    return done


def main():
    ap = argparse.ArgumentParser(description="Fulltext-augmented refined pipeline runner.")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", required=True, help="CSV path (question_id/question/answer)")
    ap.add_argument("--model", required=True, help="e.g. mistral-small-latest, mistral-large-latest")
    ap.add_argument("--num", type=int, default=None, help="limit number of questions")
    ap.add_argument("--indices", type=int, nargs="+", default=None)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--per-doc-budget", type=int, default=PER_DOC_BUDGET_DEFAULT)
    ap.add_argument("--global-budget", type=int, default=GLOBAL_BUDGET_DEFAULT)
    ap.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    ap.add_argument("--refinement-temp", type=float, default=0.1)
    ap.add_argument("--generation-temp", type=float, default=0.2)
    ap.add_argument("--overwrite", action="store_true", help="ignore existing answers in output csv")
    ap.add_argument("--dry-run", action="store_true", help="build excerpts + prompt but don't call LLM")
    ap.add_argument("--stats-json", default=None, help="optional path to dump per-question excerpt stats")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = _setup_logging(args.verbose)

    # lazy imports so the file still parses without the full env
    from core.utils.helpers import get_llm_model
    from core.utils.aql_parser import parse_aql_results
    from core.utils.fulltext_indexer import FullTextIndexer
    from core.agents.ontology_agent import OntologyConstructionAgent
    from core.agents.refinement_agent_fulltext import RefinementAgent1PassFullText
    from core.agents.generation_agent import GenerationAgent

    # ---------- indexer ----------
    fti = FullTextIndexer(cache_dir=Path(args.cache_dir))

    # ---------- agents ----------
    ont_llm = get_llm_model(model=args.model, temperature=args.refinement_temp)
    ref_llm = get_llm_model(model=args.model, temperature=args.refinement_temp)
    gen_llm = get_llm_model(model=args.model, temperature=args.generation_temp)

    ontology_agent = OntologyConstructionAgent(ont_llm, prompt_dir="prompts/ontology")
    refinement_agent = RefinementAgent1PassFullText(ref_llm, prompt_dir="prompts/refinement")
    generation_agent = GenerationAgent(gen_llm, prompt_dir="prompts/generation")

    # ---------- data ----------
    rows = _load_questions(args.input, args.num, args.indices)
    if not rows:
        log.error("no rows loaded from %s", args.input)
        sys.exit(1)

    out_path = Path(args.output)
    done = {} if args.overwrite else _load_already_done(out_path)
    if done:
        log.info("resume: %d answers already in %s, skipping those", len(done), out_path)

    all_stats: List[Dict[str, Any]] = []
    errors = 0
    successes = 0
    t_start = time.time()

    for i, row in enumerate(rows, start=1):
        qid = str(row.get("question_id") or i)
        question = row.get("question", "").strip()
        if not question:
            log.warning("row %d has empty question, skipping", i)
            continue

        if qid in done:
            log.info("[%s] already done, skip", qid)
            continue

        print(f"\n[{i}/{len(rows)}] qid={qid}: {question[:80]}...")

        # ---- parse AQL -> docs ----
        aql_raw = row.get("aql_results", "") or ""
        docs: List[Dict[str, Any]] = []
        parsed_aql_str = ""
        if aql_raw:
            parsed_aql_str = parse_aql_results(aql_raw)
            try:
                parsed = json.loads(parsed_aql_str)
                if isinstance(parsed, list):
                    docs = parsed
            except json.JSONDecodeError:
                docs = []
        log.info("  aql docs: %d", len(docs))

        # ---- ontology ----
        t0 = time.time()
        try:
            ontology = ontology_agent.process(question, include_relationships=False)
            if ontology and not getattr(ontology, "should_use_ontology", True):
                ontology = None
        except Exception as exc:
            log.error("  ontology failed: %s", exc)
            ontology = None
        print(f"  ontology: {'ok' if ontology else 'skipped'} ({time.time()-t0:.1f}s)")

        # ---- fulltext excerpts ----
        t0 = time.time()
        try:
            excerpts, stats = fti.select_excerpts_for_question(
                question=question,
                ontology=ontology,
                documents=docs,
                per_doc_budget=args.per_doc_budget,
                global_budget=args.global_budget,
                top_k_per_doc=args.top_k,
            )
        except Exception as exc:
            log.error("  excerpt selection failed: %s", exc)
            excerpts, stats = [], {"error": str(exc)}
        excerpts_text = fti.render_excerpts_block(excerpts)
        stats["qid"] = qid
        stats["excerpt_chars"] = len(excerpts_text)
        stats["excerpt_tokens_est"] = len(excerpts_text) // 4
        all_stats.append(stats)
        print(
            f"  excerpts: {stats.get('n_excerpts',0)} chunks "
            f"from {stats.get('n_docs_with_chunks',0)}/{stats.get('n_docs',0)} docs "
            f"(~{stats.get('total_tokens',0)} tok, {time.time()-t0:.1f}s)"
        )

        if args.dry_run:
            print("  (dry-run: skipping LLM calls)")
            continue

        # ---- refinement with injected excerpts ----
        refinement_agent.set_excerpts(excerpts_text)
        t0 = time.time()
        try:
            refined = refinement_agent.process_context(
                question=question,
                structured_context="",
                ontology=ontology,
                include_ontology=True,
                aql_results_str=parsed_aql_str or aql_raw,
                context_filter="full",
            )
            refined_text = refined.enriched_context or ""
        except Exception as exc:
            log.error("  refinement failed: %s", exc)
            errors += 1
            _write_row(out_path, {"question_id": qid, "question": question, "answer": f"ERROR: refinement failed: {exc}"})
            continue
        print(f"  refinement: {len(refined_text)} chars ({time.time()-t0:.1f}s)")

        # ---- generation ----
        t0 = time.time()
        try:
            ans_obj = generation_agent.generate(
                question=question,
                text_context=refined_text,
                ontology=ontology,
            )
            answer = ans_obj.answer or ""
        except Exception as exc:
            log.error("  generation failed: %s", exc)
            errors += 1
            _write_row(out_path, {"question_id": qid, "question": question, "answer": f"ERROR: generation failed: {exc}"})
            continue
        print(f"  generation: {len(answer)} chars ({time.time()-t0:.1f}s)")

        _write_row(out_path, {"question_id": qid, "question": question, "answer": answer})
        successes += 1

    # final summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"fulltext pipeline complete: ok={successes}, err={errors}, elapsed={elapsed/60:.1f} min")
    print(f"output: {out_path}")
    if args.stats_json:
        Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_json).write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
        print(f"stats:  {args.stats_json}")
    print("=" * 72)

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
