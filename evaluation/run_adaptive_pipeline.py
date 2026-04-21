"""run_adaptive_pipeline.py

generate the answers csv for the set5 adaptive pipeline. per question:
    1. profile     -- small llm, fused av-pair + profile json
    2. route       -- core.policy.Router picks PipelineConfig
    3. evidence    -- abstracts (no extra work) | excerpts via FullTextIndexer
    4. refinement  -- 1pass-refined or 1pass-fulltext, depending on evidence
    5. generation  -- direct or structured, with adaptive directives

the output csv has exactly the columns needed by run_judge.py
(question_id, question, answer), plus telemetry columns so we can
post-hoc inspect the routing decisions. telemetry columns are ignored
by run_judge.py.

usage:
    # full run (4 parallel workers)
    python3 evaluation/run_adaptive_pipeline.py \
        --output evaluation/results/set5_adaptive/answers_adaptive.csv

    # custom worker count
    python3 evaluation/run_adaptive_pipeline.py --workers 2 \
        --output evaluation/results/set5_adaptive/answers_adaptive.csv

    # dry-run: profile + route only, no LLM calls downstream (prints histogram)
    python3 evaluation/run_adaptive_pipeline.py --dry-run \
        --output evaluation/results/set5_adaptive/routing_preview.csv

    # prebuild fulltext cache first (recommended):
    python3 -m core.utils.fulltext_indexer prebuild \
        --input data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

csv.field_size_limit(int(1e8))

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv"
DEFAULT_OUTPUT = "evaluation/results/set5_adaptive/answers_adaptive.csv"
DEFAULT_CACHE = "cache/fulltext"
DEFAULT_RULES = "core/policy/rules.yaml"
DEFAULT_PROMPTS = "prompts"
DEFAULT_WORKERS = 4  # parallel workers; stays within Mistral rate limits

# telemetry columns added beyond the run_judge.py-required triple
_FIELDNAMES = [
    "question_id", "question", "answer",
    "rule_hit", "model_name", "evidence_mode",
    "expected_length", "answer_shape",
    "complexity", "quantitativity",
    "spatial_specificity", "temporal_specificity",
    "profile_confidence", "profile_json",
]

# minimum answer length (chars) to consider a row "done"
_MIN_DONE_ANSWER_CHARS = 200

# global lock for thread-safe csv writes
_CSV_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> logging.Logger:
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("run_adaptive_pipeline")


def _load_questions(
    input_csv: str,
    num: Optional[int],
    indices: Optional[List[int]],
) -> List[Dict[str, Any]]:
    """load the dlr csv and preserve 0-based row index for stable qids."""
    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["_row_index"] = i
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(r)
    if indices:
        deduped = [r for r in deduped if r["_row_index"] in set(indices)]
    if num:
        deduped = deduped[:num]
    return deduped


def _write_row(out_path: Path, row: Dict[str, Any]) -> None:
    """thread-safe incremental csv writer; replaces existing row for same qid."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_qid = str(row.get("question_id", "")).strip()
    with _CSV_LOCK:
        if out_path.exists():
            existing: List[Dict[str, str]] = []
            with open(out_path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if str(r.get("question_id", "")).strip() == new_qid:
                        continue
                    existing.append({k: r.get(k, "") for k in _FIELDNAMES})
            existing.append({k: str(row.get(k, "")) for k in _FIELDNAMES})
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                w.writeheader()
                for r in existing:
                    w.writerow(r)
        else:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                w.writeheader()
                w.writerow({k: str(row.get(k, "")) for k in _FIELDNAMES})


def _load_already_done(out_path: Path) -> Dict[str, str]:
    if not out_path.exists():
        return {}
    done: Dict[str, str] = {}
    with open(out_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = (row.get("question_id") or "").strip()
            ans = (row.get("answer") or "").strip()
            if len(ans) < _MIN_DONE_ANSWER_CHARS:
                continue
            if qid and ans and not ans.startswith("ERROR"):
                done[qid] = ans
    return done


def _print_histogram(counter: Counter, title: str) -> None:
    print(f"\n{title}")
    total = sum(counter.values()) or 1
    for key, n in counter.most_common():
        bar = "#" * int(40 * n / total)
        print(f"  {str(key):28s} {n:4d}  {bar}")
    print()


# ---------------------------------------------------------------------------
# per-question worker
# ---------------------------------------------------------------------------

def _process_row(
    i: int,
    total: int,
    row: Dict[str, Any],
    pipeline,
    out_path: Path,
    dry_run: bool,
    log: logging.Logger,
):
    """profile, route, and generate for a single question row.
    returns (rule_hit, evidence_mode, model_name, answer_shape, is_error).
    """
    from core.utils.aql_parser import parse_aql_results

    orig_idx = row.get("_row_index")
    qid = str(row.get("question_id") or (int(orig_idx) + 1 if orig_idx is not None else i))
    question = (row.get("question") or "").strip()

    print(f"[{i}/{total}] qid={qid}: {question[:80]}...")

    aql_raw = row.get("aql_results", "") or ""
    parsed_aql_str = ""
    docs: List[Dict[str, Any]] = []
    if aql_raw:
        parsed_aql_str = parse_aql_results(aql_raw)
        try:
            parsed = json.loads(parsed_aql_str)
            if isinstance(parsed, list):
                docs = parsed
        except json.JSONDecodeError:
            docs = []

    if dry_run:
        try:
            _ontology, profile, cfg = pipeline.profile_and_route(question)
        except Exception as exc:
            log.error("profile_and_route failed qid=%s: %s", qid, exc)
            return None, None, None, None, True
        print(f"  -> rule={cfg.rule_hit} model={cfg.model_name} evidence={cfg.evidence_mode} shape={profile.answer_shape}")
        _write_row(out_path, {
            "question_id": qid, "question": question, "answer": "(dry-run)",
            "rule_hit": cfg.rule_hit, "model_name": cfg.model_name,
            "evidence_mode": cfg.evidence_mode, "expected_length": profile.expected_length,
            "answer_shape": profile.answer_shape, "complexity": profile.complexity,
            "quantitativity": profile.quantitativity,
            "spatial_specificity": profile.spatial_specificity,
            "temporal_specificity": profile.temporal_specificity,
            "profile_confidence": profile.confidence,
            "profile_json": profile.model_dump_json(),
        })
        return cfg.rule_hit, cfg.evidence_mode, cfg.model_name, profile.answer_shape, False

    t0 = time.time()
    try:
        result = pipeline.run(
            question=question,
            aql_results_str=parsed_aql_str or aql_raw,
            docs=docs,
        )
    except Exception as exc:
        log.error("adaptive run failed qid=%s: %s", qid, exc)
        _write_row(out_path, {"question_id": qid, "question": question, "answer": f"ERROR: {exc}"})
        return None, None, None, None, True

    prof = result.profile
    cfg = result.pipeline_config
    print(
        f"  -> rule={cfg.rule_hit if cfg else '?'} model={cfg.model_name if cfg else '?'} "
        f"evidence={cfg.evidence_mode if cfg else '?'} "
        f"answer_chars={len(result.answer)} ({time.time()-t0:.1f}s)"
    )
    _write_row(out_path, {
        "question_id": qid, "question": question, "answer": result.answer,
        "rule_hit": cfg.rule_hit if cfg else "",
        "model_name": cfg.model_name if cfg else "",
        "evidence_mode": cfg.evidence_mode if cfg else "",
        "expected_length": prof.expected_length if prof else "",
        "answer_shape": prof.answer_shape if prof else "",
        "complexity": prof.complexity if prof else "",
        "quantitativity": prof.quantitativity if prof else "",
        "spatial_specificity": prof.spatial_specificity if prof else "",
        "temporal_specificity": prof.temporal_specificity if prof else "",
        "profile_confidence": prof.confidence if prof else "",
        "profile_json": prof.model_dump_json() if prof else "",
    })
    return (
        cfg.rule_hit if cfg else "?",
        cfg.evidence_mode if cfg else "?",
        cfg.model_name if cfg else "?",
        prof.answer_shape if prof else "?",
        False,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="adaptive pipeline runner (set5).")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--num", type=int, default=None, help="limit number of questions")
    ap.add_argument("--indices", type=int, nargs="+", default=None)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--prompts-root", default=DEFAULT_PROMPTS)
    ap.add_argument("--overwrite", action="store_true", help="re-run already-done questions")
    ap.add_argument("--dry-run", action="store_true",
                    help="profile + route only, no generation (prints histogram)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="parallel worker threads (default: 4, use 1 to disable)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = _setup_logging(args.verbose)

    from core.pipelines.adaptive_pipeline import AdaptivePipeline

    pipeline = AdaptivePipeline(
        rules_path=args.rules,
        cache_dir=args.cache_dir,
        prompts_root=args.prompts_root,
    )

    rows = _load_questions(args.input, args.num, args.indices)
    if not rows:
        log.error("no rows loaded from %s", args.input)
        sys.exit(1)

    out_path = Path(args.output)
    done = {} if (args.overwrite or args.dry_run) else _load_already_done(out_path)
    if done:
        log.info("resume: %d already done in %s", len(done), out_path)

    pending = []
    for i, row in enumerate(rows, start=1):
        orig_idx = row.get("_row_index")
        qid = str(row.get("question_id") or (int(orig_idx) + 1 if orig_idx is not None else i))
        if qid in done:
            log.info("[%s] skip (already done)", qid)
        else:
            pending.append((i, row))

    log.info("%d questions to process (%d workers)", len(pending), args.workers)

    rule_hist: Counter = Counter()
    evidence_hist: Counter = Counter()
    model_hist: Counter = Counter()
    shape_hist: Counter = Counter()
    errors = 0
    successes = 0
    total = len(rows)
    t_start = time.time()

    max_workers = 1 if args.dry_run else max(1, args.workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_row, idx, total, row, pipeline, out_path, args.dry_run, log): (idx, row)
            for idx, row in pending
        }
        for fut in concurrent.futures.as_completed(futures):
            rule, evidence, model, shape, is_err = fut.result()
            if is_err:
                errors += 1
            else:
                successes += 1
                if rule:     rule_hist[rule] += 1
                if evidence: evidence_hist[evidence] += 1
                if model:    model_hist[model] += 1
                if shape:    shape_hist[shape] += 1

    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    mode = "dry-run" if args.dry_run else "full run"
    print(f"adaptive pipeline {mode} complete: ok={successes}, err={errors}, elapsed={elapsed/60:.1f} min")
    print(f"output: {out_path}")
    _print_histogram(rule_hist, "rule_hit distribution")
    _print_histogram(evidence_hist, "evidence_mode distribution")
    _print_histogram(model_hist, "model_name distribution")
    _print_histogram(shape_hist, "answer_shape distribution")
    print("=" * 72)

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
