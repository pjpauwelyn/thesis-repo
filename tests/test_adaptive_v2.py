"""3-phase adaptive pipeline test suite.

phase 1: profile + tier routing on all questions  (inspection only -- no assertions)
phase 2: document filter dry-run on up to 20 questions with docs
phase 3: full-pipeline generation on all questions that have docs
         (append/resume: already-completed questions are skipped on restart)

Run all phases:
    pytest tests/test_adaptive_v2.py -v -s

Run a single phase:
    pytest tests/test_adaptive_v2.py::test_phase1_profiles -v -s
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from core.pipelines.pipeline import Pipeline
from core.utils.logger import (
    configure_pipeline_logging,
    log_question_end,
    log_question_start,
)

# bootstrap logging for the test session
configure_pipeline_logging(log_file="logs/test_adaptive.log")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def _find_dlr_csv() -> Optional[str]:
    candidates = [
        "data/dlr/questions.csv",
        "data/dlr/dlr_questions.csv",
        "data/questions.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    data_root = Path("data")
    if data_root.exists():
        for csv_path in sorted(data_root.rglob("*.csv")):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    header = f.readline()
                if "aql_results" in header or "question" in header.lower():
                    return str(csv_path)
            except Exception:
                pass
    return None


def _load_questions(csv_path: str) -> List[Dict[str, Any]]:
    csv.field_size_limit(int(1e8))
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _parse_docs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = row.get("aql_results", "") or ""
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def _get_question(row: Dict[str, Any]) -> str:
    for key in ("question", "Question", "query", "Query"):
        if key in row and row[key]:
            return row[key].strip()
    return ""


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("tests/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_CSV_PATH: Optional[str] = _find_dlr_csv()
_ROWS_RAW: List[Dict[str, Any]] = _load_questions(_CSV_PATH) if _CSV_PATH else []
_seen: set = set()
_ROWS: List[Dict[str, Any]] = []
for _r in _ROWS_RAW:
    _key = _get_question(_r).strip().lower()
    if _key and _key not in _seen:
        _seen.add(_key)
        _ROWS.append(_r)
_PIPELINE = Pipeline()


# ---------------------------------------------------------------------------
# phase 1: profile + tier routing on all questions
# ---------------------------------------------------------------------------

def test_phase1_profiles() -> None:
    """profile every question and write routing decisions to tests/output/.
    no assertions -- purely for inspection."""
    if not _ROWS:
        pytest.skip("no DLR CSV found")

    jsonl_path = OUTPUT_DIR / "phase1_profiles.jsonl"
    txt_path   = OUTPUT_DIR / "phase1_summary.txt"

    tier_counts: Dict[str, int] = {}

    with open(jsonl_path, "w", encoding="utf-8") as jf, \
         open(txt_path,   "w", encoding="utf-8") as tf:

        for i, row in enumerate(_ROWS, start=1):
            question = _get_question(row)
            if not question:
                continue

            ontology, profile, cfg = _PIPELINE.profile_and_route(question)
            tier_counts[cfg.rule_hit] = tier_counts.get(cfg.rule_hit, 0) + 1

            record = {
                "q_index":    i,
                "question":   question,
                "profile":    profile.model_dump(),
                "tier":       cfg.rule_hit,
                "confidence": profile.confidence,
                "use_draft":  cfg.use_draft,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            conf_str = (
                f"{profile.confidence:.2f}"
                if profile.confidence is not None
                else "None"
            )
            tf.write(
                f"  type={profile.question_type:<12} "
                f"complexity={profile.complexity:.2f}  "
                f"quant={profile.quantitativity:.2f}  "
                f"meth={profile.methodological_depth:.2f}  "
                f"conf={conf_str}  "
                f"tier={cfg.rule_hit}  "
                f"use_draft={cfg.use_draft}\n"
            )

        tf.write("\nTier distribution:\n")
        for tier, count in sorted(tier_counts.items()):
            tf.write(f"  {tier}: {count}\n")

    log.info("phase 1 complete -- %d questions profiled, tiers: %s", len(_ROWS), tier_counts)
    print(f"\nPhase 1 output: {txt_path}")


# ---------------------------------------------------------------------------
# phase 2: document filter dry-run (20 questions)
# ---------------------------------------------------------------------------

def test_phase2_filter() -> None:
    """run filter_documents on up to 20 questions that have docs.
    asserts: at least min_keep docs survive the filter."""
    if not _ROWS:
        pytest.skip("no DLR CSV found")

    rows_with_docs = [r for r in _ROWS if _parse_docs(r)][:20]
    if not rows_with_docs:
        pytest.skip("no rows with non-empty aql_results found")

    jsonl_path = OUTPUT_DIR / "phase2_filter.jsonl"
    txt_path   = OUTPUT_DIR / "phase2_filter_readable.txt"

    with open(jsonl_path, "w", encoding="utf-8") as jf, \
         open(txt_path,   "w", encoding="utf-8") as tf:

        for i, row in enumerate(rows_with_docs, start=1):
            question = _get_question(row)
            docs     = _parse_docs(row)
            if not question or not docs:
                continue

            ontology, profile, cfg, filter_summary = \
                _PIPELINE.profile_and_route_with_filter(question, docs=docs)

            record = {
                "q_index":        i,
                "question":       question,
                "tier":           cfg.rule_hit,
                "q_type":         profile.question_type,
                "filter_summary": filter_summary,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_kept = filter_summary.get("n_full", 0) + filter_summary.get("n_abstract", 0)
            tf.write(
                f"Q{i} [{cfg.rule_hit} | {profile.question_type}]: "
                f"{question[:90]}\n"
                f"  full={filter_summary.get('n_full', 0)}  "
                f"abstract={filter_summary.get('n_abstract', 0)}  "
                f"dropped={filter_summary.get('n_drop', 0)}  "
                f"(of {filter_summary.get('n_total', 0)})\n"
            )

            for entry in filter_summary.get("full_titles", []):
                title = entry if isinstance(entry, str) else entry.get("title", "?")
                sim   = entry.get("sim", "") if isinstance(entry, dict) else ""
                flag  = entry.get("title_match", "") if isinstance(entry, dict) else ""
                sim_str = f" sim={sim:.3f}" if sim != "" else ""
                tf.write(f"    [FULL]     {title}{sim_str}{flag}\n")

            for entry in filter_summary.get("abstract_titles", []):
                title = entry if isinstance(entry, str) else entry.get("title", "?")
                sim   = entry.get("sim", "") if isinstance(entry, dict) else ""
                flag  = entry.get("title_match", "") if isinstance(entry, dict) else ""
                sim_str = f" sim={sim:.3f}" if sim != "" else ""
                tf.write(f"    [ABSTRACT] {title}{sim_str}{flag}\n")

            for entry in filter_summary.get("drop_titles", []):
                title = entry if isinstance(entry, str) else entry.get("title", "?")
                sim   = entry.get("sim", "") if isinstance(entry, dict) else ""
                flag  = entry.get("title_match", "") if isinstance(entry, dict) else ""
                sim_str = f" sim={sim:.3f}" if sim != "" else ""
                tf.write(f"    [DROP]     {title}{sim_str}{flag}\n")

            tf.write("---\n")

            assert n_kept >= cfg.doc_filter_min_keep or n_kept == len(docs), (
                f"filter dropped too many docs: kept {n_kept} of {len(docs)} "
                f"(min_keep={cfg.doc_filter_min_keep})"
            )

    log.info("phase 2 complete")
    print(f"\nPhase 2 output: {txt_path}")


# ---------------------------------------------------------------------------
# phase 3 helpers
# ---------------------------------------------------------------------------

def _load_completed_questions(jsonl_path: Path) -> Set[str]:
    """Return the set of lowercase question strings already saved in the JSONL output.

    Keys are lowercased to match the deduplication applied to _ROWS, so a
    question that was saved with different casing is still recognised as done.
    Returns an empty set if the file does not exist.
    """
    done: Set[str] = set()
    if not jsonl_path.exists():
        return done
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    q = rec.get("question", "")
                    if q:
                        done.add(q.strip().lower())
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return done


def _write_readable_answer(
    tf,
    i: int,
    question: str,
    ans,
    elapsed: float,
) -> None:
    """Append a single Q&A block to the human-readable text file."""
    tf.write("=" * 60 + "\n")
    tf.write(
        f"Q{i} [actual={ans.rule_hit} "
        f"| use_draft={ans.pipeline_config.use_draft if ans.pipeline_config else 'N/A'}"
        f"| enriched={ans.enriched_context_chars if hasattr(ans, 'enriched_context_chars') else len(ans.enriched_context)}"
        f"chars | {elapsed:.1f}s]\n"
    )
    tf.write(f"QUESTION: {question}\n\n")
    tf.write("ANSWER:\n")
    tf.write(ans.answer + "\n")
    tf.write("=" * 60 + "\n\n")


# ---------------------------------------------------------------------------
# phase 3: full-pipeline generation on all questions
# ---------------------------------------------------------------------------

def test_phase3_generation() -> None:
    """Run the full pipeline on every question that has associated docs.

    Append/resume behaviour: if the JSONL output already exists from a
    previous (partial) run, questions whose answers are already recorded
    are skipped so the run can be continued without re-doing completed work.
    """
    if not _ROWS:
        pytest.skip("no DLR CSV found")

    rows_with_docs = [r for r in _ROWS if _parse_docs(r)]
    if not rows_with_docs:
        pytest.skip("no rows with non-empty aql_results found")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUTPUT_DIR / f"phase3_answers_{ts}.jsonl"
    txt_path   = OUTPUT_DIR / f"phase3_answers_readable_{ts}.txt"
    log_path   = Path("logs") / f"phase3_run_{ts}.log"

    configure_pipeline_logging(log_file=str(log_path))

    completed = _load_completed_questions(jsonl_path)
    n_skipped = 0
    n_done    = 0
    n_fail    = 0

    with open(jsonl_path, "a", encoding="utf-8") as jf, \
         open(txt_path,   "a", encoding="utf-8") as tf:

        for i, row in enumerate(rows_with_docs, start=1):
            question = _get_question(row)
            docs     = _parse_docs(row)
            if not question or not docs:
                continue

            if question.strip().lower() in completed:
                n_skipped += 1
                log.debug("phase3: skipping already-completed Q%d '%s...'", i, question[:50])
                continue

            log_question_start(log, i, question)
            t0 = time.perf_counter()

            try:
                ans = _PIPELINE.run(question=question, docs=docs)
                status = "ok"
            except Exception as exc:
                status = "fail"
                n_fail += 1
                log.error("phase3 Q%d FAILED: %s", i, exc, exc_info=True)
                elapsed = time.perf_counter() - t0
                log_question_end(log, i, status, elapsed)

                fail_record = {
                    "q_index":              i,
                    "question":             question,
                    "status":               "fail",
                    "error":                str(exc),
                    "answer":               "",
                    "enriched_context_chars": 0,
                    "excerpt_stats":        {},
                    "references":           [],
                    "formatted_references": [],
                }
                jf.write(json.dumps(fail_record, ensure_ascii=False) + "\n")
                jf.flush()
                continue  # isolate failure; remaining questions still run

            elapsed = time.perf_counter() - t0
            log_question_end(log, i, status, elapsed)
            n_done += 1

            record = {
                "q_index":                i,
                "question":               question,
                "expected_tier":          "N/A",
                "actual_tier":            ans.rule_hit,
                "use_draft":              ans.pipeline_config.use_draft if ans.pipeline_config else None,
                "answer":                 ans.answer,
                "enriched_context_chars": len(ans.enriched_context),
                "excerpt_stats":          ans.excerpt_stats,
                "references":             ans.references,
                "formatted_references":   ans.formatted_references,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            jf.flush()

            tf.write("=" * 60 + "\n")
            tf.write(
                f"Q{i} [actual={ans.rule_hit} "
                f"| use_draft={ans.pipeline_config.use_draft if ans.pipeline_config else 'N/A'}"
                f"| enriched={len(ans.enriched_context)}chars"
                f" | {elapsed:.1f}s]\n"
            )
            tf.write(f"QUESTION: {question}\n\n")
            tf.write("ANSWER:\n")
            tf.write(ans.answer + "\n")
            tf.write("=" * 60 + "\n\n")
            tf.flush()

    log.info(
        "phase 3 complete -- done=%d  skipped=%d  failed=%d",
        n_done, n_skipped, n_fail,
    )
    print(f"\nPhase 3 outputs:\n  JSONL:    {jsonl_path}\n  Readable: {txt_path}\n  Log:      {log_path}")
    assert n_fail == 0 or n_done > 0, (
        f"phase 3: every question failed (n_fail={n_fail}, n_done={n_done})"
    )
