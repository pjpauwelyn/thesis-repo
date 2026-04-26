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
            jf.write(json.dumps(record) + "\n")

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
            jf.write(json.dumps(record) + "\n")

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
    """Return the set of question strings already saved in the JSONL output.

    Used to skip completed questions when resuming an interrupted run.
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
                    q = rec.get("question", "").strip()
                    if q:
                        done.add(q)
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        log.warning("could not read existing phase3 output (%s) -- starting fresh", exc)
        return set()
    return done


def _pick_all_questions_with_docs() -> List[Tuple[str, str]]:
    """Return (question, aql_results_str) for every row that has aql_results.

    Preserves CSV order. Used by test_phase3_generation for the full run.
    """
    result: List[Tuple[str, str]] = []
    for row in _ROWS:
        q   = _get_question(row)
        aql = row.get("aql_results", "") or ""
        if q and aql.strip():
            result.append((q, aql))
    return result


def _pick_questions_by_tier(
    target_counts: Dict[str, int],
) -> List[Tuple[str, str, str]]:
    """Return a sample of (question, aql_results_str, tier_hit) spread across
    tiers.  Used for small targeted smoke tests.

    Reads tier assignments from phase1 output if available; otherwise
    runs inline profiling so the test is self-contained.
    """
    phase1_path = OUTPUT_DIR / "phase1_profiles.jsonl"
    selected: List[Tuple[str, str, str]] = []

    if phase1_path.exists():
        tier_map: Dict[str, str] = {}
        with open(phase1_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    tier_map[rec["question"]] = rec["tier"]
                except Exception:
                    pass

        counts: Dict[str, int] = {}
        for row in _ROWS:
            q    = _get_question(row)
            tier = tier_map.get(q, "")
            need = target_counts.get(tier, 0)
            if need and counts.get(tier, 0) < need:
                selected.append((q, row.get("aql_results", "") or "", tier))
                counts[tier] = counts.get(tier, 0) + 1
            if sum(counts.values()) >= sum(target_counts.values()):
                break
    else:
        counts: Dict[str, int] = {}
        for row in _ROWS:
            q = _get_question(row)
            if not q:
                continue
            _, _, cfg = _PIPELINE.profile_and_route(q)
            tier = cfg.rule_hit
            need = target_counts.get(tier, 0)
            if need and counts.get(tier, 0) < need:
                selected.append((q, row.get("aql_results", "") or "", tier))
                counts[tier] = counts.get(tier, 0) + 1
            if sum(counts.values()) >= sum(target_counts.values()):
                break

    return selected


# ---------------------------------------------------------------------------
# phase 3: full-pipeline generation (all questions with docs, append/resume)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(7200)
def test_phase3_generation() -> None:
    """Run full pipeline on every question that has aql_results docs.

    APPEND / RESUME behaviour
    -------------------------
    On first run: creates phase3_answers.jsonl and phase3_answers_readable.txt
    fresh.
    On restart after interruption: reads phase3_answers.jsonl to find
    already-completed questions and skips them.  New results are appended so
    no completed work is lost.

    Assertions (per question)
    -------------------------
    - answer length > 50 chars
    - rule_hit is a valid tier string

    Timeout: 7200s (2 h) for ~70 questions including tier-3 (~360s each).
    """
    if not _ROWS:
        pytest.skip("no DLR CSV found")

    all_questions = _pick_all_questions_with_docs()
    if not all_questions:
        pytest.skip("no rows with non-empty aql_results found")

    jsonl_path = OUTPUT_DIR / "phase3_answers.jsonl"
    txt_path   = OUTPUT_DIR / "phase3_answers_readable.txt"

    # ------------------------------------------------------------------
    # resume: load already-completed questions
    # ------------------------------------------------------------------
    completed: Set[str] = _load_completed_questions(jsonl_path)
    n_skip = len(completed)
    if n_skip:
        log.info(
            "phase3 resume: %d questions already completed, %d remaining",
            n_skip, len(all_questions) - n_skip,
        )

    # Fix 6 adds tier-1-def-parse-rescue for parse-failure definitional questions.
    # Fix 7 (branch D) keeps mechanism+mid-meth at tier-3 (already in set).
    valid_tiers = {
        "tier-1", "tier-1-def", "tier-1-def-parse-rescue",
        "tier-2", "tier-2a", "tier-2b",
        "tier-3", "tier-m", "safety-tier3", "fallback",
    }

    # open both outputs in append mode so existing content is preserved
    with open(jsonl_path, "a", encoding="utf-8") as jf, \
         open(txt_path,   "a", encoding="utf-8") as tf:

        # write a run header so individual runs are distinguishable in the txt
        import datetime
        run_ts = datetime.datetime.now().isoformat(timespec="seconds")
        if n_skip:
            tf.write(f"\n{'#'*60}\n# RESUMED RUN  {run_ts}  skip={n_skip}\n{'#'*60}\n\n")
        else:
            tf.write(f"{'#'*60}\n# NEW RUN  {run_ts}\n{'#'*60}\n\n")

        n_done = 0
        n_failed = 0

        for i, (question, aql_results_str) in enumerate(all_questions, start=1):
            if question.strip() in completed:
                continue  # already saved -- skip

            docs: List[Dict[str, Any]] = []
            for row in _ROWS:
                if _get_question(row) == question:
                    docs = _parse_docs(row)
                    break

            # Clear LLM client cache between questions to prevent cross-question
            # draft bleed (stale LLM client conversation state).
            # NOTE: do NOT call reset_llm_cache() here -- that method also wipes
            # _session_full_doc_uris, which would disable the P6 cross-question
            # full-text throttle for the entire run. Directly clear _llm_cache
            # so P6 keeps accumulating URIs across questions as intended.
            _PIPELINE._llm_cache = {}

            log_question_start(log, i, question)
            t0 = time.perf_counter()
            status = "success"
            err_msg = None

            try:
                ans = _PIPELINE.run(
                    question,
                    aql_results_str,
                    docs=docs or None,
                )
            except Exception as exc:
                status = "error"
                err_msg = str(exc)
                n_failed += 1
                log_question_end(log, i, status, time.perf_counter() - t0, err_msg)
                # write a failure record so the question is not re-attempted
                # on the next run (avoids infinite retries on hard failures)
                fail_record = {
                    "q_index":              i,
                    "question":             question,
                    "expected_tier":        "unknown",
                    "actual_tier":          "error",
                    "use_draft":            None,
                    "answer":               f"[ERROR] {err_msg[:200]}",
                    "enriched_context_chars": 0,
                    "excerpt_stats":        {},
                    "references":           [],
                    "formatted_references": [],
                }
                jf.write(json.dumps(fail_record) + "\n")
                jf.flush()
                raise  # still fail the test so pytest reports it

            elapsed = time.perf_counter() - t0
            log_question_end(log, i, status, elapsed)
            n_done += 1

            record = {
                "q_index":                i,
                "question":               question,
                "expected_tier":          "N/A",  # no tier target in full-run mode
                "actual_tier":            ans.rule_hit,
                "use_draft":              ans.pipeline_config.use_draft if ans.pipeline_config else None,
                "answer":                 ans.answer,
                "enriched_context_chars": len(ans.enriched_context),
                "excerpt_stats":          ans.excerpt_stats,
                "references":             ans.references,
                "formatted_references":   ans.formatted_references,
            }
            jf.write(json.dumps(record) + "\n")
            jf.flush()  # flush after every question so progress is not lost

            tf.write("=" * 60 + "\n")
            tf.write(
                f"Q{i} [actual={ans.rule_hit} "
                f"use_draft={ans.pipeline_config.use_draft if ans.pipeline_config else '?'}]: "
                f"{question}\n"
            )
            tf.write("-" * 60 + "\n")
            tf.write(ans.answer + "\n")
            tf.write("-" * 60 + "\n")
            # Guard: ans.answer already contains ## References (appended by
            # pipeline.py). Only write formatted_references separately when
            # the answer does NOT already include the block, to prevent the
            # double-reference-block artifact seen in phase3_answers_readable.txt.
            if "## References" not in ans.answer:
                tf.write("\n".join(ans.formatted_references) + "\n")
            tf.write(
                f"context={len(ans.enriched_context)} chars | "
                f"excerpts={ans.excerpt_stats.get('n_excerpts', 0)}\n"
            )
            tf.write("=" * 60 + "\n\n")
            tf.flush()  # flush after every question

            assert len(ans.answer) > 50, (
                f"answer too short ({len(ans.answer)} chars) for: {question[:60]}"
            )
            assert ans.rule_hit in valid_tiers, (
                f"unexpected rule_hit '{ans.rule_hit}' for: {question[:60]}"
            )

    log.info(
        "phase 3 complete: %d new, %d skipped (already done), %d failed",
        n_done, n_skip, n_failed,
    )
    print(f"\nPhase 3 output: {txt_path}  (new={n_done} skip={n_skip} fail={n_failed})")
