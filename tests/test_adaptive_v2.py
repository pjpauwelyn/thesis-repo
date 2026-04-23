"""3-phase adaptive pipeline test suite.

phase 1: profile + tier routing on all questions  (inspection only -- no assertions)
phase 2: document filter dry-run on up to 10 questions with docs
phase 3: full-pipeline generation on 5 representative questions (smoke assertions)

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
from typing import Any, Dict, List, Optional, Tuple

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
    """Profile every question and write routing decisions to tests/output/.
    No assertions -- purely for inspection."""
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
# phase 2: document filter dry-run
# ---------------------------------------------------------------------------

def test_phase2_filter() -> None:
    """Run filter_documents on up to 10 questions that have docs.
    Asserts: at least min_keep docs survive the filter."""
    if not _ROWS:
        pytest.skip("no DLR CSV found")

    rows_with_docs = [r for r in _ROWS if _parse_docs(r)][:10]
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
                "filter_summary": filter_summary,
            }
            jf.write(json.dumps(record) + "\n")

            n_kept = filter_summary.get("n_full", 0) + filter_summary.get("n_abstract", 0)
            tf.write(
                f"Q{i}: {question[:80]}\n"
                f"  tier={cfg.rule_hit} | "
                f"full={filter_summary.get('n_full',0)}  "
                f"abstract={filter_summary.get('n_abstract',0)}  "
                f"dropped={filter_summary.get('n_drop',0)}  "
                f"(of {filter_summary.get('n_total',0)})\n"
            )
            for t in filter_summary.get("full_titles", []):
                tf.write(f"    + {t}\n")
            for t in filter_summary.get("abstract_titles", []):
                tf.write(f"    ~ {t}\n")
            for t in filter_summary.get("drop_titles", []):
                tf.write(f"    - {t}\n")
            tf.write("---\n")

            assert n_kept >= cfg.doc_filter_min_keep or n_kept == len(docs), (
                f"filter dropped too many docs: kept {n_kept} of {len(docs)} "
                f"(min_keep={cfg.doc_filter_min_keep})"
            )

    log.info("phase 2 complete")
    print(f"\nPhase 2 output: {txt_path}")


# ---------------------------------------------------------------------------
# phase 3: full-pipeline generation (smoke test)
# ---------------------------------------------------------------------------

def _pick_questions_by_tier(
    target_counts: Dict[str, int],
) -> List[Tuple[str, str, str]]:
    """Return list of (question, aql_results_str, tier_hit).
    Reads tier assignments from phase1 output if available; otherwise
    runs inline profiling so the test is self-contained."""
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


@pytest.mark.timeout(480)
def test_phase3_generation() -> None:
    """Run full pipeline on ~5 questions, one per tier where possible.
    Asserts: non-empty answer, valid rule_hit, answer length > 50 chars."""
    if not _ROWS:
        pytest.skip("no DLR CSV found")

    # Use current tier names (tier-2a / tier-2b, not the stale tier-2 label)
    target_counts = {"tier-1": 2, "tier-2a": 1, "tier-2b": 1, "tier-3": 1}
    questions = _pick_questions_by_tier(target_counts)
    if not questions:
        pytest.skip("could not select representative questions")

    jsonl_path = OUTPUT_DIR / "phase3_answers.jsonl"
    txt_path   = OUTPUT_DIR / "phase3_answers_readable.txt"

    valid_tiers = {"tier-1", "tier-1-def", "tier-2", "tier-2a", "tier-2b",
                   "tier-3", "tier-m", "safety-tier3", "fallback"}

    with open(jsonl_path, "w", encoding="utf-8") as jf, \
         open(txt_path,   "w", encoding="utf-8") as tf:

        for i, (question, aql_results_str, expected_tier) in enumerate(questions, start=1):
            docs: List[Dict[str, Any]] = []
            for row in _ROWS:
                if _get_question(row) == question:
                    docs = _parse_docs(row)
                    break

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
                log_question_end(log, i, status, time.perf_counter() - t0, err_msg)
                raise

            elapsed = time.perf_counter() - t0
            log_question_end(log, i, status, elapsed)

            record = {
                "q_index":                i,
                "question":               question,
                "expected_tier":          expected_tier,
                "actual_tier":            ans.rule_hit,
                "use_draft":              ans.pipeline_config.use_draft if ans.pipeline_config else None,
                "answer":                 ans.answer,
                "enriched_context_chars": len(ans.enriched_context),
                "excerpt_stats":          ans.excerpt_stats,
                "references":             ans.references,
                "formatted_references":   ans.formatted_references,
            }
            jf.write(json.dumps(record) + "\n")

            tf.write("=" * 60 + "\n")
            tf.write(
                f"Q{i} [expected={expected_tier} actual={ans.rule_hit} "
                f"use_draft={ans.pipeline_config.use_draft if ans.pipeline_config else '?'}]: "
                f"{question}\n"
            )
            tf.write("-" * 60 + "\n")
            tf.write(ans.answer + "\n")
            tf.write("-" * 60 + "\n")
            tf.write("\n".join(ans.formatted_references) + "\n")
            tf.write(
                f"context={len(ans.enriched_context)} chars | "
                f"excerpts={ans.excerpt_stats.get('n_excerpts', 0)}\n"
            )
            tf.write("=" * 60 + "\n\n")

            assert len(ans.answer) > 50, (
                f"answer too short ({len(ans.answer)} chars) for: {question[:60]}"
            )
            assert ans.rule_hit in valid_tiers, (
                f"unexpected rule_hit '{ans.rule_hit}' for: {question[:60]}"
            )

    log.info("phase 3 complete")
    print(f"\nPhase 3 output: {txt_path}")
