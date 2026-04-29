"""regenerate_answers.py

Selectively regenerates flagged questions from the phase-3 run and merges
all 70 answers into a single clean CSV ready for evaluation.

Usage
-----
# Full regeneration + merge (recommended first run):
    pytest tests/regenerate_answers.py::test_regenerate_flagged -v -s

# Merge only (regeneration already done, just rebuild the CSV):
    python tests/regenerate_answers.py --merge

# Re-generate a single question by question text (spot fix):
    python tests/regenerate_answers.py --regen "<question text here>"
"""

from __future__ import annotations

import ast
import csv
import datetime
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

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

configure_pipeline_logging(log_file="logs/regenerate.log")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# questions flagged for regeneration
#
# Criteria for inclusion (any one is sufficient):
#   WRONG_DOCS  – wrong-domain documents leaked into context or reference list
#   TIER_M      – routed to tier-m (generic intro question; specialist pass needed)
#   SHALLOW     – thin answer: ≤3 usable references or visibly incomplete body
# ---------------------------------------------------------------------------

REGEN_QUESTIONS: List[str] = [
    # Q13 – TIER_M: intro question; frozen-soil refs leaked into reference list
    "what is solar activity and how does it influence the earth's ionosphere?",

    # Q14 – TIER_M: only 5 docs retrieved, 3 alreadyfailed → sparse context
    "how does solar activity influence the dynamics of the earth's ionosphere"
    " and magnetosphere?",

    # Q15 – TIER_M + WRONG_DOCS: sea-turtle paper in reference list
    "what is the relationship between solar activity levels and variations in"
    " ionosphere dynamics, particularly during solar maximum and minimum phases?",

    # Q17 – TIER_M + SHALLOW: only 6 refs; irrigation table appeared mid-answer
    "how do the surface radiative properties of permafrost regions compare to"
    " those of seasonally frozen ground in terms of their influence on soil"
    " temperature dynamics?",

    # Q18 – TIER_M + SHALLOW: tier-1-def with only 3 refs; answer very thin
    "what are the radiative properties of the land surface that influence the"
    " scattering, absorption, and reflection of electromagnetic radiation?",

    # Q19 – TIER_M: answer good but microwave irrigation docs leaked into context
    "how does the reflectivity of frozen ground influence surface radiative"
    " properties in permafrost regions?",

    # Q20 – TIER_M + WRONG_DOCS: irrigation/cotton farming table injected mid-answer
    "what is the relationship between surface radiative properties and the rate"
    " of thawing in frozen ground?",

    # Q27 – TIER_M + SHALLOW: only 2 geographic regions; no long-term data
    "how do the chemical characteristics of groundwater compare to those of"
    " surface water in regions influenced by glacial melt?",

    # Q29 – TIER_M + WRONG_DOCS: aerosol-cloud indirect-effect docs leaked
    "how does groundwater contamination influence the chemical characteristics"
    " of nearby surface water systems?",

    # Q34 – TIER_M + WRONG_DOCS: 4/5 docs alreadyfailed; sea-turtle paper present
    "how does ocean circulation influence the distribution of chemical"
    " constituents in seawater?",

    # Q35 – TIER_M + WRONG_DOCS: 4/6 docs failed; same North-Pacific doc as Q34
    "what is the relationship between ocean circulation patterns and the"
    " distribution of chemical constituents in seawater?",

    # Q36 – TIER_M + WRONG_DOCS: ice-shelf/permafrost/sea-ice context injected
    "how do variations in vegetation structure influence the genomic and"
    " metabolomic profiles of associated microbial communities in different"
    " ecological systems?",

    # Q45 – SHALLOW: irrigation docs leaked into adjacent question context
    "what is the relationship between soil moisture levels and crop yield"
    " variability in agricultural plant science?",

    # Q59 – TIER_M + WRONG_DOCS: irrigation/cotton/crop docs in context
    "how does the melting of sea ice influence ocean circulation patterns?",

    # Q60 – TIER_M + WRONG_DOCS: soil moisture + cotton irrigation docs leaked
    "what is the relationship between changes in sea ice extent and the"
    " stability of glaciers and ice sheets in polar regions?",
]

# normalise once at import time for fast lookup
_REGEN_SET: Set[str] = {q.strip().lower() for q in REGEN_QUESTIONS}

assert len(_REGEN_SET) == 15, (
    f"Expected exactly 15 unique REGEN_QUESTIONS entries, got {len(_REGEN_SET)}"
)

# ---------------------------------------------------------------------------
# output paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("tests/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE3_JSONL   = OUTPUT_DIR / "phase3_answers.jsonl"
SET5_V2_JSONL  = OUTPUT_DIR / "set5-V2.jsonl"
SET5_V1_JSON   = OUTPUT_DIR / "set5-V1.json"
REGEN_JSONL    = OUTPUT_DIR / "regen_answers.jsonl"
FINAL_CSV      = OUTPUT_DIR / "final_answers.csv"

# ---------------------------------------------------------------------------
# helpers copied verbatim from test_adaptive_v2.py (self-contained)
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
# module-level CSV + pipeline setup
# ---------------------------------------------------------------------------

_CSV_PATH: Optional[str] = _find_dlr_csv()
_ROWS_RAW: List[Dict[str, Any]] = _load_questions(_CSV_PATH) if _CSV_PATH else []

# deduplicate (same logic as test_adaptive_v2.py)
_seen_keys: Set[str] = set()
_ROWS: List[Dict[str, Any]] = []
for _r in _ROWS_RAW:
    _key = _get_question(_r).strip().lower()
    if _key and _key not in _seen_keys:
        _seen_keys.add(_key)
        _ROWS.append(_r)

_PIPELINE = Pipeline()

# ---------------------------------------------------------------------------
# load helpers for existing answer banks
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> Dict[str, dict]:
    """Load a JSONL file into a dict keyed by lowercased question text."""
    result: Dict[str, dict] = {}
    if not path.exists():
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("question", "").strip().lower()
                if key:
                    result[key] = rec
            except json.JSONDecodeError:
                pass
    return result


def _load_set5_v1(path: Path) -> Dict[str, dict]:
    """Load the set5-V1.json fallback (may be a list or dict)."""
    result: Dict[str, dict] = {}
    if not path.exists():
        return result
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("answers", [])
    for rec in records:
        key = rec.get("question", "").strip().lower()
        if key:
            result[key] = rec
    return result


def _load_existing_answers() -> Dict[str, dict]:
    """Merge phase3 + set5-V2 + set5-V1 into one dict.

    Priority order (highest wins): phase3 > set5-V2 > set5-V1.
    Questions in REGEN_SET are excluded so they are always regenerated.
    """
    v1  = _load_set5_v1(SET5_V1_JSON)
    v2  = _load_jsonl(SET5_V2_JSONL)
    p3  = _load_jsonl(PHASE3_JSONL)

    merged: Dict[str, dict] = {}
    for bank in (v1, v2, p3):          # lower priority first → higher overwrites
        for key, rec in bank.items():
            if key not in _REGEN_SET:
                merged[key] = rec
    return merged


# ---------------------------------------------------------------------------
# merge + write final CSV
# ---------------------------------------------------------------------------

def _merge_and_write_final_csv() -> Path:
    """Merge existing + regen answers and write tests/output/final_answers.csv.

    Returns the path of the written file.
    Raises ValueError for any missing question or [ERROR]-prefixed answer.
    """
    existing: Dict[str, dict] = _load_existing_answers()
    regen:    Dict[str, dict] = _load_jsonl(REGEN_JSONL)

    # regen entries overwrite existing for the same question
    combined: Dict[str, dict] = {**existing, **regen}

    if not _CSV_PATH:
        raise RuntimeError("No DLR questions CSV found – cannot determine canonical order.")

    rows = _load_questions(_CSV_PATH)

    # deduplicate canonical order
    seen: Set[str] = set()
    canonical: List[str] = []
    for row in rows:
        q = _get_question(row).strip().lower()
        if q and q not in seen:
            seen.add(q)
            canonical.append(q)

    if len(canonical) != 70:
        log.warning("DLR CSV has %d unique questions (expected 70)", len(canonical))

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["q_index", "question", "actual_tier", "use_draft",
                        "answer", "formatted_references"],
        )
        writer.writeheader()

        for idx, q_key in enumerate(canonical, start=1):
            if q_key not in combined:
                raise ValueError(
                    f"Q{idx}: no answer found for question: "
                    f"{q_key[:120]!r}"
                )
            rec = combined[q_key]
            answer = rec.get("answer", "")
            if answer.startswith("[ERROR]"):
                raise ValueError(
                    f"Q{idx}: answer starts with [ERROR] – fix before merging. "
                    f"Question: {q_key[:120]!r}\nAnswer: {answer[:200]!r}"
                )
            writer.writerow({
                "q_index":             rec.get("q_index", idx),
                "question":            rec.get("question", q_key),
                "actual_tier":         rec.get("actual_tier", rec.get("rule_hit", "unknown")),
                "use_draft":           rec.get("use_draft", ""),
                "answer":              answer,
                "formatted_references": json.dumps(
                    rec.get("formatted_references", []),
                    ensure_ascii=False,
                ),
            })

    log.info("final_answers.csv written with %d rows to %s", len(canonical), FINAL_CSV)
    return FINAL_CSV


# ---------------------------------------------------------------------------
# internal: run one question through the pipeline
# ---------------------------------------------------------------------------

def _run_one_question(
    question: str,
    aql_results_str: str,
    docs: List[Dict[str, Any]],
    q_index: int,
) -> dict:
    """Run the pipeline for a single question and return a record dict.

    Clears _llm_cache before running (same as test_phase3_generation).
    """
    _PIPELINE._llm_cache = {}

    valid_tiers = {
        "tier-1", "tier-1-def", "tier-1-def-parse-rescue",
        "tier-2", "tier-2a", "tier-2b",
        "tier-3", "tier-m", "safety-tier3", "fallback",
    }

    log_question_start(log, q_index, question)
    t0 = time.perf_counter()

    ans = _PIPELINE.run(question, aql_results_str, docs=docs or None)

    elapsed = time.perf_counter() - t0
    log_question_end(log, q_index, "success", elapsed)

    assert len(ans.answer) > 50, (
        f"answer too short ({len(ans.answer)} chars) for: {question[:60]}"
    )
    assert ans.rule_hit in valid_tiers, (
        f"unexpected rule_hit '{ans.rule_hit}' for: {question[:60]}"
    )

    return {
        "q_index":                q_index,
        "question":               question,
        "actual_tier":            ans.rule_hit,
        "use_draft":              ans.pipeline_config.use_draft if ans.pipeline_config else None,
        "answer":                 ans.answer,
        "enriched_context_chars": len(ans.enriched_context),
        "excerpt_stats":          ans.excerpt_stats,
        "references":             ans.references,
        "formatted_references":   ans.formatted_references,
    }


# ---------------------------------------------------------------------------
# pytest test: regenerate all flagged questions
# ---------------------------------------------------------------------------

@pytest.mark.timeout(7200)
def test_regenerate_flagged() -> None:
    """Regenerate all questions in REGEN_QUESTIONS and produce final_answers.csv.

    Append / resume behaviour
    -------------------------
    Already-regenerated questions in regen_answers.jsonl are skipped on
    restart, so the test can be safely interrupted and re-run.

    Assertions per question
    -----------------------
    - len(answer) > 50
    - rule_hit in valid_tiers
    """
    if not _ROWS:
        pytest.skip("no DLR CSV found")

    # build a quick lookup: lowercase question → (q_index, aql_str, docs)
    row_lookup: Dict[str, Tuple[int, str, List[Dict[str, Any]]]] = {}
    for i, row in enumerate(_ROWS, start=1):
        q = _get_question(row).strip().lower()
        if q:
            row_lookup[q] = (
                i,
                row.get("aql_results", "") or "",
                _parse_docs(row),
            )

    # load already-regenerated questions so we can skip them on resume
    already_done: Dict[str, dict] = _load_jsonl(REGEN_JSONL)

    run_ts = datetime.datetime.now().isoformat(timespec="seconds")
    n_skip = sum(1 for q in REGEN_QUESTIONS if q.strip().lower() in already_done)
    log.info(
        "regen run %s – %d to do, %d already done",
        run_ts, len(REGEN_QUESTIONS) - n_skip, n_skip,
    )

    with open(REGEN_JSONL, "a", encoding="utf-8") as jf:
        for regen_q in REGEN_QUESTIONS:
            q_key = regen_q.strip().lower()

            if q_key in already_done:
                log.info("skipping (already regenerated): %s", q_key[:80])
                continue

            if q_key not in row_lookup:
                pytest.fail(
                    f"REGEN_QUESTIONS entry not found in DLR CSV: {regen_q!r}"
                )

            q_index, aql_str, docs = row_lookup[q_key]
            # Use the canonical question text from the CSV row
            canonical_q = _get_question(_ROWS[q_index - 1])

            record = _run_one_question(canonical_q, aql_str, docs, q_index)
            jf.write(json.dumps(record) + "\n")
            jf.flush()

    # after all regen, produce the merged final CSV
    final_path = _merge_and_write_final_csv()
    print(f"\nfinal_answers.csv written to {final_path}")

    # verify row count
    with open(final_path, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in csv.DictReader(f))
    assert row_count == 70, (
        f"final_answers.csv has {row_count} rows (expected 70)"
    )
    print(f"Verified: {row_count} rows in final_answers.csv")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--regen" in sys.argv:
        idx_flag = sys.argv.index("--regen")
        if idx_flag + 1 >= len(sys.argv):
            print("Usage: python tests/regenerate_answers.py --regen \"<question text>\"")
            sys.exit(1)

        target_q_raw = sys.argv[idx_flag + 1]
        target_key   = target_q_raw.strip().lower()

        # find matching row in CSV (case-insensitive)
        match_row: Optional[Dict[str, Any]] = None
        match_idx: int = 0
        for i, row in enumerate(_ROWS, start=1):
            q = _get_question(row).strip().lower()
            if q == target_key:
                match_row = row
                match_idx = i
                break

        if match_row is None:
            # fuzzy fallback: check if the target is a substring of any question
            for i, row in enumerate(_ROWS, start=1):
                q = _get_question(row).strip().lower()
                if target_key in q or q in target_key:
                    match_row = row
                    match_idx = i
                    print(f"Fuzzy match found: {_get_question(row)!r}")
                    break

        if match_row is None:
            print(f"ERROR: question not found in DLR CSV: {target_q_raw!r}")
            sys.exit(1)

        canonical_q = _get_question(match_row)
        aql_str     = match_row.get("aql_results", "") or ""
        docs        = _parse_docs(match_row)

        print(f"Regenerating Q{match_idx}: {canonical_q[:80]}…")
        record = _run_one_question(canonical_q, aql_str, docs, match_idx)

        # load existing regen output, overwrite this question, rewrite file
        existing_regen: Dict[str, dict] = _load_jsonl(REGEN_JSONL)
        existing_regen[canonical_q.strip().lower()] = record

        with open(REGEN_JSONL, "w", encoding="utf-8") as jf:
            for rec in existing_regen.values():
                jf.write(json.dumps(rec) + "\n")
        print(f"Saved to {REGEN_JSONL}")

        # re-merge
        final_path = _merge_and_write_final_csv()
        print(f"Merged CSV re-written to {final_path}")

    elif "--merge" in sys.argv:
        path = _merge_and_write_final_csv()
        print(f"Merged CSV written to {path}")

    else:
        print(
            "Usage:\n"
            "  python tests/regenerate_answers.py --merge\n"
            "  python tests/regenerate_answers.py --regen \"<question text>\""
        )
        sys.exit(1)
