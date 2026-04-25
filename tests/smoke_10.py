"""Standalone 10-question smoke test for the adaptive RAG pipeline.

Runs a representative set of 10 questions covering every active tier and all
fix canaries from Sessions 4-5.  Saves full diagnostics to a timestamped
subdirectory under tests/output/smoke_10/ so re-runs never overwrite earlier
results.

Usage (from repo root):
    python tests/smoke_10.py

Outputs (all in tests/output/smoke_10/YYYYMMDD_HHMMSS/):
    smoke_10_answers.jsonl      -- one JSON record per question
    smoke_10_readable.txt       -- human-readable answers + per-question log
    smoke_10_scorecard.txt      -- pass/fail table for all 9 check dimensions
    smoke_10.log                -- full pipeline log (routing, filter, refine,
                                   generation) mirrored from the root log

No pytest required -- run as a plain Python script.
"""

from __future__ import annotations

import ast
import csv
import datetime
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# resolve repo root so the script works from any cwd
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# ---------------------------------------------------------------------------
# output directory  (timestamped so re-runs are non-destructive)
# ---------------------------------------------------------------------------
RUN_TS   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = REPO_ROOT / "tests" / "output" / "smoke_10" / RUN_TS
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH     = OUT_DIR / "smoke_10_answers.jsonl"
READABLE_PATH  = OUT_DIR / "smoke_10_readable.txt"
SCORECARD_PATH = OUT_DIR / "smoke_10_scorecard.txt"
LOG_PATH       = OUT_DIR / "smoke_10.log"

# ---------------------------------------------------------------------------
# logging  (console + file)
# ---------------------------------------------------------------------------
from core.utils.logger import configure_pipeline_logging  # noqa: E402
configure_pipeline_logging(log_file=str(LOG_PATH))
log = logging.getLogger("smoke_10")

# ---------------------------------------------------------------------------
# pipeline imports
# ---------------------------------------------------------------------------
from core.pipelines.pipeline import Pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# data loading  (reuse helpers from test_adaptive_v2)
# ---------------------------------------------------------------------------
csv.field_size_limit(int(1e8))


def _find_csv() -> Optional[str]:
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


def _load_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = _get_question(row).strip().lower()
            if q and q not in seen:
                seen.add(q)
                rows.append(dict(row))
    return rows


def _get_question(row: Dict[str, Any]) -> str:
    for key in ("question", "Question", "query", "Query"):
        if key in row and row[key]:
            return row[key].strip()
    return ""


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


# ---------------------------------------------------------------------------
# 10 target questions
# ---------------------------------------------------------------------------
# Selection rationale:
#   tier-1-def (2) : Fix 2 stub canary + Fix 7 empty-refinement canary
#   tier-m     (4) : Fix 1 bleed, Fix 3 renumber, Fix 4 use_draft, deep mech
#   tier-2a/b  (2) : answer depth + reference integrity
#   tier-3     (1) : length floor (>= 2000 chars) + multi-paper synthesis
#   high-drop  (1) : 69% filter drop-rate stress test (ontology quality)
TARGET_QUESTIONS: List[Tuple[str, str]] = [
    # (question_text, label_for_scorecard)
    (
        "What is LIDAR and how does it measure the properties of distant targets?",
        "tier-1-def | Fix2+Fix7 canary",
    ),
    (
        "What are the radiative properties of the land surface that influence "
        "the scattering, absorption, and reflection of electromagnetic radiation?",
        "tier-1-def | Fix2+Fix7 canary",
    ),
    (
        "How does solar activity influence the dynamics of the Earth's ionosphere "
        "and magnetosphere?",
        "tier-m | Fix1+Fix3+Fix4 canary",
    ),
    (
        "How do variations in cryospheric indicators, such as glacier retreat and "
        "sea ice extent, influence carbon flux dynamics and land surface heat "
        "absorption patterns?",
        "tier-m | Fix3 renumber canary",
    ),
    (
        "How do variations in soil composition, surface radiative properties, and "
        "the presence of frozen ground influence the land surface energy exchange "
        "and carbon cycling processes?",
        "tier-m | mechanism depth",
    ),
    (
        "What is the relationship between solar activity levels and variations in "
        "ionospheric electron density across different solar cycle phases?",
        "tier-m | solar cycle mechanism",
    ),
    (
        "How do machine learning algorithms improve the accuracy of land cover "
        "classification using multispectral satellite imagery?",
        "tier-2a/b | answer depth",
    ),
    (
        "What are the key challenges in retrieving soil moisture from synthetic "
        "aperture radar (SAR) observations?",
        "tier-2a/b | reference integrity",
    ),
    (
        "How do coupled climate-vegetation models represent feedbacks between land "
        "surface processes and the atmosphere in the context of global warming?",
        "tier-3 | length floor + synthesis",
    ),
    (
        "How do aerosol optical properties influence radiative forcing and climate "
        "sensitivity estimates in Earth system models?",
        "tier-m/2 | high drop-rate stress",
    ),
]

TARGET_Q_SET = {q for q, _ in TARGET_QUESTIONS}

# ---------------------------------------------------------------------------
# bleed signal words  (from known cross-question contamination event)
# ---------------------------------------------------------------------------
BLEED_SIGNALS = [
    "niche conservatism", "phylogenetic signal", "vertebrate", "invertebrate",
    "precision irrigation", "niche evolution",
]

# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

DIM_LABELS = [
    "fix7_guard",        # a) no empty-refinement warning
    "body_length",       # b) >= 500 chars (>= 2000 for tier-3)
    "sentence_ending",   # c) body does not end on ]
    "seq_refs",          # d) inline [N] gap-free + ## References matches
    "use_draft",         # e) use_draft matches tier expectation
    "bleed_check",       # f) no off-topic text
    "ref_uri_sanity",    # g) no Unknown entries in ## References
    "context_size",      # h) enriched_context large enough for tier
    "answer_quality",    # i) heuristic completeness signal
]


def _score_answer(
    question: str,
    label: str,
    ans,                    # PipelineResult
    log_lines: List[str],   # captured pipeline log output for this question
) -> Dict[str, Any]:
    """Return a dict of {dim: True/False/None} and a list of warning strings."""
    body    = ans.answer.split("## References")[0].strip()
    ref_blk = ans.answer.split("## References")[1] if "## References" in ans.answer else ""
    inline  = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", body)))
    ref_nums = sorted(int(n) for n in re.findall(r"^\[(\d+)\]", ref_blk, re.MULTILINE))
    expected = list(range(1, len(inline) + 1))
    tier     = ans.rule_hit

    scores: Dict[str, Any] = {}
    warnings: List[str] = []

    # a) Fix 7 guard: no empty-refinement warning in pipeline log
    empty_refine = any(
        "refinement produced empty context" in l.lower()
        or "aql_results_str is empty" in l.lower()
        for l in log_lines
    )
    scores["fix7_guard"] = not empty_refine
    if empty_refine:
        warnings.append("FIX7: refinement fell back to raw abstracts (aql_results_str was empty)")

    # b) body length
    min_len = 2000 if tier == "tier-3" else 500
    scores["body_length"] = len(body) >= min_len
    if not scores["body_length"]:
        warnings.append(f"BODY_LEN: {len(body)} chars < {min_len} floor for {tier}")

    # c) sentence ending
    ends_on_citation = body.rstrip().endswith("]")
    scores["sentence_ending"] = not ends_on_citation
    if ends_on_citation:
        warnings.append(f"SENTENCE_END: answer body ends on citation marker, last 80: ...{body[-80:]}")

    # d) sequential refs
    gap_free   = (inline == expected) if inline else True
    refs_match = (inline == ref_nums) if inline else (not ref_nums)
    scores["seq_refs"] = gap_free and refs_match
    if not gap_free:
        warnings.append(f"SEQ_REFS: inline markers not sequential: {inline} (expected {expected})")
    if not refs_match:
        warnings.append(f"SEQ_REFS: inline {inline} != ## References {ref_nums}")

    # e) use_draft flag
    cfg = ans.pipeline_config
    if cfg is not None:
        draft_tier_true  = {"tier-1-def", "tier-1", "fallback"}
        draft_tier_false = {"tier-m", "tier-2a", "tier-2b", "tier-3"}
        if tier in draft_tier_true:
            scores["use_draft"] = cfg.use_draft is True
        elif tier in draft_tier_false:
            scores["use_draft"] = cfg.use_draft is False
        else:
            scores["use_draft"] = None  # unknown tier -- skip
        if scores["use_draft"] is False:
            warnings.append(f"USE_DRAFT: expected {'True' if tier in draft_tier_true else 'False'} for {tier}, got {cfg.use_draft}")
    else:
        scores["use_draft"] = None

    # f) draft bleed
    bleed_hit = [w for w in BLEED_SIGNALS if w in body.lower()]
    scores["bleed_check"] = not bleed_hit
    if bleed_hit:
        warnings.append(f"BLEED: off-topic signal words found: {bleed_hit}")

    # g) ref URI sanity
    ref_lines = [l.strip() for l in ref_blk.strip().splitlines() if l.strip().startswith("[")]
    unknown_refs = [l for l in ref_lines if "unknown" in l.lower() or len(l) < 10]
    scores["ref_uri_sanity"] = not unknown_refs
    if unknown_refs:
        warnings.append(f"REF_URI: {len(unknown_refs)} suspicious reference entries: {unknown_refs[:3]}")

    # h) enriched context size
    ctx_chars = len(ans.enriched_context)
    ctx_floor = 2000 if tier in ("tier-m", "tier-2a", "tier-2b", "tier-3") else 0
    scores["context_size"] = ctx_chars >= ctx_floor
    if not scores["context_size"]:
        warnings.append(f"CTX_SIZE: enriched_context {ctx_chars} chars < {ctx_floor} floor for {tier}")

    # i) answer quality heuristic
    # Definition questions: must mention definition + mechanism + application
    # Mechanism questions: must contain at least 2 of: mechanism step words
    quality_ok = True
    quality_note = ""
    body_lower = body.lower()
    if "tier-1-def" in tier or "definition" in label:
        has_def  = any(w in body_lower for w in ["is a", "refers to", "defined as", "is an", "stands for"])
        has_mech = any(w in body_lower for w in ["by ", "through ", "using ", "measures", "emits", "detects", "works by"])
        has_app  = any(w in body_lower for w in ["application", "used for", "enables", "allows", "such as"])
        if not (has_def and has_mech):
            quality_ok = False
            quality_note = f"def={has_def} mech={has_mech} app={has_app}"
    elif "mechanism" in label or tier in ("tier-m",):
        mech_words = ["because", "therefore", "results in", "causes", "leads to",
                      "driven by", "due to", "via", "through", "mechanism"]
        hits = sum(1 for w in mech_words if w in body_lower)
        if hits < 2:
            quality_ok = False
            quality_note = f"only {hits}/2 mechanism reasoning words found"
    elif tier == "tier-3":
        # tier-3: expect multiple sections (## headings)
        n_headers = len(re.findall(r"^#{2,3} ", body, re.MULTILINE))
        if n_headers < 2:
            quality_ok = False
            quality_note = f"only {n_headers} section headings (expected >= 2 for tier-3)"
    scores["answer_quality"] = quality_ok
    if not quality_ok:
        warnings.append(f"QUALITY: {quality_note}")

    return {"scores": scores, "warnings": warnings, "body_len": len(body),
            "inline": inline, "ref_nums": ref_nums, "ctx_chars": ctx_chars}


# ---------------------------------------------------------------------------
# log capture handler
# ---------------------------------------------------------------------------

class _ListHandler(logging.Handler):
    """Accumulate log records into a list for post-hoc inspection."""
    def __init__(self):
        super().__init__()
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))

    def flush_and_reset(self) -> List[str]:
        captured = list(self.lines)
        self.lines.clear()
        return captured


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    csv_path = _find_csv()
    if not csv_path:
        print("ERROR: no questions CSV found under data/", file=sys.stderr)
        sys.exit(1)

    rows = _load_rows(csv_path)
    row_map: Dict[str, Dict[str, Any]] = {_get_question(r): r for r in rows}

    pipeline = Pipeline()

    # attach a list handler to the root logger so we can capture per-question
    # log output for scoring (Fix 7 guard etc.)
    list_handler = _ListHandler()
    list_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(list_handler)

    all_scores: List[Dict[str, Any]] = []

    with open(JSONL_PATH,    "w", encoding="utf-8") as jf, \
         open(READABLE_PATH, "w", encoding="utf-8") as tf:

        run_ts_str = datetime.datetime.now().isoformat(timespec="seconds")
        header = (
            f"{'='*70}\n"
            f"SMOKE-10 RUN  {run_ts_str}\n"
            f"Repo: {REPO_ROOT}\n"
            f"Output: {OUT_DIR}\n"
            f"{'='*70}\n\n"
        )
        tf.write(header)
        print(header, end="")

        for q_idx, (question, label) in enumerate(TARGET_QUESTIONS, start=1):
            row = row_map.get(question)
            if row is None:
                # fuzzy fallback: find closest question in CSV by prefix
                for rq, r in row_map.items():
                    if rq.startswith(question[:60]):
                        row = r
                        break
            if row is None:
                print(f"  Q{q_idx} SKIPPED -- question not found in CSV: {question[:70]}")
                continue

            docs = _parse_docs(row)
            aql  = row.get("aql_results", "") or ""

            # reset LLM cache between questions (Fix 1)
            pipeline.reset_llm_cache()
            # flush captured log lines from previous question
            list_handler.flush_and_reset()

            print(f"\nQ{q_idx}/{len(TARGET_QUESTIONS)} [{label}]")
            print(f"  {question[:90]}")

            t0 = time.perf_counter()
            error_msg: Optional[str] = None
            ans = None
            try:
                ans = pipeline.run(question, aql, docs=docs or None)
            except Exception as exc:
                error_msg = str(exc)
                log.error("pipeline.run() raised: %s", exc)

            elapsed = time.perf_counter() - t0
            log_lines = list_handler.flush_and_reset()

            # ----------------------------------------------------------------
            # score
            # ----------------------------------------------------------------
            if ans is None:
                score_result = {
                    "scores": {d: False for d in DIM_LABELS},
                    "warnings": [f"PIPELINE_ERROR: {error_msg}"],
                    "body_len": 0, "inline": [], "ref_nums": [], "ctx_chars": 0,
                }
                tier = "error"
                use_draft = None
            else:
                score_result = _score_answer(question, label, ans, log_lines)
                tier      = ans.rule_hit
                use_draft = ans.pipeline_config.use_draft if ans.pipeline_config else None

            scores   = score_result["scores"]
            warnings = score_result["warnings"]
            dims_ok  = [v for v in scores.values() if v is True]
            dims_fail = [k for k, v in scores.items() if v is False]
            status   = "✅" if not dims_fail else "⚠️"

            # ----------------------------------------------------------------
            # print quick summary to stdout
            # ----------------------------------------------------------------
            print(f"  tier={tier}  use_draft={use_draft}  "
                  f"body={score_result['body_len']}c  "
                  f"ctx={score_result['ctx_chars']}c  "
                  f"elapsed={elapsed:.1f}s  {status}")
            for w in warnings:
                print(f"    ⚠️  {w}")

            # ----------------------------------------------------------------
            # write to readable TXT
            # ----------------------------------------------------------------
            sep = "-" * 70
            tf.write(f"{'='*70}\n")
            tf.write(f"Q{q_idx} [{tier} | use_draft={use_draft}] {label}\n")
            tf.write(f"{question}\n")
            tf.write(f"elapsed={elapsed:.1f}s  body={score_result['body_len']}c  "
                     f"ctx={score_result['ctx_chars']}c\n")
            tf.write(f"{sep}\n")

            # per-dimension score block
            for dim in DIM_LABELS:
                v = scores.get(dim)
                mark = "✅" if v is True else ("⚠️" if v is False else "--")
                tf.write(f"  {mark} {dim}\n")
            if warnings:
                tf.write("\nWarnings:\n")
                for w in warnings:
                    tf.write(f"  ⚠️  {w}\n")

            tf.write(f"\nInline refs : {score_result['inline']}\n")
            tf.write(f"Ref block   : {score_result['ref_nums']}\n")
            tf.write(f"{sep}\n")

            # full answer
            if ans is not None:
                tf.write(ans.answer + "\n")
            else:
                tf.write(f"[ERROR] {error_msg}\n")

            tf.write(f"{'='*70}\n\n")
            tf.flush()

            # ----------------------------------------------------------------
            # write JSONL record
            # ----------------------------------------------------------------
            record = {
                "q_index":                q_idx,
                "question":               question,
                "label":                  label,
                "tier":                   tier,
                "use_draft":              use_draft,
                "elapsed_s":              round(elapsed, 2),
                "body_len":               score_result["body_len"],
                "ctx_chars":              score_result["ctx_chars"],
                "inline_refs":            score_result["inline"],
                "ref_block_nums":         score_result["ref_nums"],
                "scores":                 {k: v for k, v in scores.items()},
                "warnings":               warnings,
                "answer":                 ans.answer if ans else f"[ERROR] {error_msg}",
                "formatted_references":   ans.formatted_references if ans else [],
                "enriched_context_chars": score_result["ctx_chars"],
                "excerpt_stats":          ans.excerpt_stats if ans else {},
                "pipeline_log_lines":     log_lines,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            jf.flush()

            all_scores.append({
                "q_idx": q_idx, "label": label, "tier": tier,
                "scores": scores, "warnings": warnings,
                "status": status,
            })

    # -----------------------------------------------------------------------
    # scorecard
    # -----------------------------------------------------------------------
    with open(SCORECARD_PATH, "w", encoding="utf-8") as sf:
        run_ts_str = datetime.datetime.now().isoformat(timespec="seconds")
        sf.write(f"SMOKE-10 SCORECARD  {run_ts_str}\n")
        sf.write(f"Output dir: {OUT_DIR}\n\n")

        # header row
        dim_abbrev = [
            "fix7", "len", "end", "seq", "dft", "bld", "uri", "ctx", "qly"
        ]
        sf.write(f"{'Q':>3}  {'tier':<12}  ")
        sf.write("  ".join(f"{a:<4}" for a in dim_abbrev))
        sf.write("  status  label\n")
        sf.write("-" * 100 + "\n")

        n_pass = 0
        for s in all_scores:
            scores = s["scores"]
            cells  = []
            for dim in DIM_LABELS:
                v = scores.get(dim)
                cells.append("✅" if v is True else ("⚠️" if v is False else "--"))
            row_str = (
                f"{s['q_idx']:>3}  {s['tier']:<12}  "
                + "   ".join(cells)
                + f"  {s['status']}  {s['label']}"
            )
            sf.write(row_str + "\n")
            for w in s["warnings"]:
                sf.write(f"       ⚠️  {w}\n")
            if s["status"] == "✅":
                n_pass += 1

        sf.write("-" * 100 + "\n")
        sf.write(f"\nRESULT: {n_pass}/{len(all_scores)} questions passed all checks\n")

        # dimension failure summary
        dim_fails: Dict[str, int] = {d: 0 for d in DIM_LABELS}
        for s in all_scores:
            for d, v in s["scores"].items():
                if v is False:
                    dim_fails[d] += 1
        sf.write("\nFailures per dimension:\n")
        for d, n in dim_fails.items():
            sf.write(f"  {d:<20}: {n}\n")

    # final stdout summary
    print(f"\n{'='*70}")
    print(f"SMOKE-10 RESULT: {n_pass}/{len(all_scores)} passed")
    print(f"Scorecard : {SCORECARD_PATH}")
    print(f"Answers   : {READABLE_PATH}")
    print(f"JSONL     : {JSONL_PATH}")
    print(f"Log       : {LOG_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
