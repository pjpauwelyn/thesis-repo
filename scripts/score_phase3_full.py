"""
S7 -- full 70-question scorecard for phase3_answers_v2.jsonl.

Applies the same nine scoring dimensions as smoke_10.py::_score_answer()
to every answer in the pre-generated JSONL output.  No pipeline execution
required -- this is a purely static analysis tool.

Dimensions scored (matching smoke_10.py DIM_LABELS):
  a) fix7_guard       -- skipped (no pipeline log in pre-generated output)
  b) body_length      -- tier-specific floors: tier-3>=2000, tier-m>=1000, else>=500
  c) sentence_ending  -- body must not end on ]
  d) seq_refs         -- inline [N] sequential and matching ## References block
  e) use_draft        -- skipped (no pipeline_config in pre-generated output)
  f) bleed_check      -- no off-topic bleed signal words
  g) ref_uri_sanity   -- no blank/malformed ## References entries
  h) context_size     -- skipped (enriched_context not in pre-generated JSONL)
  i) answer_quality   -- heuristic definition / mechanism / tier-3 completeness

Outputs:
  tests/output/phase3_scorecard_full.txt  -- full per-question scorecard

Usage (from repo root):
    python scripts/score_phase3_full.py
    python scripts/score_phase3_full.py --jsonl path/to/phase3_answers_v2.jsonl
    python scripts/score_phase3_full.py --out   path/to/scorecard.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO_ROOT / "tests" / "output" / "phase3_answers_v2.jsonl"
DEFAULT_OUT   = REPO_ROOT / "tests" / "output" / "phase3_scorecard_full.txt"

# ---------------------------------------------------------------------------
# bleed signals  (kept in sync with smoke_10.py)
# ---------------------------------------------------------------------------
BLEED_SIGNALS = [
    "niche conservatism", "phylogenetic signal", "vertebrate", "invertebrate",
    "precision irrigation", "niche evolution",
]

# ---------------------------------------------------------------------------
# bad-ref predicate  (kept in sync with smoke_10.py::_is_bad_ref_line)
# ---------------------------------------------------------------------------

def _is_bad_ref_line(line: str) -> bool:
    """Return True if a ## References line looks blank or malformed."""
    l = line.strip()
    if "unknown" in l.lower():
        return True
    if l.startswith(". ."):
        return True
    if "[no title" in l.lower():
        return True
    if re.match(r"\[\d+\]\s*\.", l):
        return True
    return False


# ---------------------------------------------------------------------------
# dimension labels and abbreviations
# ---------------------------------------------------------------------------
DIM_LABELS = [
    "fix7_guard",
    "body_length",
    "sentence_ending",
    "seq_refs",
    "use_draft",
    "bleed_check",
    "ref_uri_sanity",
    "context_size",
    "answer_quality",
]
DIM_ABBREV = ["fix7", "len", "end", "seq", "dft", "bld", "uri", "ctx", "qly"]


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

def _score_record(rec: Dict[str, Any]) -> Tuple[Dict[str, Optional[bool]], List[str]]:
    """Score one JSONL record across 9 dimensions.

    Returns (scores_dict, warnings_list).  Dimensions that cannot be scored
    from the pre-generated JSONL (fix7_guard, use_draft, context_size) are
    set to None (SKIP) rather than True/False.
    """
    answer = rec.get("answer", "")
    tier   = rec.get("actual_tier") or rec.get("tier", "")

    body    = answer.split("## References")[0].strip() if "## References" in answer else answer.strip()
    ref_blk = answer.split("## References")[1] if "## References" in answer else ""
    inline  = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", body)))
    ref_nums = sorted(int(n) for n in re.findall(r"^\[(\d+)\]", ref_blk, re.MULTILINE))
    expected = list(range(1, len(inline) + 1))

    scores: Dict[str, Optional[bool]] = {}
    warnings: List[str] = []

    # a) fix7_guard -- not available from pre-generated JSONL
    scores["fix7_guard"] = None

    # b) body_length -- tier-specific floors
    if tier == "tier-3":
        min_len = 2000
    elif tier == "tier-m":
        min_len = 1000
    else:
        min_len = 500
    scores["body_length"] = len(body) >= min_len
    if not scores["body_length"]:
        warnings.append(f"BODY_LEN: {len(body)} chars < {min_len} for {tier}")

    # c) sentence ending
    ends_on_citation = body.rstrip().endswith("]")
    scores["sentence_ending"] = not ends_on_citation
    if ends_on_citation:
        warnings.append(f"SENTENCE_END: body ends on citation, last 60: ...{body[-60:]}")

    # d) sequential refs
    gap_free   = (inline == expected) if inline else True
    refs_match = (inline == ref_nums) if inline else (not ref_nums)
    scores["seq_refs"] = gap_free and refs_match
    if not gap_free:
        warnings.append(f"SEQ_REFS: not sequential: {inline[:8]} (expected {expected[:8]})")
    if not refs_match:
        warnings.append(f"SEQ_REFS: inline {inline[:8]} != refs {ref_nums[:8]}")

    # e) use_draft -- not available
    scores["use_draft"] = None

    # f) bleed check
    body_lower = body.lower()
    bleed_hit  = [w for w in BLEED_SIGNALS if w in body_lower]
    scores["bleed_check"] = not bleed_hit
    if bleed_hit:
        warnings.append(f"BLEED: {bleed_hit}")

    # g) ref URI sanity
    ref_lines = [l.strip() for l in ref_blk.strip().splitlines() if l.strip().startswith("[")]
    bad_refs  = [l for l in ref_lines if _is_bad_ref_line(l)]
    scores["ref_uri_sanity"] = not bad_refs
    if bad_refs:
        warnings.append(f"REF_URI: {len(bad_refs)} bad entries: {bad_refs[:2]}")

    # h) context_size -- not available
    scores["context_size"] = None

    # i) answer quality heuristic
    quality_ok   = True
    quality_note = ""
    if tier == "tier-1-def":
        has_def = any(w in body_lower for w in [
            "is a", "refers to", "defined as", "is an", "stands for",
            "are characterized", "is characterized", "include", "consist of",
            "quantif", "describe", "represent",
        ])
        has_mech = any(w in body_lower for w in [
            "by ", "through ", "using ", "measures", "emits", "detects", "works by",
        ])
        if not (has_def and has_mech):
            quality_ok   = False
            quality_note = f"def={has_def} mech={has_mech}"
    elif tier == "tier-m":
        mech_words = [
            "because", "therefore", "results in", "causes", "leads to",
            "driven by", "due to", "via", "through", "mechanism",
        ]
        hits = sum(1 for w in mech_words if w in body_lower)
        if hits < 2:
            quality_ok   = False
            quality_note = f"only {hits}/2 mechanism words"
    elif tier == "tier-3":
        n_headers = len(re.findall(r"^#{2,3} ", body, re.MULTILINE))
        if n_headers < 2:
            quality_ok   = False
            quality_note = f"only {n_headers} section headings"
    scores["answer_quality"] = quality_ok
    if not quality_ok:
        warnings.append(f"QUALITY: {quality_note}")

    return scores, warnings


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Score all phase3 answers across 9 dimensions.")
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--out",   default=str(DEFAULT_OUT))
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    out_path   = Path(args.out)

    if not jsonl_path.exists():
        print(
            f"WARNING: JSONL not found at {jsonl_path}\n"
            f"  Run this script locally after generating phase3_answers_v2.jsonl."
        )
        return 0

    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed line: {e}", file=sys.stderr)

    if not records:
        print(f"ERROR: no records loaded from {jsonl_path}")
        return 2

    all_results: List[Dict[str, Any]] = []
    for rec in records:
        scores, warnings = _score_record(rec)
        tier    = rec.get("actual_tier") or rec.get("tier", "?")
        q_idx   = rec.get("q_index", "?")
        question = rec.get("question", "")[:70]
        dims_fail = [k for k, v in scores.items() if v is False]
        status    = "OK" if not dims_fail else "WARN"
        all_results.append({
            "q_index": q_idx, "tier": tier, "question": question,
            "scores": scores, "warnings": warnings, "status": status,
        })

    # -----------------------------------------------------------------------
    # build report
    # -----------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    import datetime
    w(f"PHASE 3 FULL SCORECARD  {datetime.datetime.now().isoformat(timespec='seconds')}")
    w(f"Source: {jsonl_path}")
    w(f"Records: {len(records)}")
    w("Note: fix7_guard (a), use_draft (e), context_size (h) are SKIP")
    w("      (not available in pre-generated JSONL -- run smoke_10.py for live scoring)")
    w()

    # header row
    w(f"{'Q':>3}  {'tier':<12}  " + "   ".join(f"{a:<4}" for a in DIM_ABBREV) + "  status")
    w("-" * 90)

    n_pass = 0
    for r in all_results:
        cells = []
        for dim in DIM_LABELS:
            v = r["scores"].get(dim)
            cells.append("PASS" if v is True else ("FAIL" if v is False else "SKIP"))
        row = (
            f"{r['q_index']:>3}  {r['tier']:<12}  "
            + "   ".join(cells)
            + f"  {r['status']}"
        )
        w(row)
        for warn in r["warnings"]:
            w(f"       {warn}")
        if r["status"] == "OK":
            n_pass += 1

    w("-" * 90)
    n_scorable = len(all_results)
    w(f"\nRESULT: {n_pass}/{n_scorable} questions passed all scorable checks")
    w()

    # per-dimension failure summary
    dim_fails: Dict[str, int] = {d: 0 for d in DIM_LABELS}
    for r in all_results:
        for d, v in r["scores"].items():
            if v is False:
                dim_fails[d] += 1

    w("Failures per dimension:")
    for dim, abbrev in zip(DIM_LABELS, DIM_ABBREV):
        n = dim_fails[dim]
        skipped = all(r["scores"].get(dim) is None for r in all_results)
        if skipped:
            w(f"  {abbrev:<6} {dim:<20}: SKIP (not available in pre-generated output)")
        else:
            pct = 100 * n / n_scorable if n_scorable else 0
            w(f"  {abbrev:<6} {dim:<20}: {n:>3} / {n_scorable}  ({pct:.0f}%)")
    w()

    # per-dimension question lists for fails
    for dim in DIM_LABELS:
        failing = [r for r in all_results if r["scores"].get(dim) is False]
        if not failing:
            continue
        w(f"Questions failing {dim}:")
        for r in failing:
            w(f"  Q{r['q_index']:>3} [{r['tier']:<12}]  {r['question']}")
        w()

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nFull scorecard written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
