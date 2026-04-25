"""
S6 -- systematic blank-reference and structural failure scan.

Reads tests/output/phase3_answers_v2.jsonl (not committed; run locally)
and scans every answer for the failure patterns identified in the
Phase-3-vs-Exp4 comparison report:

  1. Blank / malformed ## References entries
     - ". . <year>"  blank-metadata artifact (Q12 pattern)
     - "[No title"   Agent A sentinel (ac7fb8) -- title is missing
     - "Unknown"     old pipeline sentinel
     - "[N] ."       empty title rendered as lone period

  2. Body ending on "]"  (dim c -- truncation at citation cluster)

  3. Body length < 500 chars  (dim b -- below minimum floor)

  4. Non-sequential inline [N] markers  (dim d -- renumber failure)
     - Gaps in the sequence 1..K
     - Inline set does not match ## References set

Writes a plain-text summary to tests/output/phase3_audit.txt.

Usage (from repo root):
    python scripts/audit_phase3.py
    python scripts/audit_phase3.py --jsonl path/to/phase3_answers_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO_ROOT / "tests" / "output" / "phase3_answers_v2.jsonl"
DEFAULT_OUT   = REPO_ROOT / "tests" / "output" / "phase3_audit.txt"

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
# per-answer audit
# ---------------------------------------------------------------------------

def _audit_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    answer = rec.get("answer", "")
    body   = answer.split("## References")[0].strip() if "## References" in answer else answer.strip()
    ref_blk = answer.split("## References")[1] if "## References" in answer else ""

    inline   = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", body)))
    expected = list(range(1, len(inline) + 1))
    ref_nums = sorted(int(n) for n in re.findall(r"^\[(\d+)\]", ref_blk, re.MULTILINE))

    ref_lines = [l.strip() for l in ref_blk.strip().splitlines() if l.strip().startswith("[")]
    bad_refs  = [l for l in ref_lines if _is_bad_ref_line(l)]

    return {
        "q_index":       rec.get("q_index", "?"),
        "question":      rec.get("question", "")[:80],
        "tier":          rec.get("actual_tier") or rec.get("tier", "?"),
        "body_len":      len(body),
        "ends_on_bracket": body.rstrip().endswith("]"),
        "short_body":    len(body) < 500,
        "bad_refs":      bad_refs,
        "inline":        inline,
        "ref_nums":      ref_nums,
        "seq_ok":        (inline == expected) if inline else True,
        "match_ok":      (inline == ref_nums) if inline else (not ref_nums),
        "has_ref_block": bool(ref_blk.strip()),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Audit phase3 JSONL for structural failures.")
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
                print(f"WARNING: skipping malformed JSON line: {e}", file=sys.stderr)

    audits = [_audit_record(r) for r in records]

    # aggregate
    bad_ref_qs    = [a for a in audits if a["bad_refs"]]
    truncated_qs  = [a for a in audits if a["ends_on_bracket"]]
    short_qs      = [a for a in audits if a["short_body"]]
    seq_fail_qs   = [a for a in audits if not a["seq_ok"] or not a["match_ok"]]
    no_ref_qs     = [a for a in audits if not a["has_ref_block"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("=" * 70)
    w(f"PHASE 3 AUDIT REPORT  ({jsonl_path.name})")
    w(f"Total records: {len(audits)}")
    w("=" * 70)
    w()
    w(f"SUMMARY")
    w(f"  Blank/malformed ## References entries : {len(bad_ref_qs)} question(s)")
    w(f"  Body ending on ']' (truncation)       : {len(truncated_qs)} question(s)")
    w(f"  Body length < 500 chars               : {len(short_qs)} question(s)")
    w(f"  Non-sequential/mismatched citations   : {len(seq_fail_qs)} question(s)")
    w(f"  Missing ## References block           : {len(no_ref_qs)} question(s)")
    w()

    def _section(title: str, items: List[Dict[str, Any]], detail_fn) -> None:
        w("-" * 70)
        w(f"{title} ({len(items)})")
        w("-" * 70)
        if not items:
            w("  (none)")
        for a in items:
            w(f"  Q{a['q_index']} [{a['tier']}]  {a['question']}")
            detail_fn(a)
        w()

    _section(
        "1. Blank/malformed ## References entries",
        bad_ref_qs,
        lambda a: [w(f"       BAD REF: {r}") for r in a["bad_refs"]],
    )
    _section(
        "2. Body ending on ']' (truncation at citation cluster)",
        truncated_qs,
        lambda a: w(f"       body_len={a['body_len']}"),
    )
    _section(
        "3. Body length < 500 chars",
        short_qs,
        lambda a: w(f"       body_len={a['body_len']}"),
    )
    _section(
        "4. Non-sequential or mismatched inline citations",
        seq_fail_qs,
        lambda a: w(
            f"       inline={a['inline'][:10]}  ref_block={a['ref_nums'][:10]}  "
            f"seq_ok={a['seq_ok']}  match_ok={a['match_ok']}"
        ),
    )
    _section(
        "5. Missing ## References block",
        no_ref_qs,
        lambda a: w(f"       body_len={a['body_len']}"),
    )

    w("=" * 70)
    w("END OF AUDIT")
    w("=" * 70)

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nAudit written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
