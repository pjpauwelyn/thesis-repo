"""Merge phase3_answers-3.jsonl + regen_answers.jsonl → phase3_final.jsonl

The regen file wins on any q_index collision (it contains fresh answers
generated after the pipeline fixes).  The final file is sorted by q_index.
A coverage check verifies every question in the CSV has an answer and
prints a clear summary of any gaps.

Usage:
    python tests/merge_phase3.py                        # uses defaults below
    python tests/merge_phase3.py --base  path/to/base.jsonl \\
                                  --regen path/to/regen.jsonl \\
                                  --out   path/to/output.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path("tests/output")
_DEFAULT_BASE  = _OUTPUT_DIR / "phase3_answers-3.jsonl"
_DEFAULT_REGEN = _OUTPUT_DIR / "regen_answers.jsonl"
_DEFAULT_OUT   = _OUTPUT_DIR / "phase3_final.jsonl"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> Dict[str, dict]:
    """Return {lowercase_question: record} for every non-error record in path."""
    records: Dict[str, dict] = {}
    if not path.exists():
        print(f"  [skip] {path} does not exist", file=sys.stderr)
        return records
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [warn] {path}:{lineno} JSON decode error: {exc}", file=sys.stderr)
                continue
            actual_tier = rec.get("actual_tier", "")
            if actual_tier == "error":
                print(
                    f"  [skip] {path}:{lineno}  q_index={rec.get('q_index')} "
                    f"is an error record -- excluded from merge",
                    file=sys.stderr,
                )
                continue
            key = rec.get("question", "").strip().lower()
            if key:
                records[key] = rec
    return records


def _find_dlr_csv() -> Optional[Path]:
    candidates = [
        Path("data/dlr/questions.csv"),
        Path("data/dlr/dlr_questions.csv"),
        Path("data/questions.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    data_root = Path("data")
    if data_root.exists():
        for csv_path in sorted(data_root.rglob("*.csv")):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    header = f.readline()
                if "aql_results" in header or "question" in header.lower():
                    return csv_path
            except Exception:
                pass
    return None


def _load_csv_questions(csv_path: Path) -> List[str]:
    """Return ordered unique question strings from the CSV."""
    csv.field_size_limit(int(1e8))
    seen: set = set()
    questions: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = ""
            for key in ("question", "Question", "query", "Query"):
                if key in row and row[key]:
                    q = row[key].strip()
                    break
            if not q:
                continue
            aql = row.get("aql_results", "") or ""
            if not aql.strip():
                continue  # no docs → pipeline would have skipped it anyway
            key = q.lower()
            if key not in seen:
                seen.add(key)
                questions.append(q)
    return questions


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",  default=str(_DEFAULT_BASE),  help="Base JSONL (phase3 main run)")
    parser.add_argument("--regen", default=str(_DEFAULT_REGEN), help="Regen JSONL (flagged questions re-run)")
    parser.add_argument("--out",   default=str(_DEFAULT_OUT),   help="Output path for merged JSONL")
    args = parser.parse_args()

    base_path  = Path(args.base)
    regen_path = Path(args.regen)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"base  : {base_path}")
    print(f"regen : {regen_path}")
    print(f"out   : {out_path}")

    # ------------------------------------------------------------------
    # load both files; regen takes priority on collision
    # ------------------------------------------------------------------
    base_records  = _load_jsonl(base_path)
    regen_records = _load_jsonl(regen_path)

    merged: Dict[str, dict] = {**base_records, **regen_records}  # regen wins
    overwritten = set(base_records) & set(regen_records)
    if overwritten:
        print(f"\nregen overrode {len(overwritten)} question(s):")
        for q in sorted(overwritten):
            print(f"  · {q[:80]}")

    # ------------------------------------------------------------------
    # sort by q_index (fall back to insertion order for missing index)
    # ------------------------------------------------------------------
    sorted_records = sorted(merged.values(), key=lambda r: r.get("q_index", 99999))

    # ------------------------------------------------------------------
    # write output
    # ------------------------------------------------------------------
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in sorted_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(sorted_records)} records → {out_path}")

    # ------------------------------------------------------------------
    # coverage check against CSV
    # ------------------------------------------------------------------
    csv_path = _find_dlr_csv()
    if not csv_path:
        print("\n[warn] no DLR CSV found -- skipping coverage check")
        return 0

    expected = _load_csv_questions(csv_path)
    answered = set(merged.keys())
    missing  = [q for q in expected if q.lower() not in answered]
    extra    = [k for k in answered if k not in {q.lower() for q in expected}]

    print(f"\ncoverage: {len(expected)} questions with docs in CSV")
    print(f"          {len(answered)} answered in merged output")

    if missing:
        print(f"\n  MISSING ({len(missing)}) -- not in merged output:")
        for q in missing:
            print(f"    · {q[:90]}")
        print(
            "\n  → re-run test_phase3_generation against phase3_final.jsonl "
            "(it will skip existing answers and only process missing ones):"
        )
        print(
            "      cp tests/output/phase3_final.jsonl tests/output/phase3_answers.jsonl"
        )
        print("      pytest tests/test_adaptive_v2.py::test_phase3_generation -v -s")
    else:
        print("\n  ✓ all questions covered -- phase3_final.jsonl is complete")

    if extra:
        print(f"\n  extra records in output not in CSV ({len(extra)}) -- harmless:")
        for q in extra:
            print(f"    · {q[:90]}")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
