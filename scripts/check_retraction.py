"""
S5 -- OpenAlex retraction checker for Q43.

Reads phase3_answers_v2.jsonl (not committed; run locally after generation),
extracts OpenAlex URIs from the specified question's ## References block,
and hits the OpenAlex REST API to check the is_retracted field.

Usage (from repo root):
    python scripts/check_retraction.py
    python scripts/check_retraction.py --jsonl path/to/phase3_answers_v2.jsonl
    python scripts/check_retraction.py --q-index 43

Exit codes:
    0 -- no retracted works found (or JSONL not present -- see warning)
    1 -- one or more retracted works found; IDs printed for Agent A blocklist
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO_ROOT / "tests" / "output" / "phase3_answers_v2.jsonl"
OPENALEX_WORKS = "https://api.openalex.org/works/{wid}"
_OPENALEX_URI_RE = re.compile(r"https?://openalex\.org/(W\d+)", re.IGNORECASE)
_RETRY_DELAY = 1.0   # seconds between API calls (polite crawling)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_question(jsonl_path: Path, q_index: int) -> Optional[dict]:
    """Return the JSONL record for the given 1-based q_index, or None."""
    if not jsonl_path.exists():
        return None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Support both q_index (smoke_10 schema) and positional
            if rec.get("q_index") == q_index:
                return rec
    return None


def _extract_openalex_ids(answer_text: str) -> List[Tuple[str, str]]:
    """Return list of (W-ID, full URI) found in the ## References block."""
    if "## References" not in answer_text:
        return []
    ref_blk = answer_text.split("## References", 1)[1]
    return [(m.group(1), m.group(0)) for m in _OPENALEX_URI_RE.finditer(ref_blk)]


def _check_retraction(wid: str) -> Tuple[bool, Optional[str]]:
    """Return (is_retracted, title) by querying OpenAlex.

    Returns (False, None) on network error so a connectivity issue never
    produces a false positive.
    """
    url = OPENALEX_WORKS.format(wid=wid)
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "thesis-retraction-check/1.0"})
        if resp.status_code == 404:
            return False, "[not found in OpenAlex]"
        resp.raise_for_status()
        data = resp.json()
        is_retracted = bool(data.get("is_retracted", False))
        title = data.get("title", "[no title]")
        return is_retracted, title
    except Exception as exc:
        print(f"  WARNING: could not fetch {wid}: {exc}", file=sys.stderr)
        return False, None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check OpenAlex retraction status for references in a phase3 answer."
    )
    parser.add_argument(
        "--jsonl",
        default=str(DEFAULT_JSONL),
        help=f"Path to phase3 JSONL output (default: {DEFAULT_JSONL})",
    )
    parser.add_argument(
        "--q-index",
        type=int,
        default=43,
        metavar="N",
        help="1-based question index to check (default: 43)",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    q_index    = args.q_index

    print(f"S5 retraction check  q_index={q_index}  jsonl={jsonl_path}")
    print("-" * 60)

    if not jsonl_path.exists():
        print(
            f"WARNING: JSONL not found at {jsonl_path}\n"
            f"  The phase3_answers_v2.jsonl file is not committed to the repo.\n"
            f"  Run this script locally after generating the 70-question output.\n"
            f"  Expected location: tests/output/phase3_answers_v2.jsonl"
        )
        return 0  # not an error -- script is correct, data just not here

    rec = _load_question(jsonl_path, q_index)
    if rec is None:
        print(f"ERROR: q_index={q_index} not found in {jsonl_path}")
        return 2

    question = rec.get("question", "[unknown]")
    answer   = rec.get("answer", "")
    print(f"Question: {question[:100]}")
    print()

    ids = _extract_openalex_ids(answer)
    if not ids:
        print("No OpenAlex URIs found in ## References block.")
        print("Check that the answer contains a ## References section with openalex.org links.")
        return 0

    print(f"Found {len(ids)} OpenAlex URI(s) in ## References:")
    retracted_ids: List[str] = []

    for wid, uri in ids:
        time.sleep(_RETRY_DELAY)
        is_retracted, title = _check_retraction(wid)
        status = "RETRACTED \u26a0" if is_retracted else "ok"
        print(f"  {status:20s}  {wid}  {(title or '')[:80]}")
        if is_retracted:
            retracted_ids.append(wid)

    print()
    if retracted_ids:
        print("=" * 60)
        print(f"RESULT: {len(retracted_ids)} RETRACTED work(s) found.")
        print()
        print("ACTION REQUIRED -- request for Agent A:")
        print("  Please add the following ID(s) to the RETRACTED_WORK_URIS")
        print("  blocklist in _build_verified_references() in")
        print("  core/pipelines/pipeline.py so that these works are skipped")
        print("  entirely during reference building.")
        print()
        for wid in retracted_ids:
            print(f"    https://openalex.org/{wid}")
        print()
        print("  Suggested implementation in pipeline.py:")
        print("    RETRACTED_WORK_URIS = {")
        for wid in retracted_ids:
            print(f"        \"https://openalex.org/{wid}\",  # flagged by check_retraction.py")
        print("    }")
        print("    # inside _build_verified_references(), before title/uri extraction:")
        print("    # if doc.get('uri') in RETRACTED_WORK_URIS: continue")
        print("=" * 60)
        return 1
    else:
        print(f"RESULT: no retracted works found in Q{q_index} references.")
        print("Item S5 closed -- no blocklist entry needed for this question.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
