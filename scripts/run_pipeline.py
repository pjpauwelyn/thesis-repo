"""parallel pipeline runner. see --help for usage."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as _futures_wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipelines.pipeline import Pipeline

csv.field_size_limit(int(1e8))
log = logging.getLogger("run_pipeline")

# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------
_BAD_ESCAPE_RE = re.compile(r'\\u(?![0-9a-fA-F]{4})')

def _sanitise(s: str) -> str:
    """Escape bare \\u not followed by 4 hex digits so json.dumps is safe.

    The LLM occasionally emits strings like \\units or \\upward which are
    not valid JSON Unicode escapes and cause json.JSONDecodeError when the
    JSONL file is read back.
    """
    if not s:
        return s
    return _BAD_ESCAPE_RE.sub(r'\\\\u', s)


_CSV_CANDIDATES = [
    "data/dlr/questions.csv",
    "data/dlr/dlr_questions.csv",
    "data/questions.csv",
    "data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv",
]


def _find_dlr_csv() -> Optional[str]:
    for p in _CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    data_root = Path("data")
    if data_root.exists():
        for csv_path in sorted(data_root.rglob("*.csv")):
            try:
                header = csv_path.open("r", encoding="utf-8").readline()
            except Exception:
                continue
            if "aql_results" in header or "question" in header.lower():
                return str(csv_path)
    return None


def _get_question(row: Dict[str, Any]) -> str:
    for key in ("question", "Question", "query", "Query"):
        val = row.get(key)
        if val:
            return str(val).strip()
    return ""


def _parse_docs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = row.get("aql_results", "") or ""
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _load_rows(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in rows:
        q = _get_question(r).strip().lower()
        if q and q not in seen:
            seen.add(q)
            deduped.append(r)
    return deduped


def _select_questions(
    rows: List[Dict[str, Any]],
    pipeline: Pipeline,
    target_counts: Dict[str, int],
    explicit_indices: Optional[List[int]],
) -> List[Tuple[int, str, str, str, List[Dict[str, Any]]]]:
    """Return list of (display_idx, question, aql_results_str, expected_tier, docs)."""
    if explicit_indices:
        out = []
        for idx in explicit_indices:
            if 1 <= idx <= len(rows):
                row = rows[idx - 1]
                q = _get_question(row)
                aql = row.get("aql_results", "") or ""
                out.append((idx, q, aql, "explicit", _parse_docs(row)))
        return out

    phase1_path = Path("tests/output/phase1_profiles.jsonl")
    tier_map: Dict[str, str] = {}
    if phase1_path.exists():
        with phase1_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    tier_map[rec["question"]] = rec["tier"]
                except Exception:
                    pass

    selected: List[Tuple[int, str, str, str, List[Dict[str, Any]]]] = []
    counts: Dict[str, int] = {}
    target_total = sum(target_counts.values())

    for i, row in enumerate(rows, start=1):
        q = _get_question(row)
        if not q:
            continue
        tier = tier_map.get(q)
        if tier is None:
            try:
                _, _, cfg = pipeline.profile_and_route(q)
                tier = cfg.rule_hit
            except Exception as exc:
                log.warning("inline profile failed for q%d: %s", i, exc)
                continue
        need = target_counts.get(tier, 0)
        if need and counts.get(tier, 0) < need:
            selected.append((i, q, row.get("aql_results", "") or "", tier, _parse_docs(row)))
            counts[tier] = counts.get(tier, 0) + 1
        if sum(counts.values()) >= target_total:
            break

    return selected


class _StartSpacer:
    def __init__(self, min_gap_s: float):
        self._gap = max(0.0, min_gap_s)
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._gap <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.monotonic()
            self._next_allowed = now + self._gap


@contextmanager
def _acquire(sem: threading.Semaphore):
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


_TIER_TO_BUCKET: Dict[str, str] = {
    "tier-1":       "small",
    "fallback":     "small",
    "tier-m":       "medium",
    "tier-2a":      "medium",
    "tier-2b":      "medium",
    "tier-3":       "large",
    "safety-tier3": "large",
}


def _tier_bucket(tier: str) -> str:
    return _TIER_TO_BUCKET.get(tier, "medium")


_TIER_ORDER_5 = ["tier-1", "tier-m", "tier-2a", "tier-2b", "tier-3"]
_TIER_ORDER_3 = ["tier-1", "tier-2b", "tier-3"]


def _parse_tier_mix(s: str) -> Dict[str, int]:
    """Accept either 3 ints (legacy: t1,t2b,t3) or 5 ints (t1,tm,t2a,t2b,t3)."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 5:
        return {tier: int(count) for tier, count in zip(_TIER_ORDER_5, parts)}
    if len(parts) == 3:
        return {tier: int(count) for tier, count in zip(_TIER_ORDER_3, parts)}
    raise argparse.ArgumentTypeError(
        "--tier-mix must be 3 ints (tier-1,tier-2b,tier-3) "
        "or 5 ints (tier-1,tier-m,tier-2a,tier-2b,tier-3)"
    )


def _run_one(
    job_idx: int,
    display_idx: int,
    question: str,
    aql: str,
    expected_tier: str,
    docs: List[Dict[str, Any]],
    pipeline: Pipeline,
    small_sem: threading.Semaphore,
    medium_sem: threading.Semaphore,
    large_sem: threading.Semaphore,
    spacer: _StartSpacer,
    max_retries: int,
) -> Dict[str, Any]:
    precomputed_route = None
    actual_tier = expected_tier
    try:
        precomputed_route = pipeline.profile_and_route(question)
        _, _, cfg = precomputed_route
        actual_tier = cfg.rule_hit
    except Exception as exc:
        log.warning(
            "[job=%d q=%d] pre-profile failed (%s); using expected_tier=%s",
            job_idx, display_idx, exc, expected_tier,
        )

    bucket = _tier_bucket(actual_tier)
    sem = {"small": small_sem, "medium": medium_sem, "large": large_sem}[bucket]

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        with _acquire(sem):
            spacer.wait()
            t0 = time.time()
            try:
                log.info(
                    "[job=%d q=%d expected=%s actual=%s bucket=%s attempt=%d] starting",
                    job_idx, display_idx, expected_tier, actual_tier, bucket, attempt,
                )
                ans = pipeline.run(
                    question,
                    aql,
                    docs=docs or None,
                    precomputed_route=precomputed_route,
                )
                elapsed = time.time() - t0
                log.info(
                    "[job=%d q=%d] done tier=%s chars=%d refs=%d elapsed=%.1fs",
                    job_idx, display_idx, ans.rule_hit,
                    len(ans.answer), len(ans.formatted_references), elapsed,
                )
                return {
                    "job_idx": job_idx,
                    "q_index": display_idx,
                    "question": question,
                    "expected_tier": expected_tier,
                    "actual_tier": ans.rule_hit,
                    "answer": ans.answer,
                    "enriched_context": ans.enriched_context,
                    "enriched_context_chars": len(ans.enriched_context),
                    "excerpt_stats": ans.excerpt_stats,
                    "references": ans.references,
                    "formatted_references": ans.formatted_references,
                    "elapsed_s": round(elapsed, 2),
                    "error": None,
                }
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                is_rate = any(tok in msg for tok in ("429", "rate limit", "too many requests", "rate_limit"))
                backoff = (2 ** (attempt - 1)) * (3.0 if is_rate else 1.0)
                backoff += random.uniform(0, 0.5)
                log.warning(
                    "[job=%d q=%d attempt=%d] %s: %s -- sleeping %.1fs",
                    job_idx, display_idx, attempt,
                    "RATE LIMIT" if is_rate else "ERROR", exc, backoff,
                )
        if attempt < max_retries:
            time.sleep(backoff)

    return {
        "job_idx": job_idx,
        "q_index": display_idx,
        "question": question,
        "expected_tier": expected_tier,
        "actual_tier": "ERROR",
        "answer": f"ERROR after {max_retries} attempts: {last_exc}",
        "enriched_context": "",
        "enriched_context_chars": 0,
        "excerpt_stats": {},
        "references": [],
        "formatted_references": [],
        "elapsed_s": 0.0,
        "error": repr(last_exc),
    }


# ---------------------------------------------------------------------------
# Generation lineage helpers
# ---------------------------------------------------------------------------
_MIN_QUESTIONS_TO_KEEP = 4
_GEN_DIR_PREFIX = "full_gen_attempt-"


def _next_gen_index(base_dir: Path) -> int:
    """Return the next available full_gen_attempt-N index (1-based)."""
    existing = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith(_GEN_DIR_PREFIX)
    ] if base_dir.exists() else []
    indices = []
    for d in existing:
        try:
            indices.append(int(d.name[len(_GEN_DIR_PREFIX):]))
        except ValueError:
            pass
    return max(indices, default=0) + 1


def _write_outputs(
    records: List[Dict[str, Any]],
    output_dir: Path,
    save_context: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Write outputs into a versioned full_gen_attempt-N subdirectory.

    Skips writing entirely when fewer than _MIN_QUESTIONS_TO_KEEP records are
    present -- those runs are treated as throwaway smoke-tests.
    Returns (txt_path, jsonl_path) or (None, None) when skipped.
    """
    if len(records) < _MIN_QUESTIONS_TO_KEEP:
        log.warning(
            "only %d record(s) produced; skipping lineage write (threshold=%d)",
            len(records), _MIN_QUESTIONS_TO_KEEP,
        )
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    gen_index = _next_gen_index(output_dir)
    gen_dir = output_dir / f"{_GEN_DIR_PREFIX}{gen_index}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    txt_path   = gen_dir / "answers_readable.txt"
    jsonl_path = gen_dir / "answers.jsonl"
    # Fix 1: sort by stable CSV row number (q_index = display_idx from _run_one)
    # instead of job_idx (submission order) so output order matches CSV row order.
    records = sorted(records, key=lambda r: r["q_index"])

    with jsonl_path.open("w", encoding="utf-8") as jf, \
         txt_path.open("w", encoding="utf-8") as tf:
        # Fix 1: iterate records directly; q_index already holds display_idx.
        for rec in records:
            row = {
                "q_index": rec["q_index"],
                "question": rec["question"],
                "expected_tier": rec["expected_tier"],
                "actual_tier": rec["actual_tier"],
                "answer": _sanitise(rec["answer"]),
                "enriched_context_chars": rec["enriched_context_chars"],
                "excerpt_stats": rec["excerpt_stats"],
                "references": rec["references"],
                "formatted_references": rec["formatted_references"],
                "elapsed_s": rec["elapsed_s"],
                "error": rec["error"],
            }
            if save_context:
                row["enriched_context"] = _sanitise(rec.get("enriched_context", ""))
            jf.write(json.dumps(row) + "\n")
            n_excerpts = (
                rec["excerpt_stats"].get("n_excerpts", 0)
                if isinstance(rec["excerpt_stats"], dict) else 0
            )
            tf.write("=" * 60 + "\n")
            tf.write(
                f"Q{rec['q_index']} [expected={rec['expected_tier']} actual={rec['actual_tier']}]: "
                f"{rec['question']}\n"
            )
            tf.write("-" * 60 + "\n")
            tf.write("ANSWER:\n" + rec["answer"] + "\n")
            tf.write("-" * 60 + "\n")
            tf.write("FORMATTED REFERENCES:\n" + "\n".join(rec["formatted_references"]) + "\n")
            tf.write("-" * 60 + "\n")
            tf.write(
                f"context={rec['enriched_context_chars']} chars | "
                f"excerpts={n_excerpts} | elapsed={rec['elapsed_s']}s\n"
            )
            tf.write("=" * 60 + "\n\n")

    log.info("lineage: wrote attempt-%d (%d questions) -> %s", gen_index, len(records), gen_dir)
    return txt_path, jsonl_path


# ---------------------------------------------------------------------------
# Fix 2: resume-mode helpers for --output-jsonl
# ---------------------------------------------------------------------------

def _load_done_from_jsonl(path: str) -> Set[int]:
    """Return the set of q_index values already present in *path*.

    Used by --output-jsonl resume mode so the runner can skip questions
    whose answers are already written and only regenerate missing slots.
    """
    done: Set[int] = set()
    p = Path(path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qi = rec.get("q_index")
                if isinstance(qi, int):
                    done.add(qi)
            except Exception:
                pass
    return done


def _merge_into_jsonl(path: str, new_records: List[Dict[str, Any]]) -> None:
    """Upsert *new_records* into an existing JSONL file keyed by q_index.

    Existing lines are preserved unless their q_index matches a new record,
    in which case the new record replaces the old one.  The result is sorted
    by q_index and written back to *path* in-place.

    Workflow:
        1. Delete the bad entry from the JSONL manually (e.g. remove q_index=18).
        2. Run:  python scripts/run_pipeline.py --indices 18 --output-jsonl <path>
        3. The regenerated record fills back in at q_index=18.
    """
    p = Path(path)
    existing: Dict[int, Dict[str, Any]] = {}
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    qi = rec.get("q_index")
                    if isinstance(qi, int):
                        existing[qi] = rec
                except Exception:
                    pass
    for rec in new_records:
        qi = rec.get("q_index")
        if isinstance(qi, int):
            existing[qi] = rec
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in sorted(existing.values(), key=lambda r: r["q_index"]):
            f.write(json.dumps(rec) + "\n")
    log.info("_merge_into_jsonl: wrote %d records to %s", len(existing), path)


def main() -> int:
    ap = argparse.ArgumentParser(description="parallel pipeline runner")
    ap.add_argument("--workers", "-w", type=int, default=4)
    ap.add_argument("--small-concurrency",  type=int, default=4,
                    help="max concurrent small-model jobs (default: 4)")
    ap.add_argument("--medium-concurrency", type=int, default=2,
                    help="max concurrent medium-model jobs (default: 2)")
    ap.add_argument("--large-concurrency",  type=int, default=1,
                    help="max concurrent large-model jobs (default: 1)")
    ap.add_argument("--min-gap-ms", type=int, default=0)
    ap.add_argument("--small-timeout-s",  type=int, default=90,
                    help="job ceiling for tier-1/fallback (default: 90s)")
    ap.add_argument("--medium-timeout-s", type=int, default=300,
                    help="job ceiling for tier-m/2a/2b (default: 300s)")
    ap.add_argument("--large-timeout-s",  type=int, default=600,
                    help="job ceiling for tier-3/safety (default: 600s)")
    ap.add_argument(
        "--tier-mix", type=_parse_tier_mix, default="1,1,1,1,1",
        help=(
            "counts per tier: 5-int (tier-1,tier-m,tier-2a,tier-2b,tier-3) or "
            "3-int legacy (tier-1,tier-2b,tier-3). default: '1,1,1,1,1'"
        ),
    )
    ap.add_argument("--n", type=int, default=None,
                    help="override total question count; weights from --tier-mix")
    ap.add_argument("--indices",   type=int, nargs="+", default=None)
    ap.add_argument("--questions", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--output-dir", default="tests/output")
    ap.add_argument("--save-context", action="store_true",
                    help="write enriched_context into output JSONL for post-run debugging")
    ap.add_argument(
        "--output-jsonl", default=None,
        help=(
            "path to an existing JSONL to resume from and merge results into.  "
            "Q-indices already present in that file are skipped; new results are "
            "upserted back into it sorted by q_index.  "
            "Workflow: delete bad entries from the JSONL, then re-run with "
            "--indices <bad_indices> --output-jsonl <path>."
        ),
    )
    ap.add_argument("--csv", default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.indices is not None and args.questions is not None:
        ap.error("--indices and --questions are mutually exclusive")
    args.indices = args.indices or args.questions

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    csv_path = args.csv or _find_dlr_csv()
    if not csv_path:
        log.error("could not find DLR CSV (looked in %s)", _CSV_CANDIDATES)
        return 1
    log.info("using CSV: %s", csv_path)

    rows = _load_rows(csv_path)
    if not rows:
        log.error("no rows loaded from %s", csv_path)
        return 1

    target_counts = args.tier_mix
    if args.n is not None:
        total_mix = sum(target_counts.values()) or 1
        scaled = {
            t: max(0, round(args.n * target_counts[t] / total_mix))
            for t in target_counts
        }
        diff = args.n - sum(scaled.values())
        scaled["tier-m"] = scaled.get("tier-m", 0) + diff
        target_counts = scaled
        log.info("scaled tier mix for n=%d: %s", args.n, target_counts)

    pipeline = Pipeline()
    selected = _select_questions(rows, pipeline, target_counts, args.indices)
    if not selected:
        log.error("no questions selected")
        return 1

    # Fix 2: resume mode -- skip q_indices already present in --output-jsonl.
    if args.output_jsonl:
        done_indices = _load_done_from_jsonl(args.output_jsonl)
        if done_indices:
            before = len(selected)
            selected = [s for s in selected if s[0] not in done_indices]
            log.info(
                "--output-jsonl resume: skipping %d already-done q_indices %s; %d remaining",
                before - len(selected), sorted(done_indices), len(selected),
            )
        if not selected:
            log.info("all selected questions already in %s; nothing to do", args.output_jsonl)
            return 0

    log.info("selected %d questions:", len(selected))
    for display_idx, q, _aql, tier, _docs in selected:
        log.info("  q%d [%s]: %s", display_idx, tier, q[:70])

    small_sem  = threading.Semaphore(max(1, args.small_concurrency))
    medium_sem = threading.Semaphore(max(1, args.medium_concurrency))
    large_sem  = threading.Semaphore(max(1, args.large_concurrency))
    spacer = _StartSpacer(args.min_gap_ms / 1000.0)

    bucket_timeout: Dict[str, int] = {
        "small":  args.small_timeout_s,
        "medium": args.medium_timeout_s,
        "large":  args.large_timeout_s,
    }

    t_start = time.time()
    records: List[Dict[str, Any]] = []

    pool = ThreadPoolExecutor(
        max_workers=max(1, args.workers),
        thread_name_prefix="pipeline",
    )
    try:
        futures = {
            pool.submit(
                _run_one,
                job_idx=ji,
                display_idx=display_idx,
                question=q,
                aql=aql,
                expected_tier=tier,
                docs=docs,
                pipeline=pipeline,
                small_sem=small_sem,
                medium_sem=medium_sem,
                large_sem=large_sem,
                spacer=spacer,
                max_retries=args.max_retries,
            ): (ji, display_idx, q, tier)
            for ji, (display_idx, q, aql, tier, docs) in enumerate(selected, start=1)
        }

        pending = set(futures.keys())
        deadline_per_job: Dict[Any, float] = {
            f: time.time() + bucket_timeout[_tier_bucket(futures[f][3])]
            for f in futures
        }

        while pending:
            now = time.time()
            next_deadline = min(deadline_per_job[f] for f in pending)
            wait_s = max(0.1, next_deadline - now)
            done, _ = _futures_wait(list(pending), timeout=wait_s, return_when=FIRST_COMPLETED)

            if done:
                for fut in done:
                    pending.discard(fut)
                    ji, display_idx, q, tier = futures[fut]
                    try:
                        records.append(fut.result())
                    except Exception as exc:
                        log.error("job %d crashed outside retry loop: %s", ji, exc)
                        records.append({
                            "job_idx": ji, "q_index": display_idx,
                            "question": q, "expected_tier": tier, "actual_tier": "ERROR",
                            "answer": f"ERROR: crash: {exc}",
                            "enriched_context": "", "enriched_context_chars": 0,
                            "excerpt_stats": {}, "references": [], "formatted_references": [],
                            "elapsed_s": 0.0, "error": repr(exc),
                        })
                continue

            now = time.time()
            for fut in list(pending):
                if now >= deadline_per_job[fut]:
                    ji, display_idx, q, tier = futures[fut]
                    timed_out_s = bucket_timeout[_tier_bucket(tier)]
                    log.error(
                        "job %d (q=%d tier=%s bucket=%s) exceeded timeout of %ds; abandoning",
                        ji, display_idx, tier, _tier_bucket(tier), timed_out_s,
                    )
                    fut.cancel()
                    pending.discard(fut)
                    records.append({
                        "job_idx": ji, "q_index": display_idx,
                        "question": q, "expected_tier": tier, "actual_tier": "TIMEOUT",
                        "answer": f"ERROR: exceeded {_tier_bucket(tier)}-bucket timeout of {timed_out_s}s",
                        "enriched_context": "", "enriched_context_chars": 0,
                        "excerpt_stats": {}, "references": [], "formatted_references": [],
                        "elapsed_s": float(timed_out_s),
                        "error": f"timeout_{timed_out_s}s_{_tier_bucket(tier)}",
                    })
    finally:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)

    elapsed = time.time() - t_start
    txt_path, jsonl_path = _write_outputs(records, Path(args.output_dir), save_context=args.save_context)

    # Fix 2: merge successful records into the target JSONL (resume mode).
    if args.output_jsonl and records:
        ok_records = [r for r in records if r.get("error") is None]
        if ok_records:
            _merge_into_jsonl(args.output_jsonl, ok_records)

    ok  = sum(1 for r in records if r["error"] is None)
    err = len(records) - ok
    print("\n" + "=" * 72)
    print(f"pipeline complete: ok={ok}, err={err}, total_elapsed={elapsed:.1f}s")
    if txt_path:
        print(f"output txt:   {txt_path}")
        print(f"output jsonl: {jsonl_path}")
    else:
        print(f"output: skipped (fewer than {_MIN_QUESTIONS_TO_KEEP} questions)")
    print("=" * 72)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
