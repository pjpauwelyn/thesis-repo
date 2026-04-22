"""phase3_parallel.py — lightweight parallel phase-3 runner.

Drops into the adaptive-pipeline workflow as a faster alternative to
    pytest tests/test_adaptive_v2.py::test_phase3_generation -v -s

Does NOT modify tests/test_adaptive_v2.py. Produces the same output files:
    tests/output/phase3_answers_readable.txt
    tests/output/phase3_answers.jsonl

Per-question work (profile + route + filter + refinement + generation) is
run in parallel with a ThreadPoolExecutor. Mistral API calls are the bottleneck
and are I/O-bound, so threads are sufficient (no GIL contention on network).

Rate-limit safety:
    * global concurrency cap via --workers (default: 4)
    * per-model concurrency caps: --small-concurrency / --medium-concurrency /
      --large-concurrency (serialise large to avoid server disconnects)
    * optional minimum spacing between request starts via --min-gap-ms
    * exponential backoff + retry on any exception from AdaptivePipeline.run()

Per-job timeouts are derived from the routed tier so that small-model jobs
never receive a large-model ceiling and vice versa:
    tier-1/fallback : --small-timeout-s   (default  90s)
    tier-m/2a/2b    : --medium-timeout-s  (default 300s)
    tier-3/safety   : --large-timeout-s   (default 600s)

Usage:
    python scripts/phase3_parallel.py
    python scripts/phase3_parallel.py --workers 4 --indices 1 8 13 18 28
    python scripts/phase3_parallel.py --questions 1,5,12,27
    python scripts/phase3_parallel.py --tier-mix 2,2,2 --n 6
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as _futures_wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipelines.adaptive_pipeline import AdaptivePipeline

csv.field_size_limit(int(1e8))
log = logging.getLogger("phase3_parallel")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# question selection
# ---------------------------------------------------------------------------

def _select_questions(
    rows: List[Dict[str, Any]],
    pipeline: AdaptivePipeline,
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


# ---------------------------------------------------------------------------
# rate-limit primitives
# ---------------------------------------------------------------------------

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


# Map tier names to semaphore bucket and job-timeout bucket.
# tier-m / tier-2a / tier-2b all have medium refinement so they use the
# medium semaphore for back-pressure; their generation is small or medium
# but medium is the bottleneck slot.
_TIER_TO_BUCKET: Dict[str, str] = {
    "tier-1":     "small",
    "fallback":   "small",
    "tier-m":     "medium",
    "tier-2a":    "medium",
    "tier-2b":    "medium",
    "tier-3":     "large",
    "safety-tier3": "large",
}


def _tier_bucket(tier: str) -> str:
    return _TIER_TO_BUCKET.get(tier, "medium")


# ---------------------------------------------------------------------------
# per-question worker
# ---------------------------------------------------------------------------

def _run_one(
    job_idx: int,
    display_idx: int,
    question: str,
    aql: str,
    expected_tier: str,
    docs: List[Dict[str, Any]],
    pipeline: AdaptivePipeline,
    small_sem: threading.Semaphore,
    medium_sem: threading.Semaphore,
    large_sem: threading.Semaphore,
    spacer: _StartSpacer,
    max_retries: int,
) -> Dict[str, Any]:
    # Pre-profile so the semaphore choice and the pipeline's internal route
    # are guaranteed to agree. Without this, LLM nondeterminism can let a
    # large call slip under the small semaphore.
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
                    "[job=%d q=%d attempt=%d] %s: %s — sleeping %.1fs",
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
# output writers
# ---------------------------------------------------------------------------

def _write_outputs(records: List[Dict[str, Any]], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path  = output_dir / "phase3_answers_readable.txt"
    jsonl_path = output_dir / "phase3_answers.jsonl"
    records = sorted(records, key=lambda r: r["job_idx"])

    with jsonl_path.open("w", encoding="utf-8") as jf, \
         txt_path.open("w", encoding="utf-8") as tf:
        for i, rec in enumerate(records, start=1):
            jf.write(json.dumps({
                "q_index": i,
                "question": rec["question"],
                "expected_tier": rec["expected_tier"],
                "actual_tier": rec["actual_tier"],
                "answer": rec["answer"],
                "enriched_context_chars": rec["enriched_context_chars"],
                "excerpt_stats": rec["excerpt_stats"],
                "references": rec["references"],
                "formatted_references": rec["formatted_references"],
                "elapsed_s": rec["elapsed_s"],
                "error": rec["error"],
            }) + "\n")
            n_excerpts = (
                rec["excerpt_stats"].get("n_excerpts", 0)
                if isinstance(rec["excerpt_stats"], dict) else 0
            )
            tf.write("=" * 60 + "\n")
            tf.write(
                f"Q{i} [expected={rec['expected_tier']} actual={rec['actual_tier']}]: "
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

    return txt_path, jsonl_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_tier_mix(s: str) -> Dict[str, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("tier-mix must be 3 ints: t1,t2,t3")
    return {"tier-1": int(parts[0]), "tier-2": int(parts[1]), "tier-3": int(parts[2])}


def main() -> int:
    ap = argparse.ArgumentParser(description="parallel phase-3 adaptive pipeline runner")
    ap.add_argument("--workers", "-w", type=int, default=4)
    ap.add_argument("--small-concurrency",  type=int, default=4,
                    help="max concurrent small-model jobs (default: 4)")
    ap.add_argument("--medium-concurrency", type=int, default=2,
                    help="max concurrent medium-model jobs (default: 2)")
    ap.add_argument("--large-concurrency",  type=int, default=1,
                    help="max concurrent large-model jobs (default: 1)")
    ap.add_argument("--min-gap-ms", type=int, default=0)
    # Per-bucket job-level timeouts. These gate the outer deadline loop in
    # main(); the per-call socket timeouts are set inside PipelineConfig and
    # are always tighter than these job ceilings.
    ap.add_argument("--small-timeout-s",  type=int, default=90,
                    help="job ceiling for tier-1/fallback jobs (default: 90s)")
    ap.add_argument("--medium-timeout-s", type=int, default=300,
                    help="job ceiling for tier-m/2a/2b jobs (default: 300s)")
    ap.add_argument("--large-timeout-s",  type=int, default=600,
                    help="job ceiling for tier-3/safety jobs (default: 600s)")
    ap.add_argument("--tier-mix", type=_parse_tier_mix, default="2,2,1")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--indices",   type=int, nargs="+", default=None)
    ap.add_argument("--questions", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--output-dir", default="tests/output")
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
        scaled = {t: max(0, round(args.n * target_counts[t] / total_mix)) for t in target_counts}
        diff = args.n - sum(scaled.values())
        scaled["tier-2"] = scaled.get("tier-2", 0) + diff
        target_counts = scaled
        log.info("scaled tier mix for n=%d: %s", args.n, target_counts)

    pipeline = AdaptivePipeline()
    selected = _select_questions(rows, pipeline, target_counts, args.indices)
    if not selected:
        log.error("no questions selected")
        return 1

    log.info("selected %d questions:", len(selected))
    for display_idx, q, _aql, tier, _docs in selected:
        log.info("  q%d [%s]: %s", display_idx, tier, q[:70])

    small_sem  = threading.Semaphore(max(1, args.small_concurrency))
    medium_sem = threading.Semaphore(max(1, args.medium_concurrency))
    large_sem  = threading.Semaphore(max(1, args.large_concurrency))
    spacer = _StartSpacer(args.min_gap_ms / 1000.0)

    # Per-bucket job-level timeout in seconds.
    bucket_timeout: Dict[str, int] = {
        "small":  args.small_timeout_s,
        "medium": args.medium_timeout_s,
        "large":  args.large_timeout_s,
    }

    t_start = time.time()
    records: List[Dict[str, Any]] = []

    pool = ThreadPoolExecutor(
        max_workers=max(1, args.workers),
        thread_name_prefix="phase3",
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
    txt_path, jsonl_path = _write_outputs(records, Path(args.output_dir))

    ok  = sum(1 for r in records if r["error"] is None)
    err = len(records) - ok
    print("\n" + "=" * 72)
    print(f"parallel phase 3 complete: ok={ok}, err={err}, total_elapsed={elapsed:.1f}s")
    print(f"output txt:   {txt_path}")
    print(f"output jsonl: {jsonl_path}")
    print("=" * 72)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
