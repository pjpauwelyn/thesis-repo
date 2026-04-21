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
    * per-model concurrency cap via semaphores (default: 4 small + 2 large)
    * optional minimum spacing between request starts via --min-gap-ms
    * exponential backoff + retry on any exception from AdaptivePipeline.run()

Usage:
    # default: 5 phase-3-style questions (2 tier-1 + 2 tier-2 + 1 tier-3)
    python scripts/phase3_parallel.py

    # faster: 6 workers, tighter spacing
    python scripts/phase3_parallel.py --workers 6 --min-gap-ms 150

    # different mix / more questions
    python scripts/phase3_parallel.py --tier-mix 3,3,2 --n 8

    # specific question indices (1-based into deduped DLR rows)
    python scripts/phase3_parallel.py --indices 1 5 12 27

    # cap mistral-large concurrency because it's the scarce tier
    python scripts/phase3_parallel.py --large-concurrency 1
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

# make "core.*" importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipelines.adaptive_pipeline import AdaptivePipeline

csv.field_size_limit(int(1e8))

log = logging.getLogger("phase3_parallel")


# ---------------------------------------------------------------------------
# data loading — mirrors tests/test_adaptive_v2.py so selection is identical
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
# question selection — reuse phase1 output if present, otherwise inline profile
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
    """enforce a minimum gap between successive call starts across all threads."""

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


def _infer_model(tier: str) -> str:
    return "mistral-large-latest" if tier == "tier-3" else "mistral-small-latest"


# ---------------------------------------------------------------------------
# per-question worker
# ---------------------------------------------------------------------------

def _wait_any(futures, timeout: float):
    """Wait for at least one future to complete or for the timeout to elapse.
    Returns (done, not_done) sets. Thin wrapper so the main loop reads clean.
    """
    return _futures_wait(list(futures), timeout=timeout, return_when=FIRST_COMPLETED)


def _run_one(
    job_idx: int,
    display_idx: int,
    question: str,
    aql: str,
    expected_tier: str,
    docs: List[Dict[str, Any]],
    pipeline: AdaptivePipeline,
    small_sem: threading.Semaphore,
    large_sem: threading.Semaphore,
    spacer: _StartSpacer,
    max_retries: int,
) -> Dict[str, Any]:

    # Pre-profile (cheap mistral-small call) so we know the ACTUAL model the
    # pipeline will use for this question, and PASS the resulting route into
    # pipeline.run() so the semaphore decision and the actual model used are
    # guaranteed to agree. Without this, LLM nondeterminism in the profile
    # step can let a second mistral-large call slip under the 'small'
    # semaphore and trigger server disconnects.
    precomputed_route = None
    try:
        precomputed_route = pipeline.profile_and_route(question)
        _, _, cfg = precomputed_route
        actual_model = cfg.model_name
        actual_tier = cfg.rule_hit
    except Exception as exc:
        log.warning("[job=%d q=%d] pre-profile failed (%s); falling back to expected_tier=%s",
                    job_idx, display_idx, exc, expected_tier)
        actual_model = _infer_model(expected_tier)
        actual_tier = expected_tier

    sem = large_sem if actual_model == "mistral-large-latest" else small_sem

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        with _acquire(sem):
            spacer.wait()
            t0 = time.time()
            try:
                log.info(
                    "[job=%d q=%d expected=%s actual=%s model=%s attempt=%d] starting",
                    job_idx, display_idx, expected_tier, actual_tier, actual_model, attempt,
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
                    job_idx, display_idx, ans.rule_hit, len(ans.answer),
                    len(ans.formatted_references), elapsed,
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
                backoff += random.uniform(0, 0.5)  # jitter
                log.warning(
                    "[job=%d q=%d attempt=%d] %s: %s — sleeping %.1fs",
                    job_idx, display_idx, attempt,
                    "RATE LIMIT" if is_rate else "ERROR",
                    exc, backoff,
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
# output writers — match tests/test_adaptive_v2.py format
# ---------------------------------------------------------------------------

def _write_outputs(records: List[Dict[str, Any]], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "phase3_answers_readable.txt"
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

            tf.write("=" * 60 + "\n")
            tf.write(
                f"Q{i} [expected={rec['expected_tier']} actual={rec['actual_tier']}]: "
                f"{rec['question']}\n"
            )
            tf.write("-" * 60 + "\n")
            tf.write("ANSWER:\n")
            tf.write(rec["answer"] + "\n")
            tf.write("-" * 60 + "\n")
            tf.write("FORMATTED REFERENCES:\n")
            tf.write("\n".join(rec["formatted_references"]) + "\n")
            tf.write("-" * 60 + "\n")
            n_excerpts = rec["excerpt_stats"].get("n_excerpts", 0) if isinstance(rec["excerpt_stats"], dict) else 0
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
    """'2,2,1' -> {tier-1:2, tier-2:2, tier-3:1}"""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("tier-mix must be 3 ints: t1,t2,t3")
    return {"tier-1": int(parts[0]), "tier-2": int(parts[1]), "tier-3": int(parts[2])}


def main() -> int:
    ap = argparse.ArgumentParser(description="parallel phase-3 adaptive pipeline runner")
    ap.add_argument("--workers", "-w", type=int, default=4,
                    help="global concurrency cap (default: 4)")
    ap.add_argument("--small-concurrency", type=int, default=4,
                    help="max concurrent mistral-small calls (default: 4)")
    ap.add_argument("--large-concurrency", type=int, default=1,
                    help="max concurrent mistral-large calls (default: 1 — "
                         "serialise large-model work to avoid server disconnects)")
    ap.add_argument("--min-gap-ms", type=int, default=0,
                    help="minimum milliseconds between call starts (default: 0)")
    ap.add_argument("--job-timeout-s", type=int, default=300,
                    help="per-question wall-clock timeout in seconds "
                         "(default: 300). A stuck request is abandoned and "
                         "recorded as an error, so one bad call does not "
                         "block the whole run.")
    ap.add_argument("--tier-mix", type=_parse_tier_mix, default="2,2,1",
                    help="'t1,t2,t3' count of questions per tier (default: 2,2,1)")
    ap.add_argument("--n", type=int, default=None,
                    help="override total question count (uniform-ish spread across tiers)")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="1-based question indices into deduped DLR rows; overrides tier-mix")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--output-dir", default="tests/output")
    ap.add_argument("--csv", default=None, help="explicit DLR CSV path")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

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
        # redistribute args.n across tiers, weighted by tier-mix proportions
        total_mix = sum(target_counts.values()) or 1
        scaled = {t: max(0, round(args.n * target_counts[t] / total_mix)) for t in target_counts}
        diff = args.n - sum(scaled.values())
        scaled["tier-2"] = scaled.get("tier-2", 0) + diff  # absorb rounding into tier-2
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

    small_sem = threading.Semaphore(max(1, args.small_concurrency))
    large_sem = threading.Semaphore(max(1, args.large_concurrency))
    spacer = _StartSpacer(args.min_gap_ms / 1000.0)

    t_start = time.time()
    records: List[Dict[str, Any]] = []

    # daemon threads so a stuck one doesn't keep the process alive
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
                large_sem=large_sem,
                spacer=spacer,
                max_retries=args.max_retries,
            ): (ji, display_idx, q, tier)
            for ji, (display_idx, q, aql, tier, docs) in enumerate(selected, start=1)
        }

        pending = set(futures.keys())
        deadline_per_job: Dict[Any, float] = {f: time.time() + args.job_timeout_s for f in futures}

        while pending:
            # next soonest deadline
            now = time.time()
            next_deadline = min(deadline_per_job[f] for f in pending)
            wait_s = max(0.1, next_deadline - now)
            done, _ = _wait_any(pending, timeout=wait_s)

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

            # no future completed within wait_s; check for timeouts
            now = time.time()
            for fut in list(pending):
                if now >= deadline_per_job[fut]:
                    ji, display_idx, q, tier = futures[fut]
                    log.error(
                        "job %d (q=%d tier=%s) exceeded --job-timeout-s=%ds; abandoning",
                        ji, display_idx, tier, args.job_timeout_s,
                    )
                    fut.cancel()  # no-op if already running; best-effort
                    pending.discard(fut)
                    records.append({
                        "job_idx": ji, "q_index": display_idx,
                        "question": q, "expected_tier": tier, "actual_tier": "TIMEOUT",
                        "answer": f"ERROR: exceeded job timeout of {args.job_timeout_s}s",
                        "enriched_context": "", "enriched_context_chars": 0,
                        "excerpt_stats": {}, "references": [], "formatted_references": [],
                        "elapsed_s": float(args.job_timeout_s),
                        "error": f"timeout_{args.job_timeout_s}s",
                    })
    finally:
        # don't block shutdown on a stuck worker. cancel_futures was added in
        # py3.9; fall back gracefully on older interpreters.
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)

    elapsed = time.time() - t_start
    txt_path, jsonl_path = _write_outputs(records, Path(args.output_dir))

    ok = sum(1 for r in records if r["error"] is None)
    err = len(records) - ok
    print("\n" + "=" * 72)
    print(f"parallel phase 3 complete: ok={ok}, err={err}, total_elapsed={elapsed:.1f}s")
    print(f"output txt:   {txt_path}")
    print(f"output jsonl: {jsonl_path}")
    print("=" * 72)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
