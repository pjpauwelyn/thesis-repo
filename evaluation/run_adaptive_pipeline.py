"""run_adaptive_pipeline.py

generate the answers csv for the set5 adaptive pipeline. per question:
    1. profile     -- small llm, fused av-pair + profile json
    2. route       -- core.policy.Router picks PipelineConfig
    3. evidence    -- abstracts (no extra work) | excerpts via FullTextIndexer
    4. refinement  -- 1pass-refined or 1pass-fulltext, depending on evidence
    5. generation  -- direct or structured, with adaptive directives

the output csv has exactly the columns needed by run_judge.py
(question_id, question, answer), plus telemetry columns so we can
post-hoc inspect the routing decisions. telemetry columns are ignored
by run_judge.py.

usage:
    # full run (4 parallel workers)
    python3 evaluation/run_adaptive_pipeline.py \
        --output evaluation/results/set5_adaptive/answers_adaptive.csv

    # custom worker count
    python3 evaluation/run_adaptive_pipeline.py --workers 2 \
        --output evaluation/results/set5_adaptive/answers_adaptive.csv

    # dry-run: profile + route only, no LLM calls downstream (prints histogram)
    python3 evaluation/run_adaptive_pipeline.py --dry-run \
        --output evaluation/results/set5_adaptive/routing_preview.csv

    # prebuild fulltext cache first (recommended):
    python3 -m core.utils.fulltext_indexer prebuild \
        --input data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# ensure the repo root is on sys.path so `from core...` works when the script
# is invoked directly (`python3 evaluation/run_adaptive_pipeline.py`) rather
# than via `python3 -m evaluation.run_adaptive_pipeline`. No-op if already present.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

csv.field_size_limit(int(1e8))

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv"
DEFAULT_OUTPUT = "evaluation/results/set5_adaptive/answers_adaptive.csv"
DEFAULT_CACHE = "cache/fulltext"
DEFAULT_RULES = "core/policy/rules.yaml"
DEFAULT_PROMPTS = "prompts"
DEFAULT_WORKERS = 4  # parallel workers; stays within Mistral rate limits

# telemetry columns added beyond the run_judge.py-required triple.
# run_judge.py reads only {question_id, question, answer} via row.get(), so
# extra telemetry columns are always safe to append.
# NOTE: expected_length removed — QuestionProfile has no such field.
_FIELDNAMES = [
    "question_id", "question", "answer",
    "rule_hit", "model_name", "evidence_mode",
    "answer_shape",
    "complexity", "quantitativity",
    "spatial_specificity", "temporal_specificity",
    "profile_confidence", "profile_json",
    "formatted_refs_count",
]

# minimum answer length (chars) to consider a row "done"
_MIN_DONE_ANSWER_CHARS = 200

# global lock for thread-safe csv writes
_CSV_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> logging.Logger:
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("run_adaptive_pipeline")


def _load_questions(
    input_csv: str,
    num: Optional[int],
    indices: Optional[List[int]],
) -> List[Dict[str, Any]]:
    """load the dlr csv and preserve 0-based row index for stable qids."""
    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["_row_index"] = i
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(r)
    if indices:
        deduped = [r for r in deduped if r["_row_index"] in set(indices)]
    if num:
        deduped = deduped[:num]
    return deduped


def _write_row(out_path: Path, row: Dict[str, Any]) -> None:
    """thread-safe incremental csv writer; replaces existing row for same qid."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_qid = str(row.get("question_id", "")).strip()
    with _CSV_LOCK:
        if out_path.exists():
            existing: List[Dict[str, str]] = []
            with open(out_path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if str(r.get("question_id", "")).strip() == new_qid:
                        continue
                    existing.append({k: r.get(k, "") for k in _FIELDNAMES})
            existing.append({k: str(row.get(k, "")) for k in _FIELDNAMES})
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                w.writeheader()
                for r in existing:
                    w.writerow(r)
        else:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                w.writeheader()
                w.writerow({k: str(row.get(k, "")) for k in _FIELDNAMES})


def _load_already_done(out_path: Path) -> Dict[str, str]:
    if not out_path.exists():
        return {}
    done: Dict[str, str] = {}
    with open(out_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = (row.get("question_id") or "").strip()
            ans = (row.get("answer") or "").strip()
            if len(ans) < _MIN_DONE_ANSWER_CHARS:
                continue
            if qid and ans and not ans.startswith("ERROR"):
                done[qid] = ans
    return done


_CITATION_MARKER_RE = None  # lazy-compiled


def _strip_references_section(answer: str) -> str:
    """Return answer body with any trailing References section removed.
    Handles plain, markdown-heading, and bold forms. Kept in-sync with
    core.pipelines.adaptive_pipeline.AdaptivePipeline._strip_self_written_refs
    but tolerant of additional heading variants we may emit.
    """
    if not answer:
        return ""
    lines = answer.split("\n")
    for i, line in enumerate(lines):
        s = line.strip().strip("*#").strip()
        if s in ("References", "REFERENCES"):
            return "\n".join(lines[:i]).rstrip()
        if s.lower().startswith("### references") or s.lower() == "references:":
            return "\n".join(lines[:i]).rstrip()
    return answer


def _body_has_citation_markers(answer: str) -> bool:
    """True iff the answer body (with References section removed) contains at
    least one [N] numeric citation marker."""
    global _CITATION_MARKER_RE
    if _CITATION_MARKER_RE is None:
        import re
        _CITATION_MARKER_RE = re.compile(r"\[[0-9][0-9,;\s]*\]")
    body = _strip_references_section(answer or "")
    return bool(_CITATION_MARKER_RE.search(body))


def _flag_empty_refs_with_citations(
    out_path: Path, log: logging.Logger
) -> Path:
    """Scan the answers CSV. For each row where formatted_refs_count == 0 AND
    the body contains [N] markers, append its question_id to a plaintext file
    so the user can re-run just those questions via --indices.

    Returns the path of the generated flags file (always created, may be empty).
    """
    flags_path = out_path.with_name(out_path.stem + "_needs_retry.txt")
    if not out_path.exists():
        log.warning("flagging: answers csv not found at %s", out_path)
        flags_path.write_text("", encoding="utf-8")
        return flags_path

    flagged: List[Dict[str, str]] = []
    total = 0
    error_rows = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            qid = (row.get("question_id") or "").strip()
            answer = (row.get("answer") or "").strip()
            if answer.startswith("ERROR"):
                error_rows += 1
                continue
            # parse formatted_refs_count defensively; absent/blank → 0
            raw_cnt = (row.get("formatted_refs_count") or "").strip()
            try:
                refs_count = int(raw_cnt) if raw_cnt else 0
            except ValueError:
                refs_count = 0
            if refs_count > 0:
                continue
            if not _body_has_citation_markers(answer):
                continue
            flagged.append({
                "question_id": qid,
                "question": (row.get("question") or "").strip(),
                "model_name": (row.get("model_name") or "").strip(),
                "rule_hit": (row.get("rule_hit") or "").strip(),
            })

    # write plaintext list of qids (one per line) — consumable by --indices
    # via `xargs`, and by humans at a glance. also emit a sibling JSON with
    # richer detail.
    flags_path.write_text(
        "\n".join(r["question_id"] for r in flagged) + ("\n" if flagged else ""),
        encoding="utf-8",
    )
    json_path = flags_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(
            {
                "source_csv": str(out_path),
                "total_rows": total,
                "error_rows": error_rows,
                "flagged_count": len(flagged),
                "criterion": "formatted_refs_count == 0 AND body contains [N] markers",
                "flagged": flagged,
            },
            jf,
            ensure_ascii=False,
            indent=2,
        )

    log.info(
        "citation-hygiene flag: %d / %d rows need retry (empty refs + citations in body); "
        "qid list -> %s ; details -> %s",
        len(flagged), total, flags_path, json_path,
    )
    if flagged:
        preview = ", ".join(r["question_id"] for r in flagged[:10])
        more = "" if len(flagged) <= 10 else f" ...(+{len(flagged)-10})"
        log.info("flagged qids: %s%s", preview, more)
    return flags_path


def _print_histogram(counter: Counter, title: str) -> None:
    print(f"\n{title}")
    total = sum(counter.values()) or 1
    for key, n in counter.most_common():
        bar = "#" * int(40 * n / total)
        print(f"  {str(key):28s} {n:4d}  {bar}")
    print()


# ---------------------------------------------------------------------------
# per-question worker
# ---------------------------------------------------------------------------

def _profile_row(
    i: int,
    total: int,
    row: Dict[str, Any],
    pipeline,
    log: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """profile + route ONCE per question (cheap mistral-small call).

    This is done up-front by the scheduler so (a) the per-model semaphore
    decision and the model actually used inside pipeline.run() are guaranteed
    to agree, and (b) questions can be ordered and batched by tier before
    launching the hot retrieval/generation calls.

    Returns a dict with all derived state, or None if profiling failed.
    """
    from core.utils.aql_parser import parse_aql_results

    orig_idx = row.get("_row_index")
    qid = str(row.get("question_id") or (int(orig_idx) + 1 if orig_idx is not None else i))
    question = (row.get("question") or "").strip()

    aql_raw = row.get("aql_results", "") or ""
    parsed_aql_str = ""
    docs: List[Dict[str, Any]] = []
    if aql_raw:
        parsed_aql_str = parse_aql_results(aql_raw)
        try:
            parsed = json.loads(parsed_aql_str)
            if isinstance(parsed, list):
                docs = parsed
        except json.JSONDecodeError:
            docs = []

    try:
        ontology, profile, cfg = pipeline.profile_and_route(question)
    except Exception as exc:
        log.error("profile_and_route failed qid=%s: %s", qid, exc)
        return None

    return {
        "i": i,
        "qid": qid,
        "question": question,
        "parsed_aql_str": parsed_aql_str,
        "aql_raw": aql_raw,
        "docs": docs,
        "precomputed_route": (ontology, profile, cfg),
        "model_name": cfg.model_name,
        "rule_hit": cfg.rule_hit,
        "evidence_mode": cfg.evidence_mode,
        "profile": profile,
        "cfg": cfg,
    }


def _run_one(
    job: Dict[str, Any],
    total: int,
    pipeline,
    out_path: Path,
    log: logging.Logger,
    small_sem: threading.Semaphore,
    large_sem: threading.Semaphore,
    dry_run: bool,
):
    """execute a single pre-profiled question under the appropriate
    per-model semaphore, write the result row, return the summary tuple.
    """
    i = job["i"]
    qid = job["qid"]
    question = job["question"]
    cfg = job["cfg"]
    profile = job["profile"]

    print(f"[{i}/{total}] qid={qid} tier={cfg.rule_hit} model={cfg.model_name}: {question[:70]}...")

    if dry_run:
        _write_row(out_path, {
            "question_id": qid, "question": question, "answer": "(dry-run)",
            "rule_hit": cfg.rule_hit, "model_name": cfg.model_name,
            "evidence_mode": cfg.evidence_mode,
            "answer_shape": profile.answer_shape, "complexity": profile.complexity,
            "quantitativity": profile.quantitativity,
            "spatial_specificity": profile.spatial_specificity,
            "temporal_specificity": profile.temporal_specificity,
            "profile_confidence": profile.confidence,
            "profile_json": profile.model_dump_json(),
            "formatted_refs_count": 0,
        })
        return cfg.rule_hit, cfg.evidence_mode, cfg.model_name, profile.answer_shape, False

    sem = large_sem if cfg.model_name == "mistral-large-latest" else small_sem

    t0 = time.time()
    try:
        sem.acquire()
        try:
            result = pipeline.run(
                question=question,
                aql_results_str=job["parsed_aql_str"] or job["aql_raw"],
                docs=job["docs"],
                precomputed_route=job["precomputed_route"],
            )
        finally:
            sem.release()
    except Exception as exc:
        log.error("adaptive run failed qid=%s: %s", qid, exc)
        _write_row(out_path, {
            "question_id": qid, "question": question, "answer": f"ERROR: {exc}",
            "rule_hit": cfg.rule_hit, "model_name": cfg.model_name,
            "evidence_mode": cfg.evidence_mode,
            "answer_shape": profile.answer_shape if profile else "",
            "complexity": profile.complexity if profile else "",
            "quantitativity": profile.quantitativity if profile else "",
            "spatial_specificity": profile.spatial_specificity if profile else "",
            "temporal_specificity": profile.temporal_specificity if profile else "",
            "profile_confidence": profile.confidence if profile else "",
            "profile_json": profile.model_dump_json() if profile else "",
            "formatted_refs_count": 0,
        })
        return None, None, None, None, True

    prof = result.profile
    cfg = result.pipeline_config
    print(
        f"  -> qid={qid} rule={cfg.rule_hit if cfg else '?'} model={cfg.model_name if cfg else '?'} "
        f"evidence={cfg.evidence_mode if cfg else '?'} "
        f"answer_chars={len(result.answer)} ({time.time()-t0:.1f}s)"
    )
    _write_row(out_path, {
        "question_id": qid, "question": question, "answer": result.answer,
        "rule_hit": cfg.rule_hit if cfg else "",
        "model_name": cfg.model_name if cfg else "",
        "evidence_mode": cfg.evidence_mode if cfg else "",
        "answer_shape": prof.answer_shape if prof else "",
        "complexity": prof.complexity if prof else "",
        "quantitativity": prof.quantitativity if prof else "",
        "spatial_specificity": prof.spatial_specificity if prof else "",
        "temporal_specificity": prof.temporal_specificity if prof else "",
        "profile_confidence": prof.confidence if prof else "",
        "profile_json": prof.model_dump_json() if prof else "",
        "formatted_refs_count": len(result.formatted_references or []),
    })
    return (
        cfg.rule_hit if cfg else "?",
        cfg.evidence_mode if cfg else "?",
        cfg.model_name if cfg else "?",
        prof.answer_shape if prof else "?",
        False,
    )


# ---------------------------------------------------------------------------
# scheduling helpers
# ---------------------------------------------------------------------------

def _is_retryable_error(exc_str: str) -> bool:
    """Heuristic: should a failed tier-3 question be retried at the end of
    the run (after the large-model endpoint has had time to recover)?

    Empty-refinement aborts and Mistral server disconnects are retryable.
    Deterministic code bugs are not (we look for a short allow-list).
    """
    s = (exc_str or "").lower()
    retryable_markers = [
        "server disconnected",
        "refinement produced empty",
        "connection",
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "503",
        "502",
        "504",
        "read timeout",
        "broken pipe",
    ]
    return any(m in s for m in retryable_markers)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="adaptive pipeline runner (set5).")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--num", type=int, default=None, help="limit number of questions")
    ap.add_argument("--indices", type=int, nargs="+", default=None)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--prompts-root", default=DEFAULT_PROMPTS)
    ap.add_argument("--overwrite", action="store_true", help="re-run already-done questions")
    ap.add_argument("--dry-run", action="store_true",
                    help="profile + route only, no generation (prints histogram)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="parallel worker threads (default: 4, use 1 to disable). "
                         "This is the global pool size; per-model concurrency is "
                         "further constrained by --large-concurrency/--small-concurrency.")
    ap.add_argument("--large-concurrency", type=int, default=1,
                    help="max concurrent mistral-large calls (default: 1 — "
                         "serialise tier-3 to avoid Mistral large-endpoint disconnects)")
    ap.add_argument("--small-concurrency", type=int, default=4,
                    help="max concurrent mistral-small calls (default: 4)")
    ap.add_argument("--job-timeout-s", type=int, default=600,
                    help="per-question wall-clock timeout in seconds (default: 600). "
                         "A stuck request is abandoned, recorded as ERROR, and "
                         "queued for a second attempt.")
    ap.add_argument("--tier3-retry", type=int, default=1,
                    help="number of extra end-of-run retries for failed tier-3 "
                         "questions with retryable errors (default: 1)")
    ap.add_argument("--flag-only", action="store_true",
                    help="skip generation; just scan --output and emit the "
                         "_needs_retry.txt / .json flag files")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log = _setup_logging(args.verbose)

    # --flag-only short-circuit: skip the pipeline entirely, just audit the csv
    if args.flag_only:
        out_path = Path(args.output)
        _flag_empty_refs_with_citations(out_path, log)
        sys.exit(0)

    from core.pipelines.adaptive_pipeline import AdaptivePipeline

    pipeline = AdaptivePipeline(
        rules_path=args.rules,
        cache_dir=args.cache_dir,
        prompts_root=args.prompts_root,
    )

    rows = _load_questions(args.input, args.num, args.indices)
    if not rows:
        log.error("no rows loaded from %s", args.input)
        sys.exit(1)

    out_path = Path(args.output)
    done = {} if (args.overwrite or args.dry_run) else _load_already_done(out_path)
    if done:
        log.info("resume: %d already done in %s", len(done), out_path)

    pending = []
    for i, row in enumerate(rows, start=1):
        orig_idx = row.get("_row_index")
        qid = str(row.get("question_id") or (int(orig_idx) + 1 if orig_idx is not None else i))
        if qid in done:
            log.info("[%s] skip (already done)", qid)
        else:
            pending.append((i, row))

    log.info("%d questions to process (workers=%d, large_conc=%d, small_conc=%d)",
             len(pending), args.workers, args.large_concurrency, args.small_concurrency)

    # --- stage 1: pre-profile every pending question (cheap mistral-small) ---
    # Doing this up-front has three benefits:
    #   * we can bucket questions by tier and schedule small-model work first
    #   * the route used here is passed into pipeline.run() verbatim so the
    #     semaphore decision and actual model used are guaranteed to agree
    #   * a flaky large-endpoint no longer blocks profile calls (which route
    #     through mistral-small-latest)
    log.info("stage 1: pre-profiling %d questions...", len(pending))
    t_profile = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(args.small_concurrency, args.workers))
    ) as pex:
        profile_futures = {
            pex.submit(_profile_row, idx, len(rows), row, pipeline, log): (idx, row)
            for idx, row in pending
        }
        jobs: List[Dict[str, Any]] = []
        for fut in concurrent.futures.as_completed(profile_futures):
            res = fut.result()
            if res is not None:
                jobs.append(res)
    log.info("stage 1: profiled %d/%d questions in %.1fs",
             len(jobs), len(pending), time.time() - t_profile)

    # bucket by model for a visible summary
    n_large = sum(1 for j in jobs if j["model_name"] == "mistral-large-latest")
    n_small = len(jobs) - n_large
    log.info("route summary: %d small-model questions, %d large-model questions", n_small, n_large)

    # --- stage 2: execute under per-model semaphores with a watchdog -------
    # Submit small-model jobs first so they can't be starved by a tier-3
    # backoff storm on the large endpoint; large-model jobs trickle through
    # one at a time (--large-concurrency=1 by default).
    jobs.sort(key=lambda j: (0 if j["model_name"] != "mistral-large-latest" else 1, j["i"]))

    rule_hist: Counter = Counter()
    evidence_hist: Counter = Counter()
    model_hist: Counter = Counter()
    shape_hist: Counter = Counter()
    errors = 0
    successes = 0
    total = len(rows)
    t_start = time.time()

    max_workers = 1 if args.dry_run else max(1, args.workers)
    small_sem = threading.Semaphore(max(1, args.small_concurrency))
    large_sem = threading.Semaphore(max(1, args.large_concurrency))

    # retry queue: jobs we'll re-run once at the end after the large endpoint
    # has had a cooldown. populated from watchdog timeouts + retryable errors.
    retry_queue: List[Dict[str, Any]] = []
    retry_reason: Dict[str, str] = {}

    def _drain_jobs(job_list: List[Dict[str, Any]], pass_label: str) -> None:
        nonlocal successes, errors
        if not job_list:
            return
        log.info("stage 2 (%s): running %d jobs", pass_label, len(job_list))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {
                pool.submit(
                    _run_one, job, total, pipeline, out_path, log,
                    small_sem, large_sem, args.dry_run,
                ): job
                for job in job_list
            }
            pending_futs = set(futures.keys())
            deadline: Dict[Any, float] = {
                f: time.time() + args.job_timeout_s for f in pending_futs
            }
            while pending_futs:
                next_deadline = min(deadline[f] for f in pending_futs)
                wait_s = max(0.5, next_deadline - time.time())
                done, _ = concurrent.futures.wait(
                    pending_futs, timeout=wait_s,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if done:
                    for fut in done:
                        pending_futs.discard(fut)
                        job = futures[fut]
                        try:
                            rule, evidence, model, shape, is_err = fut.result()
                        except Exception as exc:
                            log.error("job qid=%s crashed: %s", job["qid"], exc)
                            is_err, rule, evidence, model, shape = True, None, None, None, None
                        if is_err:
                            errors += 1
                            if pass_label == "initial" and job["model_name"] == "mistral-large-latest":
                                retry_queue.append(job)
                                retry_reason[job["qid"]] = "initial-error"
                        else:
                            successes += 1
                            if rule: rule_hist[rule] += 1
                            if evidence: evidence_hist[evidence] += 1
                            if model: model_hist[model] += 1
                            if shape: shape_hist[shape] += 1
                    continue
                # nothing completed — check for watchdog timeouts
                now = time.time()
                for fut in list(pending_futs):
                    if now >= deadline[fut]:
                        job = futures[fut]
                        log.error(
                            "job qid=%s (tier=%s) exceeded --job-timeout-s=%ds; abandoning",
                            job["qid"], job["rule_hit"], args.job_timeout_s,
                        )
                        fut.cancel()
                        pending_futs.discard(fut)
                        errors += 1
                        # write the timeout as an ERROR row immediately so the
                        # csv is never silently missing a question
                        _write_row(out_path, {
                            "question_id": job["qid"], "question": job["question"],
                            "answer": f"ERROR: job watchdog timeout after {args.job_timeout_s}s",
                            "rule_hit": job["rule_hit"], "model_name": job["model_name"],
                            "evidence_mode": job["evidence_mode"],
                            "answer_shape": job["profile"].answer_shape,
                            "complexity": job["profile"].complexity,
                            "quantitativity": job["profile"].quantitativity,
                            "spatial_specificity": job["profile"].spatial_specificity,
                            "temporal_specificity": job["profile"].temporal_specificity,
                            "profile_confidence": job["profile"].confidence,
                            "profile_json": job["profile"].model_dump_json(),
                            "formatted_refs_count": 0,
                        })
                        if pass_label == "initial" and job["model_name"] == "mistral-large-latest":
                            retry_queue.append(job)
                            retry_reason[job["qid"]] = "initial-timeout"
        finally:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pool.shutdown(wait=False)

    _drain_jobs(jobs, "initial")

    # --- stage 3 (optional): retry failed tier-3 jobs once at the end ------
    # After the large-model endpoint has been idle for the small-model phase,
    # a second attempt often succeeds. This is capped at --tier3-retry passes.
    for retry_pass in range(1, max(0, args.tier3_retry) + 1):
        if not retry_queue:
            break
        log.info("stage 3 (retry pass %d): retrying %d tier-3 jobs",
                 retry_pass, len(retry_queue))
        # errors already include these — subtract so successful retries
        # don't double-count, and re-count failures fresh.
        errors -= len(retry_queue)
        this_pass = retry_queue
        retry_queue = []
        _drain_jobs(this_pass, f"retry-{retry_pass}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    mode = "dry-run" if args.dry_run else "full run"
    print(f"adaptive pipeline {mode} complete: ok={successes}, err={errors}, elapsed={elapsed/60:.1f} min")
    print(f"output: {out_path}")
    _print_histogram(rule_hist, "rule_hit distribution")
    _print_histogram(evidence_hist, "evidence_mode distribution")
    _print_histogram(model_hist, "model_name distribution")
    _print_histogram(shape_hist, "answer_shape distribution")
    print("=" * 72)

    # post-run citation-hygiene audit: flag rows with citation markers but
    # zero validated references so the user can target them with --indices.
    # we only run this for real generation, not --dry-run (answers are stubs).
    if not args.dry_run:
        _flag_empty_refs_with_citations(out_path, log)

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
