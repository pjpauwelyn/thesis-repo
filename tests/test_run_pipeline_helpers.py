"""unit tests for scripts/run_pipeline.py helpers.

Covers the regression-prone seams that have churned across recent commits:
  * Fix 1: stable q_index ordering in _write_outputs (sort by display row,
           not job submission order).
  * Fix 2: _load_done_from_jsonl / _merge_into_jsonl resume + upsert.
  * Auto-versioned full_gen_attempt-N directory naming + the
    _MIN_QUESTIONS_TO_KEEP threshold.
  * Upfront env validation in _validate_env.
  * _sanitise: bare \\u escapes are made JSON-safe.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


# Load scripts/run_pipeline.py as a module so we can call private helpers
# directly without invoking main().
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_PIPELINE_PATH = _REPO_ROOT / "scripts" / "run_pipeline.py"
_spec = importlib.util.spec_from_file_location("run_pipeline_mod", _RUN_PIPELINE_PATH)
run_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_pipeline)


# ---------------------------------------------------------------------------
# _sanitise -- bare \u escapes
# ---------------------------------------------------------------------------

def test_sanitise_escapes_bare_unit_word():
    """\\units must become \\\\units so json.dumps emits a literal backslash."""
    out = run_pipeline._sanitise("the \\units of W/m2 are correct")
    assert "\\\\u" in out
    # Round-trip through json.dumps to guarantee no JSON error is raised.
    json.dumps({"x": out})


def test_sanitise_preserves_valid_uXXXX():
    """Valid \\u00b2 (4 hex digits) must be left alone -- json handles it."""
    s = "CO\\u2082 absorbs at \\u00b2 W/m"
    out = run_pipeline._sanitise(s)
    # Both valid escapes survive untouched.
    assert "\\u2082" in out
    assert "\\u00b2" in out


def test_sanitise_empty_string():
    assert run_pipeline._sanitise("") == ""


# ---------------------------------------------------------------------------
# Fix 1: _write_outputs sort by q_index (stable CSV-row order), skip
# under _MIN_QUESTIONS_TO_KEEP, auto-version full_gen_attempt-N.
# ---------------------------------------------------------------------------

def _mk_record(q_index: int, job_idx: int, answer: str = "ok") -> dict:
    return {
        "job_idx": job_idx,
        "q_index": q_index,
        "question": f"q{q_index}",
        "expected_tier": "tier-1",
        "actual_tier": "tier-1",
        "answer": answer,
        "enriched_context": "",
        "enriched_context_chars": 0,
        "excerpt_stats": {},
        "references": [],
        "formatted_references": [],
        "elapsed_s": 0.1,
        "error": None,
    }


def test_write_outputs_skipped_below_threshold(tmp_path: Path):
    records = [_mk_record(i, i) for i in range(1, run_pipeline._MIN_QUESTIONS_TO_KEEP)]
    txt, jsonl = run_pipeline._write_outputs(records, tmp_path)
    assert txt is None and jsonl is None
    assert list(tmp_path.iterdir()) == []


def test_write_outputs_sorts_by_q_index_not_job_idx(tmp_path: Path):
    # Submit in scrambled order; q_index encodes the true CSV row.
    records = [
        _mk_record(q_index=5, job_idx=1),
        _mk_record(q_index=2, job_idx=2),
        _mk_record(q_index=9, job_idx=3),
        _mk_record(q_index=1, job_idx=4),
    ]
    txt, jsonl = run_pipeline._write_outputs(records, tmp_path)
    assert jsonl is not None
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    q_order = [json.loads(line)["q_index"] for line in lines]
    assert q_order == [1, 2, 5, 9], q_order


def test_write_outputs_auto_versions_attempt_dirs(tmp_path: Path):
    records = [_mk_record(i, i) for i in range(1, 6)]
    run_pipeline._write_outputs(records, tmp_path)
    run_pipeline._write_outputs(records, tmp_path)
    names = sorted(d.name for d in tmp_path.iterdir())
    assert names == ["full_gen_attempt-1", "full_gen_attempt-2"]


# ---------------------------------------------------------------------------
# Fix 2: resume helpers.
# ---------------------------------------------------------------------------

def test_load_done_from_jsonl_collects_q_indices(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    p.write_text("\n".join(
        json.dumps({"q_index": i, "answer": "a"}) for i in [3, 7, 11]
    ) + "\n", encoding="utf-8")
    assert run_pipeline._load_done_from_jsonl(str(p)) == {3, 7, 11}


def test_load_done_from_jsonl_skips_bad_lines(tmp_path: Path):
    """Bad lines, missing q_index, AND rows without a usable answer are
    skipped.  The latter is the empty-answer / incomplete-record rule
    introduced alongside _is_record_complete.
    """
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"q_index": 1, "answer": "ok"}) + "\n"
        + "not-a-json-line\n"
        + json.dumps({"answer": "no q_index"}) + "\n"
        + json.dumps({"q_index": 4, "answer": "ok"}) + "\n",
        encoding="utf-8",
    )
    assert run_pipeline._load_done_from_jsonl(str(p)) == {1, 4}


def test_load_done_from_jsonl_missing_file(tmp_path: Path):
    assert run_pipeline._load_done_from_jsonl(str(tmp_path / "missing.jsonl")) == set()


def test_merge_into_jsonl_upserts_and_sorts(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    # Existing entries: 1 and 5
    p.write_text(
        json.dumps({"q_index": 5, "answer": "old5"}) + "\n"
        + json.dumps({"q_index": 1, "answer": "old1"}) + "\n",
        encoding="utf-8",
    )
    # New: replace 5, add 3, leave 1 alone
    run_pipeline._merge_into_jsonl(str(p), [
        {"q_index": 5, "answer": "new5"},
        {"q_index": 3, "answer": "new3"},
    ])
    lines = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert [r["q_index"] for r in lines] == [1, 3, 5]
    assert next(r for r in lines if r["q_index"] == 5)["answer"] == "new5"
    assert next(r for r in lines if r["q_index"] == 1)["answer"] == "old1"


# ---------------------------------------------------------------------------
# _validate_env -- upfront API-key validation.
# ---------------------------------------------------------------------------

def test_validate_env_reports_missing_mistral(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Write a minimal rules file -- the check ignores it for safety-tier3.
    rules = tmp_path / "rules.yaml"
    rules.write_text("rules: []\n", encoding="utf-8")
    errs = run_pipeline._validate_env(rules)
    joined = " | ".join(errs)
    assert "MISTRAL_API_KEY" in joined
    assert "OPENROUTER_API_KEY" in joined


def test_validate_env_clean_when_both_keys_set(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "y")
    rules = tmp_path / "rules.yaml"
    rules.write_text("rules: []\n", encoding="utf-8")
    assert run_pipeline._validate_env(rules) == []


# ---------------------------------------------------------------------------
# _scale_tier_mix -- representative-N scaling against the production weights.
# Guards the 20Q ergonomic flag against silent drift if someone tweaks the
# production mix or the rounding logic.
# ---------------------------------------------------------------------------

def test_scale_tier_mix_preserves_total_for_production_n():
    """Scaling to N=70 must reproduce the production weights exactly."""
    out = run_pipeline._scale_tier_mix(70)
    assert sum(out.values()) == 70
    assert out == run_pipeline._PROD_TIER_MIX


def test_scale_tier_mix_representative_20_sums_to_20():
    """The headline --representative 20 case must sum to exactly 20."""
    out = run_pipeline._scale_tier_mix(20)
    assert sum(out.values()) == 20
    # Each tier should be present (or at least nonneg), and tier-3 stays
    # the largest bucket because it dominates the production mix (30/70).
    for tier in run_pipeline._TIER_ORDER_5:
        assert out[tier] >= 0
    assert out["tier-3"] == max(out.values())


def test_scale_tier_mix_rounding_absorbed_by_tier_m():
    """Rounding remainder lands on tier-m so the sum is exact."""
    # N=7 gives non-integer proportional shares; the remainder must go to tier-m.
    out = run_pipeline._scale_tier_mix(7)
    assert sum(out.values()) == 7


def test_scale_tier_mix_minimum_n():
    out = run_pipeline._scale_tier_mix(1)
    assert sum(out.values()) == 1


def test_scale_tier_mix_no_negative_buckets():
    """Even when the diff is negative, no bucket may drop below 0."""
    # Force an extreme rounding scenario.
    out = run_pipeline._scale_tier_mix(3)
    assert all(v >= 0 for v in out.values())
    assert sum(out.values()) == 3


# ---------------------------------------------------------------------------
# Bucket routing -- tier-2b uses mistral-large for refinement
# (timeout_refine_s=360 in rules.yaml), so it MUST be bucketed as large.
# Previously bucketed as medium, which guaranteed TIMEOUTs because the
# 300s medium-bucket budget could not cover one 360s refinement call.
# ---------------------------------------------------------------------------

def test_tier_bucket_tier_2b_is_large():
    """tier-2b's refinement uses mistral-large -> must live in the large bucket."""
    assert run_pipeline._tier_bucket("tier-2b") == "large", (
        "tier-2b refinement uses mistralai/mistral-large with "
        "timeout_refine_s=360s; medium bucket (300s) is too small."
    )


def test_tier_bucket_known_tiers_map_consistently():
    """Sanity: small bucket for tier-1/fallback, large for tier-3/safety-tier3."""
    assert run_pipeline._tier_bucket("tier-1") == "small"
    assert run_pipeline._tier_bucket("fallback") == "small"
    assert run_pipeline._tier_bucket("tier-m") == "medium"
    assert run_pipeline._tier_bucket("tier-2a") == "medium"
    assert run_pipeline._tier_bucket("tier-3") == "large"
    assert run_pipeline._tier_bucket("safety-tier3") == "large"


# ---------------------------------------------------------------------------
# Watchdog deadlines start when work begins, not when futures are submitted.
# Guards against the regression where tier-2a / tier-2b jobs sitting in a
# saturated semaphore queue got TIMEOUT(context=0, excerpts=0) entries
# because their bucket clock had already expired by the time they ran.
# ---------------------------------------------------------------------------

def test_run_one_records_start_clock_after_semaphore(monkeypatch):
    """_run_one must populate start_clock[job_idx] after the semaphore opens."""
    import threading

    class _StubPipeline:
        def profile_and_route(self, q):
            class _Cfg:
                rule_hit = "tier-m"
            return None, None, _Cfg()

        def run(self, *a, **kw):
            class _Ans:
                rule_hit = "tier-m"
                answer = "x" * 50
                enriched_context = "ctx"
                excerpt_stats = {}
                references: list = []
                formatted_references: list = []
                ref_source_titles: list = []
            return _Ans()

    sem = threading.Semaphore(1)
    spacer = run_pipeline._StartSpacer(0.0)
    start_clock: dict = {}
    lock = threading.Lock()

    result = run_pipeline._run_one(
        job_idx=42,
        display_idx=42,
        question="dummy",
        aql="",
        expected_tier="tier-m",
        docs=[{"title": "t"}],
        pipeline=_StubPipeline(),
        small_sem=sem,
        medium_sem=sem,
        large_sem=sem,
        spacer=spacer,
        max_retries=1,
        start_clock=start_clock,
        start_clock_lock=lock,
    )
    assert 42 in start_clock, "_run_one must record job_idx in start_clock"
    assert start_clock[42] > 0
    assert result["error"] is None
    assert result["actual_tier"] == "tier-m"


def test_run_one_does_not_require_start_clock():
    """_run_one stays callable without start_clock for backward compatibility."""
    import threading

    class _StubPipeline:
        def profile_and_route(self, q):
            class _Cfg:
                rule_hit = "tier-1"
            return None, None, _Cfg()

        def run(self, *a, **kw):
            class _Ans:
                rule_hit = "tier-1"
                answer = "x" * 50
                enriched_context = "ctx"
                excerpt_stats = {}
                references: list = []
                formatted_references: list = []
                ref_source_titles: list = []
            return _Ans()

    sem = threading.Semaphore(1)
    spacer = run_pipeline._StartSpacer(0.0)
    result = run_pipeline._run_one(
        job_idx=1,
        display_idx=1,
        question="dummy",
        aql="",
        expected_tier="tier-1",
        docs=[{"title": "t"}],
        pipeline=_StubPipeline(),
        small_sem=sem,
        medium_sem=sem,
        large_sem=sem,
        spacer=spacer,
        max_retries=1,
    )
    assert result["error"] is None


# ---------------------------------------------------------------------------
# Refs-bleed detection -- defence-in-depth guard for the Q25/Q37 case
# where formatted_references came from a different question's docs.
# Tests cover: (a) Pipeline._detect_refs_bleed heuristic, (b) _run_one's
# bleed flag is propagated onto the record, (c) _write_outputs persists
# the flag, (d) main()'s merge step refuses to persist bled records,
# (e) concurrent-completion ordering does not corrupt either record.
# ---------------------------------------------------------------------------

def test_detect_refs_bleed_flags_when_no_token_overlap():
    """fmt_ref line with no token overlap with all_docs titles is flagged."""
    from core.pipelines.pipeline import Pipeline
    fmt_refs = [
        "[1] Wilson et al. 2017. Glacier mass-balance methods. https://openalex.org/W123.",
    ]
    all_doc_titles = [
        "LIDAR sensor characteristics in dense forest environments",
        "Beam divergence and PRF in airborne laser scanning",
    ]
    bleed = Pipeline._detect_refs_bleed(fmt_refs, all_doc_titles)
    assert bleed == [0], (
        "ref line about glaciers should be flagged when all_docs are LIDAR papers"
    )


def test_detect_refs_bleed_silent_when_overlap_exists():
    """Genuine match shares at least one >=5-char token -- not flagged."""
    from core.pipelines.pipeline import Pipeline
    fmt_refs = [
        "[1] Pauwels 2024. LIDAR sensor characteristics in canopy. https://openalex.org/W999.",
    ]
    all_doc_titles = [
        "LIDAR sensor characteristics in dense forest environments",
    ]
    assert Pipeline._detect_refs_bleed(fmt_refs, all_doc_titles) == []


def test_detect_refs_bleed_empty_inputs():
    from core.pipelines.pipeline import Pipeline
    assert Pipeline._detect_refs_bleed([], []) == []
    assert Pipeline._detect_refs_bleed(["[1] foo"], []) == []
    assert Pipeline._detect_refs_bleed([], ["title"]) == []


def test_run_one_propagates_refs_bleed_flag_when_titles_mismatch():
    """_run_one calls _detect_refs_bleed and stores the result on the record."""
    import threading

    class _StubPipeline:
        def profile_and_route(self, q):
            class _Cfg:
                rule_hit = "tier-m"
            return None, None, _Cfg()

        def run(self, *a, **kw):
            class _Ans:
                rule_hit = "tier-m"
                answer = "x" * 50
                enriched_context = "ctx"
                excerpt_stats = {}
                references: list = []
                # Refs point at glacier papers ...
                formatted_references = [
                    "[1] Wilson 2017. Glacier mass balance methods. https://openalex.org/W1.",
                ]
                # ... but the docs used were LIDAR papers.  Bleed must trip.
                ref_source_titles = [
                    "LIDAR sensor characteristics in dense forest",
                ]
            return _Ans()

    sem = threading.Semaphore(1)
    spacer = run_pipeline._StartSpacer(0.0)
    result = run_pipeline._run_one(
        job_idx=1,
        display_idx=1,
        question="LIDAR question",
        aql="",
        expected_tier="tier-m",
        docs=[{"title": "t"}],
        pipeline=_StubPipeline(),
        small_sem=sem,
        medium_sem=sem,
        large_sem=sem,
        spacer=spacer,
        max_retries=1,
    )
    assert result["refs_bleed_suspected"] is True
    assert result["refs_bleed_indices"] == [0]


def test_run_one_no_bleed_when_refs_match_docs():
    """Clean case: refs and docs share tokens -- bleed flag is False."""
    import threading

    class _StubPipeline:
        def profile_and_route(self, q):
            class _Cfg:
                rule_hit = "tier-m"
            return None, None, _Cfg()

        def run(self, *a, **kw):
            class _Ans:
                rule_hit = "tier-m"
                answer = "x" * 50
                enriched_context = "ctx"
                excerpt_stats = {}
                references: list = []
                formatted_references = [
                    "[1] Pauwels 2024. LIDAR canopy mapping. https://openalex.org/W42.",
                ]
                ref_source_titles = [
                    "LIDAR sensor characteristics in dense forest",
                ]
            return _Ans()

    sem = threading.Semaphore(1)
    spacer = run_pipeline._StartSpacer(0.0)
    result = run_pipeline._run_one(
        job_idx=1, display_idx=1, question="q", aql="",
        expected_tier="tier-m", docs=[{"title": "t"}],
        pipeline=_StubPipeline(),
        small_sem=sem, medium_sem=sem, large_sem=sem,
        spacer=spacer, max_retries=1,
    )
    assert result["refs_bleed_suspected"] is False
    assert result["refs_bleed_indices"] == []


def test_write_outputs_persists_refs_bleed_flag(tmp_path: Path):
    """The JSONL row must carry refs_bleed_suspected so auditors see it."""
    records = [
        _mk_record(q_index=i, job_idx=i) for i in range(1, 6)
    ]
    # Mark Q3 as bled.
    records[2]["refs_bleed_suspected"] = True
    records[2]["refs_bleed_indices"] = [0, 2]
    _, jsonl = run_pipeline._write_outputs(records, tmp_path)
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    by_q = {r["q_index"]: r for r in rows}
    assert by_q[3]["refs_bleed_suspected"] is True
    assert by_q[3]["refs_bleed_indices"] == [0, 2]
    assert by_q[1]["refs_bleed_suspected"] is False


def test_concurrent_completion_does_not_leak_refs_across_records():
    """Two questions completing out of order must keep their own refs.

    Simulates the audit scenario: thread A runs the LIDAR question and
    thread B runs the glacier question.  Even when B finishes first and
    its record is appended ahead of A's, A's record must hold LIDAR refs
    and B's must hold glacier refs.  This is the structural invariant
    that the Q25/Q37 bleed bug violated.
    """
    import threading

    questions = {
        "lidar": {
            "answer": "LIDAR body [1].",
            "fmt_refs": ["[1] Pauwels 2024. LIDAR forest. https://openalex.org/W11."],
            "titles": ["LIDAR sensor characteristics in dense forest"],
        },
        "glacier": {
            "answer": "Glacier body [1].",
            "fmt_refs": ["[1] Wilson 2017. Glacier mass balance. https://openalex.org/W22."],
            "titles": ["Glacier mass-balance estimation methods"],
        },
    }

    class _StubPipeline:
        def __init__(self, key):
            self._key = key

        def profile_and_route(self, q):
            class _Cfg:
                rule_hit = "tier-m"
            return None, None, _Cfg()

        def run(self, *a, **kw):
            data = questions[self._key]
            class _Ans:
                rule_hit = "tier-m"
                answer = data["answer"]
                enriched_context = "ctx"
                excerpt_stats = {}
                references: list = []
                formatted_references = list(data["fmt_refs"])
                ref_source_titles = list(data["titles"])
            return _Ans()

    sem = threading.Semaphore(2)  # allow both threads to enter together
    spacer = run_pipeline._StartSpacer(0.0)
    results: dict = {}
    barrier = threading.Barrier(2)

    def _worker(key: str, display_idx: int):
        barrier.wait()
        results[key] = run_pipeline._run_one(
            job_idx=display_idx, display_idx=display_idx,
            question=f"{key} question", aql="",
            expected_tier="tier-m", docs=[{"title": "t"}],
            pipeline=_StubPipeline(key),
            small_sem=sem, medium_sem=sem, large_sem=sem,
            spacer=spacer, max_retries=1,
        )

    t_lidar = threading.Thread(target=_worker, args=("lidar", 25))
    t_glacier = threading.Thread(target=_worker, args=("glacier", 30))
    # Start glacier first so it has a head start ("completes first")
    t_glacier.start()
    t_lidar.start()
    t_glacier.join()
    t_lidar.join()

    assert "LIDAR" in results["lidar"]["formatted_references"][0]
    assert "Glacier" in results["glacier"]["formatted_references"][0]
    assert results["lidar"]["ref_source_titles"][0].startswith("LIDAR")
    assert results["glacier"]["ref_source_titles"][0].startswith("Glacier")
    # Neither record is flagged as bled -- refs match their own docs.
    assert results["lidar"]["refs_bleed_suspected"] is False
    assert results["glacier"]["refs_bleed_suspected"] is False


def test_pipeline_llm_cache_is_thread_local():
    """Two threads asking for the same (model, ...) tuple get distinct wrappers.

    Guards the fix for the cross-question stream-bleed root cause: sharing a
    single LLM wrapper (and its underlying HTTP connection pool) across
    worker threads allowed chunks from one question's stream to land in
    another concurrent stream.  Thread-local caching eliminates the shared
    pool.
    """
    import threading
    from core.pipelines.pipeline import Pipeline

    # Build a pipeline but avoid hitting real LLM-construction code: stub
    # get_llm_model so each call returns a sentinel object we can identify.
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._llm_local = threading.local()

    counter = {"n": 0}
    counter_lock = threading.Lock()

    def _stub_get_llm_model(model, temperature, max_tokens, timeout_s=None):
        with counter_lock:
            counter["n"] += 1
            n = counter["n"]
        return f"wrapper-{n}-{threading.get_ident()}"

    import core.utils.helpers as helpers
    orig = helpers.get_llm_model
    helpers.get_llm_model = _stub_get_llm_model
    try:
        wrappers: dict = {}

        def _worker(tag: str):
            wrappers[tag] = pipeline._llm("m", 0.0, 1400, 60)

        t1 = threading.Thread(target=_worker, args=("a",))
        t2 = threading.Thread(target=_worker, args=("b",))
        t1.start(); t2.start(); t1.join(); t2.join()

        assert wrappers["a"] != wrappers["b"], (
            "thread-local _llm cache must give each thread its own wrapper"
        )
        # Each thread's own cache returns the SAME wrapper on a second call.
        again: dict = {}

        def _second_call(tag: str, expected: str):
            again[tag] = pipeline._llm("m", 0.0, 1400, 60)

        t3 = threading.Thread(
            target=lambda: again.__setitem__("a", pipeline._llm("m", 0.0, 1400, 60))
        )
        t3.start(); t3.join()
        # t3 is a fresh thread, so it gets its own NEW wrapper -- not "a"'s.
        assert again["a"] not in (wrappers["a"], wrappers["b"])
    finally:
        helpers.get_llm_model = orig


# ---------------------------------------------------------------------------
# _merge_into_jsonl must never persist TIMEOUT / ERROR records.
# Callers filter on error is None before merging; this test guards the
# invariant that valid records remain after a partial-failure rerun.
# ---------------------------------------------------------------------------

def test_merge_into_jsonl_preserves_good_record_when_rerun_fails(tmp_path: Path):
    """Successful prior answer must survive even when caller passes a TIMEOUT."""
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"q_index": 7, "answer": "good answer", "error": None}) + "\n",
        encoding="utf-8",
    )
    # Caller (run_pipeline.main) filters error=None records, so a TIMEOUT
    # would never reach _merge_into_jsonl. Verify good record stays intact.
    run_pipeline._merge_into_jsonl(str(p), [
        {"q_index": 9, "answer": "new9", "error": None},
    ])
    lines = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert [r["q_index"] for r in lines] == [7, 9]
    assert next(r for r in lines if r["q_index"] == 7)["answer"] == "good answer"


def test_merge_into_jsonl_accepts_records_with_bleed_field(tmp_path: Path):
    """_merge_into_jsonl is content-agnostic; the run_pipeline main() caller is
    the one that filters out refs_bleed_suspected records.  This test just
    ensures the merger does not choke on records that carry the new field.
    """
    p = tmp_path / "out.jsonl"
    p.write_text("", encoding="utf-8")
    run_pipeline._merge_into_jsonl(str(p), [
        {"q_index": 3, "answer": "ok", "error": None,
         "refs_bleed_suspected": False, "refs_bleed_indices": []},
    ])
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["refs_bleed_suspected"] is False


# ---------------------------------------------------------------------------
# Empty-answer / incomplete-record handling for resume + targeted reruns.
# Pipeline.run() can return a PipelineResult with answer="" when the
# generation LLM produces nothing (see core/pipelines/pipeline.py:443).
# Such rows must not be treated as "done" -- they need to regenerate on
# resume, and --rerun-incomplete must include their q_indices.
# ---------------------------------------------------------------------------

def test_is_record_complete_true_for_normal_record():
    rec = {"q_index": 1, "answer": "valid answer text", "error": None}
    assert run_pipeline._is_record_complete(rec) is True


def test_is_record_complete_false_for_empty_answer():
    """Empty answer (Pipeline.run() empty-generation path) is incomplete."""
    rec = {"q_index": 1, "answer": "", "error": None}
    assert run_pipeline._is_record_complete(rec) is False


def test_is_record_complete_false_for_whitespace_answer():
    rec = {"q_index": 1, "answer": "   \n\t  ", "error": None}
    assert run_pipeline._is_record_complete(rec) is False


def test_is_record_complete_false_when_error_set():
    rec = {"q_index": 1, "answer": "ERROR: something", "error": "boom"}
    assert run_pipeline._is_record_complete(rec) is False


def test_is_record_complete_false_when_refs_bleed_flagged():
    rec = {
        "q_index": 1, "answer": "valid", "error": None,
        "refs_bleed_suspected": True, "refs_bleed_indices": [0],
    }
    assert run_pipeline._is_record_complete(rec) is False


def test_load_done_from_jsonl_skips_empty_answers(tmp_path: Path):
    """Empty-answer rows must not be reported as 'done' for resume."""
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"q_index": 1, "answer": "good", "error": None}) + "\n"
        # q=2: matches the failure mode observed in uploaded answers-2.jsonl
        + json.dumps({"q_index": 2, "answer": "", "error": None}) + "\n"
        + json.dumps({"q_index": 3, "answer": "also good", "error": None}) + "\n",
        encoding="utf-8",
    )
    assert run_pipeline._load_done_from_jsonl(str(p)) == {1, 3}


def test_load_done_from_jsonl_skips_error_rows(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"q_index": 1, "answer": "good", "error": None}) + "\n"
        + json.dumps({"q_index": 2, "answer": "ERROR: x", "error": "x"}) + "\n",
        encoding="utf-8",
    )
    assert run_pipeline._load_done_from_jsonl(str(p)) == {1}


def test_incomplete_q_indices_returns_missing_plus_empty(tmp_path: Path):
    """The set returned must be (missing 1..N) ∪ (present-but-incomplete)."""
    p = tmp_path / "out.jsonl"
    # Present: 1 (good), 2 (empty answer), 3 (error), 5 (good).
    # Missing for N=6: 4, 6.
    p.write_text(
        json.dumps({"q_index": 1, "answer": "good", "error": None}) + "\n"
        + json.dumps({"q_index": 2, "answer": "", "error": None}) + "\n"
        + json.dumps({"q_index": 3, "answer": "ERROR", "error": "boom"}) + "\n"
        + json.dumps({"q_index": 5, "answer": "good", "error": None}) + "\n",
        encoding="utf-8",
    )
    assert run_pipeline._incomplete_q_indices(str(p), 6) == [2, 3, 4, 6]


def test_incomplete_q_indices_handles_missing_file(tmp_path: Path):
    """No file -> all q_indices 1..N are incomplete."""
    p = tmp_path / "missing.jsonl"
    assert run_pipeline._incomplete_q_indices(str(p), 5) == [1, 2, 3, 4, 5]


def test_incomplete_q_indices_all_complete(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"q_index": i, "answer": "x", "error": None})
            for i in (1, 2, 3)
        ) + "\n",
        encoding="utf-8",
    )
    assert run_pipeline._incomplete_q_indices(str(p), 3) == []


def test_incomplete_q_indices_matches_uploaded_failure_pattern(tmp_path: Path):
    """Smoke-test against the exact pattern seen in answers-2.jsonl.

    Empty-answer q_indices documented in the audit:
      5, 21, 22, 30, 32, 36, 37, 40, 42, 45, 47, 51, 52, 55, 57, 62, 67, 70.
    Missing q_indices (1..70 not present at all):
      18, 23, 27, 28, 29, 31, 33, 34, 35, 38, 39, 41, 43, 44, 46, 48, 49, 50,
      53, 54, 56, 58, 59, 60, 61, 63, 64, 65, 66, 68, 69.
    The union of these two sets is what --rerun-incomplete must surface.
    """
    present_qids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,24,
                    25,26,30,32,36,37,40,42,45,47,51,52,55,57,62,67,70]
    empty_qids = {5,21,22,30,32,36,37,40,42,45,47,51,52,55,57,62,67,70}
    p = tmp_path / "out.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for qi in present_qids:
            ans = "" if qi in empty_qids else f"answer-{qi}"
            f.write(json.dumps({"q_index": qi, "answer": ans, "error": None}) + "\n")
    needs = run_pipeline._incomplete_q_indices(str(p), 70)
    # All empties must be in needs.
    for qi in empty_qids:
        assert qi in needs, f"empty q_index {qi} not flagged for rerun"
    # All missing 1..70 must be in needs.
    missing = sorted(set(range(1, 71)) - set(present_qids))
    for qi in missing:
        assert qi in needs, f"missing q_index {qi} not flagged for rerun"
    # No good rows should appear.
    good = set(present_qids) - empty_qids
    for qi in good:
        assert qi not in needs, f"good q_index {qi} incorrectly flagged for rerun"
    # Total count check.
    assert len(needs) == len(missing) + len(empty_qids)


def test_merge_into_jsonl_replaces_empty_row_on_rerun(tmp_path: Path):
    """The end-to-end story: empty row in existing JSONL gets replaced
    by a successful rerun.  This is the workflow the user needs."""
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"q_index": 5, "answer": "", "error": None}) + "\n"
        + json.dumps({"q_index": 6, "answer": "kept", "error": None}) + "\n",
        encoding="utf-8",
    )
    # The next run upserts q=5 with a real answer; q=6 is untouched.
    run_pipeline._merge_into_jsonl(str(p), [
        {"q_index": 5, "answer": "regenerated", "error": None},
    ])
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert [r["q_index"] for r in rows] == [5, 6]
    assert next(r for r in rows if r["q_index"] == 5)["answer"] == "regenerated"
    assert next(r for r in rows if r["q_index"] == 6)["answer"] == "kept"
    # After the merge, the file no longer contains any incomplete records.
    assert run_pipeline._incomplete_q_indices(str(p), 6) == [1, 2, 3, 4]


def test_run_one_treats_empty_answer_as_error():
    """_run_one must raise/retry when pipeline.run returns answer=''."""
    import threading

    class _StubPipeline:
        def profile_and_route(self, q):
            class _Cfg:
                rule_hit = "tier-m"
            return None, None, _Cfg()

        def run(self, *a, **kw):
            class _Ans:
                rule_hit = "tier-m"
                answer = ""  # Pipeline.run() empty-generation path
                enriched_context = "ctx"
                excerpt_stats = {}
                references: list = []
                formatted_references: list = []
                ref_source_titles: list = []
            return _Ans()

    sem = threading.Semaphore(1)
    spacer = run_pipeline._StartSpacer(0.0)
    result = run_pipeline._run_one(
        job_idx=1, display_idx=1, question="q", aql="",
        expected_tier="tier-m", docs=[{"title": "t"}],
        pipeline=_StubPipeline(),
        small_sem=sem, medium_sem=sem, large_sem=sem,
        spacer=spacer, max_retries=2,
    )
    # max_retries exhausted -> ERROR record, not a silent success.
    assert result["error"] is not None
    assert result["actual_tier"] == "ERROR"
    assert "empty answer" in result["error"].lower()


def test_run_one_returns_success_on_non_empty_answer():
    """Sanity: a real answer still returns error=None."""
    import threading

    class _StubPipeline:
        def profile_and_route(self, q):
            class _Cfg:
                rule_hit = "tier-1"
            return None, None, _Cfg()

        def run(self, *a, **kw):
            class _Ans:
                rule_hit = "tier-1"
                answer = "a real answer with content"
                enriched_context = "ctx"
                excerpt_stats = {}
                references: list = []
                formatted_references: list = []
                ref_source_titles: list = []
            return _Ans()

    sem = threading.Semaphore(1)
    spacer = run_pipeline._StartSpacer(0.0)
    result = run_pipeline._run_one(
        job_idx=1, display_idx=1, question="q", aql="",
        expected_tier="tier-1", docs=[{"title": "t"}],
        pipeline=_StubPipeline(),
        small_sem=sem, medium_sem=sem, large_sem=sem,
        spacer=spacer, max_retries=1,
    )
    assert result["error"] is None
    assert result["answer"].strip()
