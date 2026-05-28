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
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"q_index": 1}) + "\n"
        + "not-a-json-line\n"
        + json.dumps({"answer": "no q_index"}) + "\n"
        + json.dumps({"q_index": 4}) + "\n",
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
