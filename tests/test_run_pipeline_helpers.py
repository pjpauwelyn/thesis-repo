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
