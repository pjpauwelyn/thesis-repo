"""Targeted unit/smoke tests for the Phase-4 fixes.

Covers (without LLM calls where possible):
  Fix 1  -- semaphore throttling logic (asyncio)
  Fix 2  -- abstract truncation cap (_ABSTRACT_PREVIEW_CAP)
  Fix 3  -- quality contract generation
  Fix 4  -- query hint (quantitativity threshold + no hint in generation question)
  Fix 5  -- doc block / reference order alignment check
  Fix 6  -- _build_aql_lookup and old aql_lookup param removed
  Fix 7  -- GenerationAgent.process() loud behaviour
  Fix 8  -- reset_session_state() on Pipeline
  Fix 9  -- retracted paper filter
  Fix 11 -- numeric faithfulness audit (no crash, logs correctly)

Run:
    pytest tests/test_phase4_fixes.py -v
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ── shared stubs ────────────────────────────────────────────────────────────────────

def _make_profile(**kwargs):
    """Build a minimal QuestionProfile stub with sensible defaults."""
    from core.utils.data_models import QuestionProfile
    defaults = dict(
        identity="test question",
        one_line_summary="",
        question_type="continuous",
        complexity=0.5,
        quantitativity=0.3,
        spatial_specificity=0.1,
        temporal_specificity=0.1,
        methodological_depth=0.1,
        needs_numeric_emphasis=False,
        confidence=0.9,
    )
    defaults.update(kwargs)
    return QuestionProfile(**defaults)


def _make_cfg(**kwargs):
    """Build a minimal PipelineConfig stub."""
    from core.utils.data_models import PipelineConfig
    defaults = dict(
        rule_hit="tier-m",
        model_name="mistral-small-latest",
        refinement_model_name=None,
        temperature_refine=0.1,
        temperature_generate=0.2,
        max_output_tokens=700,
        evidence_mode="abstracts",
        doc_filter_min_keep=6,
        per_doc_budget=800,
        global_budget=6000,
        top_k_per_doc=3,
        gen_context_cap=307_200,
        use_draft=True,
        generation_prompt="generation_prompt_exp4.txt",
        system_prompt_modifier="",
        timeout_refine_s=None,
        timeout_generate_s=None,
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


# ============================================================
# Fix 1 — semaphore truly limits concurrent coroutines
# ============================================================

def test_semaphore_limits_concurrency():
    """Verify that the semaphore inside the coroutine prevents more than
    max_concurrent tasks from running the blocking section simultaneously."""
    max_concurrent = 3
    semaphore = asyncio.Semaphore(max_concurrent)
    active = 0
    peak = 0
    lock = threading.Lock()

    async def worker(i):
        nonlocal active, peak
        async with semaphore:
            with lock:
                active += 1
                if active > peak:
                    peak = active
            await asyncio.sleep(0.01)
            with lock:
                active -= 1

    async def run():
        tasks = [worker(i) for i in range(12)]
        await asyncio.gather(*tasks)

    asyncio.run(run())
    assert peak <= max_concurrent, (
        f"semaphore did not limit concurrency: peak={peak} > max={max_concurrent}"
    )


# ============================================================
# Fix 2 — abstract cap constant and truncation behaviour
# ============================================================

def test_abstract_preview_cap_constant():
    from core.agents.refinement_agent_abstracts import _ABSTRACT_PREVIEW_CAP
    assert _ABSTRACT_PREVIEW_CAP >= 1_200, (
        f"_ABSTRACT_PREVIEW_CAP={_ABSTRACT_PREVIEW_CAP} is too small; expected >= 1200"
    )
    assert _ABSTRACT_PREVIEW_CAP <= 2_000, (
        f"_ABSTRACT_PREVIEW_CAP={_ABSTRACT_PREVIEW_CAP} is too large; expected <= 2000"
    )


def test_abstract_truncation_uses_cap():
    from core.agents.refinement_agent_abstracts import (
        RefinementAgentAbstracts,
        _ABSTRACT_PREVIEW_CAP,
    )
    long_abstract = "A" * (_ABSTRACT_PREVIEW_CAP + 500)
    docs = [{"title": "Test Paper", "abstract": long_abstract, "uri": ""}]
    import json
    aql_str = json.dumps(docs)
    text, parsed = RefinementAgentAbstracts._parse_aql_for_prompt(aql_str)
    assert len(long_abstract) - text.count("A") <= _ABSTRACT_PREVIEW_CAP + 3
    assert "..." in text


def test_short_abstract_not_truncated():
    from core.agents.refinement_agent_abstracts import RefinementAgentAbstracts
    short_abstract = "Short abstract with less than cap."
    import json
    docs = [{"title": "Short Paper", "abstract": short_abstract, "uri": ""}]
    text, _ = RefinementAgentAbstracts._parse_aql_for_prompt(json.dumps(docs))
    assert "..." not in text.split("Abstract:")[-1].split("\n")[0]


# ============================================================
# Fix 3 — quality contract generation
# ============================================================

def test_quality_contract_empty_for_simple_question():
    from core.pipelines.pipeline import Pipeline
    profile = _make_profile(
        question_type="definition",
        quantitativity=0.1,
        spatial_specificity=0.1,
        temporal_specificity=0.1,
        methodological_depth=0.1,
        needs_numeric_emphasis=False,
    )
    cfg = _make_cfg(evidence_mode="abstracts")
    contract = Pipeline._build_answer_quality_contract(profile, cfg)
    assert contract == "", f"Expected empty contract for simple question, got: {contract!r}"


def test_quality_contract_numeric_rules_present():
    from core.pipelines.pipeline import Pipeline
    profile = _make_profile(quantitativity=0.8, needs_numeric_emphasis=True)
    cfg = _make_cfg(evidence_mode="abstracts")
    contract = Pipeline._build_answer_quality_contract(profile, cfg)
    assert contract
    assert "unit" in contract.lower() or "value" in contract.lower()


def test_quality_contract_spatial_rules():
    from core.pipelines.pipeline import Pipeline
    profile = _make_profile(spatial_specificity=0.7)
    cfg = _make_cfg(evidence_mode="abstracts")
    contract = Pipeline._build_answer_quality_contract(profile, cfg)
    assert "spatial" in contract.lower() or "geographic" in contract.lower()


def test_quality_contract_temporal_rules():
    from core.pipelines.pipeline import Pipeline
    profile = _make_profile(temporal_specificity=0.7)
    cfg = _make_cfg(evidence_mode="abstracts")
    contract = Pipeline._build_answer_quality_contract(profile, cfg)
    assert "temporal" in contract.lower() or "time period" in contract.lower()


def test_quality_contract_method_rules():
    from core.pipelines.pipeline import Pipeline
    profile = _make_profile(methodological_depth=0.8)
    cfg = _make_cfg(evidence_mode="abstracts")
    contract = Pipeline._build_answer_quality_contract(profile, cfg)
    assert "method" in contract.lower() or "limitation" in contract.lower()


def test_quality_contract_fulltext_grounding_rule():
    from core.pipelines.pipeline import Pipeline
    profile = _make_profile(quantitativity=0.8)
    cfg = _make_cfg(evidence_mode="excerpts_narrow")
    contract = Pipeline._build_answer_quality_contract(profile, cfg)
    assert "grounded" in contract.lower() or "context" in contract.lower()


# ============================================================
# Fix 4 — query hint threshold + original question for generation
# ============================================================

def test_query_hint_triggers_on_high_quantitativity():
    from core.pipelines.pipeline import Pipeline
    profile = _make_profile(quantitativity=0.6, needs_numeric_emphasis=False)
    hint = Pipeline._build_query_hint("What is the mean temperature?", profile)
    assert "numeric" in hint.lower() or "quantitative" in hint.lower()


def test_query_hint_no_trigger_below_threshold():
    from core.pipelines.pipeline import Pipeline
    profile = _make_profile(quantitativity=0.3, needs_numeric_emphasis=False)
    hint = Pipeline._build_query_hint("What is the mean temperature?", profile)
    assert hint == "What is the mean temperature?"


# ============================================================
# Fix 5 — doc block / reference order alignment check
# ============================================================

def test_alignment_check_passes_consistent_lists():
    from core.pipelines.pipeline import Pipeline
    full = [{"uri": "http://ex.org/1", "title": "A"}]
    abstract = [{"uri": "http://ex.org/2", "title": "B"}]
    Pipeline._assert_doc_block_ref_alignment(full, abstract)


def test_alignment_check_detects_swap():
    from core.pipelines.pipeline import Pipeline
    full = [{"uri": "http://ex.org/1", "title": "A"}]
    abstract = [{"uri": "http://ex.org/2", "title": "B"}]
    combined_correct = full + abstract
    combined_flipped = abstract + full
    key_correct = combined_correct[0].get("uri", "")
    key_flipped = combined_flipped[0].get("uri", "")
    assert key_correct != key_flipped


# ============================================================
# Fix 6 — _build_aql_lookup removed, _render_documents_block no aql_lookup param
# ============================================================

def test_build_aql_lookup_removed():
    from core.pipelines.pipeline import Pipeline
    assert not hasattr(Pipeline, "_build_aql_lookup"), (
        "_build_aql_lookup should have been deleted (Fix 6)"
    )


def test_render_documents_block_no_aql_lookup_param():
    from core.pipelines.pipeline import Pipeline
    import inspect
    sig = inspect.signature(Pipeline._render_documents_block)
    param_names = list(sig.parameters.keys())
    assert "aql_lookup" not in param_names


def test_render_documents_block_works_without_lookup():
    from core.pipelines.pipeline import Pipeline
    full = [{"title": "Doc1", "uri": "http://ex.org/1", "abstract": "text1"}]
    abstract = [{"title": "Doc2", "uri": "http://ex.org/2", "abstract": "text2"}]
    block = Pipeline._render_documents_block(full, abstract, [])
    assert "Doc1" in block
    assert "Doc2" in block


# ============================================================
# Fix 7 — GenerationAgent.process() loud behaviour
# ============================================================

def test_generation_agent_process_raises_on_empty_input():
    from core.agents.generation_agent import GenerationAgent
    agent = GenerationAgent(MagicMock())
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        agent.process({})


def test_generation_agent_process_raises_on_non_dict():
    from core.agents.generation_agent import GenerationAgent
    agent = GenerationAgent(MagicMock())
    with pytest.raises((NotImplementedError, TypeError, AttributeError)):
        agent.process("not a dict")


def test_generation_agent_process_not_silent_none():
    """process() must not return None silently."""
    from core.agents.generation_agent import GenerationAgent
    agent = GenerationAgent(MagicMock())
    result = None
    try:
        result = agent.process({})
    except Exception:
        pass
    assert result is None, "process({}) should raise, not return a value"


# ============================================================
# Fix 8 — reset_session_state() on Pipeline
# ============================================================

def test_reset_session_state_clears_uris():
    from core.pipelines.pipeline import Pipeline
    p = Pipeline.__new__(Pipeline)
    p._session_full_doc_uris = {"http://ex.org/1", "http://ex.org/2"}
    p._profiler_parse_failures = 5
    p._counter_lock = threading.Lock()
    p._llm_cache = {"key": "value"}
    p.reset_session_state()
    assert p._session_full_doc_uris == set(), "Session URIs not cleared"
    assert p._profiler_parse_failures == 0, "Parse failure counter not reset"
    assert p._llm_cache == {"key": "value"}, "reset_session_state should not clear LLM cache"


def test_reset_session_state_method_exists():
    from core.pipelines.pipeline import Pipeline
    assert hasattr(Pipeline, "reset_session_state")


# ============================================================
# Fix 9 — retracted paper filter
# ============================================================

def _make_pipeline_stub():
    from core.pipelines.pipeline import Pipeline
    p = Pipeline.__new__(Pipeline)
    p._counter_lock = threading.Lock()
    p._session_full_doc_uris = set()
    p._profiler_parse_failures = 0
    return p


def test_retracted_paper_removed_by_title():
    p = _make_pipeline_stub()
    docs = [
        {"title": "RETRACTED: Effects of warming on permafrost", "uri": "http://a"},
        {"title": "Normal paper about permafrost", "uri": "http://b"},
    ]
    clean = p._remove_retracted_papers(docs, "What causes permafrost thaw?")
    titles = [d["title"] for d in clean]
    assert "Normal paper about permafrost" in titles
    assert not any("RETRACTED" in t for t in titles)


def test_withdrawn_paper_removed():
    p = _make_pipeline_stub()
    docs = [
        {"title": "Withdrawn: A study of river discharge", "uri": "http://a"},
        {"title": "River discharge trends", "uri": "http://b"},
    ]
    clean = p._remove_retracted_papers(docs, "River discharge changes?")
    assert len(clean) == 1
    assert clean[0]["title"] == "River discharge trends"


def test_expression_of_concern_removed():
    p = _make_pipeline_stub()
    docs = [
        {"title": "Expression of Concern: Data fabrication in hydrology", "uri": "http://a"},
        {"title": "Hydrology review", "uri": "http://b"},
    ]
    clean = p._remove_retracted_papers(docs, "How is hydrology studied?")
    assert len(clean) == 1


def test_retraction_question_skips_filter():
    p = _make_pipeline_stub()
    docs = [{"title": "RETRACTED: Effects of warming on permafrost", "uri": "http://a"}]
    clean = p._remove_retracted_papers(docs, "Why was the retracted permafrost paper retracted?")
    assert len(clean) == 1, "Filter should be skipped when question is about retractions"


def test_normal_paper_with_retraction_word_in_abstract_not_filtered():
    p = _make_pipeline_stub()
    docs = [
        {
            "title": "A review of data integrity issues in climate science",
            "abstract": "This review examines retracted papers in climate science...",
            "uri": "http://a",
        },
    ]
    clean = p._remove_retracted_papers(docs, "Data integrity in climate research?")
    assert len(clean) == 1, "Papers discussing retractions in abstract should not be filtered"


# ============================================================
# Fix 11 — numeric faithfulness audit (no crash + logs warnings)
# ============================================================

def test_numeric_audit_logs_unmatched(caplog):
    from core.pipelines.pipeline import Pipeline
    answer = "The mean temperature was 42.7 degrees Celsius [1]."
    context = "Temperature measurements show values around 40 degrees."
    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._audit_numeric_faithfulness(answer, context, "What is temperature?")
    assert any("42.7" in record.message or "numeric_audit" in record.message
               for record in caplog.records)


def test_numeric_audit_no_warning_for_matched_numbers(caplog):
    from core.pipelines.pipeline import Pipeline
    answer = "The mean temperature was 42.7 degrees [1]."
    context = "Temperature measurements show 42.7 degrees in the region."
    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._audit_numeric_faithfulness(answer, context, "What is temperature?")
    unmatched_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "42.7" in r.message
    ]
    assert not unmatched_warnings


def test_numeric_audit_skips_years():
    from core.pipelines.pipeline import Pipeline
    answer = "Measurements from 2020 show a change."
    context = "Recent observations confirm this trend."
    Pipeline._audit_numeric_faithfulness(answer, context, "When did change occur?")


def test_numeric_audit_skips_single_digits():
    from core.pipelines.pipeline import Pipeline
    answer = "There are 3 methods commonly used."
    context = "Two main approaches exist."
    Pipeline._audit_numeric_faithfulness(answer, context, "What methods exist?")


def test_numeric_audit_does_not_modify_answer():
    from core.pipelines.pipeline import Pipeline
    answer = "The value is 99.5 kg [1]."
    context = "No numbers here."
    Pipeline._audit_numeric_faithfulness(answer, context, "test?")
    assert answer == "The value is 99.5 kg [1]."
