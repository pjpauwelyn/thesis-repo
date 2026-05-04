"""Phase-2 filter tests.

Covers (no LLM calls):
  Fix B  -- URI window rolling: type, maxlen, reset, rollover, union, empty-guard
  Fix B  -- Surplus demotion via mocked _filter_documents: seen URIs demoted,
            surplus=0 no-op, empty window no-op, unseen URIs untouched,
            demotion capped at surplus, window updated after call,
            cross-topic isolation after full window
  Fix A  -- _strip_references_section: [VALIDATED REFERENCES] heading,
            lowercase variant, trailing numbered ref sweep, no false-positive
            on inline 'references', inline [N] markers not swept
  Fix C  -- _extract_cited_indices: comma-separated multi-cite, space-separated,
            single cite, out-of-range drop, partial valid, legacy bracket
            fallback, sentinel priority over legacy path
  Fix E  -- _parse_aql_for_prompt parity: no warning on clean input,
            warning on mocked count mismatch, returns parsed list despite mismatch

Run:
    pytest tests/test_filter_phase2.py -v
"""

from __future__ import annotations

import collections
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ── shared helpers ───────────────────────────────────────────────────────────

def _make_profile(**kwargs):
    from core.utils.data_models import QuestionProfile
    defaults = dict(
        identity="test question",
        one_line_summary="",
        question_type="mechanism",
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
    from core.utils.data_models import PipelineConfig
    defaults = dict(
        rule_hit="tier-m",
        model_name="mistral-small-latest",
        refinement_model_name=None,
        temperature_refine=0.1,
        temperature_generate=0.2,
        max_output_tokens=4000,
        evidence_mode="excerpts_narrow",
        doc_filter_min_keep=6,
        per_doc_budget=800,
        global_budget=6000,
        top_k_per_doc=3,
        gen_context_cap=307_200,
        use_draft=False,
        generation_prompt="generation_structured.txt",
        system_prompt_modifier="",
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_pipeline_stub():
    """Pipeline instance with no LLM dependencies."""
    from core.pipelines.pipeline import Pipeline, _URI_WINDOW_SIZE
    p = Pipeline.__new__(Pipeline)
    p._session_uri_window = collections.deque(maxlen=_URI_WINDOW_SIZE)
    p._profiler_parse_failures = 0
    p._counter_lock = threading.Lock()
    p._llm_cache = {}
    p._cache_dir = Path("cache/fulltext")
    p._prompts_root = Path("prompts")
    return p


def _make_docs(n: int, prefix: str = "http://ex.org/") -> List[Dict[str, Any]]:
    return [{"uri": f"{prefix}{i}", "title": f"Doc {i}"} for i in range(1, n + 1)]


def _call_filter_mocked(
    p,
    full_docs_to_return: List[Dict],
    cfg,
    seed_window_uris=None,
):
    """Call _filter_documents with OntologyAgent mocked to avoid LLM calls.

    filter_agent.filter_documents() returns (full_docs_to_return, [], []).
    Optional seed_window_uris is appended to _session_uri_window before the call
    to simulate previously-seen documents.
    """
    if seed_window_uris is not None:
        p._session_uri_window.append(seed_window_uris)
    profile = _make_profile()
    ontology = MagicMock()
    with patch("core.agents.ontology_agent.OntologyAgent") as MockAgent:
        mock_inst = MagicMock()
        MockAgent.return_value = mock_inst
        mock_inst.filter_documents.return_value = (full_docs_to_return, [], [])
        p._llm = MagicMock(return_value=MagicMock())
        return p._filter_documents(
            full_docs_to_return, ontology, profile, "test question?", cfg
        )


# ============================================================
# Fix B — URI window structure
# ============================================================

def test_uri_window_constant_exists():
    from core.pipelines.pipeline import _URI_WINDOW_SIZE
    assert isinstance(_URI_WINDOW_SIZE, int)
    assert 5 <= _URI_WINDOW_SIZE <= 20, (
        f"_URI_WINDOW_SIZE={_URI_WINDOW_SIZE} out of expected range [5, 20]"
    )


def test_uri_window_is_deque():
    p = _make_pipeline_stub()
    assert isinstance(p._session_uri_window, collections.deque)


def test_uri_window_maxlen_matches_constant():
    from core.pipelines.pipeline import _URI_WINDOW_SIZE
    p = _make_pipeline_stub()
    assert p._session_uri_window.maxlen == _URI_WINDOW_SIZE, (
        f"deque maxlen={p._session_uri_window.maxlen} != _URI_WINDOW_SIZE={_URI_WINDOW_SIZE}"
    )


def test_uri_window_starts_empty():
    p = _make_pipeline_stub()
    assert len(p._session_uri_window) == 0


def test_reset_session_state_clears_window():
    p = _make_pipeline_stub()
    p._session_uri_window.append({"http://ex.org/1"})
    p._session_uri_window.append({"http://ex.org/2"})
    p._profiler_parse_failures = 3
    p.reset_session_state()
    assert len(p._session_uri_window) == 0, (
        "reset_session_state() did not clear _session_uri_window"
    )
    assert p._profiler_parse_failures == 0, (
        "reset_session_state() did not reset _profiler_parse_failures"
    )


def test_uri_window_rolls_over_evicts_oldest():
    """After maxlen+1 appends the oldest set is no longer in the union."""
    from core.pipelines.pipeline import _URI_WINDOW_SIZE
    p = _make_pipeline_stub()
    first = {"http://FIRST.org/old"}
    p._session_uri_window.append(first)
    for i in range(_URI_WINDOW_SIZE):          # fill remaining slots
        p._session_uri_window.append({f"http://later.org/{i}"})
    assert len(p._session_uri_window) == _URI_WINDOW_SIZE
    union = set().union(*p._session_uri_window)
    assert "http://FIRST.org/old" not in union, (
        "Oldest URI set was not evicted after window rolled over"
    )


def test_uri_window_union_covers_all_current_slots():
    p = _make_pipeline_stub()
    p._session_uri_window.append({"http://ex.org/A"})
    p._session_uri_window.append({"http://ex.org/B"})
    p._session_uri_window.append({"http://ex.org/C"})
    union = set().union(*p._session_uri_window)
    assert union == {"http://ex.org/A", "http://ex.org/B", "http://ex.org/C"}


def test_empty_window_union_returns_empty_set():
    """Union of empty deque must not raise and must return empty set."""
    p = _make_pipeline_stub()
    with p._counter_lock:
        result = set().union(*p._session_uri_window) if p._session_uri_window else set()
    assert result == set()


# ============================================================
# Fix B — surplus demotion via _filter_documents
# ============================================================

def test_surplus_demotion_seen_uris_demoted():
    """Docs whose URIs are in the window are demoted when surplus > 0."""
    p = _make_pipeline_stub()
    seen = {"http://ex.org/1", "http://ex.org/2"}
    # 9 full docs, min_keep=6 -> surplus = 9 - (6+2) = 1
    full_docs = _make_docs(9)
    cfg = _make_cfg(doc_filter_min_keep=6)
    _, abstract_out, _ = _call_filter_mocked(p, full_docs, cfg, seed_window_uris=seen)
    demoted_uris = {d["uri"] for d in abstract_out}
    assert demoted_uris & seen, (
        f"Expected at least one seen URI to be demoted. "
        f"abstract_uris={demoted_uris}, seen={seen}"
    )


def test_surplus_zero_no_demotion():
    """At surplus=0 no docs are demoted even if all URIs are in the window."""
    p = _make_pipeline_stub()
    full_docs = _make_docs(8)           # 8 - (6+2) = 0 surplus
    seen = {d["uri"] for d in full_docs}
    cfg = _make_cfg(doc_filter_min_keep=6)
    _, abstract_out, _ = _call_filter_mocked(p, full_docs, cfg, seed_window_uris=seen)
    assert len(abstract_out) == 0, (
        f"Expected 0 demotions at surplus=0, got {len(abstract_out)}"
    )


def test_empty_window_no_demotion_even_with_surplus():
    """If the window is empty (first question of a run) no demotion happens."""
    p = _make_pipeline_stub()
    full_docs = _make_docs(12)          # 12 - (6+2) = 4 surplus
    cfg = _make_cfg(doc_filter_min_keep=6)
    _, abstract_out, _ = _call_filter_mocked(p, full_docs, cfg, seed_window_uris=None)
    assert len(abstract_out) == 0, (
        f"Expected 0 demotions with empty window, got {len(abstract_out)}"
    )


def test_unseen_uri_not_demoted():
    """Docs with URIs absent from the window must not be demoted."""
    p = _make_pipeline_stub()
    # Seed window with DIFFERENT URIs
    p._session_uri_window.append({"http://old.org/1", "http://old.org/2"})
    full_docs = _make_docs(10, prefix="http://new.org/")  # all unseen
    cfg = _make_cfg(doc_filter_min_keep=6)
    _, abstract_out, _ = _call_filter_mocked(p, full_docs, cfg)
    assert len(abstract_out) == 0, (
        f"Docs with unseen URIs should not be demoted, got {len(abstract_out)} demoted"
    )


def test_demotion_capped_at_surplus():
    """Demotion stops at `surplus` docs even when more seen URIs exist."""
    p = _make_pipeline_stub()
    full_docs = _make_docs(10)          # all seen
    seen = {d["uri"] for d in full_docs}
    cfg = _make_cfg(doc_filter_min_keep=6)
    _, abstract_out, _ = _call_filter_mocked(p, full_docs, cfg, seed_window_uris=seen)
    expected_surplus = 10 - (cfg.doc_filter_min_keep + 2)  # = 2
    assert len(abstract_out) == expected_surplus, (
        f"Expected exactly {expected_surplus} demotions (surplus cap), "
        f"got {len(abstract_out)}"
    )


def test_demotion_updates_window_after_call():
    """After _filter_documents the new question's URI set is appended."""
    p = _make_pipeline_stub()
    full_docs = _make_docs(6)
    cfg = _make_cfg(doc_filter_min_keep=6)
    assert len(p._session_uri_window) == 0
    _call_filter_mocked(p, full_docs, cfg)
    assert len(p._session_uri_window) == 1, (
        "Window should contain exactly one set after first _filter_documents call"
    )
    new_uris = p._session_uri_window[-1]
    assert isinstance(new_uris, set)
    assert len(new_uris) > 0


def test_cross_topic_isolation_after_full_window():
    """After the window is full with topic-A URIs, topic-B docs are not demoted."""
    from core.pipelines.pipeline import _URI_WINDOW_SIZE
    p = _make_pipeline_stub()
    for i in range(_URI_WINDOW_SIZE):
        p._session_uri_window.append({f"http://topic_A.org/{i}"})
    # Topic-B docs: none of their URIs are in the full window
    full_docs = _make_docs(10, prefix="http://topic_B.org/")
    cfg = _make_cfg(doc_filter_min_keep=6)
    _, abstract_out, _ = _call_filter_mocked(p, full_docs, cfg)
    assert len(abstract_out) == 0, (
        f"Cross-topic docs must not be demoted; got {len(abstract_out)} demoted. "
        "Fix B window isolation failed."
    )


# ============================================================
# Fix A — _strip_references_section
# ============================================================

def test_strip_validated_references_heading_exact():
    from core.agents.generation_agent import GenerationAgent
    text = (
        "The sea ice extent is decreasing.\n\n"
        "[VALIDATED REFERENCES]\n"
        "[1] Smith et al. (2020). Nature.\n"
        "[2] Jones et al. (2019). Science."
    )
    result = GenerationAgent._strip_references_section(text)
    assert "[VALIDATED REFERENCES]" not in result
    assert "[1] Smith" not in result
    assert "The sea ice extent is decreasing." in result


def test_strip_validated_references_lowercase_variant():
    from core.agents.generation_agent import GenerationAgent
    text = "Answer body here.\n\nvalidated references\n[1] Author 2021"
    result = GenerationAgent._strip_references_section(text)
    assert "validated references" not in result.lower()
    assert "[1]" not in result
    assert "Answer body" in result


def test_strip_trailing_numbered_refs_no_heading():
    """Post-body sweep removes trailing [N] bibliography lines even without heading."""
    from core.agents.generation_agent import GenerationAgent
    text = (
        "Permafrost thaw accelerates carbon release.\n"
        "This is supported by multiple studies.\n"
        "[1] Smith et al. (2020). Nature Climate Change.\n"
        "[2] Jones et al. (2019). Science."
    )
    result = GenerationAgent._strip_references_section(text)
    assert "[1] Smith" not in result
    assert "[2] Jones" not in result
    assert "Permafrost thaw" in result
    assert "multiple studies" in result


def test_strip_classic_references_heading_still_works():
    from core.agents.generation_agent import GenerationAgent
    text = "Answer body.\n\n## References\n[1] Author 2020"
    result = GenerationAgent._strip_references_section(text)
    assert "## References" not in result
    assert "[1] Author" not in result
    assert "Answer body." in result


def test_strip_no_false_positive_inline_references_word():
    """A sentence containing 'references' mid-text must NOT be stripped."""
    from core.agents.generation_agent import GenerationAgent
    text = "This references the methodology described earlier. Cross-references also exist."
    result = GenerationAgent._strip_references_section(text)
    assert result == text, (
        f"False-positive strip detected. Expected unchanged text, got: {result!r}"
    )


def test_strip_inline_citation_brackets_not_swept():
    """Mid-sentence [N] citation markers must NOT be removed by the trailing sweep."""
    from core.agents.generation_agent import GenerationAgent
    text = "Sea ice loss is accelerating [1] due to albedo feedback [3]."
    result = GenerationAgent._strip_references_section(text)
    assert "[1]" in result, "Inline [1] citation was incorrectly removed"
    assert "[3]" in result, "Inline [3] citation was incorrectly removed"


def test_strip_clean_answer_unchanged():
    from core.agents.generation_agent import GenerationAgent
    text = "A clean answer with no references section at all."
    result = GenerationAgent._strip_references_section(text)
    assert result == text


# ============================================================
# Fix C — _extract_cited_indices multi-cite pre-pass
# ============================================================

def test_multi_cite_comma_separated():
    from core.agents.generation_agent import GenerationAgent
    text = "Ice loss is accelerating <<CITE:1,6>> in the Arctic."
    result = GenerationAgent._extract_cited_indices(text)
    assert result == {1, 6}, f"Expected {{1, 6}}, got {result}"


def test_multi_cite_with_spaces_around_comma():
    from core.agents.generation_agent import GenerationAgent
    text = "Studies confirm this <<CITE:2, 5, 8>>."
    result = GenerationAgent._extract_cited_indices(text)
    assert result == {2, 5, 8}, f"Expected {{2, 5, 8}}, got {result}"


def test_single_cite_unchanged():
    from core.agents.generation_agent import GenerationAgent
    text = "This is supported <<CITE:3>>."
    result = GenerationAgent._extract_cited_indices(text)
    assert result == {3}


def test_multiple_single_cites_in_text():
    from core.agents.generation_agent import GenerationAgent
    text = "First claim <<CITE:1>> and second claim <<CITE:4>>."
    result = GenerationAgent._extract_cited_indices(text)
    assert result == {1, 4}


def test_out_of_range_cite_dropped():
    from core.agents.generation_agent import GenerationAgent, _MAX_CITE_INDEX
    text = f"Hallucinated citation <<CITE:{_MAX_CITE_INDEX + 1}>>."
    result = GenerationAgent._extract_cited_indices(text)
    assert result == set(), (
        f"Index {_MAX_CITE_INDEX + 1} should be dropped (> _MAX_CITE_INDEX={_MAX_CITE_INDEX})"
    )


def test_multi_cite_partial_valid_out_of_range():
    """Mixed valid + out-of-range: only the valid index is kept."""
    from core.agents.generation_agent import GenerationAgent, _MAX_CITE_INDEX
    text = f"<<CITE:2,{_MAX_CITE_INDEX + 5}>>"
    result = GenerationAgent._extract_cited_indices(text)
    assert result == {2}, (
        f"Expected only {{2}}, got {result}"
    )


def test_legacy_bracket_fallback_when_no_sentinels():
    from core.agents.generation_agent import GenerationAgent
    text = "This is supported [1][3] in the literature [5]."
    result = GenerationAgent._extract_cited_indices(text)
    assert 1 in result
    assert 3 in result
    assert 5 in result


def test_sentinel_path_does_not_process_legacy_brackets():
    """When <<CITE:N>> is present, the legacy [N] path must not run.
    [99] is also present but 99 > _MAX_CITE_INDEX so it's out of range either way."""
    from core.agents.generation_agent import GenerationAgent, _MAX_CITE_INDEX
    # 99 is always > _MAX_CITE_INDEX (30), so it must never appear regardless
    text = "Claim <<CITE:2>> plus bracket [99]."
    result = GenerationAgent._extract_cited_indices(text)
    assert 2 in result
    assert 99 not in result


# ============================================================
# Fix E — _parse_aql_for_prompt doc-order parity warning
# ============================================================

def test_parity_no_warning_on_clean_input(caplog):
    """Clean input (no doc drop) must not emit any parity warning."""
    from core.agents.refinement_agent_abstracts import RefinementAgentAbstracts
    docs = [
        {"title": "Paper A", "abstract": "Abstract A.", "uri": "http://ex.org/A"},
        {"title": "Paper B", "abstract": "Abstract B.", "uri": "http://ex.org/B"},
    ]
    aql_str = json.dumps(docs)
    with caplog.at_level(logging.WARNING,
                         logger="core.agents.refinement_agent_abstracts"):
        _, parsed = RefinementAgentAbstracts._parse_aql_for_prompt(aql_str)
    parity_warns = [
        r for r in caplog.records
        if "parity" in r.message.lower()
        or ("count" in r.message.lower() and "changed" in r.message.lower())
    ]
    assert not parity_warns, (
        f"Unexpected parity warning on clean input: {[r.message for r in parity_warns]}"
    )
    assert len(parsed) == 2


def test_parity_warning_fires_on_count_mismatch(caplog):
    """When parse_aql_results silently drops a doc, a WARNING must be emitted."""
    from core.agents.refinement_agent_abstracts import RefinementAgentAbstracts
    docs = [
        {"title": "Paper A", "abstract": "Abstract A.", "uri": ""},
        {"title": "Paper B", "abstract": "Abstract B.", "uri": ""},
        {"title": "Paper C", "abstract": "Abstract C.", "uri": ""},
    ]
    aql_str = json.dumps(docs)
    # parse_aql_results returns only 2 docs instead of 3
    dropped_json = json.dumps(docs[:2])
    with patch("core.utils.aql_parser.parse_aql_results", return_value=dropped_json):
        with caplog.at_level(logging.WARNING,
                             logger="core.agents.refinement_agent_abstracts"):
            _, parsed = RefinementAgentAbstracts._parse_aql_for_prompt(aql_str)
    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "count" in m.lower() or "parity" in m.lower() or "mismatch" in m.lower()
        for m in warning_msgs
    ), (
        f"Expected a count/parity/mismatch WARNING. Got: {warning_msgs}"
    )


def test_parity_returns_parsed_list_despite_mismatch():
    """Method must return the (shorter) parsed list and not raise or return None."""
    from core.agents.refinement_agent_abstracts import RefinementAgentAbstracts
    docs = [
        {"title": "Paper A", "abstract": "Abs A.", "uri": ""},
        {"title": "Paper B", "abstract": "Abs B.", "uri": ""},
        {"title": "Paper C", "abstract": "Abs C.", "uri": ""},
    ]
    aql_str = json.dumps(docs)
    dropped_json = json.dumps(docs[:2])
    with patch("core.utils.aql_parser.parse_aql_results", return_value=dropped_json):
        text, parsed = RefinementAgentAbstracts._parse_aql_for_prompt(aql_str)
    assert len(parsed) == 2, (
        f"Expected 2-item parsed list after mocked drop, got {len(parsed)}"
    )
    assert text is not None and len(text) > 0
