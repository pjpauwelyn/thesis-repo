"""Unit tests for Fixes 1-4: _clean_answer_artifacts and _warn_thin_sources.

All tests are pure unit tests (no LLM calls, no mocks, no fixtures).
They exercise Pipeline static methods directly.
"""
import logging

import pytest

from core.pipelines.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Fix 1 -- TITLE section header bleed
# ---------------------------------------------------------------------------

def test_clean_strips_title_section_header():
    raw = (
        "Glacier retreat mobilises carbon.\n"
        "TITLE Key Mechanisms Linking Cryospheric Change to Carbon Flux\n"
        "Permafrost thaw releases CO\u2082.\n"
    )
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "TITLE Key Mechanisms" not in cleaned
    assert "Glacier retreat mobilises carbon." in cleaned
    assert "Permafrost thaw releases" in cleaned


def test_clean_strips_title_references_header():
    raw = "Some claim [1].\nTITLE References\n1. Smith et al. 2020."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "TITLE References" not in cleaned
    assert "Some claim [1]." in cleaned


def test_clean_strips_multiple_title_headers():
    raw = (
        "TITLE Land Surface Heat Absorption Patterns\n"
        "Surface albedo determines absorption.\n"
        "TITLE Uncertainties and Data Gaps\n"
        "Coverage is limited.\n"
        "TITLE Implications for Earth System Models and Monitoring\n"
    )
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "TITLE Land Surface" not in cleaned
    assert "TITLE Uncertainties" not in cleaned
    assert "TITLE Implications" not in cleaned
    assert "Surface albedo determines absorption." in cleaned
    assert "Coverage is limited." in cleaned


def test_clean_does_not_strip_inline_title_word():
    """'TITLE' mid-sentence (not at line start) must not be stripped."""
    raw = "The paper's TITLE references the dataset directly."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert cleaned == raw


# ---------------------------------------------------------------------------
# Fix 2 -- OpenAlex URL bleed
# ---------------------------------------------------------------------------

def test_clean_strips_https_openalex_url_bleed():
    raw = "See httpsopenalex.orgW4308372574 for details."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "httpsopenalex.org" not in cleaned
    assert "for details." in cleaned


def test_clean_strips_http_openalex_url_bleed():
    raw = "Source: httpopenalex.orgW1234567890."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "httpopenalex.org" not in cleaned


def test_clean_preserves_well_formed_openalex_url():
    """A correctly formed https://openalex.org/... URL must be left intact."""
    raw = "See https://openalex.org/W4308372574 for the full record."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "https://openalex.org/W4308372574" in cleaned


# ---------------------------------------------------------------------------
# Fix 3 -- broken unit strings
# ---------------------------------------------------------------------------

def test_clean_fixes_wm_unit():
    raw = "Net absorption increased from 13.5Wm to 54Wm over the decade."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "13.5Wm" not in cleaned
    assert "54Wm" not in cleaned
    assert "W/m\u00b2" in cleaned


def test_clean_fixes_mday_unit():
    raw = "Melt rate was 0.5mday during summer."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "0.5mday" not in cleaned
    assert "m/day" in cleaned


def test_clean_does_not_alter_correct_prose():
    raw = "Surface temperature is 0.55\u00b0C. Albedo is 0.86. The year 2015 saw record loss."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert cleaned == raw


# ---------------------------------------------------------------------------
# Fix 4 -- thin-source logging (log-only, no filtering)
# ---------------------------------------------------------------------------

def test_thin_source_warning_emitted(caplog):
    excerpt_stats = {
        "per_doc": [
            {"work_id": "W4308372574", "title": "Summer Dynamics", "kept_tokens": 13},
            {"work_id": "W4406439341", "title": "Greening of Svalbard", "kept_tokens": 3436},
        ]
    }
    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._warn_thin_sources(excerpt_stats)
    thin = [r for r in caplog.records if "thin_source" in r.message]
    assert len(thin) == 1
    assert "W4308372574" in thin[0].message
    assert "13" in thin[0].message


def test_thin_source_no_warning_above_threshold(caplog):
    excerpt_stats = {
        "per_doc": [
            {"work_id": "W1111111111", "title": "Normal Doc", "kept_tokens": 3436},
            {"work_id": "W2222222222", "title": "Another Doc", "kept_tokens": 2800},
        ]
    }
    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._warn_thin_sources(excerpt_stats)
    assert not any("thin_source" in r.message for r in caplog.records)


def test_thin_source_boundary_at_threshold(caplog):
    """kept_tokens == 49 triggers warning; kept_tokens == 50 does not."""
    below = {"per_doc": [{"work_id": "W1", "title": "T", "kept_tokens": 49}]}
    at = {"per_doc": [{"work_id": "W2", "title": "T", "kept_tokens": 50}]}

    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._warn_thin_sources(below)
    assert any("thin_source" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._warn_thin_sources(at)
    assert not any("thin_source" in r.message for r in caplog.records)


def test_thin_source_empty_input_safe(caplog):
    """_warn_thin_sources must not raise on missing or empty per_doc."""
    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._warn_thin_sources({})
        Pipeline._warn_thin_sources({"per_doc": []})
    assert not any("thin_source" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Fix E -- non-citation <<...>> placeholder leak (Q7-style "<<CAVEATS & GAPS>>")
# ---------------------------------------------------------------------------

def test_clean_strips_caveats_and_gaps_sentinel():
    """Q7 regression: '<<CAVEATS & GAPS>>' leaked into a finished answer body."""
    raw = (
        "with limited data for the Southern Hemisphere, oceanic plates, "
        "and polar regions <<CAVEATS & GAPS>>."
    )
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "<<CAVEATS" not in cleaned
    assert "GAPS>>" not in cleaned
    assert "limited data for the Southern Hemisphere" in cleaned
    assert "polar regions." in cleaned


def test_clean_strips_ontology_summary_sentinel():
    raw = "Intro paragraph. <<ONTOLOGY SUMMARY>> Body continues."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "<<ONTOLOGY" not in cleaned
    assert "Intro paragraph." in cleaned
    assert "Body continues." in cleaned


def test_clean_strips_topics_and_information_sentinel():
    raw = "Lead-in. <<TOPICS AND INFORMATION>> Then the rest."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "<<TOPICS" not in cleaned
    assert "Lead-in." in cleaned
    assert "Then the rest." in cleaned


def test_clean_strips_single_closing_angle_variant():
    """The LLM sometimes emits '<<X>' with only one closing > -- strip both forms."""
    raw = "Trailing artifact <<CAVEATS & GAPS> here."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "<<CAVEATS" not in cleaned
    assert "Trailing artifact here." in cleaned


def test_clean_preserves_already_converted_citation_brackets():
    """Citation sentinels are converted to [N] BEFORE _clean_answer_artifacts.
    The new Fix E must not damage [N] markers."""
    raw = "Body of claim [1] and another claim [42] with refs."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "[1]" in cleaned
    assert "[42]" in cleaned
    assert cleaned == raw


def test_clean_does_not_strip_single_angle_quotation():
    """A lone '<' (e.g. inequality, French quotation) must survive."""
    raw = "Concentrations < 5 ppm were excluded. Range was 2 < x < 9."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert cleaned == raw


def test_clean_does_not_swallow_body_text_on_unclosed_marker():
    """A pathological '<<' with no closing >> within 80 chars must not eat body text."""
    long_body = "a" * 200
    raw = f"<<{long_body} and then more"
    cleaned = Pipeline._clean_answer_artifacts(raw)
    # The unclosed marker is left intact rather than swallowing the body.
    assert long_body in cleaned


def test_clean_strips_caveats_with_punctuation_spacing():
    """After stripping the marker, leading whitespace before punctuation is tidied."""
    raw = "polar regions <<CAVEATS & GAPS>> ."
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "<<" not in cleaned
    assert ">>" not in cleaned
    assert "polar regions." in cleaned


def test_clean_strips_multiple_distinct_sentinels_in_one_body():
    raw = (
        "Para one <<ONTOLOGY SUMMARY>>. Para two has data "
        "<<TOPICS AND INFORMATION>>. Para three notes gaps <<CAVEATS & GAPS>>."
    )
    cleaned = Pipeline._clean_answer_artifacts(raw)
    assert "<<" not in cleaned
    assert ">>" not in cleaned
    assert "Para one." in cleaned
    assert "Para two has data." in cleaned
    assert "Para three notes gaps." in cleaned
