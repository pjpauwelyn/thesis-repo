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
        "perdoc": [
            {"workid": "W4308372574", "title": "Summer Dynamics", "kepttokens": 13},
            {"workid": "W4406439341", "title": "Greening of Svalbard", "kepttokens": 3436},
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
        "perdoc": [
            {"workid": "W1111111111", "title": "Normal Doc", "kepttokens": 3436},
            {"workid": "W2222222222", "title": "Another Doc", "kepttokens": 2800},
        ]
    }
    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._warn_thin_sources(excerpt_stats)
    assert not any("thin_source" in r.message for r in caplog.records)


def test_thin_source_boundary_at_threshold(caplog):
    """kepttokens == 49 triggers warning; kepttokens == 50 does not."""
    below = {"perdoc": [{"workid": "W1", "title": "T", "kepttokens": 49}]}
    at = {"perdoc": [{"workid": "W2", "title": "T", "kepttokens": 50}]}

    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._warn_thin_sources(below)
    assert any("thin_source" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._warn_thin_sources(at)
    assert not any("thin_source" in r.message for r in caplog.records)


def test_thin_source_empty_input_safe(caplog):
    """_warn_thin_sources must not raise on missing or empty perdoc."""
    with caplog.at_level(logging.WARNING, logger="core.pipelines.pipeline"):
        Pipeline._warn_thin_sources({})
        Pipeline._warn_thin_sources({"perdoc": []})
    assert not any("thin_source" in r.message for r in caplog.records)
