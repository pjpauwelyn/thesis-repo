"""tests for OntologyConstructionAgent.process_with_profile().

we don't call mistral here -- we monkey-patch self.llm.invoke() to return
hand-crafted json fixtures, then verify:
  - the QuestionProfile is parsed correctly
  - the router's decision matches the fixture's expected rule_hit
  - parse failures yield confidence=None which lands on safety-tier3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from core.agents.ontology_agent import OntologyConstructionAgent
from core.policy.router import Router
from core.utils.data_models import QuestionProfile

_FIXTURE = Path(__file__).parent / "fixtures" / "handcrafted_profiles.json"


@pytest.fixture(scope="module")
def fixtures():
    return json.loads(_FIXTURE.read_text())


@pytest.fixture(scope="module")
def router():
    return Router()


def _agent_with_response(payload: Dict[str, Any]) -> OntologyConstructionAgent:
    llm = MagicMock()
    llm.invoke.return_value = payload
    agent = OntologyConstructionAgent(llm, prompt_dir="prompts/ontology")
    return agent


def test_fixture_file_loads(fixtures):
    assert isinstance(fixtures, list) and len(fixtures) >= 6


@pytest.mark.parametrize("idx", range(6))
def test_profile_parses_and_routes(fixtures, router, idx):
    case = fixtures[idx]
    agent = _agent_with_response(case["raw_response"])
    ontology, profile = agent.process_with_profile(case["question"])

    assert isinstance(profile, QuestionProfile)
    assert profile.identity == case["question"]

    if "expected_question_type" in case:
        assert profile.question_type == case["expected_question_type"]
    if "expected_complexity_min" in case:
        assert profile.complexity >= case["expected_complexity_min"]
    if "expected_complexity_max" in case:
        assert profile.complexity <= case["expected_complexity_max"]
    if "expected_quantitativity_min" in case:
        assert profile.quantitativity >= case["expected_quantitativity_min"]
    if "expected_quantitativity_max" in case:
        assert profile.quantitativity <= case["expected_quantitativity_max"]

    cfg = router.select(profile)
    assert cfg.rule_hit == case["expected_rule_hit"], (
        f"q={case['question']!r}: profile={profile.model_dump()} -> {cfg.rule_hit}"
    )


def test_invalid_response_falls_back_to_safety_tier3(router):
    """non-dict llm response (e.g. parse failure) -> confidence=None -> safety-tier3."""
    llm = MagicMock()
    llm.invoke.return_value = "not a json object"
    agent = OntologyConstructionAgent(llm, prompt_dir="prompts/ontology")
    # process() is also mocked by the same llm; make it return a benign None
    # so the fallback inside process_with_profile produces a default ontology.
    llm.invoke.return_value = None

    _, profile = agent.process_with_profile("garbled question")
    assert profile.confidence is None
    cfg = router.select(profile)
    assert cfg.rule_hit == "safety-tier3"
    assert cfg.model_name == "mistral-large-latest"


def test_confidence_below_floor_triggers_safety(router):
    """profiler returns a valid response but with low confidence."""
    payload = {
        "pairs": [],
        "profile": {
            "one_line_summary": "ambiguous",
            "question_type": "continuous",
            "complexity": 0.5,
            "quantitativity": 0.3,
            "spatial_specificity": 0.1,
            "temporal_specificity": 0.1,
            "methodological_depth": 0.1,
            "confidence": 0.3,
        },
    }
    agent = _agent_with_response(payload)
    _, profile = agent.process_with_profile("foo")
    assert profile.confidence == 0.3
    assert router.select(profile).rule_hit == "safety-tier3"


def test_profile_json_round_trip():
    """profile.model_dump_json() must be valid round-trip-able json (csv column)."""
    p = QuestionProfile(
        identity="q",
        question_type="quantitative",
        complexity=0.7,
        quantitativity=0.8,
        spatial_specificity=0.6,
        temporal_specificity=0.4,
        methodological_depth=0.3,
        confidence=0.9,
    )
    s = p.model_dump_json()
    d = json.loads(s)
    assert d["question_type"] == "quantitative"
    assert d["complexity"] == 0.7
    assert d["confidence"] == 0.9
