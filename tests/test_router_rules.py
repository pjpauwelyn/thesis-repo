"""router unit tests -- 24 parameterised cases covering every tier and the
safety net. order matters in rules.yaml: tier-2b appears before tier-3, so a
high-quant focal question hits tier-2b even when complexity + question_type
would also satisfy tier-3 (cases 15 and 16). this is intentional -- a
quantitative focal question is better served by small + narrow than
large + full, and we want the cheaper route to win.
"""

from __future__ import annotations

import pytest

from core.policy.router import Router
from core.utils.data_models import QuestionProfile


@pytest.fixture(scope="module")
def router() -> Router:
    return Router()


def _profile(**overrides) -> QuestionProfile:
    """build a profile with sensible defaults; only set what's under test."""
    base = dict(
        identity="q",
        one_line_summary="",
        question_type="continuous",
        complexity=0.2,
        quantitativity=0.2,
        spatial_specificity=0.0,
        temporal_specificity=0.0,
        methodological_depth=0.0,
        confidence=0.85,
    )
    base.update(overrides)
    return QuestionProfile(**base)


CASES = [
    # id, profile kwargs, expected rule_hit
    ("01", dict(complexity=0.2, quantitativity=0.2),                                            "tier-1"),
    ("02", dict(complexity=0.3, quantitativity=0.3),                                            "tier-1"),
    ("03", dict(complexity=0.34, quantitativity=0.34),                                          "tier-1"),
    ("04", dict(complexity=0.36, quantitativity=0.2),                                           "tier-m"),
    ("05", dict(complexity=0.5, quantitativity=0.3),                                            "tier-m"),
    ("06", dict(complexity=0.59, quantitativity=0.3),                                           "tier-m"),
    ("07", dict(quantitativity=0.6, spatial_specificity=0.3, temporal_specificity=0.3),         "tier-2a"),
    ("08", dict(quantitativity=0.55, spatial_specificity=0.3),                                  "tier-2a"),
    ("09", dict(quantitativity=0.6, spatial_specificity=0.6),                                   "tier-2b"),
    ("10", dict(quantitativity=0.6, temporal_specificity=0.6),                                  "tier-2b"),
    ("11", dict(complexity=0.6, question_type="mechanism"),                                     "tier-3"),
    ("12", dict(complexity=0.7, question_type="method_eval"),                                   "tier-3"),
    ("13", dict(complexity=0.65, question_type="comparison"),                                   "tier-3"),
    ("14", dict(complexity=0.6, question_type="definition"),                                    "tier-m"),
    # ordering check: tier-2b appears before tier-3 in rules.yaml -> wins
    ("15", dict(complexity=0.6, question_type="quantitative",
                quantitativity=0.6, spatial_specificity=0.6),                                   "tier-2b"),
    ("16", dict(complexity=0.8, question_type="mechanism",
                quantitativity=0.8, spatial_specificity=0.7),                                   "tier-2b"),
    ("17", dict(confidence=None),                                                               "safety-tier3"),
    ("18", dict(confidence=0.4),                                                                "safety-tier3"),
    ("19", dict(confidence=0.59),                                                               "safety-tier3"),
    ("20", dict(confidence=0.6, complexity=0.2, quantitativity=0.2),                            "tier-1"),
    ("21", dict(complexity=0.4, quantitativity=0.3),                                            "tier-m"),
    ("22", dict(complexity=0.1, quantitativity=0.1),                                            "tier-1"),
    ("23", dict(complexity=0.0, quantitativity=0.0),                                            "tier-1"),
    ("24", dict(quantitativity=0.9, spatial_specificity=0.0, temporal_specificity=0.0),         "tier-2a"),
]


@pytest.mark.parametrize("case_id,profile_kwargs,expected", CASES, ids=[c[0] for c in CASES])
def test_router_routes_correctly(router, case_id, profile_kwargs, expected):
    cfg = router.select(_profile(**profile_kwargs))
    assert cfg.rule_hit == expected, (
        f"case {case_id}: profile={profile_kwargs} -> got {cfg.rule_hit}, expected {expected}"
    )


def test_safety_tier3_uses_large_model(router):
    cfg = router.select(_profile(confidence=None))
    assert cfg.model_name == "mistral-large-latest"
    assert cfg.evidence_mode == "excerpts_full"


def test_fallback_matches_set2(router):
    """unmatched profile lands on the set2-equivalent fallback."""
    # something that satisfies no rule: high complexity, type=application, no quant
    cfg = router.select(_profile(complexity=0.8, quantitativity=0.0,
                                 question_type="application"))
    # neither tier-3 (wrong type) nor tier-2 (low quant) fires -> fallback
    assert cfg.rule_hit in ("fallback",)
    assert cfg.evidence_mode == "abstracts"
    assert cfg.generation_prompt == "generation_prompt_exp4.txt"
