"""router unit tests -- 26 parameterised cases covering every tier and the
safety net.

Key ordering notes (rules.yaml evaluated top-to-bottom, first match wins):
  - tier-1-def fires before tier-m: definitional + complexity < 0.60 + quant < 0.45
  - tier-2b fires before tier-3: high-quant focal beats complex-type
  - tier-m range: complexity in [0.40, 0.60)
  - tier-1 ceiling: complexity < 0.40
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

    # --- tier-1: complexity < 0.40 ---
    ("01", dict(complexity=0.2,  quantitativity=0.2),                                           "tier-1"),
    ("02", dict(complexity=0.3,  quantitativity=0.3),                                           "tier-1"),
    ("03", dict(complexity=0.34, quantitativity=0.34),                                          "tier-1"),
    # 0.36 < 0.40 -> tier-1 (tier-m requires >= 0.40)
    ("04", dict(complexity=0.36, quantitativity=0.2),                                           "tier-1"),

    # --- tier-m: complexity in [0.40, 0.60) ---
    # 0.40 is the lower bound (inclusive)
    ("21", dict(complexity=0.4,  quantitativity=0.3),                                           "tier-m"),
    ("05", dict(complexity=0.5,  quantitativity=0.3),                                           "tier-m"),
    ("06", dict(complexity=0.59, quantitativity=0.3),                                           "tier-m"),

    # --- tier-1-def: definition + complexity < 0.60 + quant < 0.45 ---
    # complexity=0.5 -> tier-1-def fires before tier-m
    ("25", dict(complexity=0.5,  question_type="definition", quantitativity=0.2),               "tier-1-def"),
    # complexity=0.6 is NOT < 0.60 -> tier-1-def misses; not mechanism/comparison -> fallback
    ("14", dict(complexity=0.6,  question_type="definition"),                                   "fallback"),

    # --- tier-2a: standalone high quantitativity (>= 0.55) ---
    ("07", dict(quantitativity=0.6,  spatial_specificity=0.3, temporal_specificity=0.3),        "tier-2a"),
    ("08", dict(quantitativity=0.55, spatial_specificity=0.3),                                  "tier-2a"),
    ("24", dict(quantitativity=0.9,  spatial_specificity=0.0, temporal_specificity=0.0),        "tier-2a"),

    # --- tier-2b: quant >= 0.60 + spatial or temporal >= 0.60 ---
    ("09", dict(quantitativity=0.6,  spatial_specificity=0.6),                                  "tier-2b"),
    ("10", dict(quantitativity=0.6,  temporal_specificity=0.6),                                 "tier-2b"),

    # --- tier-3: complexity >= 0.60 + mechanism/method_eval/comparison ---
    ("11", dict(complexity=0.6,  question_type="mechanism"),                                    "tier-3"),
    ("12", dict(complexity=0.7,  question_type="method_eval"),                                  "tier-3"),
    ("13", dict(complexity=0.65, question_type="comparison"),                                   "tier-3"),

    # --- ordering: tier-2b beats tier-3 when both could match ---
    ("15", dict(complexity=0.6,  question_type="quantitative",
                quantitativity=0.6, spatial_specificity=0.6),                                   "tier-2b"),
    ("16", dict(complexity=0.8,  question_type="mechanism",
                quantitativity=0.8, spatial_specificity=0.7),                                   "tier-2b"),

    # --- safety-tier3: low confidence (handled in router.py, not rules.yaml) ---
    ("17", dict(confidence=None),                                                               "safety-tier3"),
    ("18", dict(confidence=0.4),                                                                "safety-tier3"),
    ("19", dict(confidence=0.59),                                                               "safety-tier3"),

    # --- confidence boundary: 0.60 is sufficient, routes normally ---
    ("20", dict(confidence=0.6,  complexity=0.2, quantitativity=0.2),                           "tier-1"),

    # --- edge cases ---
    ("22", dict(complexity=0.1,  quantitativity=0.1),                                           "tier-1"),
    ("23", dict(complexity=0.0,  quantitativity=0.0),                                           "tier-1"),
    # tier-1-def: low complexity definition
    ("26", dict(complexity=0.3,  question_type="definition", quantitativity=0.1),               "tier-1-def"),
]


@pytest.mark.parametrize("case_id,profile_kwargs,expected", CASES, ids=[c[0] for c in CASES])
def test_router_routes_correctly(router, case_id, profile_kwargs, expected):
    cfg = router.select(_profile(**profile_kwargs))
    assert cfg.rule_hit == expected, (
        f"case {case_id}: profile={profile_kwargs} -> got {cfg.rule_hit!r}, expected {expected!r}"
    )


def test_safety_tier3_uses_large_model(router):
    cfg = router.select(_profile(confidence=None))
    assert cfg.model_name == "mistral-large-latest"
    assert cfg.evidence_mode == "excerpts_full"


def test_tier_1_def_uses_small_abstracts(router):
    cfg = router.select(_profile(question_type="definition", complexity=0.3, quantitativity=0.2))
    assert cfg.rule_hit == "tier-1-def"
    assert cfg.model_name == "mistral-small-latest"
    assert cfg.evidence_mode == "abstracts"


def test_tier_m_uses_medium_excerpts(router):
    cfg = router.select(_profile(complexity=0.5, quantitativity=0.3))
    assert cfg.rule_hit == "tier-m"
    assert cfg.model_name == "mistral-medium-latest"
    assert cfg.evidence_mode == "excerpts_narrow"


def test_fallback_properties(router):
    """unmatched profile: high complexity, type=application, no quant -> fallback."""
    cfg = router.select(_profile(complexity=0.8, quantitativity=0.0,
                                 question_type="application"))
    assert cfg.rule_hit == "fallback"
    assert cfg.evidence_mode == "abstracts"
    assert cfg.generation_prompt == "generation_prompt_exp4.txt"
