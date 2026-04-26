"""router unit tests -- updated for rules.yaml v8 and Fix 6/7.

Key ordering notes (rules.yaml v8, evaluated top-to-bottom, first match wins):
  - tier-1-def fires before tier-m: definitional + complexity < 0.60 + quant < 0.45
  - tier-2b fires before tier-3: high-quant focal beats complex-type
  - tier-3 gate (v8 -- any one of a/b/c/d):
      a) meth >= 0.55
      b) cx >= 0.75 AND meth >= 0.45
      c) type in comparison/method_eval AND meth >= 0.40
      d) type == mechanism AND meth >= 0.45 AND cx >= 0.60  [new in v8]
  - tier-m: catch-all for complexity >= 0.40 (NO upper bound)
  - tier-1 ceiling: complexity < 0.40

v7 changes (cases 32-34):
  Branch-c gate lowered from meth>=0.45 to meth>=0.40 for comparison/method_eval.

v8 changes (cases 35-40):
  Branch-d added: mechanism + meth>=0.45 + cx>=0.60 -> tier-3.
  Case 28 updated: mechanism+meth=0.54+cx=0.70 now routes tier-3 (was tier-m).
  Case 28b added as regression guard: mechanism+meth=0.44+cx=0.70 stays tier-m.
  Fix 6: tier-1-def-parse-rescue cases added (36-38).
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

    # --- tier-m: complexity >= 0.40, no meth/quant threshold hit ---
    # 0.40 is the lower bound (inclusive), no upper bound
    ("21", dict(complexity=0.4,  quantitativity=0.3),                                           "tier-m"),
    ("05", dict(complexity=0.5,  quantitativity=0.3),                                           "tier-m"),
    ("06", dict(complexity=0.59, quantitativity=0.3),                                           "tier-m"),
    # tier-m catches high-complexity mechanism questions with low meth
    ("27", dict(complexity=0.8,  question_type="mechanism",  methodological_depth=0.30),        "tier-m"),
    # v8 regression guard: mechanism+meth=0.44 is BELOW branch-D gate (0.45) -> tier-m
    ("28b",dict(complexity=0.70, question_type="mechanism",  methodological_depth=0.44),        "tier-m"),
    # v7: comparison+meth=0.35 is below new gate (0.40) -> tier-m (regression guard)
    ("29", dict(complexity=0.75, question_type="comparison", methodological_depth=0.35),        "tier-m"),

    # --- tier-1-def: definition + complexity < 0.60 + quant < 0.45 ---
    # complexity=0.5 -> tier-1-def fires before tier-m
    ("25", dict(complexity=0.5,  question_type="definition", quantitativity=0.2),               "tier-1-def"),
    # complexity=0.6 is NOT < 0.60 -> tier-1-def misses; uncapped tier-m catches cx=0.6
    ("14", dict(complexity=0.6,  question_type="definition"),                                   "tier-m"),

    # --- tier-2a: standalone high quantitativity (>= 0.55), meth < 0.55 ---
    ("07", dict(quantitativity=0.6,  spatial_specificity=0.3, temporal_specificity=0.3),        "tier-2a"),
    ("08", dict(quantitativity=0.55, spatial_specificity=0.3),                                  "tier-2a"),
    ("24", dict(quantitativity=0.9,  spatial_specificity=0.0, temporal_specificity=0.0),        "tier-2a"),

    # --- tier-2b: quant >= 0.60 + spatial or temporal >= 0.60 ---
    ("09", dict(quantitativity=0.6,  spatial_specificity=0.6),                                  "tier-2b"),
    ("10", dict(quantitativity=0.6,  temporal_specificity=0.6),                                 "tier-2b"),

    # --- tier-3: methodology synthesis gate ---
    # a) pure meth depth >= 0.55
    ("11", dict(complexity=0.6,  question_type="mechanism",  methodological_depth=0.55),        "tier-3"),
    ("12", dict(complexity=0.7,  question_type="method_eval",methodological_depth=0.60),        "tier-3"),
    # b) comparison + meth >= 0.45 (still fires under v7 gate of 0.40)
    ("13", dict(complexity=0.65, question_type="comparison", methodological_depth=0.50),        "tier-3"),
    ("30", dict(complexity=0.65, question_type="comparison", methodological_depth=0.45),        "tier-3"),
    # c) cx >= 0.75 + meth >= 0.45
    ("31", dict(complexity=0.75, question_type="mechanism",  methodological_depth=0.45),        "tier-3"),
    # d) v8 branch-D: mechanism + meth >= 0.45 + cx >= 0.60
    # case 28 updated: mechanism+meth=0.54+cx=0.70 now routes tier-3 (was tier-m under v7)
    ("28", dict(complexity=0.70, question_type="mechanism",  methodological_depth=0.54),        "tier-3"),
    # branch-D boundary: meth=0.45 (at gate), cx=0.60 (at gate) -> tier-3
    ("35", dict(complexity=0.60, question_type="mechanism",  methodological_depth=0.45),        "tier-3"),
    # branch-D floor: cx=0.59 is BELOW the 0.60 floor -> tier-m
    ("36", dict(complexity=0.59, question_type="mechanism",  methodological_depth=0.45),        "tier-m"),

    # --- ordering: tier-2b beats tier-3 when both could match ---
    ("15", dict(complexity=0.6,  question_type="quantitative",
                quantitativity=0.6, spatial_specificity=0.6),                                   "tier-2b"),
    ("16", dict(complexity=0.8,  question_type="mechanism",
                quantitativity=0.8, spatial_specificity=0.7),                                   "tier-2b"),

    # --- safety-tier3: low confidence (handled in router.py, not rules.yaml) ---
    # Fix 6: definition+low-quant+low-cx parse-failure rescues to tier-1-def-parse-rescue,
    # NOT safety-tier3. Cases 17-19 use non-definitional types so safety-tier3 still fires.
    ("17", dict(confidence=None, question_type="mechanism"),                                    "safety-tier3"),
    ("18", dict(confidence=0.4,  question_type="mechanism"),                                    "safety-tier3"),
    ("19", dict(confidence=0.59, question_type="mechanism"),                                    "safety-tier3"),

    # --- confidence boundary: 0.60 is sufficient, routes normally ---
    ("20", dict(confidence=0.6,  complexity=0.2, quantitativity=0.2),                           "tier-1"),

    # --- edge cases ---
    ("22", dict(complexity=0.1,  quantitativity=0.1),                                           "tier-1"),
    ("23", dict(complexity=0.0,  quantitativity=0.0),                                           "tier-1"),
    # tier-1-def: low complexity definition
    ("26", dict(complexity=0.3,  question_type="definition", quantitativity=0.1),               "tier-1-def"),

    # --- v7 new cases: tier-3 branch-c gate lowered to meth>=0.40 ---
    # comparison + meth=0.40 -> tier-3 (new threshold, was tier-m under v6)
    ("32", dict(complexity=0.70, question_type="comparison", methodological_depth=0.40),        "tier-3"),
    # method_eval + meth=0.40 -> tier-3 (gate applies to method_eval too)
    ("33", dict(complexity=0.65, question_type="method_eval", methodological_depth=0.40),       "tier-3"),
    # comparison + meth=0.35 -> tier-m (below new gate -- regression guard)
    ("34", dict(complexity=0.70, question_type="comparison", methodological_depth=0.35),        "tier-m"),

    # --- Fix 6: tier-1-def-parse-rescue (confidence < 0.60 but definitional profile) ---
    # definition + cx<0.60 + quant<0.45 + conf=0.0 -> rescue to tier-1-def-parse-rescue
    ("37", dict(confidence=0.0, question_type="definition", complexity=0.4, quantitativity=0.2),
           "tier-1-def-parse-rescue"),
    # factual + cx<0.60 + quant<0.45 + conf=0.0 -> rescue
    ("38", dict(confidence=0.0, question_type="factual", complexity=0.3, quantitativity=0.3),
           "tier-1-def-parse-rescue"),
    # mechanism + conf=0.0 -> NOT rescued (non-definitional) -> safety-tier3
    ("39", dict(confidence=0.0, question_type="mechanism", complexity=0.4, quantitativity=0.2),
           "safety-tier3"),
    # definition + quant>=0.45 + conf=0.0 -> NOT rescued (quant too high) -> safety-tier3
    ("40", dict(confidence=0.0, question_type="definition", complexity=0.4, quantitativity=0.5),
           "safety-tier3"),
]


@pytest.mark.parametrize("case_id,profile_kwargs,expected", CASES, ids=[c[0] for c in CASES])
def test_router_routes_correctly(router, case_id, profile_kwargs, expected):
    cfg = router.select(_profile(**profile_kwargs))
    assert cfg.rule_hit == expected, (
        f"case {case_id}: profile={profile_kwargs} -> got {cfg.rule_hit!r}, expected {expected!r}"
    )


def test_safety_tier3_uses_large_model(router):
    cfg = router.select(_profile(confidence=None, question_type="mechanism"))
    assert cfg.rule_hit == "safety-tier3"
    # safety-tier3 routes via OpenRouter; model name carries the mistralai/ prefix
    assert "mistral-large" in cfg.model_name
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


def test_tier_m_catches_high_cx_low_meth(router):
    """cx=0.80 mechanism with meth=0.20 must NOT go to tier-3."""
    cfg = router.select(_profile(complexity=0.8, question_type="mechanism",
                                 methodological_depth=0.20))
    assert cfg.rule_hit == "tier-m"
    assert cfg.model_name == "mistral-medium-latest"


def test_tier3_requires_meth_not_just_complexity(router):
    """cx=0.80 mechanism with meth=0.0 must NOT go to tier-3 (old broken gate)."""
    cfg = router.select(_profile(complexity=0.8, question_type="mechanism",
                                 methodological_depth=0.0))
    assert cfg.rule_hit != "tier-3", (
        "tier-3 fired on meth=0.0 -- complexity-only gate is back, check rules.yaml tier_3 when:"
    )


def test_tier3_comparison_meth_040(router):
    """v7: comparison + meth=0.40 must hit tier-3 (gate lowered from 0.45)."""
    cfg = router.select(_profile(complexity=0.70, question_type="comparison",
                                 methodological_depth=0.40))
    assert cfg.rule_hit == "tier-3", (
        "comparison+meth=0.40 should hit tier-3 under v7 gate -- "
        "check tier_3 branch-c in rules.yaml (methodological_depth ge 0.40)"
    )


def test_tier_m_comparison_below_gate(router):
    """v7 regression guard: comparison + meth=0.35 must stay in tier-m."""
    cfg = router.select(_profile(complexity=0.70, question_type="comparison",
                                 methodological_depth=0.35))
    assert cfg.rule_hit == "tier-m", (
        "comparison+meth=0.35 should stay in tier-m -- gate lowered too far if this fails"
    )


def test_tier3_mechanism_branch_d_mid_meth(router):
    """v8 branch-D: mechanism + meth=0.50 + cx=0.65 must hit tier-3."""
    cfg = router.select(_profile(complexity=0.65, question_type="mechanism",
                                 methodological_depth=0.50))
    assert cfg.rule_hit == "tier-3", (
        "mechanism+meth=0.50+cx=0.65 should hit tier-3 via branch-D -- "
        "check tier_3 branch-d in rules.yaml"
    )


def test_tier_m_mechanism_below_branch_d_meth(router):
    """v8 regression guard: mechanism + meth=0.44 (< 0.45) + cx=0.70 stays tier-m."""
    cfg = router.select(_profile(complexity=0.70, question_type="mechanism",
                                 methodological_depth=0.44))
    assert cfg.rule_hit == "tier-m", (
        "mechanism+meth=0.44 (< 0.45) should stay in tier-m -- branch-D gate too loose if fails"
    )


def test_tier_m_mechanism_below_branch_d_cx(router):
    """v8 regression guard: mechanism + meth=0.45 + cx=0.59 (< 0.60) stays tier-m."""
    cfg = router.select(_profile(complexity=0.59, question_type="mechanism",
                                 methodological_depth=0.45))
    assert cfg.rule_hit == "tier-m", (
        "mechanism+meth=0.45+cx=0.59 (< 0.60) should stay in tier-m -- "
        "branch-D cx floor too low if fails"
    )


def test_parse_rescue_definition_routes_tier1_def(router):
    """Fix 6: conf=0.0 definition+low-cx+low-quant -> tier-1-def-parse-rescue."""
    cfg = router.select(_profile(
        confidence=0.0, question_type="definition",
        complexity=0.4, quantitativity=0.2,
    ))
    assert cfg.rule_hit == "tier-1-def-parse-rescue", (
        "definition+low-cx+low-quant+conf=0.0 should rescue to tier-1-def, not safety-tier3"
    )
    assert cfg.model_name == "mistral-small-latest"
    assert cfg.evidence_mode == "abstracts"


def test_parse_rescue_mechanism_still_safety_tier3(router):
    """Fix 6: conf=0.0 mechanism -> NOT rescued, still safety-tier3."""
    cfg = router.select(_profile(
        confidence=0.0, question_type="mechanism",
        complexity=0.4, quantitativity=0.2,
    ))
    assert cfg.rule_hit == "safety-tier3", (
        "mechanism+conf=0.0 should NOT rescue -- non-definitional types must stay at safety-tier3"
    )


def test_is_tier1_def_rescue_none_fields():
    """_is_tier1_def_rescue returns False when any field is None (conservative)."""
    from core.policy.router import Router
    p = _profile(confidence=0.0, question_type=None, complexity=0.4, quantitativity=0.2)
    assert not Router._is_tier1_def_rescue(p)


def test_fallback_properties(router):
    """unmatched profile: cx=0.1, type=application, no quant -> tier-1 (cx < 0.40)."""
    cfg = router.select(_profile(complexity=0.1, quantitativity=0.0,
                                 question_type="application"))
    # application type does not match any tier-specific type check;
    # cx=0.1 < 0.40 -> tier-1 catches it before fallback.
    assert cfg.rule_hit == "tier-1"


def test_fallback_truly_unmatched(router):
    """verify the fallback YAML rule has the expected config values."""
    import yaml
    from pathlib import Path
    rules = yaml.safe_load((Path(__file__).parent.parent / "core/policy/rules.yaml").read_text())["rules"]
    fallback = next(r for r in rules if r["name"] == "fallback")
    assert fallback["config"]["evidence_mode"] == "abstracts"
    assert fallback["config"]["generation_prompt"] == "generation_direct.txt"
    assert fallback["config"]["use_draft"] is True
