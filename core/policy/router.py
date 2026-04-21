"""pure-python policy router. no llm calls, deterministic, yaml-driven."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from core.utils.data_models import PipelineConfig, QuestionProfile

_DEFAULT_RULES = Path(__file__).parent / "rules.yaml"
_log = logging.getLogger(__name__)

# below this profiler confidence we ignore the rule table entirely and route
# to the safety net (large + full excerpts). also fires on confidence=None,
# which is what process_with_profile() returns on parse failure.
_CONFIDENCE_FLOOR = 0.6

# ---------------------------------------------------------------------------
# deterministic answer_shape mapping
# the LLM profiler tends to default to structured_long regardless of question
# type, so we override answer_shape here based on question_type + complexity.
# this keeps answer_shape consistent and removes LLM non-compliance as a
# variable.
# ---------------------------------------------------------------------------

def _resolve_answer_shape(profile: QuestionProfile) -> str:
    qt = profile.question_type
    c  = profile.complexity
    if qt in ("definition", "application"):
        return "direct_paragraph"
    if qt == "continuous":
        return "short_explainer"
    if qt == "comparison":
        return "comparison_table"
    if qt == "mechanism":
        # simple causal chain -> step-by-step; multi-mechanism -> headed sections
        return "mechanism_walkthrough" if c < 0.6 else "structured_long"
    # quantitative, method_eval, fallback
    return "structured_long"


class Router:
    def __init__(self, rules_path: Union[str, Path] = _DEFAULT_RULES):
        with open(rules_path, "r", encoding="utf-8") as f:
            self._rules = yaml.safe_load(f)["rules"]

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def select(self, profile: QuestionProfile) -> PipelineConfig:
        # override answer_shape deterministically before routing so downstream
        # generation_structured.txt gets the correct directive.
        profile.answer_shape = _resolve_answer_shape(profile)

        if profile.confidence is None or profile.confidence < _CONFIDENCE_FLOOR:
            return PipelineConfig(
                model_name="mistral-large-latest",
                evidence_mode="excerpts_full",
                top_k_per_doc=10,
                per_doc_budget=9000,
                global_budget=60000,
                refinement_prompt="refinement_1pass_refined_fulltext.txt",
                generation_prompt="generation_structured.txt",
                rule_hit="safety-tier3",
                reason=f"low/missing profiler confidence ({profile.confidence}) -> safety tier-3",
            )

        _log.debug(
            "router.select: type=%s complexity=%.2f quant=%.2f spatial=%.2f "
            "temporal=%.2f conf=%.2f",
            profile.question_type,
            profile.complexity,
            profile.quantitativity,
            profile.spatial_specificity,
            profile.temporal_specificity,
            profile.confidence or 0.0,
        )

        for rule in self._rules:
            if self._matches(rule["when"], profile):
                _log.debug("router.select: matched rule '%s'", rule["name"])
                return PipelineConfig(**rule["config"])

        # the yaml always ends with always:true, so this is unreachable
        raise RuntimeError("router: no rule matched and no fallback found in rules.yaml")

    # ------------------------------------------------------------------
    # condition evaluation
    # ------------------------------------------------------------------

    def _matches(self, when: Dict[str, Any], p: QuestionProfile) -> bool:
        for key, cond in when.items():
            if key == "always":
                return bool(cond)
            if key == "any":
                if not any(self._eval(k, v, p) for k, v in cond.items()):
                    return False
            elif not self._eval(key, cond, p):
                return False
        return True

    def _eval(self, field: str, cond: Any, p: QuestionProfile) -> bool:
        # categorical membership: {in: [a, b, c]}
        if isinstance(cond, dict) and "in" in cond:
            return getattr(p, field, None) in cond["in"]

        # complexity_lt is an alias so a single rule can express two
        # inequalities on the complexity field without a custom yaml schema.
        actual = "complexity" if field == "complexity_lt" else field
        val = getattr(p, actual, None)
        if val is None or not isinstance(cond, dict):
            return False

        for op, threshold in cond.items():
            if op == "lt" and not val < threshold:
                return False
            if op == "le" and not val <= threshold:
                return False
            if op == "gt" and not val > threshold:
                return False
            if op == "ge" and not val >= threshold:
                return False
        return True
