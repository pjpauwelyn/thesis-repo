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

# 60% of the 128k token window shared by mistral-small-latest and
# mistral-large-latest. expressed in chars (tokens x 4) so generation_agent
# can compare directly against len(text_context).
_CONTEXT_CAP_CHARS = 307_200  # 76_800 tokens x 4


class Router:
    def __init__(self, rules_path: Union[str, Path] = _DEFAULT_RULES):
        with open(rules_path, "r", encoding="utf-8") as f:
            self._rules = yaml.safe_load(f)["rules"]

    def select(self, profile: QuestionProfile) -> PipelineConfig:
        if profile.confidence is None or profile.confidence < _CONFIDENCE_FLOOR:
            # Fix 6: before escalating to safety-tier3, check whether a
            # parse-failure profile still carries a clear definitional/low-cx
            # signal. If so, route to tier-1-def instead of safety-tier3 to
            # avoid ~6x cost overrun on simple definitional questions whose
            # profiler JSON wrapper was malformed but payload was intact.
            if self._is_tier1_def_rescue(profile):
                _log.warning(
                    "router.select: parse-failure profile matches tier-1-def rescue heuristic "
                    "(type=%s complexity=%.2f quant=%.2f conf=%.2f) -> tier-1-def-parse-rescue",
                    profile.question_type,
                    profile.complexity,
                    profile.quantitativity,
                    profile.confidence or 0.0,
                )
                return PipelineConfig(
                    model_name="mistral-small-latest",
                    evidence_mode="abstracts",
                    top_k_per_doc=0,
                    per_doc_budget=0,
                    global_budget=0,
                    refinement_prompt="refinement_1pass_refined_exp4.txt",
                    generation_prompt="generation_direct.txt",
                    gen_context_cap=_CONTEXT_CAP_CHARS,
                    max_output_tokens=2000,
                    system_prompt_modifier="",
                    doc_filter_min_keep=3,
                    scope_filter=False,
                    synthesis_mode="homogeneous",
                    timeout_refine_s=30,
                    timeout_generate_s=45,
                    use_draft=True,
                    rule_hit="tier-1-def-parse-rescue",
                    reason=(
                        f"parse-failure but profile matches tier-1-def "
                        f"(type={profile.question_type} cx={profile.complexity:.2f} "
                        f"quant={profile.quantitativity:.2f}) -> rescue to tier-1-def"
                    ),
                )

            _log.warning(
                "router.select: low/missing confidence (%.2f) -> safety-tier3 "
                "(type=%s complexity=%.2f quant=%.2f)",
                profile.confidence or 0.0,
                profile.question_type,
                profile.complexity,
                profile.quantitativity,
            )
            # P1b: max_output_tokens raised from 1400 -> 4000 to match all other
            # full-text tiers.  safety-tier3 uses mistral-large + excerpts_full
            # (35k-60k token context); 1400 tok output was causing truncation on
            # exactly the questions (low-confidence escalations) that need the
            # most thorough answers.  4000 tok @ 47 tok/s = 85s < 480s timeout.
            return PipelineConfig(
                model_name="mistralai/mistral-large",
                refinement_model_name="mistralai/mistral-large",
                evidence_mode="excerpts_full",
                top_k_per_doc=10,
                per_doc_budget=9000,
                global_budget=60000,
                refinement_prompt="refinement_1pass_refined_fulltext.txt",
                generation_prompt="generation_structured.txt",
                gen_context_cap=_CONTEXT_CAP_CHARS,
                max_output_tokens=4000,
                system_prompt_modifier="",
                doc_filter_min_keep=8,
                scope_filter=False,
                synthesis_mode="homogeneous",
                timeout_refine_s=360,
                timeout_generate_s=480,
                use_draft=False,
                rule_hit="safety-tier3",
                reason=f"low/missing profiler confidence ({profile.confidence}) -> safety tier-3",
            )

        _log.info(
            "router.select: type=%s complexity=%.2f quant=%.2f spatial=%.2f "
            "temporal=%.2f meth=%.2f conf=%.2f",
            profile.question_type,
            profile.complexity,
            profile.quantitativity,
            profile.spatial_specificity,
            profile.temporal_specificity,
            profile.methodological_depth,
            profile.confidence or 0.0,
        )

        for rule in self._rules:
            if self._matches(rule["when"], profile):
                _log.info("router.select: matched rule '%s'", rule["name"])
                return PipelineConfig(**rule["config"])

        # no rule matched -- yaml fallback (always: true) should have caught
        # this, so reaching here is a rules.yaml misconfiguration. return a
        # safe default and warn loudly.
        _log.warning(
            "router.select: NO rule matched for profile "
            "(type=%s complexity=%.2f quant=%.2f spatial=%.2f temporal=%.2f meth=%.2f) "
            "-- returning fallback. check rules.yaml for coverage gaps.",
            profile.question_type,
            profile.complexity,
            profile.quantitativity,
            profile.spatial_specificity,
            profile.temporal_specificity,
            profile.methodological_depth,
        )
        # P1c: max_output_tokens raised from 700 -> 2000 to match the fallback
        # rule in rules.yaml (which was already updated by P1).
        return PipelineConfig(
            model_name="mistral-small-latest",
            evidence_mode="abstracts",
            top_k_per_doc=0,
            per_doc_budget=0,
            global_budget=0,
            refinement_prompt="refinement_1pass_refined_exp4.txt",
            generation_prompt="generation_direct.txt",
            gen_context_cap=_CONTEXT_CAP_CHARS,
            max_output_tokens=2000,
            system_prompt_modifier="",
            doc_filter_min_keep=6,
            scope_filter=False,
            synthesis_mode="homogeneous",
            timeout_refine_s=60,
            timeout_generate_s=60,
            use_draft=True,
            rule_hit="fallback",
            reason="no rule matched -- conservative fallback",
        )

    @staticmethod
    def _is_tier1_def_rescue(profile: QuestionProfile) -> bool:
        """Return True when a parse-failure profile safely matches tier-1-def.

        Fix 6: intentionally mirrors the tier-1-def when: block in rules.yaml
        (question_type in [definition, factual] AND quant < 0.45 AND cx < 0.60)
        so the rescue heuristic stays in sync if the rule thresholds change.
        Returns False when any field is None (conservative: unknown -> safety-tier3).
        """
        qt = getattr(profile, "question_type", None)
        cx = getattr(profile, "complexity", None)
        quant = getattr(profile, "quantitativity", None)
        if qt is None or cx is None or quant is None:
            return False
        return (
            qt in ("definition", "factual")
            and quant < 0.45
            and cx < 0.60
        )

    def _matches(self, when: Dict[str, Any], p: QuestionProfile) -> bool:
        """evaluate a when: block -- all top-level keys must pass.

        supports:
          field: {op: value}          plain scalar comparison
          question_type: {in: [...]}  membership test
          any: {k: v, ...}            flat-dict OR  (legacy tier-2b style)
          any: [{k: v}, {k: v}, ...]  list-of-dicts OR  (new style, supports nesting)
          all: [{k: v}, {k: v}, ...]  list-of-dicts AND block
          always: true                unconditional match (fallback)
        """
        for key, cond in when.items():
            if key == "always":
                return bool(cond)
            if key == "any":
                items = cond if isinstance(cond, list) else [{k: v} for k, v in cond.items()]
                if not any(self._matches_block(block, p) for block in items):
                    return False
            elif key == "all":
                items = cond if isinstance(cond, list) else [{k: v} for k, v in cond.items()]
                if not all(self._matches_block(block, p) for block in items):
                    return False
            elif not self._eval(key, cond, p):
                return False
        return True

    def _matches_block(self, block: Dict[str, Any], p: QuestionProfile) -> bool:
        """evaluate a single condition block.

        a block may itself contain all:/any: combinators (nested inside an any:
        list item). delegate back to _matches so the full evaluator handles them
        rather than calling _eval on 'all' or 'any' as if they were profile
        field names (which would always return False and silently break the gate).

        example that was broken before this fix:

          any:
            - all:
                question_type: {in: [comparison, method_eval]}
                methodological_depth: {ge: 0.45}

        the list item is {'all': {'question_type': ..., 'methodological_depth': ...}}.
        old _matches_block called _eval('all', {...}, p) -> getattr(p, 'all') -> None
        -> False, so the whole any: always evaluated to False for this branch.
        """
        if any(k in block for k in ("all", "any", "always")):
            return self._matches(block, p)
        return all(self._eval(k, v, p) for k, v in block.items())

    def _eval(self, field: str, cond: Any, p: QuestionProfile) -> bool:
        if isinstance(cond, dict) and "in" in cond:
            return getattr(p, field, None) in cond["in"]
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
