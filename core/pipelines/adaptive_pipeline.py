"""adaptive pipeline -- routes per-question to the right evidence mode + model.

zero changes to exp 1-4 pipelines. all behaviour is selected at runtime by
the policy router based on the question profile.

flow per question:
  1. profile     -- fused av-pair + profile call (small, temp=0)
  2. route       -- pure-python policy router selects PipelineConfig
  3. evidence    -- abstracts (no extra work) | excerpts via FullTextIndexer
  4. refinement  -- 1pass-refined or 1pass-fulltext, depending on evidence
  5. generation  -- direct or structured prompt, with shape + synthesis directives
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.policy.router import Router
from core.utils.data_models import (
    DynamicOntology,
    PipelineConfig,
    QuestionProfile,
)

log = logging.getLogger(__name__)

# directive maps -- injected into generation_structured.txt's two placeholders
_ANSWER_SHAPE_DIRECTIVES = {
    "direct_paragraph":      "Write a direct paragraph answer without section headings.",
    "short_explainer":       "Write a short explanation in 2-4 sentences.",
    "structured_long":       "Structure your answer with clear section headings matched to the question type.",
    "comparison_table":      "Present your answer as a markdown comparison table followed by one explanatory paragraph per row.",
    "mechanism_walkthrough": "Walk through the mechanism step by step, one numbered step per causal link.",
    "raw":                   "",
}
_SYNTHESIS_DIRECTIVES = {
    "homogeneous": "",
    "focused": (
        "Focus your synthesis strictly on the specific region and/or time period "
        "named in the question. Do not extrapolate findings to other regions or periods."
    ),
}


@dataclass
class AdaptiveResult:
    answer: str
    references: List[str] = field(default_factory=list)
    formatted_references: List[str] = field(default_factory=list)
    profile: Optional[QuestionProfile] = None
    pipeline_config: Optional[PipelineConfig] = None
    enriched_context: str = ""
    rule_hit: str = ""
    excerpt_stats: Dict[str, Any] = field(default_factory=dict)


class AdaptivePipeline:
    def __init__(
        self,
        rules_path: Union[str, Path] = "core/policy/rules.yaml",
        cache_dir: Union[str, Path] = "cache/fulltext",
        prompts_root: Union[str, Path] = "prompts",
    ):
        self.router = Router(rules_path)
        self._cache_dir = Path(cache_dir)
        self._prompts_root = Path(prompts_root)
        self._llm_cache: Dict[Tuple[str, float], Any] = {}
        self._indexer = None  # lazy

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------

    def profile_and_route(
        self, question: str
    ) -> Tuple[DynamicOntology, QuestionProfile, PipelineConfig]:
        """profile a question and return the routing decision -- no refinement,
        no generation. used by the runner's --dry-run flag.
        """
        from core.agents.ontology_agent import OntologyConstructionAgent

        ont_agent = OntologyConstructionAgent(
            self._llm("mistral-small-latest", 0.0),
            prompt_dir=str(self._prompts_root / "ontology"),
        )
        ontology, profile = ont_agent.process_with_profile(question)
        cfg = self.router.select(profile)
        return ontology, profile, cfg

    def run(
        self,
        question: str,
        aql_results_str: str,
        docs: Optional[List[Dict[str, Any]]] = None,
    ) -> AdaptiveResult:
        from core.agents.generation_agent import GenerationAgent
        from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
        from core.agents.refinement_agent_fulltext import RefinementAgent1PassFullText

        ontology, profile, cfg = self.profile_and_route(question)
        log.info(
            "q=%s... -> rule=%s model=%s evidence=%s",
            question[:60], cfg.rule_hit, cfg.model_name, cfg.evidence_mode,
        )

        # 1. evidence
        excerpts_text, excerpt_stats = self._build_evidence(cfg, question, ontology, docs)

        # 2. refinement
        ref_llm = self._llm(cfg.model_name, cfg.temperature_refine)
        if cfg.evidence_mode == "abstracts":
            ref_agent = RefinementAgent1PassRefined(
                ref_llm, prompt_dir=str(self._prompts_root / "refinement"),
            )
        else:
            ref_agent = RefinementAgent1PassFullText(
                ref_llm, prompt_dir=str(self._prompts_root / "refinement"),
            )
            ref_agent.set_excerpts(excerpts_text)
        # scope_filter is signalled to the agent if it supports it; otherwise
        # synthesis_mode=focused already enforces the constraint at generation.
        if hasattr(ref_agent, "set_scope_filter"):
            ref_agent.set_scope_filter(cfg.scope_filter)

        refined = ref_agent.process_context(
            question=question,
            structured_context="",
            ontology=ontology,
            include_ontology=True,
            aql_results_str=aql_results_str,
            context_filter="full",
        )
        enriched = refined.enriched_context or ""

        # 3. generation -- override template + inject adaptive directives
        gen_agent = GenerationAgent(
            self._llm(cfg.model_name, cfg.temperature_generate),
            prompt_dir=str(self._prompts_root / "generation"),
        )
        template = gen_agent._load_template(cfg.generation_prompt)
        if not template:
            raise FileNotFoundError(f"generation prompt not found: {cfg.generation_prompt}")
        # pre-fill our two placeholders before GenerationAgent.generate() does
        # the rest of the .replace() chain. extra placeholders that aren't in
        # this template are silently ignored by str.replace.
        template = template.replace(
            "{answer_shape_directives}",
            _ANSWER_SHAPE_DIRECTIVES.get(profile.answer_shape, ""),
        ).replace(
            "{synthesis_mode_directives}",
            _SYNTHESIS_DIRECTIVES.get(cfg.synthesis_mode, ""),
        )
        gen_agent.refinement_template = template

        ans = gen_agent.generate(question=question, text_context=enriched, ontology=ontology)

        return AdaptiveResult(
            answer=ans.answer,
            references=ans.references,
            formatted_references=ans.formatted_references,
            profile=profile,
            pipeline_config=cfg,
            enriched_context=enriched,
            rule_hit=cfg.rule_hit,
            excerpt_stats=excerpt_stats,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _llm(self, model: str, temperature: float):
        key = (model, temperature)
        if key not in self._llm_cache:
            from core.utils.helpers import get_llm_model
            self._llm_cache[key] = get_llm_model(model=model, temperature=temperature)
        return self._llm_cache[key]

    def _build_evidence(
        self,
        cfg: PipelineConfig,
        question: str,
        ontology: DynamicOntology,
        docs: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, Dict[str, Any]]:
        if cfg.evidence_mode == "abstracts" or not docs:
            return "", {"mode": cfg.evidence_mode, "n_excerpts": 0, "n_docs": len(docs or [])}

        if self._indexer is None:
            from core.utils.fulltext_indexer import FullTextIndexer
            self._indexer = FullTextIndexer(cache_dir=self._cache_dir)

        excerpts, stats = self._indexer.select_excerpts_for_question(
            question=question,
            ontology=ontology,
            documents=docs,
            per_doc_budget=cfg.per_doc_budget,
            global_budget=cfg.global_budget,
            top_k_per_doc=cfg.top_k_per_doc,
        )
        text = self._indexer.render_excerpts_block(excerpts)
        stats = dict(stats)
        stats.update({
            "mode": cfg.evidence_mode,
            "excerpt_chars": len(text),
            "excerpt_tokens_est": len(text) // 4,
        })
        return text, stats
