"""adaptive pipeline -- routes per-question to the right evidence mode + model.

zero changes to exp 1-4 pipelines. all behaviour is selected at runtime by
the policy router based on the question profile.

flow per question:
  1. profile     -- fused av-pair + profile call (small, temp=0)
  2. route       -- pure-python policy router selects PipelineConfig
  3. filter      -- llm-based document relevance filter (all tiers)
  4. evidence    -- abstracts (no extra work) | excerpts via FullTextIndexer
  5. refinement  -- 1pass-refined or 1pass-fulltext, depending on evidence
  6. generation  -- direct or structured prompt, with shape + synthesis directives
"""

from __future__ import annotations

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

_ANSWER_SHAPE_DIRECTIVES = {
    "direct_paragraph":      (
        "Write a direct paragraph answer of 3–5 sentences. "
        "No section headings. No bullet lists. No Implications section."
    ),
    "short_explainer":       "Write a short explanation in 2-4 sentences.",
    "structured_long":       (
        "Structure your answer with 3–5 section headings that directly match "
        "the sub-questions implied by the question. "
        "Open with a 2–3 sentence direct answer before the first heading. "
        "Do not add headings that are not needed to answer the question."
    ),
    "comparison_table":      (
        "Your answer MUST include a markdown comparison table as the first "
        "structured element after the opening paragraph. "
        "The table must have one row per entity being compared and columns for "
        "the key dimensions the question asks about. "
        "Follow the table with one short paragraph per major finding."
    ),
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
        """profile + route only -- no refinement, no generation.
        used by the runner's --dry-run flag."""
        from core.agents.ontology_agent import OntologyConstructionAgent

        ont_agent = OntologyConstructionAgent(
            self._llm("mistral-small-latest", 0.0),
            prompt_dir=str(self._prompts_root / "ontology"),
        )
        ontology, profile = ont_agent.process_with_profile(question)
        cfg = self.router.select(profile)
        return ontology, profile, cfg

    def profile_and_route_with_filter(
        self,
        question: str,
        docs: Optional[List[Dict[str, Any]]] = None,
        aql_results_str: Optional[str] = None,
    ) -> Tuple[DynamicOntology, QuestionProfile, PipelineConfig, Dict[str, Any]]:
        """extended dry-run: profile + route + document filter, no generation."""
        ontology, profile, cfg = self.profile_and_route(question)
        filter_summary: Dict[str, Any] = {"n_total": 0}
        if docs:
            full_docs, abstract_docs, drop_docs = self._filter_documents(
                docs, ontology, profile, question, cfg
            )
            filter_summary = {
                "n_total":         len(docs),
                "n_full":          len(full_docs),
                "n_abstract":      len(abstract_docs),
                "n_drop":          len(drop_docs),
                "full_titles":     [d.get("title", "")[:80] for d in full_docs],
                "abstract_titles": [d.get("title", "")[:80] for d in abstract_docs],
                "drop_titles":     [d.get("title", "")[:80] for d in drop_docs],
            }
        return ontology, profile, cfg, filter_summary

    def run(
        self,
        question: str,
        aql_results_str: str,
        docs: Optional[List[Dict[str, Any]]] = None,
    ) -> AdaptiveResult:
        from core.agents.generation_agent import GenerationAgent
        from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
        from core.agents.refinement_agent_fulltext import RefinementAgent1PassFullText

        # 1. profile + route
        ontology, profile, cfg = self.profile_and_route(question)
        log.info(
            "q=%s... -> rule=%s model=%s evidence=%s",
            question[:60], cfg.rule_hit, cfg.model_name, cfg.evidence_mode,
        )

        # 2. document filter (all tiers)
        all_docs = docs or []
        full_docs, abstract_docs, drop_docs = self._filter_documents(
            all_docs, ontology, profile, question, cfg
        )

        # 3. evidence -- PDF excerpts for full_docs only
        excerpts, excerpt_stats = self._build_evidence(cfg, question, ontology, full_docs)

        # 4. assemble unified documents_block for tier-2 / tier-3
        if cfg.evidence_mode != "abstracts":
            aql_lookup = self._build_aql_lookup(aql_results_str, all_docs)
            self._ensure_indexer()
            from core.utils.fulltext_indexer import FullTextIndexer
            documents_block = FullTextIndexer.render_documents_block(
                full_docs=full_docs,
                abstract_docs=abstract_docs,
                excerpts=excerpts,
                aql_lookup=aql_lookup,
            )
        else:
            documents_block = None  # tier-1/fallback: uses aql_results_str path unchanged

        # 5. refinement
        ref_llm = self._llm(cfg.model_name, cfg.temperature_refine)
        if cfg.evidence_mode == "abstracts":
            ref_agent = RefinementAgent1PassRefined(
                ref_llm,
                prompt_dir=str(self._prompts_root / "refinement"),
            )
        else:
            ref_agent = RefinementAgent1PassFullText(
                ref_llm,
                prompt_dir=str(self._prompts_root / "refinement"),
            )
            ref_agent.set_documents_block(documents_block)

        if hasattr(ref_agent, "set_scope_filter"):
            ref_agent.set_scope_filter(cfg.scope_filter)

        query_hint = self._build_query_hint(question, profile)
        refined = ref_agent.process_context(
            question=query_hint,
            structured_context="",
            ontology=ontology,
            include_ontology=True,
            aql_results_str=aql_results_str if documents_block is None else None,
            context_filter="full",
        )
        enriched = refined.enriched_context or ""

        # 6. generation
        gen_agent = GenerationAgent(
            self._llm(cfg.model_name, cfg.temperature_generate),
            prompt_dir=str(self._prompts_root / "generation"),
        )
        template = gen_agent._load_template(cfg.generation_prompt)
        if not template:
            raise FileNotFoundError(
                f"generation prompt not found: {cfg.generation_prompt}"
            )
        template = template.replace(
            "{answer_shape_directives}",
            _ANSWER_SHAPE_DIRECTIVES.get(profile.answer_shape, ""),
        ).replace(
            "{synthesis_mode_directives}",
            _SYNTHESIS_DIRECTIVES.get(cfg.synthesis_mode, ""),
        )
        gen_agent.refinement_template = template

        ans = gen_agent.generate(
            question=question,
            text_context=enriched,
            ontology=ontology,
            context_cap=cfg.gen_context_cap,
            max_output_tokens=cfg.max_output_tokens,
            system_prompt=cfg.system_prompt_modifier,
        )

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
    # private helpers
    # ------------------------------------------------------------------

    def _llm(self, model: str, temperature: float):
        key = (model, temperature)
        if key not in self._llm_cache:
            from core.utils.helpers import get_llm_model
            self._llm_cache[key] = get_llm_model(
                model=model, temperature=temperature
            )
        return self._llm_cache[key]

    def _ensure_indexer(self) -> None:
        if self._indexer is None:
            from core.utils.fulltext_indexer import FullTextIndexer
            self._indexer = FullTextIndexer(cache_dir=self._cache_dir)

    def _build_evidence(
        self,
        cfg: PipelineConfig,
        question: str,
        ontology: DynamicOntology,
        docs: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if cfg.evidence_mode == "abstracts" or not docs:
            return [], {
                "mode":       cfg.evidence_mode,
                "n_excerpts": 0,
                "n_docs":     len(docs),
            }
        self._ensure_indexer()
        excerpts, stats = self._indexer.select_excerpts_for_question(
            question=question,
            ontology=ontology,
            documents=docs,
            per_doc_budget=cfg.per_doc_budget,
            global_budget=cfg.global_budget,
            top_k_per_doc=cfg.top_k_per_doc,
        )
        stats = dict(stats)
        stats.update({
            "mode":               cfg.evidence_mode,
            "excerpt_chars":      sum(len(e.get("text", "")) for e in excerpts),
            "excerpt_tokens_est": sum(len(e.get("text", "")) for e in excerpts) // 4,
        })
        return excerpts, stats

    def _filter_documents(
        self,
        docs: List[Dict[str, Any]],
        ontology: DynamicOntology,
        profile: QuestionProfile,
        question: str,
        cfg: PipelineConfig,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        if not docs:
            return [], [], []
        from core.agents.ontology_agent import OntologyConstructionAgent
        ont_agent = OntologyConstructionAgent(
            self._llm("mistral-small-latest", 0.0),
            prompt_dir=str(self._prompts_root / "ontology"),
        )
        return ont_agent.filter_documents(
            docs=docs,
            ontology=ontology,
            profile=profile,
            question=question,
            min_keep=cfg.doc_filter_min_keep,
        )

    @staticmethod
    def _build_aql_lookup(
        aql_results_str: str,
        docs: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Build a URI -> metadata dict from both the in-memory docs list
        and the raw aql_results_str (which may contain richer metadata)."""
        lookup: Dict[str, Dict[str, Any]] = {}
        for d in docs:
            uri = d.get("uri", "")
            if uri:
                lookup[uri] = d
        if aql_results_str:
            try:
                import ast as _ast
                raw = _ast.literal_eval(aql_results_str)
                if isinstance(raw, list):
                    for item in raw:
                        uri = item.get("uri", "")
                        if uri and uri not in lookup:
                            lookup[uri] = item
            except Exception:
                pass
        return lookup

    @staticmethod
    def _build_query_hint(question: str, profile: QuestionProfile) -> str:
        hints = []

        # complexity/depth signal
        if profile.complexity is not None:
            if profile.complexity < 0.50:
                hints.append(
                    "[DEPTH: concise — this is a foundational question; "
                    "answer in 3–5 sentences without exhaustive enumeration]"
                )
            elif profile.complexity >= 0.75:
                hints.append(
                    "[DEPTH: comprehensive — this is a high-complexity question; "
                    "provide mechanistic detail, quantitative evidence, and uncertainty assessment]"
                )

        # quantitative emphasis (unchanged)
        if getattr(profile, "needs_numeric_emphasis", False):
            hints.append(
                "[NUMERIC EMPHASIS: prioritise excerpts with quantitative "
                "values, rates, and units]"
            )

        # answer shape (unchanged)
        if profile.answer_shape and profile.answer_shape not in ("direct_paragraph", ""):
            hints.append(f"[SYNTHESIS TARGET: {profile.answer_shape}]")

        return ("\n".join(hints) + "\n" + question) if hints else question