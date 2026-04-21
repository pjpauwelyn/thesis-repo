"""adaptive pipeline -- routes per-question to the right evidence mode + model.

zero changes to exp 1-4 pipelines. all behaviour is selected at runtime by
the policy router based on the question profile.

flow per question:
  1. profile     -- fused av-pair + profile call (small, temp=0)
  2. route       -- pure-python policy router selects PipelineConfig
  3. retrieve    -- live ArangoDB KG (graph expansion) if available,
                    else fall back to pre-baked aql_results_str from CSV
  4. evidence    -- abstracts (no extra work) | excerpts via FullTextIndexer
  5. refinement  -- 1pass-refined or 1pass-fulltext, depending on evidence
  6. generation  -- direct or structured prompt, with shape + synthesis directives

KG retrieval is activated automatically when ARANGO_URL + ARANGO_ROOT_PASSWORD
are present in the environment.  when they are absent the pipeline falls back
to the existing CSV-based aql_results_str path unchanged.
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
    kg_docs_used: int = 0           # how many primary KG docs were retrieved
    kg_source: str = "csv"          # "live" | "csv" | "none"


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
        aql_results_str: str = "",
        docs: Optional[List[Dict[str, Any]]] = None,
        aql_params: Optional[Dict[str, Any]] = None,
    ) -> AdaptiveResult:
        """run the full adaptive pipeline for one question.

        retrieval order:
          1. live ArangoDB KG  (if ARANGO_ROOT_PASSWORD is set)
          2. pre-parsed docs   (if caller already passed docs list)
          3. aql_results_str   (CSV fallback)
        """
        from core.agents.generation_agent import GenerationAgent
        from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
        from core.agents.refinement_agent_fulltext import RefinementAgent1PassFullText
        from core.utils.aql_parser import parse_aql_results

        ontology, profile, cfg = self.profile_and_route(question)
        log.info(
            "q=%s... -> rule=%s model=%s evidence=%s",
            question[:60], cfg.rule_hit, cfg.model_name, cfg.evidence_mode,
        )

        # ----------------------------------------------------------
        # retrieval: live KG > caller docs > CSV fallback
        # ----------------------------------------------------------
        kg_source = "none"
        live_docs = self._try_live_kg(question, ontology, aql_params)
        if live_docs:
            kg_source = "live"
            docs = live_docs
            aql_results_str = self._format_kg_context(live_docs)
            log.info("KG live: %d primary docs, %d total nodes",
                     len(live_docs),
                     sum(1 + len(d.get("science_keywords", [])) +
                         sum(len(sk.get("secondary_nodes", []))
                             for sk in d.get("science_keywords", []))
                         for d in live_docs))
        elif docs:
            kg_source = "csv"
        else:
            kg_source = "csv"
            if aql_results_str:
                parsed = parse_aql_results(aql_results_str)
                try:
                    docs = json.loads(parsed)
                    if not isinstance(docs, list):
                        docs = []
                except json.JSONDecodeError:
                    docs = []

        # ----------------------------------------------------------
        # evidence (fulltext excerpts when evidence_mode != abstracts)
        # ----------------------------------------------------------
        excerpts_text, excerpt_stats = self._build_evidence(cfg, question, ontology, docs)

        # ----------------------------------------------------------
        # refinement
        # ----------------------------------------------------------
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

        if hasattr(ref_agent, "set_scope_filter"):
            ref_agent.set_scope_filter(cfg.scope_filter)

        # when we have live KG docs, pass the rich context string directly;
        # otherwise fall through to the existing aql_results_str path
        context_for_refinement = (
            aql_results_str if kg_source == "live"
            else (aql_results_str or "")
        )

        refined = ref_agent.process_context(
            question=question,
            structured_context="",
            ontology=ontology,
            include_ontology=True,
            aql_results_str=context_for_refinement,
            context_filter="full",
        )
        enriched = refined.enriched_context or ""

        # ----------------------------------------------------------
        # generation
        # ----------------------------------------------------------
        gen_agent = GenerationAgent(
            self._llm(cfg.model_name, cfg.temperature_generate),
            prompt_dir=str(self._prompts_root / "generation"),
        )
        template = gen_agent._load_template(cfg.generation_prompt)
        if not template:
            raise FileNotFoundError(f"generation prompt not found: {cfg.generation_prompt}")
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
            kg_docs_used=len(live_docs) if live_docs else (len(docs) if docs else 0),
            kg_source=kg_source,
        )

    # ------------------------------------------------------------------
    # KG retrieval
    # ------------------------------------------------------------------

    def _try_live_kg(
        self,
        question: str,
        ontology: DynamicOntology,
        aql_params: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """attempt live ArangoDB query; return [] if unavailable or unconfigured."""
        import os
        if not os.getenv("ARANGO_ROOT_PASSWORD"):
            return []

        try:
            from core.utils.arango_client import query_kg
        except ImportError:
            return []

        # build a keyword-focused search string from the ontology + question
        search_terms = self._build_search_query(question, ontology)

        params = aql_params or {}
        return query_kg(
            search_query=search_terms,
            kappa=params.get("n", 16),
            phi=params.get("k", 3),
            psi=params.get("s", 2),
            theta_expand=params.get("k_threshold", 0.3),
        )

    def _build_search_query(self, question: str, ontology: DynamicOntology) -> str:
        """combine question keywords with ontology attributes for BM25 search."""
        parts = [question]
        try:
            if hasattr(ontology, "attributes") and ontology.attributes:
                parts += [str(a) for a in ontology.attributes[:6]]
            if hasattr(ontology, "relationships") and ontology.relationships:
                parts += [str(r) for r in ontology.relationships[:4]]
        except Exception:
            pass
        return " ".join(parts)

    @staticmethod
    def _format_kg_context(docs: List[Dict[str, Any]]) -> str:
        """render the full KG neighbourhood as a structured text block.

        structure mirrors DLR's _structure_context() but keeps keywords and
        secondary nodes as named sections so the refinement agent can reason
        over graph-adjacent documents.
        """
        sections: List[str] = []
        for doc in docs:
            lines = [f"## {doc.get('title', 'Untitled')}"]
            if doc.get("abstract"):
                lines.append(doc["abstract"])
            if doc.get("uri"):
                lines.append(f"URI: {doc['uri']}")

            for sk in doc.get("science_keywords", []):
                kw_name = sk.get("name", "")
                kw_desc = sk.get("description", "")
                lines.append(f"\n### Keyword: {kw_name}")
                if kw_desc:
                    lines.append(kw_desc)
                for sn in sk.get("secondary_nodes", []):
                    sn_title = sn.get("title", "")
                    sn_abstract = sn.get("abstract", "")
                    if sn_title or sn_abstract:
                        lines.append(f"  - **{sn_title}**: {sn_abstract[:300]}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

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
