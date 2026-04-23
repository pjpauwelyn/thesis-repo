"""pipeline -- routes each question to the right evidence mode and model.

Routing is determined at runtime by the policy router based on the
question profile.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.policy.router import Router
from core.utils.data_models import (
    DynamicOntology,
    PipelineConfig,
    QuestionProfile,
)
from core.utils.logger import (
    log_doc_filter,
    log_excerpt_stats,
    log_generation,
    log_ontology,
    log_profile_and_route,
    log_refinement,
)

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    answer: str
    references: List[str] = field(default_factory=list)
    formatted_references: List[str] = field(default_factory=list)
    profile: Optional[QuestionProfile] = None
    pipeline_config: Optional[PipelineConfig] = None
    enriched_context: str = ""
    rule_hit: str = ""
    excerpt_stats: Dict[str, Any] = field(default_factory=dict)
    kg_docs_used: int = 0
    kg_source: str = "csv"         # "live" | "csv" | "narrow_csv" | "none"


# backward-compat alias
AdaptiveResult = PipelineResult


class Pipeline:
    def __init__(
        self,
        rules_path: Union[str, Path] = "core/policy/rules.yaml",
        cache_dir: Union[str, Path] = "cache/fulltext",
        prompts_root: Union[str, Path] = "prompts",
    ):
        self.router = Router(rules_path)
        self._cache_dir = Path(cache_dir)
        self._prompts_root = Path(prompts_root)
        self._llm_cache: Dict[Tuple[str, float, int, int], Any] = {}
        self._indexer = None  # lazy

        # Profiler parse-failure counter (thread-safe).
        # Incremented whenever the profiler LLM returns malformed JSON and
        # confidence falls back to 0.0, causing silent escalation to
        # safety-tier3 (most expensive model + evidence path).
        self._profiler_parse_failures: int = 0
        self._counter_lock = threading.Lock()

    def profile_and_route(
        self, question: str
    ) -> Tuple[DynamicOntology, QuestionProfile, PipelineConfig]:
        from core.agents.ontology_agent import OntologyAgent
        ont_agent = OntologyAgent(
            self._llm("mistral-small-latest", 0.0),
            prompt_dir=str(self._prompts_root / "ontology"),
        )
        ontology, profile = ont_agent.process_with_profile(question)
        cfg = self.router.select(profile)

        # Track questions where the profiler returned malformed JSON.
        # confidence == 0.0 is the sentinel that OntologyAgent sets on parse
        # failure before falling back to the safety-tier3 catch-all rule.
        if profile.confidence is not None and profile.confidence == 0.0:
            with self._counter_lock:
                self._profiler_parse_failures += 1
                count = self._profiler_parse_failures
            log.warning(
                "profiler JSON parse failure for question '%s...' "
                "(null-confidence fallback -> safety-tier3). "
                "Session total: %d",
                question[:60], count,
            )

        return ontology, profile, cfg

    def profile_and_route_with_filter(
        self,
        question: str,
        docs: Optional[List[Dict[str, Any]]] = None,
        aql_results_str: Optional[str] = None,
    ) -> Tuple[DynamicOntology, QuestionProfile, PipelineConfig, Dict[str, Any]]:
        """Profile + route + document filter without generation."""
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
        aql_results_str: str = "",
        docs: Optional[List[Dict[str, Any]]] = None,
        aql_params: Optional[Dict[str, Any]] = None,
        precomputed_route: Optional[Tuple[DynamicOntology, QuestionProfile, PipelineConfig]] = None,
    ) -> PipelineResult:
        """Run the full pipeline for one question.

        Retrieval order:
          1. Live ArangoDB KG  (if ARANGO_ROOT_PASSWORD is set)
          2. Pre-parsed docs   (if caller already passed a docs list)
          3. aql_results_str   (CSV fallback, JSON then ast.literal_eval)

        Pass precomputed_route to skip a redundant profiling LLM call when
        the caller has already run profile_and_route().
        """
        from core.agents.generation_agent import GenerationAgent
        from core.agents.refinement_agent_abstracts import RefinementAgentAbstracts
        from core.agents.refinement_agent_fulltext import RefinementAgent1PassFullText
        from core.utils.aql_parser import parse_aql_results

        # -- 1. profile + route -----------------------------------------------
        t0 = time.perf_counter()
        if precomputed_route is not None:
            ontology, profile, cfg = precomputed_route
            log.debug(
                "q=%s... -> rule=%s refine=%s gen=%s evidence=%s (precomputed route)",
                question[:60], cfg.rule_hit,
                cfg.refinement_model_name or cfg.model_name,
                cfg.model_name, cfg.evidence_mode,
            )
        else:
            ontology, profile, cfg = self.profile_and_route(question)
            log.debug(
                "q=%s... -> rule=%s refine=%s gen=%s evidence=%s",
                question[:60], cfg.rule_hit,
                cfg.refinement_model_name or cfg.model_name,
                cfg.model_name, cfg.evidence_mode,
            )
        ontology_elapsed = time.perf_counter() - t0
        log_ontology(log, ontology, elapsed=ontology_elapsed)
        log_profile_and_route(log, profile, cfg, elapsed=ontology_elapsed)

        # -- 2. retrieve documents --------------------------------------------
        kg_source = "none"
        live_docs = self._try_live_kg(question, ontology, aql_params)
        if live_docs:
            kg_source = "live"
            docs = live_docs
            aql_results_str = self._format_kg_context(live_docs)
            log.info(
                "  KG-live   %d primary docs  %d total nodes",
                len(live_docs),
                sum(
                    1 + len(d.get("science_keywords", []))
                    + sum(len(sk.get("secondary_nodes", []))
                          for sk in d.get("science_keywords", []))
                    for d in live_docs
                ),
            )
        elif docs:
            kg_source = "csv"
            log.info("  KG-csv    %d pre-parsed docs", len(docs))
        else:
            if aql_results_str:
                parsed = parse_aql_results(aql_results_str)
                docs = self._parse_docs_from_str(parsed)
            else:
                docs = []
            kg_source = "csv" if docs else "none"
            if docs:
                log.info("  KG-csv    %d docs parsed from aql_results_str", len(docs))
            else:
                log.warning(
                    "  KG        no docs available (aql_results_str len=%d)",
                    len(aql_results_str or ""),
                )

        if not docs:
            raise RuntimeError(
                f"Pipeline.run(): no documents available for question "
                f"'{question[:80]}...' (kg_source={kg_source}). "
                "Cannot proceed -- refinement and generation require document context."
            )

        # -- 3. document filter -----------------------------------------------
        t0 = time.perf_counter()
        full_docs: List[Dict] = list(docs)
        abstract_docs: List[Dict] = []
        drop_docs: List[Dict] = []

        if len(docs) > cfg.doc_filter_min_keep:
            full_docs, abstract_docs, drop_docs = self._filter_documents(
                docs, ontology, profile, question, cfg
            )
        log_doc_filter(log, full_docs, abstract_docs, drop_docs, elapsed=time.perf_counter() - t0)

        # -- 4. excerpt selection ---------------------------------------------
        excerpts: List[Any] = []
        excerpt_stats: Dict[str, Any] = {}
        documents_block = ""

        if cfg.evidence_mode in ("excerpts_narrow", "excerpts_full"):
            t0 = time.perf_counter()
            indexer = self._get_indexer()
            excerpts, excerpt_stats = indexer.select_excerpts_for_question(
                question=question,
                ontology=ontology,
                documents=full_docs,
                per_doc_budget=cfg.per_doc_budget,
                global_budget=cfg.global_budget,
                top_k_per_doc=cfg.top_k_per_doc,
            )
            log_excerpt_stats(log, excerpt_stats, elapsed=time.perf_counter() - t0)
            aql_lookup = self._build_aql_lookup(aql_results_str, docs)
            documents_block = self._render_documents_block(
                full_docs, abstract_docs, excerpts, aql_lookup
            )
        else:
            log_excerpt_stats(log, {}, elapsed=0.0)

        # -- 5. refinement ----------------------------------------------------
        refine_llm = self._llm(
            cfg.refinement_model_name or cfg.model_name,
            cfg.temperature_refine,
            timeout_s=cfg.timeout_refine_s,
        )

        t0 = time.perf_counter()
        if cfg.evidence_mode == "abstracts":
            refine_agent = RefinementAgentAbstracts(
                refine_llm,
                prompt_dir=str(self._prompts_root / "refinement"),
            )
            query_hint = self._build_query_hint(question, profile)
            refined = refine_agent.process_context(
                question=query_hint,
                structured_context="",
                ontology=ontology,
                include_ontology=True,
                aql_results_str=aql_results_str,
                context_filter="full",
            )
        else:
            refine_agent = RefinementAgent1PassFullText(
                refine_llm,
                prompt_dir=str(self._prompts_root / "refinement"),
            )
            refine_agent.set_documents_block(documents_block)
            query_hint = self._build_query_hint(question, profile)
            refined = refine_agent.process_context(
                question=query_hint,
                structured_context="",
                ontology=ontology,
                include_ontology=True,
                aql_results_str=None,
                context_filter="full",
            )

        enriched_context = refined.enriched_context or ""
        log_refinement(log, enriched_context, elapsed=time.perf_counter() - t0)

        if not enriched_context.strip():
            raise RuntimeError(
                f"Pipeline.run(): refinement produced empty enriched_context for "
                f"question '{question[:80]}'. Aborting generation."
            )

        # -- 6. generation ----------------------------------------------------
        gen_llm = self._llm(
            cfg.model_name,
            cfg.temperature_generate,
            timeout_s=cfg.timeout_generate_s,
        )
        gen_agent = GenerationAgent(
            gen_llm,
            prompt_dir=str(self._prompts_root / "generation"),
        )
        t0 = time.perf_counter()
        answer_obj = gen_agent.generate(
            question=question,
            text_context=enriched_context,
            ontology=ontology,
            context_cap=cfg.gen_context_cap,
            max_output_tokens=cfg.max_output_tokens,
            system_prompt=cfg.system_prompt_modifier,
            use_draft=cfg.use_draft,
        )
        log_generation(log, answer_obj, elapsed=time.perf_counter() - t0)

        # Log session profiler failure count after every question so operators
        # can spot misrouting early without waiting for the run to finish.
        with self._counter_lock:
            failure_count = self._profiler_parse_failures
        if failure_count > 0:
            log.info(
                "session profiler_parse_failures=%d "
                "(questions silently escalated to safety-tier3 "
                "due to profiler JSON errors)",
                failure_count,
            )

        return PipelineResult(
            answer=answer_obj.answer,
            references=answer_obj.references,
            formatted_references=answer_obj.formatted_references,
            profile=profile,
            pipeline_config=cfg,
            enriched_context=enriched_context,
            rule_hit=cfg.rule_hit,
            excerpt_stats=excerpt_stats,
            kg_docs_used=len(full_docs) + len(abstract_docs),
            kg_source=kg_source,
        )

    def _filter_documents(
        self,
        docs: List[Dict[str, Any]],
        ontology: DynamicOntology,
        profile: QuestionProfile,
        question: str,
        cfg: PipelineConfig,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        from core.agents.ontology_agent import OntologyAgent
        filter_agent = OntologyAgent(
            self._llm("mistral-small-latest", 0.0),
            prompt_dir=str(self._prompts_root / "ontology"),
        )
        return filter_agent.filter_documents(
            docs=docs,
            ontology=ontology,
            profile=profile,
            question=question,
            min_keep=cfg.doc_filter_min_keep,
            evidence_mode=cfg.evidence_mode,
            rule_hit=cfg.rule_hit,
        )

    def _get_indexer(self):
        if self._indexer is None:
            from core.utils.fulltext_indexer import FullTextIndexer
            self._indexer = FullTextIndexer(cache_dir=self._cache_dir)
        return self._indexer

    def _llm(self, model: str, temperature: float, max_tokens: int = 1400, timeout_s: Optional[int] = None):
        from core.utils.helpers import get_llm_model
        key = (model, temperature, max_tokens, timeout_s or 0)
        if key not in self._llm_cache:
            self._llm_cache[key] = get_llm_model(
                model, temperature, max_tokens, timeout_s=timeout_s
            )
        return self._llm_cache[key]

    @staticmethod
    def _try_live_kg(
        question: str,
        ontology: DynamicOntology,
        aql_params: Optional[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        import os
        if not os.getenv("ARANGO_ROOT_PASSWORD"):
            return None
        try:
            from core.kg.arango_retriever import ArangoRetriever
            retriever = ArangoRetriever()
            return retriever.retrieve(question, ontology, aql_params)
        except Exception as exc:
            log.warning("live KG retrieval failed, falling back to CSV: %s", exc)
            return None

    @staticmethod
    def _format_kg_context(docs: List[Dict[str, Any]]) -> str:
        try:
            return json.dumps(docs, ensure_ascii=False)
        except Exception:
            return str(docs)

    @staticmethod
    def _parse_docs_from_str(parsed_str: str) -> List[Dict[str, Any]]:
        if not parsed_str:
            return []
        try:
            result = json.loads(parsed_str)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(parsed_str)
            if isinstance(result, list):
                return result
        except Exception:
            pass
        return []

    @staticmethod
    def _build_aql_lookup(
        aql_results_str: str,
        docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        lookup: Dict[str, Any] = {}
        for doc in docs:
            key = doc.get("uri") or doc.get("id") or doc.get("title", "")
            if key:
                lookup[key] = doc
        return lookup

    @staticmethod
    def _render_documents_block(
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
        excerpts: List[Any],
        aql_lookup: Dict[str, Any],
    ) -> str:
        excerpt_map: Dict[str, List[Any]] = {}
        for exc in excerpts:
            key = exc.get("work_id") or exc.get("uri") or ""
            if key:
                excerpt_map.setdefault(key, []).append(exc)

        lines: List[str] = []
        for i, doc in enumerate(full_docs, 1):
            title = doc.get("title", f"Document {i}")
            uri = doc.get("uri") or doc.get("id") or ""
            lines.append(f"\n=== Document [{i}]: {title} ===")
            if uri:
                lines.append(f"URI: {uri}")
            doc_excerpts = excerpt_map.get(uri, [])
            if doc_excerpts:
                for exc in doc_excerpts:
                    section = exc.get("section", "")
                    text = exc.get("text", "")
                    lines.append(f"\n<<< Section: {section} >>>")
                    lines.append(text)
            else:
                abstract = doc.get("abstract", "")
                if abstract:
                    lines.append("\n[abstract only]")
                    lines.append(abstract)

        for i, doc in enumerate(abstract_docs, len(full_docs) + 1):
            title = doc.get("title", f"Document {i}")
            uri = doc.get("uri") or doc.get("id") or ""
            lines.append(f"\n=== Document [{i}]: {title} (abstract) ===")
            if uri:
                lines.append(f"URI: {uri}")
            abstract = doc.get("abstract", "")
            if abstract:
                lines.append(abstract)

        return "\n".join(lines)

    @staticmethod
    def _build_query_hint(question: str, profile: QuestionProfile) -> str:
        parts = [question]
        if getattr(profile, "needs_numeric_emphasis", False):
            parts.append("[emphasis: numeric/quantitative precision]")
        if getattr(profile, "methodological_depth", 0) > 0.6:
            parts.append("[emphasis: methodological detail]")
        return " ".join(parts)


# backward-compat alias
AdaptivePipeline = Pipeline
