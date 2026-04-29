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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

_RETRACTION_TITLE_TOKENS = frozenset([
    "retracted",
    "retraction",
    "retraction notice",
    "withdrawn",
    "expression of concern",
])


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
    kg_source: str = "csv"


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
        self._indexer = None
        self._session_full_doc_uris: Set[str] = set()
        self._profiler_parse_failures: int = 0
        self._counter_lock = threading.Lock()

    def reset_session_state(self) -> None:
        """Reset run/session state so a new orchestrator.run() starts clean."""
        with self._counter_lock:
            self._session_full_doc_uris = set()
            self._profiler_parse_failures = 0
        log.debug("Pipeline.reset_session_state() called")

    def reset_llm_cache(self) -> None:
        """Clear the LLM instance cache."""
        self._llm_cache = {}
        self._session_full_doc_uris = set()
        log.debug("Pipeline._llm_cache cleared")

    def profile_and_route(
        self, question: str
    ) -> Tuple[DynamicOntology, QuestionProfile, PipelineConfig]:
        from core.agents.ontology_agent import OntologyAgent

        ont_agent = OntologyAgent(
            self._llm("mistral-small-latest", 0.0),
            prompt_dir=str(self._prompts_root / "ontology"),
        )
        ontology, profile = ont_agent.process_with_profile(question)

        if profile.confidence is not None and profile.confidence == 0.0:
            log.warning(
                "profiler parse failure on first attempt -- retrying with temp=0.1 for '%s...'",
                question[:60],
            )
            retry_llm = self._llm("mistral-small-latest", 0.1)
            retry_agent = OntologyAgent(
                retry_llm,
                prompt_dir=str(self._prompts_root / "ontology"),
            )
            ontology, profile = retry_agent.process_with_profile(question)

        cfg = self.router.select(profile)

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

        if not docs:
            raise RuntimeError(
                f"Pipeline.run(): no documents available for question "
                f"'{question[:80]}...' (kg_source={kg_source})."
            )

        # -- 3. document filter -----------------------------------------------
        # Retraction filter runs unconditionally so retracted papers are
        # excluded even when the doc count is at or below doc_filter_min_keep.
        t0 = time.perf_counter()
        docs = self._remove_retracted_papers(docs, question)
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
            _doc_key_snapshot = self._key_snapshot(full_docs, abstract_docs)
            excerpts, excerpt_stats = indexer.select_excerpts_for_question(
                question=question,
                ontology=ontology,
                documents=full_docs,
                per_doc_budget=cfg.per_doc_budget,
                global_budget=cfg.global_budget,
                top_k_per_doc=cfg.top_k_per_doc,
            )
            log_excerpt_stats(log, excerpt_stats, elapsed=time.perf_counter() - t0)
            self._assert_doc_block_ref_alignment(full_docs, abstract_docs, _doc_key_snapshot)
            documents_block = self._render_documents_block(full_docs, abstract_docs, excerpts)
        else:
            log_excerpt_stats(log, {}, elapsed=0.0)

        # -- 5. refinement ----------------------------------------------------
        refine_max_tokens = (
            4000 if cfg.evidence_mode in ("excerpts_narrow", "excerpts_full") else 2000
        )
        refine_llm = self._llm(
            cfg.refinement_model_name or cfg.model_name,
            cfg.temperature_refine,
            max_tokens=refine_max_tokens,
            timeout_s=cfg.timeout_refine_s,
        )
        filtered_aql = self._format_kg_context(full_docs + abstract_docs)

        t0 = time.perf_counter()
        try:
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
                    aql_results_str=filtered_aql,
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
                    aql_results_str=filtered_aql,
                    context_filter="full",
                )
        except Exception as refine_exc:
            refine_msg = str(refine_exc).lower()
            if "context length exceeded" in refine_msg:
                raise RuntimeError(
                    f"Pipeline.run(): context length exceeded during refinement for "
                    f"question '{question[:80]}' (rule={cfg.rule_hit}, "
                    f"model={cfg.refinement_model_name or cfg.model_name})."
                ) from refine_exc
            log.error(
                "refinement raised an exception for '%s...' (rule=%s): %s -- "
                "attempting abstract fallback",
                question[:60], cfg.rule_hit, refine_exc,
            )
            refined = None

        enriched_context = (refined.enriched_context or "") if refined is not None else ""
        log_refinement(log, enriched_context, elapsed=time.perf_counter() - t0)

        if not enriched_context.strip():
            fallback_ctx = "\n\n".join(
                d.get("abstract", "") for d in (full_docs + abstract_docs)
                if d.get("abstract", "").strip()
            )
            if not fallback_ctx.strip():
                raise RuntimeError(
                    f"Pipeline.run(): refinement produced empty enriched_context and no "
                    f"abstract fallback available for question '{question[:80]}'."
                )
            log.warning(
                "refinement returned empty context for '%s...' (rule=%s) -- "
                "falling back to raw abstracts",
                question[:60], cfg.rule_hit,
            )
            enriched_context = fallback_ctx

        # -- 6. generation ----------------------------------------------------
        quality_contract = self._build_answer_quality_contract(profile, cfg)
        system_prompt = (
            (cfg.system_prompt_modifier + "\n\n" + quality_contract).strip()
            if quality_contract
            else cfg.system_prompt_modifier
        )

        gen_llm = self._llm(
            cfg.model_name,
            cfg.temperature_generate,
            max_tokens=cfg.max_output_tokens,
            timeout_s=cfg.timeout_generate_s,
        )
        gen_agent = GenerationAgent(
            gen_llm,
            prompt_dir=str(self._prompts_root / "generation"),
        )
        t0 = time.perf_counter()
        try:
            answer_obj = gen_agent.generate(
                question=question,
                text_context=enriched_context,
                ontology=ontology,
                context_cap=cfg.gen_context_cap,
                max_output_tokens=cfg.max_output_tokens,
                draft_max_tokens=cfg.draft_max_tokens,
                system_prompt=system_prompt,
                use_draft=cfg.use_draft,
                generation_prompt=cfg.generation_prompt,
            )
        except Exception as gen_exc:
            gen_msg = str(gen_exc).lower()
            if "context length exceeded" in gen_msg:
                raise RuntimeError(
                    f"Pipeline.run(): context length exceeded during generation for "
                    f"question '{question[:80]}' (rule={cfg.rule_hit}, "
                    f"model={cfg.model_name})."
                ) from gen_exc
            log.error(
                "generation raised an exception for '%s...' (rule=%s, model=%s): %s",
                question[:60], cfg.rule_hit, cfg.model_name, gen_exc,
            )
            answer_obj = None

        log_generation(log, answer_obj, elapsed=time.perf_counter() - t0)

        if answer_obj is None or not getattr(answer_obj, "answer", "").strip():
            log.error(
                "generation returned no answer for '%s...' (rule=%s, model=%s)",
                question[:60], cfg.rule_hit, cfg.model_name,
            )
            return PipelineResult(
                answer="",
                rule_hit=cfg.rule_hit,
                profile=profile,
                pipeline_config=cfg,
                enriched_context=enriched_context,
                excerpt_stats=excerpt_stats,
                kg_docs_used=len(full_docs) + len(abstract_docs),
                kg_source=kg_source,
            )

        self._audit_numeric_faithfulness(answer_obj.answer, enriched_context, question)

        # -- 7. build verified references + sequential renumbering -----------
        all_docs = full_docs + abstract_docs

        # Normalise first: collapses [doc N] -> [N] and fixes decimal edge cases
        # before cited_indices extraction. answer_obj.cited_indices is derived
        # from the pre-normalised text so we re-extract here from the clean body.
        normalised_body = self._normalize_citation_format(answer_obj.answer)
        cited_indices = self._extract_cited_indices(normalised_body)

        if not cited_indices:
            log.warning(
                "no inline citations extracted for '%s...' (rule=%s) -- "
                "all %d docs will be attached as references",
                question[:60], cfg.rule_hit, len(all_docs),
            )

        fmt_refs, plain_refs, index_remap = self._build_verified_references(
            all_docs, cited_indices
        )
        answer_text = self._renumber_inline_citations(normalised_body, index_remap)
        # Collapse adjacent identical brackets produced by title-dedup
        # (e.g. [3][3] -> [3] when two old indices map to the same new index).
        answer_text = re.sub(r'(\[\d+\])(?:\1)+', r'\1', answer_text)
        answer_text = re.sub(r'(\[\d+\])+\s*$', '', answer_text).rstrip()

        if fmt_refs:
            refs_block = "\n\n## References\n" + "\n".join(fmt_refs)
            answer_text = answer_text.rstrip() + refs_block
        else:
            log.warning(
                "_build_verified_references returned empty list for '%s...' "
                "(rule=%s, n_docs=%d, cited=%s)",
                question[:60], cfg.rule_hit, len(all_docs), sorted(cited_indices),
            )

        with self._counter_lock:
            failure_count = self._profiler_parse_failures
        if failure_count > 0:
            log.info("session profiler_parse_failures=%d", failure_count)

        return PipelineResult(
            answer=answer_text,
            references=plain_refs,
            formatted_references=fmt_refs,
            profile=profile,
            pipeline_config=cfg,
            enriched_context=enriched_context,
            rule_hit=cfg.rule_hit,
            excerpt_stats=excerpt_stats,
            kg_docs_used=len(full_docs) + len(abstract_docs),
            kg_source=kg_source,
        )

    @staticmethod
    def _build_answer_quality_contract(profile: QuestionProfile, cfg: PipelineConfig) -> str:
        rules: List[str] = []
        if cfg.evidence_mode in ("excerpts_narrow", "excerpts_full"):
            rules.append("Every factual claim must be grounded in the provided context passages.")
        quant = getattr(profile, "quantitativity", 0.0) or 0.0
        needs_numeric = getattr(profile, "needs_numeric_emphasis", False)
        if quant >= 0.5 or needs_numeric:
            rules.append(
                "Numeric claims must include: the numeric value, its unit, "
                "the spatial/temporal scope it applies to, and an inline citation [N]."
            )
            rules.append(
                "Do not paraphrase numeric values -- state them exactly as reported in the sources."
            )
        if (getattr(profile, "spatial_specificity", 0.0) or 0.0) >= 0.5:
            rules.append(
                "State the spatial/geographic scope explicitly "
                "(e.g., basin name, country, coordinates) for every spatial claim."
            )
        if (getattr(profile, "temporal_specificity", 0.0) or 0.0) >= 0.5:
            rules.append(
                "State the time period or observation window explicitly "
                "for every temporal or trend claim."
            )
        if (getattr(profile, "methodological_depth", 0.0) or 0.0) >= 0.6:
            rules.append(
                "When describing methods, name the specific method and state its "
                "key limitations or applicability constraints."
            )
        if not rules:
            return ""
        header = "ANSWER QUALITY CONTRACT (follow strictly):"
        numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
        return f"{header}\n{numbered}"

    @staticmethod
    def _normalize_citation_format(text: str) -> str:
        # Collapse [doc N] / [Doc N] tokens emitted by the refinement agent
        # into plain [N] so all downstream passes see a uniform format.
        text = re.sub(r"\[(?:doc|Doc)\s+(\d+)\]", r"[\1]", text)

        def _expand_line_start(m: re.Match) -> str:
            nums = re.split(r"[\s,]+", m.group(1).strip())
            valid = [n for n in nums if n.isdigit() and 1 <= int(n) <= 30]
            if not valid:
                return m.group(0)
            return "".join(f"[{n}]" for n in valid)

        text = re.sub(
            r"(?m)^(\d{1,2}(?:\s*,\s*\d{1,2}){1,4})(?=\s+[A-Za-z])",
            _expand_line_start,
            text,
        )

        def _expand_multi_bracket(m: re.Match) -> str:
            nums = re.split(r"[\s,]+", m.group(1).strip())
            return "".join(f"[{n}]" for n in nums if n.isdigit())

        text = re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]", _expand_multi_bracket, text)

        def _expand_bare_cluster(m: re.Match) -> str:
            nums = re.split(r"[\s,]+", m.group(2).strip())
            valid = [n for n in nums if n.isdigit() and 1 <= int(n) <= 30]
            if not valid:
                return m.group(0)
            return m.group(1) + "".join(f"[{n}]" for n in valid)

        # Lookahead excludes `.\d` so a decimal like 13.5 is never mis-parsed
        # into citation [13] followed by orphaned .5.
        text = re.sub(
            r"([a-zA-Z\)\]%] )(\d{1,2}(?:\s*,\s*\d{1,2}){0,4})"
            r"(?=\s*(?:(?:\.(?!\d))|\n|,|;|$|\s*[-]{2,}|\s*\[))",
            _expand_bare_cluster,
            text,
        )
        return text

    @staticmethod
    def _extract_cited_indices(answer_body: str) -> Set[int]:
        """Return the set of 1-based integer doc indices cited inline.

        Operates on the pipeline-normalised body (after [doc N] -> [N]
        conversion) so all citation forms are captured before
        _build_verified_references runs.
        """
        indices: Set[int] = set()
        for bracket in re.findall(r"\[([\d,\s]+)\]", answer_body):
            for token in bracket.split(","):
                token = token.strip()
                if token.isdigit():
                    indices.add(int(token))
        return indices

    @staticmethod
    def _build_verified_references(
        docs: List[Dict[str, Any]],
        cited_indices: Set[int],
    ) -> Tuple[List[str], List[str], Dict[int, int]]:
        from core.utils.openalex_client import OpenAlexClient, format_reference_from_metadata

        if not cited_indices:
            indices_to_use = set(range(1, len(docs) + 1))
        else:
            indices_to_use = {i for i in cited_indices if 1 <= i <= len(docs)}
            if not indices_to_use:
                log.warning(
                    "_build_verified_references: all cited indices %s out of range "
                    "(n_docs=%d) -- attaching all docs",
                    sorted(cited_indices), len(docs),
                )
                indices_to_use = set(range(1, len(docs) + 1))

        formatted: List[str] = []
        plain: List[str] = []
        index_remap: Dict[int, int] = {}
        seq = 0
        seen_norm_titles: Set[str] = set()
        norm_title_to_seq: Dict[str, int] = {}

        for i in sorted(indices_to_use):
            doc = docs[i - 1]
            title = doc.get("title_or_name") or doc.get("title") or ""
            uri   = doc.get("uri") or doc.get("id") or ""

            if not title and not uri:
                log.warning(
                    "_build_verified_references: doc %d has no title or URI -- skipping", i
                )
                continue

            norm_title = re.sub(r"\W+", " ", title[:100].lower()).strip()

            if norm_title and norm_title in seen_norm_titles:
                index_remap[i] = norm_title_to_seq[norm_title]
                continue

            if norm_title:
                seen_norm_titles.add(norm_title)

            seq += 1
            index_remap[i] = seq

            if norm_title:
                norm_title_to_seq[norm_title] = seq

            metadata = None
            if uri and "openalex.org" in uri:
                metadata = OpenAlexClient.fetch_metadata(uri)

            if metadata and not (metadata.get("title") or "").strip():
                metadata = None

            if metadata:
                meta = dict(metadata)
                meta["position"] = seq
                line = format_reference_from_metadata(meta)
                formatted.append(line)
                plain.append(re.sub(r"^\[\d+\]\s*", "", line))
            else:
                fallback_title = title or "[No title -- see URI]"
                line = f"[{seq}] {fallback_title}. {uri}." if uri else f"[{seq}] {fallback_title}."
                formatted.append(line)
                plain.append(re.sub(r"^\[\d+\]\s*", "", line))

        return formatted, plain, index_remap

    @staticmethod
    def _renumber_inline_citations(answer_body: str, index_remap: Dict[int, int]) -> str:
        if not index_remap:
            return answer_body
        result = answer_body
        for old, new in sorted(index_remap.items(), key=lambda kv: -kv[0]):
            result = re.sub(r"\[" + str(old) + r"\]", f"__CITE_{new}__", result)
        result = re.sub(r"__CITE_(\d+)__", r"[\1]", result)
        return result

    @staticmethod
    def _key_snapshot(
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
    ) -> List[str]:
        def _key(doc: Dict[str, Any]) -> str:
            return doc.get("uri") or doc.get("id") or doc.get("title") or ""
        return [_key(d) for d in full_docs] + [_key(d) for d in abstract_docs]

    @staticmethod
    def _assert_doc_block_ref_alignment(
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
        snapshot: List[str],
    ) -> None:
        def _key(doc: Dict[str, Any]) -> str:
            return doc.get("uri") or doc.get("id") or doc.get("title") or ""
        current = [_key(d) for d in full_docs] + [_key(d) for d in abstract_docs]
        if len(snapshot) != len(current):
            raise RuntimeError(
                f"Pipeline._assert_doc_block_ref_alignment: list length mismatch -- "
                f"snapshot={len(snapshot)}, current={len(current)}."
            )
        for pos, (snap_key, cur_key) in enumerate(zip(snapshot, current), 1):
            if snap_key != cur_key:
                raise RuntimeError(
                    f"Pipeline._assert_doc_block_ref_alignment: ordering divergence "
                    f"at position {pos}. snapshot={snap_key!r}, current={cur_key!r}."
                )

    @staticmethod
    def _remove_retracted_papers(
        docs: List[Dict[str, Any]],
        question: str,
    ) -> List[Dict[str, Any]]:
        q_lower = question.lower()
        if "retract" in q_lower or "withdrawn" in q_lower or "expression of concern" in q_lower:
            return docs
        clean: List[Dict[str, Any]] = []
        for doc in docs:
            title_lower = (doc.get("title") or "").lower()
            status_lower = (doc.get("status") or "").lower()
            flagged = False
            for token in _RETRACTION_TITLE_TOKENS:
                if re.search(r"(?:^|[\s:\-])" + re.escape(token), title_lower):
                    flagged = True
                    break
                if token in status_lower:
                    flagged = True
                    break
            if flagged:
                log.warning(
                    "retraction_filter: excluding '%s'",
                    (doc.get("title") or "")[:80],
                )
            else:
                clean.append(doc)
        return clean

    def _filter_documents(
        self,
        docs: List[Dict[str, Any]],
        ontology: DynamicOntology,
        profile: QuestionProfile,
        question: str,
        cfg: PipelineConfig,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        # Note: retraction filtering is done in pipeline.run() before this
        # method is called, so docs here are already retraction-clean.
        from core.agents.ontology_agent import OntologyAgent
        filter_agent = OntologyAgent(
            self._llm("mistral-small-latest", 0.0),
            prompt_dir=str(self._prompts_root / "ontology"),
        )
        full_docs, abstract_docs, drop_docs = filter_agent.filter_documents(
            docs=docs,
            ontology=ontology,
            profile=profile,
            question=question,
            min_keep=cfg.doc_filter_min_keep,
            evidence_mode=cfg.evidence_mode,
            rule_hit=cfg.rule_hit,
        )
        surplus = len(full_docs) - (cfg.doc_filter_min_keep + 2)
        with self._counter_lock:
            session_uris_snapshot = set(self._session_full_doc_uris)
        if surplus > 0 and session_uris_snapshot:
            demoted = 0
            new_full: List[Dict] = []
            new_abstract: List[Dict] = list(abstract_docs)
            for doc in full_docs:
                uri = doc.get("uri") or doc.get("id") or ""
                if uri and uri in session_uris_snapshot and demoted < surplus:
                    new_abstract.append(doc)
                    demoted += 1
                else:
                    new_full.append(doc)
            full_docs = new_full
            abstract_docs = new_abstract
        new_uris = {
            doc.get("uri") or doc.get("id") or ""
            for doc in full_docs
            if doc.get("uri") or doc.get("id")
        }
        with self._counter_lock:
            self._session_full_doc_uris.update(new_uris)
        return full_docs, abstract_docs, drop_docs

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
    def _render_documents_block(
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
        excerpts: List[Any],
    ) -> str:
        from core.utils.fulltext_indexer import FullTextIndexer
        # Build aql_lookup from the docs themselves so render_documents_block
        # can enrich title/abstract/year/authors without a live DB call.
        all_docs = list(full_docs) + list(abstract_docs)
        aql_lookup: Dict[str, Dict[str, Any]] = {
            doc["uri"]: doc
            for doc in all_docs
            if doc.get("uri")
        }
        return FullTextIndexer.render_documents_block(
            full_docs, abstract_docs, excerpts, aql_lookup
        )

    @staticmethod
    def _build_query_hint(question: str, profile: QuestionProfile) -> str:
        parts = [question]
        quant = getattr(profile, "quantitativity", 0.0) or 0.0
        if getattr(profile, "needs_numeric_emphasis", False) or quant >= 0.5:
            parts.append("[emphasis: numeric/quantitative precision]")
        if (getattr(profile, "methodological_depth", 0.0) or 0.0) > 0.6:
            parts.append("[emphasis: methodological detail]")
        return " ".join(parts)

    @staticmethod
    def _audit_numeric_faithfulness(
        answer_text: str,
        context: str,
        question: str,
        *,
        max_year: int = 2030,
        min_year: int = 1970,
    ) -> None:
        answer_years = set(
            int(m) for m in re.findall(r"\b((?:19|20)\d{2})\b", answer_text)
            if min_year <= int(m) <= max_year
        )
        if not answer_years:
            return
        context_years = set(
            int(m) for m in re.findall(r"\b((?:19|20)\d{2})\b", context)
            if min_year <= int(m) <= max_year
        )
        unsupported = answer_years - context_years
        if unsupported:
            log.warning(
                "_audit_numeric_faithfulness: year(s) %s in answer not found in context "
                "(question='%s...'). May be hallucinated.",
                sorted(unsupported), question[:60],
            )


AdaptivePipeline = Pipeline
