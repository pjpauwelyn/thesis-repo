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

# Fix 9: retraction indicator tokens matched against the *title* (lower-cased).
# Conservative list — only exact editorial labels, not words that appear in
# ordinary paper titles or abstracts discussing retractions as a topic.
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

        # P6: session-level set of document URIs that have already been used
        # as full-text in a prior question. Used by _filter_documents to demote
        # repeated full-text docs to abstract-only, preventing the same passage
        # from dominating multiple answers (e.g. Q19/Q20 Yellow River overlap).
        self._session_full_doc_uris: Set[str] = set()

        # Profiler parse-failure counter (thread-safe).
        # Incremented whenever the profiler LLM returns malformed JSON and
        # confidence falls back to 0.0, causing silent escalation to
        # safety-tier3 (most expensive model + evidence path).
        self._profiler_parse_failures: int = 0
        self._counter_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Fix 8: session reset boundary
    # ------------------------------------------------------------------

    def reset_session_state(self) -> None:
        """Reset run/session state so a new orchestrator.run() starts clean.

        Clears _session_full_doc_uris and _profiler_parse_failures.
        Does NOT clear _llm_cache (LLM clients are expensive to recreate and
        carry no cross-question state beyond what the underlying model provides).

        Call this at the start of PipelineOrchestrator.run() for adaptive
        runs.  Do NOT call between individual questions inside a run.
        """
        with self._counter_lock:
            self._session_full_doc_uris = set()
            self._profiler_parse_failures = 0
        log.debug("Pipeline.reset_session_state() called — session URIs and parse-failure counter cleared")

    def reset_llm_cache(self) -> None:
        """Clear the LLM instance cache.

        Call this between questions in a multi-question test loop to force
        fresh LLM client instances for each question.  This prevents any
        accumulated conversation state in the underlying client object from
        leaking a prior question's draft into the next refinement context
        (draft-bleed guard).

        Also resets the session full-doc URI throttle so each test run starts
        with a clean slate (P6).

        Safe to call at any point; does not affect routing, docs, or results.
        """
        self._llm_cache = {}
        self._session_full_doc_uris = set()
        log.debug("Pipeline._llm_cache cleared (reset_llm_cache called)")

    def profile_and_route(
        self, question: str
    ) -> Tuple[DynamicOntology, QuestionProfile, PipelineConfig]:
        from core.agents.ontology_agent import OntologyAgent

        ont_agent = OntologyAgent(
            self._llm("mistral-small-latest", 0.0),
            prompt_dir=str(self._prompts_root / "ontology"),
        )
        ontology, profile = ont_agent.process_with_profile(question)

        # Fix 9: single automatic retry on profiler parse failure — use a
        # slightly warmer temperature to escape the deterministic zero-temp
        # failure loop. A retry at temp=0.0 produces the exact same malformed
        # JSON every time; temp=0.1 breaks the deterministic cycle.
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

        Reference contract (post-generation, Python-side)
        -------------------------------------------------
        After generation the pipeline calls _build_verified_references() which
        constructs formatted_references entirely from the known-good doc list
        (full_docs + abstract_docs) filtered to the indices cited by the LLM.
        The LLM's own ## References output (if any) is stripped by the agent
        before we receive the answer body.  This makes references deterministic
        and hallucination-proof: the LLM only writes [N] markers inline.

        Before renumbering, _normalize_citation_format() converts any bare
        comma-lists ("word 3, 7") or multi-index brackets ("[5,6,7]") that the
        LLM produced into canonical "[N][M]" form.  Inline markers are then
        renumbered to sequential [1]..[K] so that if the LLM cited doc indices
        {3,7,14} the output shows [1][2][3] and the ## References section lists
        [1]..[3] to match.
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

            # Fix 1: take a snapshot of doc URI/title keys BEFORE excerpt
            # selection runs so the alignment guard has a real before/after
            # window to compare against (not list-to-itself at the same instant).
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

            # Fix 1: assert the doc lists were not mutated during excerpt
            # selection. Raises RuntimeError if ordering or length changed.
            self._assert_doc_block_ref_alignment(
                full_docs, abstract_docs, _doc_key_snapshot
            )

            documents_block = self._render_documents_block(
                full_docs, abstract_docs, excerpts
            )
        else:
            log_excerpt_stats(log, {}, elapsed=0.0)

        # -- 5. refinement ----------------------------------------------------
        # Fix C: use a tier-aware refinement max_tokens instead of the
        # hardcoded 2000. Full-text tiers (excerpts_narrow / excerpts_full)
        # pass large document blocks to the refinement model; 2000 tokens is
        # not enough to summarise 6+ papers with full methodology sections.
        # 4000 tokens @ 47 tok/s = ~85s, within the 360s timeout for large.
        refine_max_tokens = (
            4000 if cfg.evidence_mode in ("excerpts_narrow", "excerpts_full") else 2000
        )
        refine_llm = self._llm(
            cfg.refinement_model_name or cfg.model_name,
            cfg.temperature_refine,
            max_tokens=refine_max_tokens,
            timeout_s=cfg.timeout_refine_s,
        )

        # Build filtered_aql from the post-filter doc list so both branches
        # always pass a non-empty JSON string to their refinement agent,
        # regardless of whether the caller supplied aql_results_str= or docs=.
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
                    f"model={cfg.refinement_model_name or cfg.model_name}). "
                    "Reduce per_doc_budget / global_budget in rules.yaml or "
                    "switch to a larger context model."
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
                    f"abstract fallback available for question '{question[:80]}'. "
                    "Aborting generation."
                )
            log.warning(
                "refinement returned empty context for '%s...' (rule=%s) -- "
                "falling back to %d raw abstracts for generation",
                question[:60], cfg.rule_hit,
                len([d for d in (full_docs + abstract_docs) if d.get("abstract", "").strip()]),
            )
            enriched_context = fallback_ctx

        # -- 6. generation ----------------------------------------------------
        # Fix 3: build answer-quality contract from profile and cfg.
        # Build a local system_prompt — never mutate cfg globally.
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
            # Fix 4: pass the original question, not query_hint, to generation.
            answer_obj = gen_agent.generate(
                question=question,
                text_context=enriched_context,
                ontology=ontology,
                context_cap=cfg.gen_context_cap,
                max_output_tokens=cfg.max_output_tokens,
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
                    f"model={cfg.model_name}). "
                    "Reduce gen_context_cap or max_output_tokens in rules.yaml."
                ) from gen_exc
            log.error(
                "generation raised an exception for '%s...' (rule=%s, model=%s): %s",
                question[:60], cfg.rule_hit, cfg.model_name, gen_exc,
            )
            answer_obj = None

        log_generation(log, answer_obj, elapsed=time.perf_counter() - t0)

        if answer_obj is None or not getattr(answer_obj, "answer", "").strip():
            log.error(
                "generation returned no answer for '%s...' "
                "(rule=%s, model=%s) -- returning empty result",
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

        # Fix 11: post-generation numeric faithfulness audit (diagnostics only).
        self._audit_numeric_faithfulness(answer_obj.answer, enriched_context, question)

        # -- 7. build verified references + sequential renumbering -----------
        # Normalise the raw answer body first: some LLM variants write citations
        # as bare comma-lists ("word 3, 7") or multi-index brackets ("[5,6,7]")
        # instead of the canonical "[N][M]" form.  _normalize_citation_format()
        # converts both forms to "[N][M]" so _renumber_inline_citations() and
        # the scorer's re.findall(r'\[(\d+)\]') both see the correct markers.
        all_docs = full_docs + abstract_docs
        cited_indices = getattr(answer_obj, "cited_indices", set())
        fmt_refs, plain_refs, index_remap = self._build_verified_references(
            all_docs, cited_indices
        )

        # Normalise citation format, then rewrite [N] to sequential [1]..[K].
        normalised_body = self._normalize_citation_format(answer_obj.answer)
        answer_text = self._renumber_inline_citations(normalised_body, index_remap)

        # P3: strip any trailing [N] cluster left by a token-limit truncation.
        # Primary fix is P1 (higher token ceiling); this is a safety net.
        answer_text = re.sub(r'(\[\d+\])+\s*$', '', answer_text).rstrip()

        # Append ## References to the answer body.
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
            log.info(
                "session profiler_parse_failures=%d "
                "(questions silently escalated to safety-tier3 "
                "due to profiler JSON errors)",
                failure_count,
            )

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

    # ------------------------------------------------------------------
    # Fix 3: answer quality contract
    # ------------------------------------------------------------------

    @staticmethod
    def _build_answer_quality_contract(profile: QuestionProfile, cfg: PipelineConfig) -> str:
        """Build a short system-level quality contract from the question profile.

        Returns an empty string when the question profile does not require any
        special handling (low quantitativity, no spatial/temporal/methodological
        emphasis) so ordinary questions receive no extra system noise.

        The contract is appended to system_prompt_modifier *locally* inside
        Pipeline.run() — cfg is never mutated.
        """
        rules: List[str] = []

        # Universal grounding rule for full-text evidence tiers.
        if cfg.evidence_mode in ("excerpts_narrow", "excerpts_full"):
            rules.append("Every factual claim must be grounded in the provided context passages.")

        # Numeric precision rules.
        quant = getattr(profile, "quantitativity", 0.0) or 0.0
        needs_numeric = getattr(profile, "needs_numeric_emphasis", False)
        if quant >= 0.5 or needs_numeric:
            rules.append(
                "Numeric claims must include: the numeric value, its unit, "
                "the spatial/temporal scope it applies to, and an inline citation [N]."
            )
            rules.append(
                "Do not paraphrase numeric values — state them exactly as reported in the sources."
            )

        # Spatial scope rule.
        if (getattr(profile, "spatial_specificity", 0.0) or 0.0) >= 0.5:
            rules.append(
                "State the spatial/geographic scope explicitly "
                "(e.g., basin name, country, coordinates) for every spatial claim."
            )

        # Temporal scope rule.
        if (getattr(profile, "temporal_specificity", 0.0) or 0.0) >= 0.5:
            rules.append(
                "State the time period or observation window explicitly "
                "for every temporal or trend claim."
            )

        # Methodological depth rule.
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

    # ------------------------------------------------------------------
    # citation normalisation (Python-side, no LLM involvement)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_citation_format(text: str) -> str:
        """Normalise LLM citation variants to canonical [N][M] form.

        Three passes, applied before renumbering:

        Pre-pass — line-start bare citation clusters:
            "\\n3, 7 confirm that..."  ->  "[3][7] confirm that..."
            Only fires when followed by a word character (not a year or decimal).
            The lookahead (?=\\s+[A-Za-z]) prevents matching years or plain
            numbers that start a sentence with a numeric topic.

        Pass 1 — multi-index bracket expansion:
            [N, M, K]  ->  [N][M][K]

        Pass 2 — bare cluster bracketing:
            "word N, M"  ->  "word [N][M]"
            Anchor:    letter / ) / ] / % followed by exactly one space.
            NOTE: "." (dot) is intentionally excluded from the anchor class.
                  Including it caused false matches on decimal values such as
                  "2.16 Ma" (the "." before "16" matched, wrapping 16 as [16])
                  and "Ms 6.4" (same issue after the decimal point).
            Guard:     each digit token must be 1..30.
            Lookahead: sentence-ending punctuation (.,;), newline, em-dash "--",
                       or an opening "[".

        All passes are no-ops when the LLM already used [N] form correctly.
        """
        # Fix 8 pre-pass: handle line-start bare citation clusters.
        # e.g. "\\n3, 7 show that..." -> "[3][7] show that..."
        # Lookahead (?=\\s+[A-Za-z]) ensures we only match citation-like patterns,
        # not years or sentences starting with a numeric subject.
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

        # Pass 1: [N, M, K] -> [N][M][K]
        def _expand_multi_bracket(m: re.Match) -> str:
            nums = re.split(r"[\s,]+", m.group(1).strip())
            return "".join(f"[{n}]" for n in nums if n.isdigit())

        text = re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]", _expand_multi_bracket, text)

        # Pass 2: bare citation clusters after a word/symbol anchor.
        def _expand_bare_cluster(m: re.Match) -> str:
            nums = re.split(r"[\s,]+", m.group(2).strip())
            valid = [n for n in nums if n.isdigit() and 1 <= int(n) <= 30]
            if not valid:
                return m.group(0)
            return m.group(1) + "".join(f"[{n}]" for n in valid)

        text = re.sub(
            r"([a-zA-Z\)\]%] )(\d{1,2}(?:\s*,\s*\d{1,2}){0,4})"
            r"(?=\s*(?:[\.\\n,;]|$|\s*[-]{2,}|\s*\[))",
            _expand_bare_cluster,
            text,
        )
        return text

    # ------------------------------------------------------------------
    # verified reference builder (Python-side, no LLM involvement)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_verified_references(
        docs: List[Dict[str, Any]],
        cited_indices: Set[int],
    ) -> Tuple[List[str], List[str], Dict[int, int]]:
        """Build formatted_references and references from the known-good doc list.

        Args:
            docs: ordered list of documents (full + abstract, 1-indexed).
            cited_indices: set of 1-based integers the LLM cited inline.
                If empty, ALL docs are used (graceful fallback for old prompts
                or questions where the LLM wrote no markers).

        Returns:
            (formatted_references, plain_references, index_remap).
            formatted_references: '[N] Author (Year). *Title*. URI.' lines,
                renumbered sequentially from [1].
            plain_references:     'Author (Year). Title. URI.' lines (no [N]).
            index_remap:          {original_doc_index: sequential_output_index}
                e.g. {3: 1, 7: 2, 14: 3} when docs 3,7,14 were cited.
                Used by _renumber_inline_citations() to rewrite the answer body.

        OpenAlex metadata is fetched when the URI is available; on failure the
        doc's title and URI from the KG are used as a reliable fallback.
        Every cited doc that has at least a title or URI is guaranteed a line.

        Fix 2 — P4 title-normalised duplicate deduplication (O(1) lookup):
            Two docs with near-identical titles (preprint + journal version of
            the same paper) are collapsed to one entry. The first occurrence wins;
            the duplicate's original index is remapped to the same sequential
            output number. A norm_title_to_seq dict is built alongside
            seen_norm_titles so duplicate lookup is O(1) — no inner scan loop,
            no fragile docs[prev_i - 1] access.
        """
        from core.utils.openalex_client import OpenAlexClient, format_reference_from_metadata

        if not cited_indices:
            indices_to_use = set(range(1, len(docs) + 1))
        else:
            indices_to_use = {
                i for i in cited_indices if 1 <= i <= len(docs)
            }
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

        # Fix 2: O(1) duplicate tracking — norm_title_to_seq maps each
        # normalised title to the sequential output position already assigned
        # to its first occurrence.  Replaces the O(N) inner scan loop.
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

            # Fix 2: O(1) duplicate lookup via norm_title_to_seq.
            if norm_title and norm_title in seen_norm_titles:
                index_remap[i] = norm_title_to_seq[norm_title]
                log.debug(
                    "_build_verified_references: doc %d title duplicate of seq %d "
                    "('%s') -- remapped",
                    i, norm_title_to_seq[norm_title], title[:60],
                )
                continue

            if norm_title:
                seen_norm_titles.add(norm_title)

            seq += 1
            index_remap[i] = seq

            # Fix 2: record the seq for this norm_title immediately (O(1)).
            if norm_title:
                norm_title_to_seq[norm_title] = seq

            metadata = None
            if uri and "openalex.org" in uri:
                metadata = OpenAlexClient.fetch_metadata(uri)

            if metadata and not (metadata.get("title") or "").strip():
                log.warning(
                    "_build_verified_references: OpenAlex metadata for doc %d "
                    "has empty title -- falling back to KG title", i
                )
                metadata = None

            if metadata:
                meta = dict(metadata)
                meta["position"] = seq
                line = format_reference_from_metadata(meta)
                formatted.append(line)
                plain.append(re.sub(r"^\[\d+\]\s*", "", line))
            else:
                fallback_title = title or "[No title \u2014 see URI]"
                line = f"[{seq}] {fallback_title}. {uri}." if uri else f"[{seq}] {fallback_title}."
                formatted.append(line)
                plain.append(re.sub(r"^\[\d+\]\s*", "", line))

        return formatted, plain, index_remap

    @staticmethod
    def _renumber_inline_citations(answer_body: str, index_remap: Dict[int, int]) -> str:
        """Rewrite inline [N] markers in answer_body using index_remap.

        Replaces each [old] with [new] where new = index_remap[old].
        Markers not present in index_remap are left unchanged (safety).

        Uses a two-pass approach (placeholder -> final) to avoid collisions
        when two indices swap values (e.g. [1]->2 and [2]->1).
        """
        if not index_remap:
            return answer_body

        result = answer_body
        for old, new in sorted(index_remap.items(), key=lambda kv: -kv[0]):
            result = re.sub(
                r"\[" + str(old) + r"\]",
                f"__CITE_{new}__",
                result,
            )

        result = re.sub(r"__CITE_(\d+)__", r"[\1]", result)
        return result

    # ------------------------------------------------------------------
    # Fix 1: doc block / reference order alignment check
    # ------------------------------------------------------------------

    @staticmethod
    def _key_snapshot(
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
    ) -> List[str]:
        """Capture URI/title keys from both doc lists before excerpt selection.

        Called in run() immediately BEFORE select_excerpts_for_question() so
        that _assert_doc_block_ref_alignment() has a real before/after window
        to compare against (not list-to-itself at the same instant).
        """
        def _key(doc: Dict[str, Any]) -> str:
            return doc.get("uri") or doc.get("id") or doc.get("title") or ""

        return [_key(d) for d in full_docs] + [_key(d) for d in abstract_docs]

    @staticmethod
    def _assert_doc_block_ref_alignment(
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
        snapshot: List[str],
    ) -> None:
        """Assert that doc lists were not mutated during excerpt selection.

        Fix 1: accepts a pre-computed snapshot taken BEFORE excerpt selection
        runs (via _key_snapshot). Compares snapshot against the current list
        state position-by-position, raising RuntimeError on any ordering or
        length divergence.

        Because both _render_documents_block and _build_verified_references
        concatenate full_docs + abstract_docs in the same order, any mutation
        here would mis-align inline [N] markers with the ## References section.

        Raises RuntimeError on mismatch.
        """
        def _key(doc: Dict[str, Any]) -> str:
            return doc.get("uri") or doc.get("id") or doc.get("title") or ""

        current = [_key(d) for d in full_docs] + [_key(d) for d in abstract_docs]

        if len(snapshot) != len(current):
            raise RuntimeError(
                f"Pipeline._assert_doc_block_ref_alignment: list length mismatch — "
                f"snapshot has {len(snapshot)} docs, current has {len(current)}. "
                "Doc list was mutated during excerpt selection. Aborting."
            )

        for pos, (snap_key, cur_key) in enumerate(zip(snapshot, current), 1):
            if snap_key != cur_key:
                raise RuntimeError(
                    f"Pipeline._assert_doc_block_ref_alignment: ordering divergence "
                    f"at position {pos}. "
                    f"snapshot key={snap_key!r}, current key={cur_key!r}. "
                    "Doc list was reordered during excerpt selection — "
                    "citation indices would be misaligned. Aborting."
                )

    # ------------------------------------------------------------------
    # document filter
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_retracted_papers(
        docs: List[Dict[str, Any]],
        question: str,
    ) -> List[Dict[str, Any]]:
        """Pre-filter: drop papers with retraction/withdrawal indicators in title.

        Fix 9: conservative approach — only title-level markers, not abstracts.
        Skipped entirely when the question itself is about retractions, so that
        meta-questions about retracted literature still have access to those docs.
        """
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
                    "retraction_filter: excluding '%s' (title/status contains retraction indicator)",
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
        docs = self._remove_retracted_papers(docs, question)

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

        # Fix 5 (strengthened): P6 cross-question full-text throttle.
        # Read _session_full_doc_uris under lock to get a consistent snapshot,
        # then use the local copy for the demotion loop (no lock held during
        # iteration — prevents lock contention on concurrent questions).
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
                    log.debug(
                        "P6 throttle: demoted '%s' (URI %s) from full to abstract",
                        doc.get("title", "")[:60], uri[:60],
                    )
                else:
                    new_full.append(doc)
            full_docs = new_full
            abstract_docs = new_abstract

        # Fix 5 (strengthened): register new full-text URIs under lock to
        # prevent data races when multiple questions run concurrently.
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
                    # Fix 10: log a warning when a full doc has no excerpt cache
                    # hit so the developer can tune FullTextIndexer population.
                    log.warning(
                        "_render_documents_block: full doc '%s' (URI=%s) has no "
                        "excerpt cache hit — serving abstract only. "
                        "Check FullTextIndexer cache population.",
                        doc.get("title", "")[:60], uri[:60],
                    )
                    lines.append("\n[abstract only — no fulltext cache hit]")
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
        """Build a query hint string for refinement only.

        Fix 4: also triggers numeric emphasis when quantitativity >= 0.5,
        not only when needs_numeric_emphasis is explicitly True.
        The hint is used ONLY for refinement (not passed to final generation).
        """
        parts = [question]
        quant = getattr(profile, "quantitativity", 0.0) or 0.0
        if getattr(profile, "needs_numeric_emphasis", False) or quant >= 0.5:
            parts.append("[emphasis: numeric/quantitative precision]")
        if (getattr(profile, "methodological_depth", 0.0) or 0.0) > 0.6:
            parts.append("[emphasis: methodological detail]")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Fix 11: numeric faithfulness audit (diagnostics only)
    # ------------------------------------------------------------------

    @staticmethod
    def _audit_numeric_faithfulness(
        answer_text: str,
        context: str,
        question: str,
    ) -> None:
        """Post-generation diagnostic: log numeric claims not found in context.

        Conservative to avoid false positives:
        - Skips 4-digit tokens in the range 1900–2100 (years).
        - Skips single-digit integers (too common and ambiguous).
        - Skips citation markers [N].
        Does NOT modify the answer. Logging only.
        """
        answer_clean = re.sub(r"\[\d+\]", "", answer_text)
        candidates = re.findall(r"\b-?\d+(?:\.\d+)?\b", answer_clean)

        unmatched: List[str] = []
        for token in candidates:
            try:
                as_int = int(float(token))
                if 1900 <= as_int <= 2100 and "." not in token:
                    continue
                if abs(as_int) < 10 and "." not in token:
                    continue
            except (ValueError, OverflowError):
                continue
            if token not in context:
                unmatched.append(token)

        if unmatched:
            log.warning(
                "numeric_audit: question='%s...' — %d numeric claim(s) not found "
                "verbatim in enriched_context: %s",
                question[:60], len(unmatched), unmatched[:10],
            )


# backward-compat alias
AdaptivePipeline = Pipeline
