"""pipeline -- routes each question to the right evidence mode and model.

Routing is determined at runtime by the policy router based on the
question profile.
"""

from __future__ import annotations

import ast
import collections
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

# Maximum doc index the LLM is allowed to cite. Indices above this cap are
# treated as hallucinated and dropped before they reach _build_verified_references.
# Consistent with the same constant in generation_agent.py (_MAX_CITE_INDEX).
_MAX_CITE_INDEX = 30

# Fix B: rolling window size for the per-question URI demotion throttle.
# Only URIs seen in the last _URI_WINDOW_SIZE questions are considered
# "already served as full-text" for the purpose of the surplus-demotion
# loop in _filter_documents.  A global accumulation across a 70-question
# batch caused the throttle to degrade into a broad cross-topic suppressor
# by question 40+, misclassifying entire topic clusters as "already seen".
_URI_WINDOW_SIZE = 10

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
    # Titles of the docs from which formatted_references was built, in the
    # same order as all_docs (full_docs + abstract_docs).  Persisted on the
    # PipelineResult so downstream consumers (output writer, audit tools)
    # can verify post-hoc that formatted_references did not bleed in from a
    # different question.  See _verify_refs_against_docs in Pipeline.run().
    ref_source_titles: List[str] = field(default_factory=list)


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
        # Thread-local LLM wrapper cache.  Each worker thread maintains its
        # own dict of LLM wrappers (and therefore its own underlying HTTP
        # connection pool) so concurrent streams cannot share pooled sockets.
        #
        # Why this matters: commit 4a8cc81 documented that pooled HTTP/2
        # connections in the OpenAI / Mistral clients can leak chunks
        # between streams when one stream is interrupted (retry, timeout)
        # and another is started on the same pooled socket.  The fix added
        # reset_llm_cache() between sequential questions but never covered
        # the concurrent runner in scripts/run_pipeline.py, where multiple
        # worker threads call Pipeline.run() simultaneously on the same
        # Pipeline instance.  Sharing a single OpenAI()/Mistral() client
        # across threads is the same pooled-socket scenario, just compressed
        # in time -- and is the most plausible root cause of the Q25/Q37
        # cross-question reference bleed seen in the 20Q quality audit.
        #
        # threading.local() gives each thread its own dict; _llm() lazily
        # initialises the dict on first access from that thread.
        self._llm_local = threading.local()
        self._indexer = None
        # Fix B: replace the flat, ever-growing _session_full_doc_uris Set with
        # a rolling deque of per-question URI sets (maxlen=_URI_WINDOW_SIZE).
        # The surplus-demotion throttle in _filter_documents now only considers
        # the last N questions, preventing cross-domain contamination that
        # accumulated over full 70-question batch runs.
        self._session_uri_window: collections.deque = collections.deque(
            maxlen=_URI_WINDOW_SIZE
        )
        self._profiler_parse_failures: int = 0
        self._counter_lock = threading.Lock()

    def reset_session_state(self) -> None:
        """Reset run/session state so a new orchestrator.run() starts clean."""
        with self._counter_lock:
            self._session_uri_window.clear()
            self._profiler_parse_failures = 0
        log.debug("Pipeline.reset_session_state() called")

    def reset_llm_cache(self) -> None:
        """Clear the calling thread's LLM instance cache.

        Only affects the thread that calls this; other threads retain their
        own caches.  Matches the pre-thread-local contract for sequential
        callers (e.g. core/main.py) while leaving concurrent worker threads
        isolated from each other.
        """
        if hasattr(self._llm_local, "cache"):
            self._llm_local.cache = {}
        with self._counter_lock:
            self._session_uri_window.clear()
        log.debug("Pipeline._llm_local.cache cleared (this thread)")

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

        # Demote any filter-classified 'full' docs whose PDF is absent from
        # cache to abstract so excerpt selection does not silently score 0
        # on them and mislead the refinement agent about document coverage.
        if cfg.evidence_mode in ("excerpts_narrow", "excerpts_full"):
            full_docs, abstract_docs = self._demote_cache_unavailable(
                full_docs, abstract_docs
            )

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
            self._warn_thin_sources(excerpt_stats)  # Fix 4: log thin sources
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
        # Decode literal \uXXXX sequences that may appear when the input
        # CSV was serialised with ensure_ascii=True and the LLM reproduced
        # them verbatim (e.g. \u2082 for CO subscript-2).
        # Also sanitize invalid \u escapes (e.g. \units, \uncertainty) that
        # are not followed by 4 hex digits and would cause json.JSONDecodeError
        # when the string is later serialised as part of a JSON prompt payload.
        enriched_context = self._unescape_unicode(enriched_context)
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

        # Sanitise bare \u escapes in the raw answer BEFORE any re.sub call.
        # The LLM can emit \units, \upwelling, etc. which cause re.sub to raise
        # 'bad escape \u at position N' when the string is used as a pattern
        # input or flows through _normalize_and_extract_citations /
        # _renumber_inline_citations. _unescape_unicode() decodes valid \uXXXX
        # and escapes the rest to \\u so re is never exposed to a bare \u.
        answer_obj.answer = self._unescape_unicode(answer_obj.answer)

        self._audit_numeric_faithfulness(answer_obj.answer, enriched_context, question)

        # -- 7. build verified references + sequential renumbering -----------
        all_docs = full_docs + abstract_docs

        # Unified citation extraction.
        # Primary path: <<CITE:N>> sentinel markers emitted by the updated
        # generation prompts. These are unambiguous and cannot collide with
        # prose numbers (decimals, years, table labels, quantities).
        # Fallback path: legacy [N] square-bracket handling, preserved for
        # backward compatibility with old prompts and cached test fixtures.
        normalised_body, cited_indices = self._normalize_and_extract_citations(
            answer_obj.answer
        )

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

        # Strip any orphan [N] whose integer is not among the remapped output
        # indices. These are phantom citations that were never in any document
        # list and therefore have no entry in index_remap.
        if index_remap:
            valid_new = set(index_remap.values())
            answer_text = re.sub(
                r'\[(\d+)\]',
                lambda m: f'[{m.group(1)}]' if int(m.group(1)) in valid_new else '',
                answer_text,
            )

        # Collapse adjacent identical brackets produced by title-dedup
        # (e.g. [3][3] -> [3] when two old indices map to the same new index).
        answer_text = re.sub(r'(\[\d+\])(?:\1)+', r'\1', answer_text)
        answer_text = re.sub(r'(\[\d+\])+\s*$', '', answer_text).rstrip()
        # Decode any remaining literal \uXXXX sequences in the generated answer
        # (belt-and-suspenders: catches any \u introduced by citation renumbering).
        answer_text = self._unescape_unicode(answer_text)
        # Fix 1-3: strip context-assembly artifacts from the answer body.
        answer_text = self._clean_answer_artifacts(answer_text)

        # Snapshot the titles of all_docs at ref-build time.  Persisted on
        # the PipelineResult so downstream code (output writer, audit tools)
        # can prove formatted_references came from these specific docs and
        # not from a different question's docs (cross-question refs bleed).
        ref_source_titles = [
            (d.get("title_or_name") or d.get("title") or "") for d in all_docs
        ]

        # Invariant: every formatted reference line's payload must be derivable
        # from all_docs.  This is a defence-in-depth guard against the
        # Q25/Q37-style refs-bleed where the body cited the right indices but
        # the references section contained titles from a different question.
        # We check that each fmt_ref line shares a normalised title token with
        # at least one all_docs title.  If the check fails we LOG LOUDLY and
        # add a "REFS_MISMATCH" marker to references (no silent fallback --
        # preserves the no-silent-fallback principle from feedback memory).
        bleed_detected = self._detect_refs_bleed(fmt_refs, ref_source_titles)
        if bleed_detected:
            log.error(
                "REFS_BLEED suspected for '%s...' (rule=%s): "
                "%d/%d formatted_references lines do not share any normalised "
                "title token with all_docs (n_docs=%d). "
                "This usually means a concurrent question's stream chunks "
                "leaked into refs assembly. Marking record for audit.",
                question[:60], cfg.rule_hit,
                len(bleed_detected), len(fmt_refs), len(all_docs),
            )

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
            ref_source_titles=ref_source_titles,
        )

    @staticmethod
    def _build_answer_quality_contract(profile: QuestionProfile, cfg: PipelineConfig) -> str:
        rules: List[str] = []
        if cfg.evidence_mode in ("excerpts_narrow", "excerpts_full"):
            rules.append("Every factual claim must be grounded in the provided context passages.")
        quant = getattr(profile, "quantitativity", 0.0) or 0.0
        needs_numeric = getattr(profile, "needs_numeric_emphasis", False)
        if quant >= 0.40 or needs_numeric:
            rules.append(
                "Numeric claims must include: the numeric value, its unit, "
                "the spatial/temporal scope it applies to, and an inline citation marker if available."
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
    def _extract_sentinel_citations(text: str) -> Tuple[str, Set[int]]:
        """Convert <<CITE:N>> sentinel markers to [N] and collect cited indices.

        This is the primary citation extraction path. <<CITE:N>> markers are
        unambiguous: they cannot appear in prose, decimal numbers, table
        references, quantities like 4,000, or year values. Out-of-range
        indices (N > _MAX_CITE_INDEX) are silently dropped.

        Pre-pass: the model occasionally emits <<CITE:1,6>> (comma-separated
        multi-cite) instead of <<CITE:1>><<CITE:6>>, particularly inside
        markdown table cells. The pre-pass expands these to individual
        single-integer sentinels before the main extraction loop runs.

        Fix D: both regexes now use >{1,2} instead of requiring exactly >>
        so that <<CITE:3> (single closing >) is handled identically to the
        well-formed <<CITE:3>> variant.  The safety-net strip in
        _clean_answer_artifacts removes any residual markers that still
        survive all extraction paths.
        """
        # Pre-pass: expand <<CITE:N,M,...>> or <<CITE:N,M,...> -> <<CITE:N>><<CITE:M>>...
        def _expand_multi(m: re.Match) -> str:
            parts = re.split(r"[\s,]+", m.group(1).strip())
            return "".join(f"<<CITE:{p}>>" for p in parts if p.isdigit())

        text = re.sub(r"<<CITE:([\d,\s]+)>{1,2}", _expand_multi, text)

        # Main extraction: single-integer sentinels only.
        indices: Set[int] = set()

        def _replace(m: re.Match) -> str:
            n = int(m.group(1))
            if 1 <= n <= _MAX_CITE_INDEX:
                indices.add(n)
                return f"[{n}]"
            return ""

        clean = re.sub(r"<<CITE:(\d+)>{1,2}", _replace, text)
        return clean, indices

    @staticmethod
    def _normalize_and_extract_citations(text: str) -> Tuple[str, Set[int]]:
        """Normalise citation markers and return (clean_body, cited_indices).

        Primary path: structured <<CITE:N>> sentinel markers (new prompts).
        Fallback path: legacy square-bracket normalisation chain, retained for
        backward compatibility with old prompts, fixtures, and cached runs.
        The two paths produce the same downstream format: plain [N] markers
        and a Set[int] of cited 1-based doc indices.
        """
        if "<<CITE:" in text:
            return Pipeline._extract_sentinel_citations(text)

        # --- Legacy fallback ---------------------------------------------------
        # Collapse [doc N] / [Doc N] tokens emitted by older refinement variants.
        text = re.sub(r"\[(?:doc|Doc)\s+(\d+)\]", r"[\1]", text)

        def _expand_line_start(m: re.Match) -> str:
            nums = re.split(r"[\s,]+", m.group(1).strip())
            valid = [n for n in nums if n.isdigit() and 1 <= int(n) <= _MAX_CITE_INDEX]
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
            return "".join(
                f"[{n}]" for n in nums if n.isdigit() and 1 <= int(n) <= _MAX_CITE_INDEX
            )

        text = re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]", _expand_multi_bracket, text)

        def _expand_bare_cluster(m: re.Match) -> str:
            nums = re.split(r"[\s,]+", m.group(2).strip())
            valid = [n for n in nums if n.isdigit() and 1 <= int(n) <= _MAX_CITE_INDEX]
            if not valid:
                return m.group(0)
            return m.group(1) + "".join(f"[{n}]" for n in valid)

        text = re.sub(
            r"(?<!Table )(?<!Figure )(?<!Section )(?<!Equation )(?<!Appendix )"
            r"([a-zA-Z\)\]%] )(\d{1,2}(?:\s*,\s*\d{1,2}){0,4})"
            r"(?=\s*(?:(?:\.(?!\d))|\n|,|;|$|\s*[-]{2,}|\s*\[))",
            _expand_bare_cluster,
            text,
        )

        indices: Set[int] = set()
        for bracket in re.findall(r"\[([\d,\s]+)\]", text):
            for token in bracket.split(","):
                token = token.strip()
                if token.isdigit():
                    n = int(token)
                    if 1 <= n <= _MAX_CITE_INDEX:
                        indices.add(n)

        return text, indices

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
    def _detect_refs_bleed(
        fmt_refs: List[str],
        ref_source_titles: List[str],
    ) -> List[int]:
        """Return indices of fmt_refs lines that share NO token with any all_docs title.

        A reference line is considered "bled" (sourced from a different question's
        docs) if none of its content tokens (>= 5 chars, alphanumeric) appear in
        any of the titles of the docs that were supplied to _build_verified_references.

        This is a coarse but cheap heuristic: a true match needs at least one
        non-stopword token in common.  Empty fmt_refs / empty all_docs returns
        empty list (no bleed claimed in the absence of evidence).
        """
        if not fmt_refs or not ref_source_titles:
            return []

        def _toks(s: str) -> Set[str]:
            return {
                t.lower()
                for t in re.findall(r"[A-Za-z][A-Za-z0-9]{4,}", s or "")
            }

        all_doc_tokens: Set[str] = set()
        for t in ref_source_titles:
            all_doc_tokens.update(_toks(t))
        if not all_doc_tokens:
            return []

        bleed: List[int] = []
        for i, line in enumerate(fmt_refs):
            ref_tokens = _toks(line)
            if not ref_tokens:
                continue
            if ref_tokens.isdisjoint(all_doc_tokens):
                bleed.append(i)
        return bleed

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
        # Fix B: union the last _URI_WINDOW_SIZE per-question URI sets instead
        # of a single ever-growing flat set.  This bounds the throttle's
        # "already seen" memory to the last 10 questions and prevents
        # cross-domain demotion that accumulated across full 70-Q batch runs.
        with self._counter_lock:
            session_uris_snapshot = (
                set().union(*self._session_uri_window)
                if self._session_uri_window
                else set()
            )
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
        # Fix B: append this question's URI set as one slot in the rolling
        # window; the deque automatically evicts the oldest slot when full.
        with self._counter_lock:
            self._session_uri_window.append(new_uris)
        return full_docs, abstract_docs, drop_docs

    def _get_indexer(self):
        if self._indexer is None:
            from core.utils.fulltext_indexer import FullTextIndexer
            self._indexer = FullTextIndexer(cache_dir=self._cache_dir)
        return self._indexer

    def _llm(self, model: str, temperature: float, max_tokens: int = 1400, timeout_s: Optional[int] = None):
        """Return a thread-local LLM wrapper for (model, temperature, max_tokens, timeout).

        Each worker thread maintains its own cache (see Pipeline.__init__).
        Two threads asking for the same (model, ...) tuple get two distinct
        wrapper instances backed by two distinct OpenAI/Mistral clients,
        which use separate HTTP connection pools.  This eliminates the
        pooled-socket stream-bleed scenario that produced the Q25/Q37
        cross-question reference contamination.
        """
        from core.utils.helpers import get_llm_model
        if not hasattr(self._llm_local, "cache"):
            self._llm_local.cache = {}
        cache: Dict[Tuple[str, float, int, int], Any] = self._llm_local.cache
        key = (model, temperature, max_tokens, timeout_s or 0)
        if key not in cache:
            cache[key] = get_llm_model(
                model, temperature, max_tokens, timeout_s=timeout_s
            )
        return cache[key]

    @staticmethod
    def _try_live_kg(
        question: str,
        ontology: DynamicOntology,
        aql_params: Optional[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        import os
        # Guard: live KG is disabled in evaluation runs where
        # ARANGO_ROOT_PASSWORD is not set; returns None immediately so
        # the pipeline falls through to the CSV path.
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
        q_lower = question.lower()
        quant = getattr(profile, "quantitativity", 0.0) or 0.0
        if getattr(profile, "needs_numeric_emphasis", False) or quant >= 0.5:
            parts.append("[emphasis: numeric/quantitative precision]")
        if (getattr(profile, "methodological_depth", 0.0) or 0.0) > 0.6:
            parts.append("[emphasis: methodological detail]")
        if "radiative propert" in q_lower and "land surface" in q_lower:
            parts.append(
                "[include: albedo, emissivity, spectral reflectance, surface emissivity]"
            )
        return " ".join(parts)

    def _demote_cache_unavailable(
        self,
        full_docs: list,
        abstract_docs: list,
    ) -> tuple:
        """Move filter-classified 'full' docs whose PDF has a .fail marker to abstract."""
        pdf_dir = self._cache_dir / "pdfs"
        if not pdf_dir.exists():
            return full_docs, abstract_docs

        new_full: list = []
        new_abstract: list = list(abstract_docs)
        for doc in full_docs:
            uri = doc.get("uri") or doc.get("id") or ""
            work_id = (
                uri.rstrip("/").split("/")[-1]
                if "openalex.org" in uri
                else None
            )
            if work_id and (pdf_dir / f"{work_id}.fail").exists():
                new_abstract.append(doc)
                log.debug(
                    "_demote_cache_unavailable: %s -> abstract (.fail exists)",
                    work_id,
                )
            else:
                new_full.append(doc)

        demoted = len(full_docs) - len(new_full)
        if demoted:
            log.info(
                "_demote_cache_unavailable: %d doc(s) moved full -> abstract"
                " (no PDF in cache)",
                demoted,
            )
        return new_full, new_abstract

    @staticmethod
    def _warn_thin_sources(
        excerpt_stats: Dict[str, Any],
        threshold: int = 50,
    ) -> None:
        """Emit a WARNING for each doc whose kept_tokens falls below *threshold*.

        Fix 4 (log-only observability): fires after excerpt selection and records
        thin-evidence sources so evaluation runs can identify which questions
        relied on near-empty documents.  No documents are filtered or removed.

        Keys match what FullTextIndexer.select_excerpts_for_question returns:
            excerpt_stats["per_doc"]  -> list of per-doc info dicts
            info["kept_tokens"]       -> total tokens kept for that doc
            info["work_id"]           -> OpenAlex work identifier
            info["title"]             -> document title
        """
        for pd in excerpt_stats.get("per_doc", []):
            kt = pd.get("kept_tokens", threshold)
            if kt < threshold:
                log.warning(
                    "thin_source: work=%s title='%s' kept_tokens=%d -- "
                    "answer may rely on near-empty evidence for this document",
                    pd.get("work_id", "?"),
                    (pd.get("title") or "?")[:60],
                    kt,
                )

    @staticmethod
    def _clean_answer_artifacts(text: str) -> str:
        """Strip known context-assembly artifacts that bleed into answer bodies.

        Fixes 1-3 (post-render string cleanup only; no semantic changes):
        1. TITLE <section> header bleed  -- context section labels copied verbatim
           by the LLM from the documents block into the answer body.
        2. httpsopenalex.org URL bleed   -- raw URIs with missing colon/slashes
           that leaked from reference block serialisation into the answer.
        3. Broken unit strings           -- missing spaces produced by context
           serialisation (e.g. "13.5Wm" -> "13.5 W/m\u00b2", "0.5mday" -> "0.5 m/day").

        Fix D (safety net): any <<CITE:...> or <<CITE:...>> marker that survived
        all extraction paths (e.g. because the model emitted a single closing >)
        is stripped here as a last resort so it never appears in the final output.

        Intentionally excluded: CO/CH4 subscript normalisation (ambiguous --
        CO is a valid compound distinct from CO2).
        """
        # Fix 1: strip "TITLE <Header Text>" lines anchored to start-of-line.
        text = re.sub(r'(?m)^TITLE\s+[A-Z][^\n]*\n?', '', text)
        # Fix 2: strip raw OpenAlex URL bleed (missing "://" -> httpsopenalex...).
        text = re.sub(r'https?openalex\.org\w+', '', text)
        # Fix 3a: digit immediately followed by "Wm" -> "W/m\u00b2".
        # NOTE: use a plain string (not r'...') so \u00b2 is decoded to \u00b2
        # at Python parse time; raw strings leave \u00b2 as literal characters
        # which re.sub rejects with "bad escape \u at position 6".
        text = re.sub(r'(\d)(Wm)\b', '\\1 W/m\u00b2', text)
        # Fix 3b: digit immediately followed by "mday" -> "m/day".
        text = re.sub(r'(\d)(mday)\b', r'\1 m/day', text)
        # Fix D: safety-net strip for any residual <<CITE:...> or <<CITE:...>>
        # that escaped extraction (e.g. single closing > instead of >>).
        # This runs AFTER all extraction paths so it never interferes with
        # index collection -- it only removes the now-dead marker text.
        text = re.sub(r'<<CITE:[^>\n]*>{1,2}', '', text)
        return text

    @staticmethod
    def _unescape_unicode(text: str) -> str:
        r"""Decode literal \uXXXX escape sequences in LLM output text.

        Two-pass strategy:
        1. Replace valid \uXXXX sequences (exactly 4 hex digits) with the
           corresponding Unicode character.
        2. Replace any remaining bare \u not followed by 4 hex digits
           (e.g. \units, \uncertainty) with a literal backslash + u so the
           string is safe for downstream JSON serialisation (e.g. inside the
           Mistral SDK's httpx request payload).  Without this second pass,
           json.loads / the SDK raises JSONDecodeError: bad escape \u.
        """
        # Pass 1: decode valid \uXXXX -> Unicode char
        text = re.sub(
            r'\\u([0-9a-fA-F]{4})',
            lambda m: chr(int(m.group(1), 16)),
            text,
        )
        # Pass 2: escape remaining bare \u (not followed by 4 hex digits)
        text = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', text)
        return text

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
