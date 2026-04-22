"""adaptive pipeline -- routes per-question to the right evidence mode + model.

zero changes to exp 1-4 pipelines. all behaviour is selected at runtime by
the policy router based on the question profile.

flow per question:
  1. profile     -- fused av-pair + profile call (small, temp=0)
  2. route       -- pure-python policy router selects PipelineConfig
  3. retrieve    -- live ArangoDB KG (graph expansion) if available,
                    else fall back to pre-baked aql_results_str from CSV
  4. filter      -- second LLM pass classifies docs as full/abstract/drop
                    (skipped when doc count <= doc_filter_min_keep)
  5. evidence    -- abstracts (no extra work) | excerpts via FullTextIndexer
                    only full_docs get PDF fetch; abstract_docs contribute
                    abstracts only; drop_docs are excluded entirely
  6. refinement  -- v2 unified documents_block (abstracts + excerpts) when
                    evidence_mode != abstracts; plain aql_results_str for
                    tier-1/abstracts mode
  7. generation  -- direct or structured prompt, with shape + synthesis

KG retrieval is activated automatically when ARANGO_URL + ARANGO_ROOT_PASSWORD
are present in the environment.  when they are absent the pipeline falls back
to the existing CSV-based aql_results_str path unchanged.

Hard-fail policy
----------------
No step silently swallows errors and falls back to generating an ungrounded
answer.  If refinement or generation fails (LLM unavailable, empty response,
empty documents_block) a RuntimeError propagates to the caller so that smoke
tests and parallel runners see a clean, retryable error rather than a short
garbage answer in the results CSV.
"""

from __future__ import annotations

import ast
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
    "direct_paragraph":      (
        "Write a direct paragraph answer of 3-5 sentences. "
        "No section headings. No bullet lists. No Implications section."
    ),
    "short_explainer":       "Write a short explanation in 2-4 sentences.",
    "structured_long":       (
        "Structure your answer with 3-5 section headings that directly match "
        "the sub-questions implied by the question. "
        "Open with a 2-3 sentence direct answer before the first heading. "
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
    kg_docs_used: int = 0
    kg_source: str = "csv"         # "live" | "csv" | "narrow_csv" | "none"


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
        aql_results_str: str = "",
        docs: Optional[List[Dict[str, Any]]] = None,
        aql_params: Optional[Dict[str, Any]] = None,
        precomputed_route: Optional[Tuple[DynamicOntology, QuestionProfile, PipelineConfig]] = None,
    ) -> AdaptiveResult:
        """run the full adaptive pipeline for one question.

        retrieval order:
          1. live ArangoDB KG  (if ARANGO_ROOT_PASSWORD is set)
          2. pre-parsed docs   (if caller already passed docs list)
          3. aql_results_str   (CSV fallback, JSON first then ast.literal_eval)

        precomputed_route: if phase3_parallel (or any caller) already ran
          profile_and_route() to choose the semaphore, pass the result here
          to skip a redundant LLM call.  Must be a 3-tuple
          (ontology, profile, cfg) as returned by profile_and_route().
        """
        from core.agents.generation_agent import GenerationAgent
        from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
        from core.agents.refinement_agent_fulltext import RefinementAgent1PassFullText
        from core.utils.aql_parser import parse_aql_results

        if precomputed_route is not None:
            ontology, profile, cfg = precomputed_route
            log.info(
                "q=%s... -> rule=%s model=%s evidence=%s (precomputed route)",
                question[:60], cfg.rule_hit, cfg.model_name, cfg.evidence_mode,
            )
        else:
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
            log.info(
                "KG live: %d primary docs, %d total nodes",
                len(live_docs),
                sum(1 + len(d.get("science_keywords", [])) +
                    sum(len(sk.get("secondary_nodes", []))
                        for sk in d.get("science_keywords", []))
                    for d in live_docs),
            )
        elif docs:
            kg_source = "csv"
        else:
            if aql_results_str:
                parsed = parse_aql_results(aql_results_str)
                docs = self._parse_docs_from_str(parsed)
            else:
                docs = []
            kg_source = "csv" if docs else "none"
            if docs:
                log.info("CSV fallback: parsed %d docs from aql_results_str", len(docs))
            else:
                log.warning(
                    "CSV fallback: could not parse any docs from aql_results_str "
                    "(len=%d)", len(aql_results_str or "")
                )

        # Guard: we must have at least some documents to proceed.
        # For tier-1 (abstracts mode) we can proceed with abstracts alone;
        # for tier-2/3 we need at least some full or abstract docs.
        if not docs:
            raise RuntimeError(
                f"AdaptivePipeline.run(): no documents available for question "
                f"'{question[:80]}...' (kg_source={kg_source}). "
                "Cannot proceed — refinement and generation require document context."
            )

        # ----------------------------------------------------------
        # document filter (second LLM pass)
        # skipped when the result set is already small enough that
        # filtering would burn an LLM call for negligible benefit.
        # ----------------------------------------------------------
        full_docs: List[Dict] = list(docs) if docs else []
        abstract_docs: List[Dict] = []
        drop_docs: List[Dict] = []
        filter_ran = False

        if docs and len(docs) > cfg.doc_filter_min_keep:
            full_docs, abstract_docs, drop_docs = self._filter_documents(
                docs, ontology, profile, question, cfg
            )
            filter_ran = True
            log.info(
                "doc filter: %d full, %d abstract, %d dropped (of %d)",
                len(full_docs), len(abstract_docs), len(drop_docs), len(docs),
            )

        # Guard: after filtering we must still have documents.
        # filter_documents() already falls back to the full set on LLM error,
        # but if it somehow returns empty we catch it here.
        if not full_docs and not abstract_docs:
            raise RuntimeError(
                f"AdaptivePipeline.run(): document filter eliminated ALL documents "
                f"for question '{question[:80]}...'. "
                "This should never happen — filter_documents() falls back to the "
                "full set on any error. Investigate ontology_agent.filter_documents()."
            )

        # rebuild aql_results_str from only the non-dropped docs so the
        # refinement prompt never receives irrelevant context
        if filter_ran and (abstract_docs or drop_docs):
            aql_results_str = self._build_filtered_aql_str(
                full_docs, abstract_docs
            )

        # ----------------------------------------------------------
        # evidence: full_docs get PDF fetch; abstract_docs are
        # abstract-only (no PDF fetch); drop_docs are excluded
        # ----------------------------------------------------------
        excerpts, excerpt_stats = self._build_evidence(
            cfg, question, ontology, full_docs
        )
        excerpt_stats["n_full_docs"]     = len(full_docs)
        excerpt_stats["n_abstract_docs"] = len(abstract_docs)
        excerpt_stats["n_dropped_docs"]  = len(drop_docs)
        excerpt_stats["filter_ran"]      = filter_ran

        # ----------------------------------------------------------
        # refinement
        # ----------------------------------------------------------
        ref_llm = self._llm(cfg.model_name, cfg.temperature_refine)

        if cfg.evidence_mode == "abstracts":
            # tier-1 / fallback: plain aql_results_str (no PDF fetch)
            ref_agent = RefinementAgent1PassRefined(
                ref_llm,
                prompt_dir=str(self._prompts_root / "refinement"),
            )
        else:
            # tier-2/3: unified documents_block (v2) with abstracts +
            # interleaved full-text excerpts
            ref_agent = RefinementAgent1PassFullText(
                ref_llm,
                prompt_dir=str(self._prompts_root / "refinement"),
            )
            # build aql_lookup: for live KG use the docs list directly
            # (aql_results_str is prose, not parseable JSON in that case)
            if kg_source == "live":
                aql_lookup = {d["uri"]: d for d in (live_docs or []) if d.get("uri")}
            else:
                aql_lookup = self._build_aql_lookup(aql_results_str, docs or [])
            documents_block = self._render_documents_block(
                full_docs, abstract_docs, excerpts, aql_lookup
            )
            # set_documents_block() raises RuntimeError if block is empty.
            ref_agent.set_documents_block(documents_block)

        # build query hint from profile for richer generation context
        query_hint = self._build_query_hint(question, profile)

        # abstracts mode: pass filtered aql_results_str
        # excerpts modes: pass None so process_context takes the v2 branch
        context_for_refinement = (
            aql_results_str
            if cfg.evidence_mode == "abstracts"
            else None
        )
        # live KG always provides its own rich context string regardless of mode
        if kg_source == "live" and cfg.evidence_mode == "abstracts":
            context_for_refinement = aql_results_str

        if cfg.evidence_mode == "abstracts" and not context_for_refinement:
            raise RuntimeError(
                f"AdaptivePipeline.run(): abstracts mode but context_for_refinement "
                f"is empty (aql_results_str len={len(aql_results_str or '')}, "
                f"kg_source={kg_source}). Cannot proceed."
            )

        # process_context() raises RuntimeError on LLM failure — let it propagate.
        refined = ref_agent.process_context(
            question=query_hint,
            structured_context="",
            ontology=ontology,
            include_ontology=True,
            aql_results_str=context_for_refinement,
            context_filter="full",
        )
        enriched = refined.enriched_context or ""

        # Double-check: enriched must be non-empty before generation.
        # This is belt-and-suspenders — process_context() should have already raised.
        if not enriched.strip():
            raise RuntimeError(
                f"AdaptivePipeline.run(): refinement produced empty enriched_context "
                f"for question '{question[:80]}...'. "
                "process_context() should have raised — this is a bug in the "
                "refinement agent. Aborting generation."
            )

        # ----------------------------------------------------------
        # generation
        # ----------------------------------------------------------
        gen_agent = GenerationAgent(
            self._llm(cfg.model_name, cfg.temperature_generate, cfg.max_output_tokens),
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

        # generate() raises RuntimeError on LLM failure — let it propagate.
        ans = gen_agent.generate(
            question=question,
            text_context=enriched,
            ontology=ontology,
            context_cap=cfg.gen_context_cap,
            max_output_tokens=cfg.max_output_tokens,
            system_prompt=cfg.system_prompt_modifier,
        )

        # citation hygiene:
        #   1. strip any References section the model wrote
        #   2. strip out-of-range [N] markers
        #   3. append authoritative References block (only cited IDs)
        final_answer = ans.answer or ""
        final_answer = AdaptivePipeline._strip_self_written_refs(final_answer)

        max_ref_id = len(ans.formatted_references or [])
        final_answer = AdaptivePipeline._strip_out_of_range_citations(
            final_answer, max_ref_id
        )

        cited_ids = AdaptivePipeline._extract_cited_ids(final_answer)
        filtered_refs: List[str] = []
        if ans.formatted_references and cited_ids:
            filtered_refs = AdaptivePipeline._filter_refs_by_cited_ids(
                ans.formatted_references, cited_ids
            )
            if filtered_refs:
                final_answer = (
                    final_answer.rstrip()
                    + "\n\nReferences\n"
                    + "\n".join(filtered_refs)
                )
        ans.formatted_references = filtered_refs

        return AdaptiveResult(
            answer=final_answer,
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
        """attempt live ArangoDB query; return [] if unavailable."""
        import os
        if not os.getenv("ARANGO_ROOT_PASSWORD"):
            return []
        try:
            from core.utils.arango_client import query_kg
        except ImportError:
            return []
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
                    sn_title   = sn.get("title", "")
                    sn_abstract = sn.get("abstract", "")
                    if sn_title or sn_abstract:
                        lines.append(f"  - **{sn_title}**: {sn_abstract[:300]}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_docs_from_str(raw: str) -> List[Dict[str, Any]]:
        if not raw or not raw.strip():
            return []
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
        except Exception:
            pass
        return []

    def _llm(self, model: str, temperature: float, max_tokens: int = 1400):
        key = (model, temperature, max_tokens)
        if key not in self._llm_cache:
            from core.utils.helpers import get_llm_model
            self._llm_cache[key] = get_llm_model(
                model=model, temperature=temperature, max_tokens=max_tokens
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
        """fetch excerpts for full_docs only. abstract_docs never reach here."""
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

    def _render_documents_block(
        self,
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
        excerpts: List[Dict[str, Any]],
        aql_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        """v2 unified documents block: full_docs with interleaved excerpts,
        followed by abstract_docs (abstract only). delegates to
        FullTextIndexer.render_documents_block for consistent formatting."""
        self._ensure_indexer()
        return self._indexer.render_documents_block(
            full_docs=full_docs,
            abstract_docs=abstract_docs,
            excerpts=excerpts,
            aql_lookup=aql_lookup,
        )

    @staticmethod
    def _render_excerpts_block(excerpts: List[Dict[str, Any]]) -> str:
        """Legacy flat excerpts block (kept for backwards compat / tier-1)."""
        if not excerpts:
            return ""
        parts: List[str] = []
        for ex in excerpts:
            header = (
                f"[Doc {ex.get('doc_index', '?')}] "
                f"{ex.get('title', 'Unknown')} | "
                f"Section: {ex.get('section', 'unknown')} | "
                f"p. {ex.get('page', '?')}"
            )
            parts.append(header)
            parts.append(ex.get("text", ""))
            parts.append("")
        return "\n".join(parts).strip()

    @staticmethod
    def _build_filtered_aql_str(
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
    ) -> str:
        """Rebuild aql_results_str from only full + abstract docs so the
        refinement prompt never receives dropped documents. Returns a
        compact JSON string matching the format expected by _parse_aql_for_prompt.
        """
        combined = list(full_docs) + list(abstract_docs)
        if not combined:
            return ""
        try:
            return json.dumps(combined, ensure_ascii=False)
        except Exception:
            return ""

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
        lookup: Dict[str, Dict[str, Any]] = {}
        for d in docs:
            uri = d.get("uri", "")
            if uri:
                lookup[uri] = d
        if aql_results_str:
            parsed_docs = AdaptivePipeline._parse_docs_from_str(aql_results_str)
            for item in parsed_docs:
                uri = item.get("uri", "")
                if uri and uri not in lookup:
                    lookup[uri] = item
        return lookup

    # ------------------------------------------------------------------
    # citation hygiene helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cited_ids(answer: str) -> List[int]:
        import re
        body = AdaptivePipeline._strip_self_written_refs(answer)
        ids = set()
        for m in re.finditer(r"\[([0-9][0-9,;\s]*)\]", body):
            for tok in re.split(r"[,;\s]+", m.group(1)):
                if tok.isdigit():
                    ids.add(int(tok))
        return sorted(ids)

    @staticmethod
    def _strip_out_of_range_citations(answer: str, max_id: int) -> str:
        import re
        lines = answer.split("\n")
        refs_start = None
        for i, line in enumerate(lines):
            s = line.strip()
            if s in ("References", "REFERENCES") or s.lower().startswith("### references"):
                refs_start = i
                break
        body_end = refs_start if refs_start is not None else len(lines)
        body = "\n".join(lines[:body_end])
        tail = "\n".join(lines[body_end:]) if refs_start is not None else ""

        def _sub(m: "re.Match") -> str:
            inner = m.group(1)
            tokens = re.split(r"([,;\s]+)", inner)
            kept_digits: List[str] = []
            for tok in tokens:
                if tok.isdigit():
                    n = int(tok)
                    if 1 <= n <= max_id:
                        kept_digits.append(tok)
            if not kept_digits:
                return ""
            return "[" + ", ".join(kept_digits) + "]"

        body = re.sub(r"\[([0-9][0-9,;\s]*)\]", _sub, body)
        body = re.sub(r"[ \t]+([,.;:])", r"\1", body)
        body = re.sub(r"[ \t]{2,}", " ", body)
        return body + (("\n" + tail) if tail else "")

    @staticmethod
    def _strip_self_written_refs(answer: str) -> str:
        import re
        lines = answer.split("\n")
        for i, line in enumerate(lines):
            s = line.strip().strip("*#").strip()
            if s in ("References", "REFERENCES"):
                return "\n".join(lines[:i]).rstrip()
        return answer

    @staticmethod
    def _filter_refs_by_cited_ids(
        formatted_refs: List[str], cited_ids: List[int]
    ) -> List[str]:
        import re
        cited = set(cited_ids)
        out: List[str] = []
        for line in formatted_refs:
            m = re.match(r"\s*\[(\d+)\]", line)
            if m and int(m.group(1)) in cited:
                out.append(line)
        return out

    @staticmethod
    def _build_query_hint(question: str, profile: QuestionProfile) -> str:
        hints = []
        if profile.complexity is not None:
            if profile.complexity < 0.50:
                hints.append(
                    "[DEPTH: concise -- this is a foundational question; "
                    "answer in 3-5 sentences without exhaustive enumeration]"
                )
            elif profile.complexity >= 0.75:
                hints.append(
                    "[DEPTH: comprehensive -- this is a high-complexity question; "
                    "provide mechanistic detail, quantitative evidence, and uncertainty assessment]"
                )
        if getattr(profile, "needs_numeric_emphasis", False):
            hints.append(
                "[NUMERIC EMPHASIS: prioritise excerpts with quantitative "
                "values, rates, and units]"
            )
        if profile.answer_shape and profile.answer_shape not in ("direct_paragraph", ""):
            hints.append(f"[SYNTHESIS TARGET: {profile.answer_shape}]")
        return ("\n".join(hints) + "\n" + question) if hints else question
