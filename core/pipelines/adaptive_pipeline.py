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
        excerpts, excerpt_stats = self._build_evidence(cfg, question, ontology, docs)

        # ----------------------------------------------------------
        # refinement
        # ----------------------------------------------------------
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
            excerpts_text = self._render_excerpts_block(excerpts) if excerpts else ""
            ref_agent.set_excerpts(excerpts_text)

        if hasattr(ref_agent, "set_scope_filter"):
            ref_agent.set_scope_filter(cfg.scope_filter)

        # when we have live KG docs, pass the rich context string directly;
        # otherwise fall through to the existing aql_results_str path
        context_for_refinement = (
            aql_results_str if kg_source == "live"
            else (aql_results_str or "")
        )

        # build query hint from profile for richer generation context
        query_hint = self._build_query_hint(question, profile)

        refined = ref_agent.process_context(
            question=query_hint,
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

        ans = gen_agent.generate(
            question=question,
            text_context=enriched,
            ontology=ontology,
            context_cap=cfg.gen_context_cap,
            max_output_tokens=cfg.max_output_tokens,
            system_prompt=cfg.system_prompt_modifier,
        )

        # Normalise the answer's citation hygiene. The generation prompt now
        # forbids the model from writing its own References section (to prevent
        # ID hallucination + truncation), and we take three defensive steps
        # here:
        #   1. strip any References section the model wrote anyway
        #   2. strip any [N] citation marker whose N is outside the validated
        #      reference range (prevents Q3-style [7] / [12] hallucinations)
        #   3. append the authoritative References block we own, trimmed to
        #      just the IDs actually cited in the final answer
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

    @staticmethod
    def _render_excerpts_block(excerpts: List[Dict[str, Any]]) -> str:
        """Render a list of excerpt dicts (from select_excerpts_for_question)
        into a plain-text block suitable for injection into the refinement
        agent prompt.

        Each excerpt dict has keys: doc_index, title, work_id, section,
        page, text, tokens, score.
        """
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
            parts.append("")  # blank line between excerpts
        return "\n".join(parts).strip()

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

    # ------------------------------------------------------------------
    # citation hygiene helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cited_ids(answer: str) -> List[int]:
        """Return sorted, deduplicated list of [N] citation IDs appearing in
        the body of the answer (ignoring any existing References section).
        """
        import re
        body = AdaptivePipeline._strip_self_written_refs(answer)
        ids = set()
        # match [1], [1, 2], [1,2,3], [1; 2]
        for m in re.finditer(r"\[([0-9][0-9,;\s]*)\]", body):
            for tok in re.split(r"[,;\s]+", m.group(1)):
                if tok.isdigit():
                    ids.add(int(tok))
        return sorted(ids)

    @staticmethod
    def _strip_out_of_range_citations(answer: str, max_id: int) -> str:
        """Remove any [N] marker in the body whose N is > max_id or <= 0.
        Leaves the References section (if any) alone.
        """
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
            # split keeping separators; keep only in-range digit tokens and
            # the separators between them
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
        # tidy double spaces / space-before-punct left by removals
        body = re.sub(r"[ \t]+([,.;:])", r"\1", body)
        body = re.sub(r"[ \t]{2,}", " ", body)
        return body + (("\n" + tail) if tail else "")

    @staticmethod
    def _strip_self_written_refs(answer: str) -> str:
        """Remove any References / REFERENCES section the LLM wrote.
        Handles markdown headings (### References), plain heading lines, and
        bold forms (**References**).
        """
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
        """Return the subset of formatted_refs whose leading [N] ID appears in
        cited_ids. Preserves original order and leading IDs verbatim.
        """
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
