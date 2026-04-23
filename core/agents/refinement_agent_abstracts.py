"""refinement agent for abstracts evidence mode.

sends the raw aql results + ontology to the llm in a single prompt and
receives a ready-to-use text context for the generation agent.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.agents.base_refinement_agent import BaseRefinementAgent
from core.utils.data_models import (
    DocumentAssessment,
    DynamicOntology,
    RefinedContext,
)
from core.utils.openalex_client import OpenAlexClient, format_reference_from_metadata

_debug_logger = logging.getLogger("raw_prompts")


class RefinementAgentAbstracts(BaseRefinementAgent):
    def __init__(self, llm, prompt_dir: str = "prompts/refinement"):
        super().__init__("RefinementAgentAbstracts", llm, prompt_dir)

    def process_context(
        self,
        question: str,
        structured_context: str,
        ontology: Optional[DynamicOntology] = None,
        include_ontology: bool = True,
        aql_results_str: Optional[str] = None,
        context_filter: str = "full",
    ) -> RefinedContext:
        if not aql_results_str:
            raise RuntimeError(
                "RefinementAgentAbstracts: aql_results_str is empty; "
                "no document context available for refinement."
            )

        prompt, documents = self._build_refined_prompt(
            question, ontology, include_ontology, aql_results_str
        )

        refined_text = self._invoke_llm_text(prompt)

        est_tokens = len(refined_text) // 4
        self.logger.info(f"refined context: {len(refined_text)} chars (~{est_tokens} tokens)")

        assessments = self._minimal_assessments(documents)
        ont_summary = self._ontology_summary(ontology) if include_ontology and ontology else ""

        return RefinedContext(
            original_context=structured_context,
            question=question,
            ontology_summary=ont_summary,
            assessments=assessments,
            summary=f"refined context: {len(refined_text)} chars, ~{est_tokens} tokens",
            enriched_context=refined_text,
        )

    def _build_refined_prompt(
        self,
        question: str,
        ontology: Optional[DynamicOntology],
        include_ontology: bool,
        aql_results_str: Optional[str],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """return (filled prompt, parsed document list)."""
        template = self.load_prompt("refinement_1pass_refined_exp4.txt")
        if not template:
            raise ValueError("refined prompt template not found")

        ontology_text = (
            self._format_ontology(ontology)
            if include_ontology and ontology
            else "No ontology provided"
        )

        documents: List[Dict[str, Any]] = []
        if aql_results_str:
            aql_text, documents = self._parse_aql_for_prompt(aql_results_str)
        else:
            aql_text = "No AQL results provided"

        documents_text = self._format_documents_for_references(documents)

        prompt = (
            template.replace("{query}", question)
            .replace("{ontology}", ontology_text)
            .replace("{aql_results}", aql_text)
            .replace("{documents}", documents_text)
            .replace("{incomplete_references}", "")
        )

        _debug_logger.debug("refinement prompt:\n%s", prompt[:500])
        return prompt, documents

    @staticmethod
    def _parse_aql_for_prompt(aql_results_str: str) -> Tuple[str, List[Dict[str, Any]]]:
        """return (human-readable text, list of doc dicts)."""
        from core.utils.aql_parser import parse_aql_results

        compact_json = parse_aql_results(aql_results_str)
        try:
            documents = json.loads(compact_json)
            if not isinstance(documents, list):
                return compact_json, []
        except json.JSONDecodeError:
            return compact_json, []

        lines: List[str] = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", f"Document {i}")
            abstract = doc.get("abstract", "No abstract")
            preview = abstract[:200] + "..." if len(abstract) > 200 else abstract
            uri = doc.get("uri", "")
            lines.append(f"[{i}] {title}")
            lines.append(f"   Abstract: {preview}")
            if uri:
                lines.append(f"   URI: {uri}")
            lines.append("")

        return "\n".join(lines), documents

    @staticmethod
    def _format_ontology(ontology: DynamicOntology) -> str:
        unique: Dict[str, Any] = {}
        for av in ontology.attribute_value_pairs:
            existing = unique.get(av.attribute)
            if existing is None or av.centrality > existing.centrality:
                unique[av.attribute] = av

        sorted_attrs = sorted(unique.values(), key=lambda x: x.centrality, reverse=True)
        lines = ["ONTOLOGY ATTRIBUTES (sorted by relevance):"]
        for av in sorted_attrs:
            line = f"- {av.attribute}: {av.value}"
            if av.description:
                line += f" ({av.description})"
            line += f" (centrality: {av.centrality:.2f})"
            lines.append(line)
        return "\n".join(lines)

    def _format_documents_for_references(self, documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "All document information is contained within the AQL results section"

        formatted: List[str] = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("title_or_name") or doc.get("title", f"Document {i}")
            uri = doc.get("uri", "")
            if not title or not uri:
                continue

            metadata = None
            if uri and "openalex.org" in uri:
                metadata = OpenAlexClient.fetch_metadata(uri)

            if metadata:
                metadata["position"] = i
                formatted.append(format_reference_from_metadata(metadata))
            else:
                ref = f"[{i}] {title}. {uri}." if uri else f"[{i}] {title}."
                formatted.append(ref)

        return "\n".join(formatted)

    def _invoke_llm_text(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        if not response:
            raise RuntimeError(
                f"{self.__class__.__name__}: LLM returned empty/None response after all retries. "
                "Aborting — no fallback to ungrounded generation."
            )
        text = response.strip()
        if not text:
            raise RuntimeError(
                f"{self.__class__.__name__}: LLM response was whitespace-only after stripping."
            )
        return text

    def _minimal_assessments(
        self, documents: List[Dict[str, Any]]
    ) -> List[DocumentAssessment]:
        return [
            DocumentAssessment(
                title_or_name=doc.get("title_or_name", doc.get("title", f"Document {i}")),
                relevance_score=0.7,
                include=True,
                position=i,
                type="publication",
                scope="",
                caveats="",
                instruction="use for context",
            )
            for i, doc in enumerate(documents, 1)
        ]

    def _assess_documents(self, question, documents, ontology, include_ontology, aql_results_str=None):
        return self._minimal_assessments(documents)

    @staticmethod
    def _ontology_summary(ontology: DynamicOntology) -> str:
        if not ontology:
            return ""
        n_crit = len(ontology.get_critical_attributes())
        n_supp = len(ontology.get_contextual_attributes())
        return f"{n_crit} critical, {n_supp} supporting attributes"


# backward-compat alias
RefinementAgent1PassRefined = RefinementAgentAbstracts
