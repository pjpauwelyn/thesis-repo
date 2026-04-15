"""refinement agent for the no-ontology experiment.

compared to the ontology-based refinement_agent_1pass_refined:
  - the ontology parameter is always ignored
  - prompt template references to "ontology" are replaced with
    "the user question" so the llm conditions only on the query
  - the ontology section in the prompt is filled with a neutral
    placeholder rather than structured attribute-value pairs
  - everything else (aql parsing, reference formatting, llm call
    flow) is identical so the comparison is fair
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


class RefinementAgentNoOntology(BaseRefinementAgent):
    """single-pass refinement that never uses an ontology."""

    def __init__(self, llm, prompt_dir: str = "prompts/refinement"):
        super().__init__("RefinementAgentNoOntology", llm, prompt_dir)

    # ------------------------------------------------------------------
    # main entry – ontology is accepted for interface compat but ignored
    # ------------------------------------------------------------------

    def process_context(
        self,
        question: str,
        structured_context: str,
        ontology: Optional[DynamicOntology] = None,
        include_ontology: bool = False,  # default False for this experiment
        aql_results_str: Optional[str] = None,
        context_filter: str = "full",
    ) -> RefinedContext:
        # changed: ontology is never forwarded; include_ontology forced to False
        if not aql_results_str:
            return self._empty_context(question, ontology=None)

        prompt, documents = self._build_prompt(question, aql_results_str)

        refined_text = self._invoke_llm_text(prompt)
        est_tokens = len(refined_text) // 4
        self.logger.info(f"refined context (no ontology): {len(refined_text)} chars (~{est_tokens} tokens)")

        assessments = self._minimal_assessments(documents)

        return RefinedContext(
            original_context=structured_context,
            question=question,
            ontology_summary="",  # changed: no ontology summary
            assessments=assessments,
            summary=f"refined context (no ontology): {len(refined_text)} chars, ~{est_tokens} tokens",
            enriched_context=refined_text,
        )

    # ------------------------------------------------------------------
    # prompt construction – ontology slot replaced with neutral text
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        aql_results_str: Optional[str],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        # changed: loads the no-ontology prompt variant
        template = self.load_prompt("refinement_1pass_no_ontology.txt")
        if not template:
            raise ValueError("no-ontology prompt template not found")

        documents: List[Dict[str, Any]] = []
        if aql_results_str:
            aql_text, documents = self._parse_aql_for_prompt(aql_results_str)
        else:
            aql_text = "No AQL results provided"

        documents_text = self._format_documents_for_references(documents)

        # changed: {ontology} placeholder filled with neutral instruction
        prompt = (
            template.replace("{query}", question)
            .replace("{ontology}", "No ontology is used in this experiment. Focus only on the user question.")
            .replace("{aql_results}", aql_text)
            .replace("{documents}", documents_text)
            .replace("{incomplete_references}", "")
        )

        _debug_logger.debug("no-ontology refinement prompt:\n%s", prompt[:500])
        return prompt, documents

    # ------------------------------------------------------------------
    # helpers reused from the ontology variant (no changes needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_aql_for_prompt(aql_results_str: str) -> Tuple[str, List[Dict[str, Any]]]:
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
        try:
            response = self.llm.invoke(prompt)
            if response:
                return response.strip()
            self.logger.warning("empty response from llm")
            return "No refined context generated"
        except Exception as e:
            self.logger.error(f"refined context generation failed: {e}")
            return "Error generating refined context"

    def _minimal_assessments(self, documents: List[Dict[str, Any]]) -> List[DocumentAssessment]:
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
