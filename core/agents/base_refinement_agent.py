"""shared logic for all refinement agent variants."""

import os
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from core.agents.base_agent import BaseAgent
from core.utils.data_models import (
    DocumentAssessment,
    DynamicOntology,
    RefinedContext,
)


class BaseRefinementAgent(BaseAgent):
    def __init__(self, name: str, llm, prompt_dir: str = "prompts/refinement"):
        super().__init__(name, llm)
        self.prompt_dir = prompt_dir

    def load_prompt(self, filename: str) -> str:
        filepath = os.path.join(self.prompt_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            self.logger.warning(f"prompt not found: {filepath}")
            return ""

    def process_context(
        self,
        question: str,
        structured_context: str,
        ontology: Optional[DynamicOntology] = None,
        include_ontology: bool = True,
        aql_results_str: Optional[str] = None,
        context_filter: str = "full",
    ) -> RefinedContext:
        documents = self._parse_documents(structured_context, aql_results_str)
        self.logger.info(f"parsed {len(documents)} documents")

        if not documents:
            return self._empty_context(question, ontology)

        if include_ontology and ontology:
            documents = self._reorder_by_ontology(documents, ontology)

        assessments = self._assess_documents(
            question, documents, ontology, include_ontology, aql_results_str
        )
        enriched = self._build_enriched_context(
            assessments, context_filter, structured_context
        )

        included = [a for a in assessments if a.include]
        excluded_n = len(assessments) - len(included)
        summary = f"assessed {len(assessments)} docs: {len(included)} included, {excluded_n} excluded"

        ontology_summary = ""
        if include_ontology and ontology:
            n_crit = len(ontology.get_critical_attributes())
            n_supp = len(ontology.get_contextual_attributes())
            ontology_summary = f"critical: {n_crit} attrs, supporting: {n_supp} attrs"

        return RefinedContext(
            original_context=structured_context,
            question=question,
            ontology_summary=ontology_summary,
            assessments=assessments,
            summary=summary,
            enriched_context=enriched,
        )

    def process(self, input_data: Any) -> RefinedContext:
        """base_agent compatibility shim."""
        if isinstance(input_data, dict):
            return self.process_context(
                question=input_data.get("question", ""),
                structured_context=input_data.get("structured_context", ""),
                ontology=input_data.get("ontology"),
                include_ontology=input_data.get("include_ontology", True),
                aql_results_str=input_data.get("aql_results_str"),
            )
        raise ValueError("input must be a dict with question, structured_context, etc.")

    @abstractmethod
    def _assess_documents(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        ontology: Optional[DynamicOntology],
        include_ontology: bool,
        aql_results_str: Optional[str],
    ) -> List[DocumentAssessment]:
        ...

    def _parse_documents(
        self, context_string: str, aql_results_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []

        if "RELATED DOCUMENTS" in context_string:
            docs_section = context_string.split("RELATED DOCUMENTS")[1]
            if "RELATED TOPICS" in docs_section:
                docs_section = docs_section.split("RELATED TOPICS")[0]

            for i, block in enumerate(docs_section.split("---"), 1):
                block = block.strip()
                if not block or block.startswith("#"):
                    continue

                title_m = re.search(r"[#\s]*Title[:\s]+([^\n]+)", block, re.MULTILINE)
                content_m = re.search(
                    r"[#\s]*Content[:\s]+(.+?)(?:---|$)", block, re.MULTILINE | re.DOTALL
                )

                if title_m:
                    title = title_m.group(1).strip()
                elif content_m:
                    first_sent = content_m.group(1).strip().split(".")[0]
                    title = f"EO Study on {first_sent[:50]}..." if first_sent else f"Document {i}"
                else:
                    title = f"Document {i}"

                content = content_m.group(1).strip() if content_m else block[:500]

                documents.append(
                    {
                        "id": f"doc{i}",
                        "title": title,
                        "title_or_name": title,
                        "position": i,
                        "type": "publication",
                        "content": content[:3000],
                        "aql_data": "",
                    }
                )

        if aql_results_str and documents:
            from core.utils.aql_parser import parse_aql_results
            clean_aql = parse_aql_results(aql_results_str)
            documents[0]["aql_data"] = clean_aql

        self.logger.info(f"parsed {len(documents)} documents from structured context")
        return documents

    def _reorder_by_ontology(
        self,
        documents: List[Dict[str, Any]],
        ontology: Optional[DynamicOntology],
    ) -> List[Dict[str, Any]]:
        """sort documents by keyword overlap with critical ontology concepts."""
        if not ontology or len(documents) <= 1:
            return documents

        key_concepts = [av.attribute.lower() for av in ontology.get_critical_attributes()]
        if not key_concepts:
            return documents

        def score(doc: Dict[str, Any]) -> int:
            haystack = (doc.get("content", "") + " " + doc.get("title", "")).lower()
            return sum(c in haystack for c in key_concepts)

        return sorted(documents, key=score, reverse=True)

    def _build_enriched_context(
        self,
        assessments: List[DocumentAssessment],
        context_filter: str = "full",
        structured_context: str = "",
    ) -> str:
        included = [a for a in assessments if a.include]
        if not included:
            return "No documents included after assessment."

        if context_filter == "scores_only":
            return self._build_scores_only_context(included, structured_context)
        if context_filter == "slim":
            return self._build_slim_context(included)
        return self._build_full_context(included)

    def _build_full_context(self, assessments: List[DocumentAssessment]) -> str:
        lines = ["# Refined Context"]
        for idx, a in enumerate(assessments, 1):
            title = self._clean_title(a.title_or_name)
            lines.append(f"\n## Document {idx}: {title}")
            lines.append(f"**Relevance:** {a.relevance_score:.2f}")
            if a.geographic_scope:
                geo = f"**Geographic Scope:** {a.geographic_scope}"
                if a.geographic_match:
                    geo += f" ({a.geographic_match})"
                lines.append(geo)
            if a.temporal_scope:
                tmp = f"**Temporal Scope:** {a.temporal_scope}"
                if a.temporal_match:
                    tmp += f" ({a.temporal_match})"
                lines.append(tmp)
            if a.what_is_relevant:
                lines.append(f"**Relevant Content:** {a.what_is_relevant}")
            if a.context_limitations:
                lines.append(f"**Limitations:** {a.context_limitations}")
            if a.raw_metrics_text:
                lines.append(f"**Metrics:** {a.raw_metrics_text}")
            if a.spatial_context_text:
                lines.append(f"**Spatial Context:** {a.spatial_context_text}")
            if a.temporal_context_text:
                lines.append(f"**Temporal Context (detailed):** {a.temporal_context_text}")
            if a.supporting_details_text:
                lines.append(f"**Supporting Details:** {a.supporting_details_text}")
            if a.generation_instruction:
                lines.append(f"**Usage Guidance:** {a.generation_instruction}")
            self._append_citation_lines(lines, a)
        return "\n".join(lines)

    def _build_slim_context(self, assessments: List[DocumentAssessment]) -> str:
        lines = ["# Refined Context (Slim)"]
        for idx, a in enumerate(assessments, 1):
            title = self._clean_title(a.title_or_name)
            lines.append(f"\n## Document {idx}: {title}")
            lines.append(f"**Relevance:** {a.relevance_score:.2f}")
            if a.geographic_scope:
                geo = f"**Geographic Scope:** {a.geographic_scope}"
                if a.geographic_match and a.geographic_match.lower() != "none":
                    geo += f" ({a.geographic_match})"
                lines.append(geo)
            if a.temporal_scope:
                tmp = f"**Temporal Scope:** {a.temporal_scope}"
                if a.temporal_match and a.temporal_match.lower() != "none":
                    tmp += f" ({a.temporal_match})"
                lines.append(tmp)
            if a.what_is_relevant:
                lines.append(f"**Relevant Content:** {a.what_is_relevant}")
            if a.supporting_details_text:
                lines.append(f"**Supporting Details:** {a.supporting_details_text}")
            if a.instruction and a.instruction.strip():
                lines.append(f"**Usage Guidance:** {a.instruction}")
            self._append_citation_lines(lines, a)
        return "\n".join(lines)

    def _build_scores_only_context(
        self,
        assessments: List[DocumentAssessment],
        structured_context: str,
    ) -> str:
        lines = [structured_context.strip(), "\n" + "=" * 60, "# Document Relevance Scores"]
        for idx, a in enumerate(assessments, 1):
            title = self._clean_title(a.title_or_name)
            lines.append(f"\n## Document {idx}: {title}")
            lines.append(f"**Relevance Score:** {a.relevance_score:.2f}")
            if a.generation_instruction:
                lines.append(f"**Usage Guidance:** {a.generation_instruction}")
            self._append_citation_lines(lines, a)
        return "\n".join(lines)

    @staticmethod
    def _clean_title(title: str) -> str:
        if not title or not title.strip():
            return "Earth Observation Study"
        cleaned = title.replace("Document Title", "").replace("Title:", "").replace(":", "").strip()
        if not cleaned or cleaned.lower() in {"unknown", "none", "n/a"}:
            return "Earth Observation Study"
        return cleaned

    @staticmethod
    def _append_citation_lines(lines: List[str], a: DocumentAssessment) -> None:
        if a.citation:
            lines.append(f"**Citation:** {a.citation}")
        if a.full_reference:
            lines.append(f"**Full Reference:** {a.full_reference}")

    def _empty_context(
        self, question: str, ontology: Optional[DynamicOntology]
    ) -> RefinedContext:
        ont_summary = ""
        if ontology:
            ont_summary = f"{len(ontology.attribute_value_pairs)} attributes extracted"
        return RefinedContext(
            original_context="",
            question=question,
            ontology_summary=ont_summary,
            assessments=[],
            summary="no documents to refine",
            enriched_context="No documents available",
        )
