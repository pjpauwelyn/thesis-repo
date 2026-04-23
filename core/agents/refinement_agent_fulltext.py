"""refinement agent variant that injects full-text excerpts into the prompt.

Reuses all of RefinementAgentAbstracts (prompt building, ontology
formatting, reference enrichment). Overrides only the prompt filename and
the prompt-construction step so it can fill either:
  - the {documents_block} placeholder  (v2, primary path)
  - the legacy {aql_results}/{fulltext_excerpts} placeholders (backwards compat)

Hard-fail policy: this agent never silently falls back to a placeholder
string. If the LLM call fails or the documents_block is empty, a
RuntimeError is raised so Pipeline.run() can propagate the failure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.agents.refinement_agent_abstracts import RefinementAgentAbstracts
from core.utils.data_models import DynamicOntology, RefinedContext
from core.utils.helpers import setup_logging

_debug_logger = logging.getLogger("raw_prompts")


class RefinementAgent1PassFullText(RefinementAgentAbstracts):
    """Drop-in replacement for RefinementAgentAbstracts that also consumes
    per-document full-text excerpts produced by FullTextIndexer."""

    PROMPT_FILENAME = "refinement_1pass_refined_fulltext.txt"

    def __init__(self, llm, prompt_dir: str = "prompts/refinement"):
        super().__init__(llm, prompt_dir=prompt_dir)
        self.name = "RefinementAgent1PassFullText"
        self.logger = setup_logging(self.name)
        self._current_excerpts_text: str = ""
        self._documents_block: Optional[str] = None

    def set_excerpts(self, excerpts_text: str) -> None:
        """Called by the runner per question before process_context (legacy path)."""
        self._current_excerpts_text = excerpts_text or ""

    def set_documents_block(self, documents_block: str) -> None:
        """Called by Pipeline.run() with the unified documents block (v2)."""
        if not documents_block or not documents_block.strip():
            raise RuntimeError(
                "RefinementAgent1PassFullText.set_documents_block(): "
                "documents_block is empty or whitespace-only. "
                "The pipeline must always provide non-empty document context "
                "before calling the refinement agent."
            )
        self._documents_block = documents_block

    def _build_refined_prompt(
        self,
        question: str,
        ontology: Optional[DynamicOntology],
        include_ontology: bool,
        aql_results_str: Optional[str],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        template = self.load_prompt(self.PROMPT_FILENAME)
        if not template:
            raise ValueError(
                f"fulltext refinement prompt not found: {self.PROMPT_FILENAME}"
            )

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

        if self._documents_block is not None:
            prompt = (
                template
                .replace("{query}", question)
                .replace("{ontology}", ontology_text)
                .replace("{documents_block}", self._documents_block)
                .replace("{aql_results}", "")
                .replace("{fulltext_excerpts}", "")
                .replace("{documents}", documents_text)
                .replace("{incomplete_references}", "")
            )
        else:
            excerpts_text = self._current_excerpts_text or (
                "No full-text excerpts available for this question (abstracts only)."
            )
            prompt = (
                template
                .replace("{query}", question)
                .replace("{ontology}", ontology_text)
                .replace("{aql_results}", aql_text)
                .replace("{documents}", documents_text)
                .replace("{fulltext_excerpts}", excerpts_text)
                .replace("{incomplete_references}", "")
            )

        _debug_logger.debug("fulltext refinement prompt head:\n%s", prompt[:500])
        return prompt, documents

    def process_context(
        self,
        question: str,
        structured_context: str,
        ontology: Optional[DynamicOntology] = None,
        include_ontology: bool = True,
        aql_results_str: Optional[str] = None,
        context_filter: str = "full",
    ) -> RefinedContext:
        if self._documents_block is not None:
            prompt, documents = self._build_refined_prompt(
                question, ontology, include_ontology, aql_results_str
            )
            refined_text = self._invoke_llm_text(prompt)
            est_tokens = len(refined_text) // 4
            self.logger.info(
                "fulltext refined context: %d chars (~%d tokens)",
                len(refined_text), est_tokens,
            )
            assessments = self._minimal_assessments(documents)
            ont_summary = (
                self._ontology_summary(ontology)
                if include_ontology and ontology else ""
            )
            return RefinedContext(
                original_context=structured_context,
                question=question,
                ontology_summary=ont_summary,
                assessments=assessments,
                summary=f"fulltext refined context: {len(refined_text)} chars",
                enriched_context=refined_text,
            )

        return super().process_context(
            question=question,
            structured_context=structured_context,
            ontology=ontology,
            include_ontology=include_ontology,
            aql_results_str=aql_results_str,
            context_filter=context_filter,
        )
