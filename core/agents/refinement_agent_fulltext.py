"""refinement agent variant that injects full-text excerpts into the prompt.

Reuses all of RefinementAgent1PassRefined (prompt building, ontology
formatting, reference enrichment).  Overrides only the prompt filename and
the prompt-construction step so it can fill either:
  - the new {documents_block} placeholder  (documents_block path, v2)
  - the legacy {aql_results}/{fulltext_excerpts} placeholders (backwards compat)

Hard-fail policy
----------------
This agent NEVER silently falls back to returning a placeholder string.
If the LLM call fails (None/empty response) or the documents_block is empty,
a RuntimeError is raised immediately so AdaptivePipeline.run() can propagate
the failure to the caller (smoke test / phase3_parallel retry loop).
There is no circumstance under which generation should proceed without a
grounded, non-empty refined context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
from core.utils.data_models import DynamicOntology, RefinedContext
from core.utils.helpers import setup_logging

_debug_logger = logging.getLogger("raw_prompts")


class RefinementAgent1PassFullText(RefinementAgent1PassRefined):
    """Drop-in replacement for the 1-pass refined agent that also consumes
    per-document full-text excerpts produced by FullTextIndexer."""

    PROMPT_FILENAME = "refinement_1pass_refined_fulltext.txt"

    def __init__(self, llm, prompt_dir: str = "prompts/refinement"):
        super().__init__(llm, prompt_dir=prompt_dir)
        # override the name inherited from RefinementAgent1PassRefined so logs
        # correctly identify this agent as RefinementAgent1PassFullText.
        self.name = "RefinementAgent1PassFullText"
        self.logger = setup_logging(self.name)
        self._current_excerpts_text: str = ""
        self._documents_block: Optional[str] = None

    # ------------------------------------------------------------------
    # runner hooks
    # ------------------------------------------------------------------

    def set_excerpts(self, excerpts_text: str) -> None:
        """Called by the runner per question before process_context (legacy path)."""
        self._current_excerpts_text = excerpts_text or ""

    def set_documents_block(self, documents_block: str) -> None:
        """Called by AdaptivePipeline.run() with the unified documents block (v2)."""
        if not documents_block or not documents_block.strip():
            raise RuntimeError(
                "RefinementAgent1PassFullText.set_documents_block(): "
                "documents_block is empty or whitespace-only. "
                "The pipeline must always provide non-empty document context "
                "before calling the refinement agent."
            )
        self._documents_block = documents_block

    # ------------------------------------------------------------------
    # overridden prompt builder
    # ------------------------------------------------------------------

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
            # v2 path: unified documents_block replaces both aql_results and
            # fulltext_excerpts slots.
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
            # legacy path: fill {aql_results} and {fulltext_excerpts} as before
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

    # ------------------------------------------------------------------
    # process_context: hard-fail on LLM error — no silent fallback
    # ------------------------------------------------------------------

    def process_context(
        self,
        question: str,
        structured_context: str,
        ontology: Optional[DynamicOntology] = None,
        include_ontology: bool = True,
        aql_results_str: Optional[str] = None,
        context_filter: str = "full",
    ) -> RefinedContext:
        # v2 path: documents_block is set; aql_results_str is None.
        # _documents_block emptiness is already checked in set_documents_block().
        if self._documents_block is not None:
            prompt, documents = self._build_refined_prompt(
                question, ontology, include_ontology, aql_results_str
            )
            # _invoke_llm_text raises RuntimeError on None/empty — let it propagate.
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

        # legacy path: delegate to parent.
        # Parent's process_context() now also hard-fails on empty aql_results_str
        # and on empty LLM response.
        return super().process_context(
            question=question,
            structured_context=structured_context,
            ontology=ontology,
            include_ontology=include_ontology,
            aql_results_str=aql_results_str,
            context_filter=context_filter,
        )
