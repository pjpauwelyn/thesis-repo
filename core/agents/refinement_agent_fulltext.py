"""refinement agent variant that injects full-text excerpts into the prompt.

This subclass reuses every piece of RefinementAgent1PassRefined (prompt building,
ontology formatting, reference enrichment) and only overrides the template name
plus the prompt-construction step so it fills the new `{fulltext_excerpts}`
placeholder.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
from core.utils.data_models import DynamicOntology, RefinedContext

_debug_logger = logging.getLogger("raw_prompts")


class RefinementAgent1PassFullText(RefinementAgent1PassRefined):
    """Drop-in replacement for the 1-pass refined agent that also consumes
    per-document full-text excerpts produced by FullTextIndexer."""

    PROMPT_FILENAME = "refinement_1pass_refined_fulltext.txt"

    def __init__(self, llm, prompt_dir: str = "prompts/refinement"):
        super().__init__(llm, prompt_dir=prompt_dir)
        # holds the excerpts for the current question; set by the runner
        self._current_excerpts_text: str = ""

    # ------------------------------------------------------------------
    # runner hook
    # ------------------------------------------------------------------

    def set_excerpts(self, excerpts_text: str) -> None:
        """Called by the runner *per question* before process_context."""
        self._current_excerpts_text = excerpts_text or ""

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

        excerpts_text = self._current_excerpts_text or (
            "No full-text excerpts available for this question (abstracts only)."
        )

        prompt = (
            template.replace("{query}", question)
            .replace("{ontology}", ontology_text)
            .replace("{aql_results}", aql_text)
            .replace("{documents}", documents_text)
            .replace("{fulltext_excerpts}", excerpts_text)
            .replace("{incomplete_references}", "")
        )

        _debug_logger.debug("fulltext refinement prompt head:\n%s", prompt[:500])
        return prompt, documents
