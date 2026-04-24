"""two-step generation agent: zero-shot draft then context-grounded refinement."""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set

from core.agents.base_agent import BaseAgent
from core.utils.data_models import DynamicOntology

# 60% of the 128k token window shared by mistral-small-latest and
# mistral-large-latest, expressed in chars. this constant is the source of
# truth; rules.yaml and router.py carry the same value for config clarity.
_CONTEXT_WINDOW_60PCT_CHARS = 307_200  # 76_800 tokens x 4


@dataclass
class Answer:
    answer: str
    references: List[str] = field(default_factory=list)
    formatted_references: List[str] = field(default_factory=list)
    # set of 1-based doc indices the LLM cited inline, e.g. {1, 3, 4}
    # populated by generate(); used by pipeline.py to build verified refs
    cited_indices: Set[int] = field(default_factory=set)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GenerationAgent(BaseAgent):
    def __init__(self, llm, prompt_dir: str = "prompts/generation"):
        super().__init__("GenerationAgent", llm)
        self.prompt_dir = prompt_dir
        self.zero_shot_template = self._load_template("zero_shot.txt")

    def _load_template(self, filename: str) -> str:
        try:
            filepath = os.path.join(self.prompt_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            self.logger.warning(f"template not found: {filename}")
            return ""

    def process(self, input_data):
        pass

    def generate(
        self,
        question: str,
        text_context: str,
        ontology: Optional[DynamicOntology] = None,
        context_cap: int = _CONTEXT_WINDOW_60PCT_CHARS,
        max_output_tokens: int = 700,
        system_prompt: str = "",
        use_draft: bool = True,
        generation_prompt: str = "generation_prompt_exp4.txt",
    ) -> Answer:
        """Generate an answer for a question.

        Args:
            use_draft: When True (default), generates a zero-shot draft first
                then refines it against the provided context. When False,
                skips the draft step entirely and goes straight to
                context-grounded generation. Set to False for high-context
                tiers (tier-2, tier-3) where a draft anchors the model to
                parametric knowledge and fights against context grounding.
            generation_prompt: filename of the generation prompt template to
                use, relative to prompt_dir. Passed from cfg.generation_prompt
                by the pipeline so each tier uses its declared prompt.
                Defaults to generation_prompt_exp4.txt (abstracts / draft path).
        """
        self.logger.info(
            f"generating answer for: {question[:80]} (use_draft={use_draft})"
        )

        # Guard: context must be non-empty.
        if not text_context or not text_context.strip():
            raise RuntimeError(
                "GenerationAgent.generate(): text_context is empty. "
                "Refusing to generate an ungrounded answer."
            )

        # Guard: context must not exceed the 60% window ceiling.
        ctx_len = len(text_context)
        effective_cap = context_cap if context_cap > 0 else _CONTEXT_WINDOW_60PCT_CHARS
        if ctx_len > effective_cap:
            raise RuntimeError(
                f"GenerationAgent.generate(): text_context length {ctx_len:,} chars "
                f"exceeds the 60% context window ceiling of {effective_cap:,} chars "
                f"({effective_cap // 4:,} tokens). "
                "This indicates runaway excerpt output upstream. "
                "Aborting -- do not truncate evidence silently."
            )

        # Load the generation prompt declared by the tier's routing config.
        template = self._load_template(generation_prompt)
        if not template:
            raise RuntimeError(
                f"GenerationAgent.generate(): generation prompt '{generation_prompt}' "
                "not found or empty. Cannot proceed."
            )

        if use_draft:
            # step 1: zero-shot draft (question only)
            zero_prompt = self.zero_shot_template.replace("{question}", question)
            draft = self._call_llm(zero_prompt, max_tokens=max_output_tokens)
        else:
            # skip draft -- context-grounded generation only
            draft = ""

        # step 2: context-grounded generation (with or without draft)
        refine_prompt = self._build_generation_prompt(
            template, question, draft, text_context, ontology, use_draft=use_draft
        )
        raw_answer = self._call_llm(
            refine_prompt, max_tokens=max_output_tokens, system=system_prompt
        )

        # Strip any ## References section the LLM may have written despite
        # the prompt instruction not to.  The pipeline appends a verified
        # ## References block built from the known-good doc list, so any
        # LLM-produced section would duplicate or conflict with it.
        answer_body = self._strip_references_section(raw_answer)

        # Extract which doc indices the LLM cited inline so the pipeline
        # can build a filtered, verified reference list.
        cited = self._extract_cited_indices(answer_body)

        self.logger.info(
            "generation produced %d chars, cited indices: %s",
            len(answer_body),
            sorted(cited) if cited else "none",
        )
        if not cited:
            self.logger.warning(
                "generation no inline [N] markers found -- "
                "pipeline will attach all available references"
            )

        # Keep legacy extraction for backward compat (tests that inspect
        # Answer.references / Answer.formatted_references directly).
        refs     = self._extract_references(raw_answer)
        fmt_refs = self._extract_formatted_references(raw_answer)

        return Answer(
            answer=answer_body,
            references=refs,
            formatted_references=fmt_refs,
            cited_indices=cited,
        )

    # ------------------------------------------------------------------
    # cited_indices extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cited_indices(answer_body: str) -> Set[int]:
        """Return the set of 1-based integer doc indices cited inline.

        Scans only the answer body (after _strip_references_section has
        removed any ## References block), so stray numbers in a reference
        list do not inflate the set.

        Matches [N] and [N, M] and [N][M] patterns.
        """
        return {int(m) for m in re.findall(r"\[(\d+)\]", answer_body)}

    # ------------------------------------------------------------------
    # references section stripping
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_references_section(text: str) -> str:
        """Remove everything from the first '## References' heading onward.

        Handles the same LLM formatting variations as _find_references_section.
        Returns the original text unchanged if no heading is found.
        """
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip().lstrip("#").strip().strip("*_ ")
            if (
                "references" in stripped.lower()
                and not stripped.lstrip("*_ ").startswith("[")
            ):
                # Keep everything before this line, strip trailing blank lines.
                body = "\n".join(lines[:i]).rstrip()
                return body
        return text

    # ------------------------------------------------------------------
    # prompt building
    # ------------------------------------------------------------------

    def _build_generation_prompt(
        self,
        template: str,
        question: str,
        draft: str,
        context: str,
        ontology: Optional[DynamicOntology],
        use_draft: bool = True,
    ) -> str:
        prompt = template
        prompt = prompt.replace("{question}", question)
        # When use_draft=False the draft is empty string; replace placeholder
        # so templates that include {draft_answer} don't break.
        prompt = prompt.replace("{draft_answer}", draft if use_draft else "")
        prompt = prompt.replace(
            "{context}",
            context.strip() if context else "No additional context available.",
        )

        # Strip unfilled directive placeholders used in generation_structured.txt.
        # These are currently not populated by the pipeline; removing them keeps
        # the prompt clean without altering any instructions.
        prompt = prompt.replace("{answer_shape_directives}", "")
        prompt = prompt.replace("{synthesis_mode_directives}", "")

        if ontology and ontology.attribute_value_pairs:
            ont_lines = [
                f"- {av.attribute}: {av.value} ({av.description})"
                for av in ontology.attribute_value_pairs
            ]
            if ontology.logical_relationships:
                ont_lines.append("\nRelationships:")
                for rel in ontology.logical_relationships:
                    ont_lines.append(
                        f"  - {rel.source_attribute} {rel.relationship_type} {rel.target_attribute}"
                    )
            prompt = prompt.replace("{ontology}", "\n".join(ont_lines))
        else:
            prompt = prompt.replace("{ontology}", "No ontology constraints.")

        return prompt

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self, prompt: str, max_tokens: int = 700, system: str = ""
    ) -> str:
        if system:
            prompt = f"SYSTEM: {system}\n\n{prompt}"
        response = self.llm.invoke(prompt)
        if not response:
            raise RuntimeError(
                "GenerationAgent._call_llm(): LLM returned empty/None response. "
                "Aborting generation -- no silent fallback."
            )
        text = response.strip()
        if not text:
            raise RuntimeError(
                "GenerationAgent._call_llm(): LLM response was whitespace-only."
            )
        return text

    # ------------------------------------------------------------------
    # legacy reference extraction (kept for backward compat)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_references_section(text: str) -> Optional[int]:
        """Return the line index of the '## References' heading, or None."""
        for i, line in enumerate(text.split("\n")):
            stripped = line.strip()
            stripped = stripped.lstrip("#").strip()
            stripped = stripped.strip("*_ ")
            if (
                "references" in stripped.lower()
                and not stripped.lstrip("*_ ").startswith("[")
            ):
                return i
        return None

    def _extract_references(self, answer: str) -> List[str]:
        """Extract plain reference strings (legacy; pipeline.py is now authoritative)."""
        lines = answer.split("\n")
        start = self._find_references_section(answer)

        if start is not None:
            refs: List[str] = []
            for line in lines[start + 1:]:
                stripped = line.strip()
                if stripped.startswith("##") and refs:
                    break
                if not stripped:
                    continue
                cleaned = stripped.strip("-\u2022* []")
                if cleaned and "No author" not in cleaned and "[EVIDENCE]" not in cleaned:
                    refs.append(" ".join(cleaned.split()))
            if refs:
                return refs

        fallback: List[str] = []
        for line in lines:
            stripped = line.strip()
            cleaned = stripped.strip("-\u2022* []")
            if (
                cleaned
                and "openalex.org/W" in cleaned
                and "[EVIDENCE]" not in cleaned
            ):
                fallback.append(" ".join(cleaned.split()))
        return fallback

    def _extract_formatted_references(self, answer: str) -> List[str]:
        """Extract formatted '[N] Author...' lines (legacy; pipeline.py is now authoritative)."""
        lines = answer.split("\n")
        start = self._find_references_section(answer)

        if start is not None:
            refs: List[str] = []
            for line in lines[start + 1:]:
                stripped = line.strip()
                if stripped.startswith("##") and refs:
                    break
                if not stripped:
                    continue
                normalised = re.sub(r"^(\d+)\.\s+", r"[\1] ", stripped)
                if (
                    normalised.startswith("[")
                    and "]" in normalised
                    and "[EVIDENCE]" not in normalised
                ):
                    refs.append(normalised)
            if refs:
                return refs

        fallback: List[str] = []
        for line in lines:
            stripped = line.strip()
            normalised = re.sub(r"^(\d+)\.\s+", r"[\1] ", stripped)
            if (
                normalised.startswith("[")
                and "]" in normalised
                and "openalex.org/W" in normalised
                and "[EVIDENCE]" not in normalised
            ):
                fallback.append(normalised)
        return fallback
