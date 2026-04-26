"""two-step generation agent: zero-shot draft then context-grounded refinement."""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from core.agents.base_agent import BaseAgent
from core.utils.data_models import DynamicOntology

# 60% of the 128k token window shared by mistral-small-latest and
# mistral-large-latest, expressed in chars. this constant is the source of
# truth; rules.yaml and router.py carry the same value for config clarity.
_CONTEXT_WINDOW_60PCT_CHARS = 307_200  # 76_800 tokens x 4


@dataclass
class Answer:
    answer: str
    # NOTE: references and formatted_references on this object are always empty
    # lists.  pipeline._build_verified_references() is the sole authority; it
    # populates PipelineResult.formatted_references from the known-good doc list.
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

    def process(self, input_data: Dict[str, Any]) -> "Answer":
        """Thin dict-based wrapper around generate().

        Either routes to generate() when the input dict is valid, or raises
        NotImplementedError with an actionable message so callers notice the
        problem immediately.

        Expected keys in input_data:
            question      (str, required)
            text_context  (str, required)
            ontology      (DynamicOntology | None, optional)
            system_prompt (str, optional)
            use_draft     (bool, optional, default True)
        """
        if not isinstance(input_data, dict):
            raise NotImplementedError(
                "GenerationAgent.process() requires a dict with at least "
                "'question' and 'text_context' keys. "
                f"Got {type(input_data).__name__}. "
                "Use generate() directly for full control."
            )
        question = input_data.get("question")
        text_context = input_data.get("text_context")
        if not question or not text_context:
            raise NotImplementedError(
                "GenerationAgent.process(): input_data must contain non-empty "
                "'question' and 'text_context'. "
                f"Got question={question!r}, text_context={str(text_context)[:40]!r}."
            )
        return self.generate(
            question=question,
            text_context=text_context,
            ontology=input_data.get("ontology"),
            system_prompt=input_data.get("system_prompt", ""),
            use_draft=input_data.get("use_draft", True),
        )

    def generate(
        self,
        question: str,
        text_context: str,
        ontology: Optional[DynamicOntology] = None,
        context_cap: int = _CONTEXT_WINDOW_60PCT_CHARS,
        max_output_tokens: int = 700,
        system_prompt: str = "",
        use_draft: bool = True,
        generation_prompt: str = "generation_structured.txt",
    ) -> "Answer":
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
                Defaults to generation_structured.txt.
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
            # step 1: zero-shot draft (question only, no system prompt)
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

        return Answer(
            answer=answer_body,
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

        Handles both separate markers ([1] [2]) and grouped forms
        ([1,2,3] or [1, 2, 3]) that the upstream citation-format guard
        prevents from being written, but which may appear in pre-existing
        CSV docs or old test fixtures.
        """
        indices: Set[int] = set()
        for bracket in re.findall(r"\[([\d,\s]+)\]", answer_body):
            for token in bracket.split(","):
                token = token.strip()
                if token.isdigit():
                    indices.add(int(token))
        return indices

    # ------------------------------------------------------------------
    # references section stripping
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_references_section(text: str) -> str:
        """Remove everything from the first '## References' heading onward.

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
                return "\n".join(lines[:i]).rstrip()
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
        """Invoke the LLM with an optional system prompt and per-call max_tokens.

        system is forwarded to llm.invoke() as a proper kwarg so both
        MistralLLMWrapper and OpenRouterLLMWrapper insert it as a
        {role: system} message before the user turn — not as raw user text.
        max_tokens is forwarded so tier-specific token ceilings are honoured
        per call rather than only at LLM construction time.
        """
        response = self.llm.invoke(prompt, system=system, max_tokens=max_tokens)
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
