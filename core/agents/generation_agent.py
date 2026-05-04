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

# Maximum doc index the LLM is allowed to cite.  Indices above this cap are
# considered hallucinated and are dropped before they reach the reference
# builder.  Consistent with the existing guard in pipeline.py (_MAX_CITE_INDEX).
_MAX_CITE_INDEX = 30


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
        draft_max_tokens: int = 1200,
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
            draft_max_tokens: token budget for the zero-shot draft call only.
                irrelevant when use_draft=False. default 1200.
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
            draft = self._call_llm(zero_prompt, max_tokens=draft_max_tokens)
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
                "generation no inline citation markers found -- "
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

        Supports both citation formats:
        - Sentinel format (primary): <<CITE:N>> markers emitted by the
          updated generation prompts. Unambiguous and collision-free.
        - Legacy format (fallback): [N] square-bracket markers from older
          prompts or cached test fixtures.

        Fix C: multi-cite pre-pass added here to mirror the expansion in
        pipeline._extract_sentinel_citations so Answer.cited_indices is
        always consistent with what the pipeline extracts from the same
        answer body.  Without the pre-pass, <<CITE:1,6>> emitted inside a
        markdown table cell would be silently skipped by the single-integer
        regex, leaving cited_indices under-counted relative to the pipeline
        extraction path.

        Indices above _MAX_CITE_INDEX (30) are treated as hallucinated and
        dropped so they never reach _build_verified_references without a
        corresponding index_remap entry.
        """
        indices: Set[int] = set()

        if "<<CITE:" in answer_body:
            # Fix C -- Pre-pass: expand <<CITE:N,M,...>> -> <<CITE:N>><<CITE:M>>...
            # The model occasionally emits comma-separated multi-cites inside
            # markdown table cells. Mirrors the pre-pass in pipeline.py's
            # _extract_sentinel_citations so both extraction paths stay in sync.
            def _expand(m: re.Match) -> str:
                parts = re.split(r"[\s,]+", m.group(1).strip())
                return "".join(f"<<CITE:{p}>>" for p in parts if p.isdigit())
            answer_body = re.sub(r"<<CITE:([\d,\s]+)>>", _expand, answer_body)

            for m in re.finditer(r"<<CITE:(\d+)>>", answer_body):
                n = int(m.group(1))
                if 1 <= n <= _MAX_CITE_INDEX:
                    indices.add(n)
            return indices

        # Legacy fallback: scan for [N] and [N,M,...] markers.
        for bracket in re.findall(r"\[([\d,\s]+)\]", answer_body):
            for token in bracket.split(","):
                token = token.strip()
                if token.isdigit():
                    n = int(token)
                    if 1 <= n <= _MAX_CITE_INDEX:
                        indices.add(n)
        return indices

    # ------------------------------------------------------------------
    # references section stripping
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_references_section(text: str) -> str:
        """Remove everything from the first References heading onward.

        Fix A: extended heading set now includes '[validated references]'
        and 'validated references' to catch the structured section header
        emitted by refinement_1pass_refined_exp4.txt when the generation
        LLM echoes it verbatim or paraphrased into the answer body.

        Post-body sweep: after the heading scan, any trailing lines that
        look like numbered bibliography entries (e.g. '[1] Smith et al.')
        are stripped from the tail of the text.  This catches reference
        blocks that leaked past the heading guard when the LLM reproduced
        the [VALIDATED REFERENCES] section content without emitting a
        recognisable section heading first.

        Uses an exact-match set to identify reference section headings.
        This prevents false-positive truncation on prose lines that merely
        contain the word 'references' mid-sentence, e.g.:
          'This references the methodology of Smith et al.'
          'Cross-references between datasets suggest...'

        Returns the original text unchanged if no heading or trailing ref
        lines are found.
        """
        _EXACT_HEADINGS = frozenset([
            "## references",
            "# references",
            "references",
            "sources",
            "bibliography",
            # Fix A: catch the structured section header from the refinement prompt
            "[validated references]",
            "validated references",
        ])
        lines = text.split("\n")
        for i, line in enumerate(lines):
            normalised = line.strip().lower().replace("*", "").replace("_", "").strip()
            if normalised in _EXACT_HEADINGS:
                return "\n".join(lines[:i]).rstrip()

        # Fix A -- Post-body sweep: strip trailing numbered bibliography lines
        # like "[1] Smith et al. ..." that leaked past the heading guard.
        # Only trims from the END of the text so in-body citation markers
        # (e.g. "Sea ice loss [1] is accelerating") are never touched.
        _REF_LINE_RE = re.compile(r"^\s*\[\d+\]\s+\S")
        while lines and _REF_LINE_RE.match(lines[-1]):
            lines.pop()
        return "\n".join(lines).rstrip()

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
        # The prompt templates already contain '### INITIAL DRAFT (refine against
        # the CONTEXT below)' as the section heading for the {draft_answer} slot.
        # The injected block must NOT prepend another '### INITIAL DRAFT' heading --
        # that would produce a duplicate nested heading in the rendered prompt:
        #
        #   ### INITIAL DRAFT (refine against the CONTEXT below)  <- template
        #   ### INITIAL DRAFT                                       <- duplicate (removed)
        #   You produced the following draft answer...
        #
        # When use_draft=True, inject only the explanatory preamble + draft text
        # directly under the template's existing heading.
        # When use_draft=False (or draft is empty), draft_block is an empty string
        # so {draft_answer} collapses cleanly; the template heading remains but has
        # no body, which the model treats as a no-op.
        if use_draft and draft:
            draft_block = (
                "You produced the following draft answer using your training knowledge. "
                "Use it as a starting point and refine it with the CONTEXT below:\n\n"
                + draft
            )
        else:
            draft_block = ""
        prompt = prompt.replace("{draft_answer}", draft_block)
        prompt = prompt.replace(
            "{context}",
            context.strip() if context else "No additional context available.",
        )

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
        {role: system} message before the user turn -- not as raw user text.
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
