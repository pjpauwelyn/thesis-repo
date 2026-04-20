"""two-step generation agent: zero-shot draft then context-grounded refinement."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from core.agents.base_agent import BaseAgent
from core.utils.data_models import DynamicOntology


@dataclass
class Answer:
    answer: str
    references: List[str] = field(default_factory=list)
    formatted_references: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GenerationAgent(BaseAgent):
    def __init__(self, llm, prompt_dir: str = "prompts/generation"):
        super().__init__("GenerationAgent", llm)
        self.prompt_dir = prompt_dir
        self.zero_shot_template = self._load_template("zero_shot.txt")
        self.refinement_template = self._load_template("generation_prompt_exp4.txt")

    def _load_template(self, filename: str) -> str:
        try:
            filepath = os.path.join(self.prompt_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            self.logger.warning(f"template not found: {filename}")
            return ""

    def process(self, input_data):
        # required by base_agent but unused; callers use generate() directly
        pass

    def generate(
        self,
        question: str,
        text_context: str,
        ontology: Optional[DynamicOntology] = None,
        context_cap: int = 60_000,
        max_output_tokens: int = 700,
        system_prompt: str = "",
    ) -> Answer:
        self.logger.info(f"generating answer for: {question[:80]}")

        # cap context before any LLM call
        if context_cap > 0:
            text_context = text_context[:context_cap]

        # step 1: zero-shot draft (question only)
        zero_prompt = self.zero_shot_template.replace("{question}", question)
        draft = self._call_llm(zero_prompt, max_tokens=max_output_tokens)

        # step 2: refine with context + ontology
        refine_prompt = self._build_refinement_prompt(
            question, draft, text_context, ontology
        )
        final_answer = self._call_llm(
            refine_prompt, max_tokens=max_output_tokens, system=system_prompt
        )

        refs     = self._extract_references(final_answer)
        fmt_refs = self._extract_formatted_references(final_answer)

        # fallback: pull references from context if the answer lacks them
        if not fmt_refs and text_context:
            fmt_refs = self._extract_references_from_context(text_context)

        return Answer(answer=final_answer, references=refs, formatted_references=fmt_refs)

    # ------------------------------------------------------------------
    # prompt helpers
    # ------------------------------------------------------------------

    def _build_refinement_prompt(
        self,
        question: str,
        draft: str,
        context: str,
        ontology: Optional[DynamicOntology],
    ) -> str:
        prompt = self.refinement_template
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{draft_answer}", draft)
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
    # llm wrapper
    # ------------------------------------------------------------------

    def _call_llm(
        self, prompt: str, max_tokens: int = 700, system: str = ""
    ) -> str:
        try:
            if system:
                prompt = f"SYSTEM: {system}\n\n{prompt}"
            response = self.llm.invoke(
                prompt, force_json=False, max_tokens=max_tokens
            )
            return response.strip() if response else "Error: empty LLM response."
        except Exception as e:
            self.logger.error(f"llm generation failed: {e}")
            return "Error: Could not generate answer."

    # ------------------------------------------------------------------
    # reference extraction  (logic identical to original)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_references_section(text: str) -> Optional[int]:
        for i, line in enumerate(text.split("\n")):
            stripped = line.strip()
            if (
                ("References" in stripped or "REFERENCES" in stripped)
                and not stripped.startswith("[")
            ):
                return i
        return None

    def _extract_references(self, answer: str) -> List[str]:
        lines = answer.split("\n")
        start = self._find_references_section(answer)
        if start is None:
            return []
        refs: List[str] = []
        for line in lines[start + 1 :]:
            line = line.strip()
            if not line:
                break
            cleaned = line.strip("-•* []")
            if cleaned and "No author" not in cleaned and "[EVIDENCE]" not in cleaned:
                refs.append(" ".join(cleaned.split()))
        return refs

    def _extract_formatted_references(self, answer: str) -> List[str]:
        lines = answer.split("\n")
        start = self._find_references_section(answer)
        if start is None:
            return []
        refs: List[str] = []
        for line in lines[start + 1 :]:
            line = line.strip()
            if not line:
                break
            if line.startswith("[") and "]" in line and "[EVIDENCE]" not in line:
                refs.append(line)
        return refs

    @staticmethod
    def _extract_references_from_context(context: str) -> List[str]:
        if "[VALIDATED REFERENCES]" not in context:
            return []
        refs: List[str] = []
        in_section = False
        for line in context.split("\n"):
            line = line.strip()
            if line.startswith("[VALIDATED REFERENCES]"):
                in_section = True
                continue
            if in_section and line:
                if line.startswith("[") and "]" in line:
                    refs.append(line)
            elif in_section and not line:
                break
        return refs