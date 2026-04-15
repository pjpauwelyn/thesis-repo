"""
synthetic data and lightweight llm mocks for offline testing.

the mock llm returns plausible-looking text or json so the full pipeline
can execute end-to-end without hitting the mistral or openai api.
prompt files that are missing on disk are also stubbed here.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tests_framework.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# mock llm – drop-in for MistralLLMWrapper
# ---------------------------------------------------------------------------


class MockLLM:
    """
    lightweight mock that mimics MistralLLMWrapper.invoke().

    returns canned responses based on prompt keywords so the pipeline
    agents get data in the right shape without an api call.
    """

    def __init__(self):
        self.call_log: List[Dict[str, Any]] = []
        self._total_tokens_estimate = 0

    # ---- MistralLLMWrapper interface ------------------------------------

    def invoke(
        self, prompt_text: str, force_json: bool = False,
    ) -> Optional[Union[dict, str]]:
        self.call_log.append({
            "prompt_length": len(prompt_text),
            "force_json": force_json,
        })
        self._total_tokens_estimate += len(prompt_text) // 4

        if force_json:
            return self._json_response(prompt_text)
        return self._text_response(prompt_text)

    # ---- internal response generators -----------------------------------

    @staticmethod
    def _json_response(prompt: str) -> dict:
        """return a json dict that matches what the agent expects."""
        prompt_lower = prompt.lower()

        # ontology agent – extract_attributes_slim
        if "attribute" in prompt_lower and "value" in prompt_lower:
            return {
                "pairs": [
                    {
                        "attribute": "satellite_mission",
                        "value": "Sentinel-3",
                        "description": "earth observation mission for sst",
                        "centrality": 0.9,
                    },
                    {
                        "attribute": "spatial_resolution",
                        "value": "1 km",
                        "description": "typical sst grid spacing",
                        "centrality": 0.7,
                    },
                ],
            }

        # ontology agent – extract_relationships
        if "relationship" in prompt_lower:
            return {
                "relationships": [
                    {
                        "source_attribute": "satellite_mission",
                        "source_value": "Sentinel-3",
                        "target_attribute": "spatial_resolution",
                        "target_value": "1 km",
                        "relationship_type": "provides",
                        "logical_constraint": "SLSTR instrument resolution",
                    },
                ],
            }

        # refinement agent – assessments (list of dicts)
        if "assess" in prompt_lower or "document" in prompt_lower:
            return {
                "assessments": [
                    {
                        "title_or_name": "Sentinel-3 SST",
                        "relevance_score": 0.85,
                        "scope": "global sst monitoring",
                        "caveats": "none",
                        "instruction": "include fully",
                        "include": True,
                    },
                ],
            }

        # generic fallback
        return {"result": "mock json response"}

    @staticmethod
    def _text_response(prompt: str) -> str:
        """return realistic-looking text for the generation agent."""
        prompt_lower = prompt.lower()

        # refinement agent – refined context text
        if "refine" in prompt_lower or "context" in prompt_lower:
            return (
                "## Refined Context\n\n"
                "The Sentinel-3 mission carries the SLSTR instrument which "
                "measures sea surface temperature at 1 km resolution. MODIS "
                "on Aqua and Terra provides complementary SST data. VIIRS on "
                "Suomi NPP achieves 750 m resolution. These missions together "
                "enable global ocean monitoring for climate research and "
                "operational weather forecasting."
            )

        # generation agent – zero-shot draft
        if "zero" in prompt_lower or "draft" in prompt_lower or "question" in prompt_lower:
            return (
                "Satellite missions commonly used for monitoring global sea "
                "surface temperature include Sentinel-3 with SLSTR at 1 km "
                "resolution, MODIS on Aqua/Terra, and VIIRS on Suomi NPP at "
                "750 m resolution."
            )

        # generation agent – final answer
        return (
            "Several satellite missions are routinely employed for monitoring "
            "global sea surface temperature (SST). The Sentinel-3 mission, "
            "equipped with the SLSTR instrument, delivers SST measurements at "
            "approximately 1 km spatial resolution. MODIS aboard the Aqua and "
            "Terra satellites provides SST data at comparable resolution. "
            "Additionally, the VIIRS instrument on the Suomi NPP satellite "
            "achieves a finer 750 m resolution. Collectively, these missions "
            "support climate research, weather forecasting, and operational "
            "oceanography.\n\n"
            "References:\n"
            "[1] Sentinel-3 Sea Surface Temperature Monitoring\n"
            "[2] VIIRS SST Products from Suomi NPP"
        )

    @property
    def token_usage_estimate(self) -> int:
        return self._total_tokens_estimate


# ---------------------------------------------------------------------------
# mock openai evaluator client
# ---------------------------------------------------------------------------


class _MockChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})()


class _MockResponse:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]


class _MockCompletions:
    def create(self, **kwargs):
        payload = json.dumps({
            "factuality": 4,
            "relevance": 4,
            "groundedness": 3,
            "helpfulness": 4,
            "depth": 3,
            "feedback": (
                "The answer correctly identifies major SST satellite "
                "missions and their spatial resolutions. The context is "
                "well grounded. Depth could be improved with temporal "
                "coverage details."
            ),
        })
        return _MockResponse(payload)


class _MockChat:
    completions = _MockCompletions()


class MockOpenAIClient:
    """
    drop-in replacement for openai.OpenAI used inside DLREvaluator.

    returns a valid json evaluation payload so the evaluator's parsing
    and validation logic can be exercised without an api call.
    """
    chat = _MockChat()


# ---------------------------------------------------------------------------
# prompt-file stubs
# ---------------------------------------------------------------------------

_PROMPT_STUBS: Dict[str, str] = {
    "prompts/ontology/extract_attributes_slim.txt": (
        "Extract attribute-value pairs from this question.\n"
        "Question: {question}\n"
        "Respond with JSON containing a 'pairs' list."
    ),
    "prompts/ontology/extract_relationships.txt": (
        "Define logical relationships between these attributes.\n"
        "Pairs: {pairs}\n"
        "Question: {question}\n"
        "Respond with JSON containing a 'relationships' list."
    ),
    "prompts/generation/zero_shot.txt": (
        "Answer the following question based on your knowledge.\n"
        "Question: {question}\n"
    ),
    "prompts/refinement/refinement_prompt_1pass_refined.txt": (
        "Refine the following context for the question.\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Ontology: {ontology}\n"
    ),
}


def ensure_prompt_stubs() -> List[str]:
    """
    create minimal prompt files for any templates the pipeline expects
    but that are missing on disk.  returns paths of files created.
    """
    created: List[str] = []
    for rel_path, content in _PROMPT_STUBS.items():
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            created.append(str(full_path))
    return created


def cleanup_prompt_stubs(paths: List[str]) -> None:
    """remove stub files created by ensure_prompt_stubs()."""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# synthetic input csv
# ---------------------------------------------------------------------------


def write_synthetic_csv(path: Path, questions: List[Dict[str, str]]) -> Path:
    """
    write a minimal csv that PipelineOrchestrator._load_csv() can read.
    columns: question_id, question, structured_context, aql_results
    """
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question_id", "question", "structured_context", "aql_results"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for q in questions:
            w.writerow({fn: q.get(fn, "") for fn in fieldnames})
    return path
