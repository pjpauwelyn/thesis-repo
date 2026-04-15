"""
dlr llm-as-a-judge evaluator.

uses openai gpt-4.1-mini to score answers on 5 criteria (factuality,
relevance, groundedness, helpfulness, depth) on a 1-5 integer scale with
equal weights, following the dlr evaluation rubric.

scoring rubric, score ranges, decision rules, and judge prompt content are
kept semantically identical to the original implementation.
"""

import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# prompt / criteria loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_criteria(path: Optional[Path] = None) -> list:
    """load dlr criteria definitions from json."""
    path = path or _PROMPTS_DIR / "dlr_criteria_scorer_desc.json"
    with open(path, "r") as fh:
        return json.load(fh)


def _load_score_prompt(path: Optional[Path] = None) -> str:
    """load the dlr scoring prompt template."""
    path = path or _PROMPTS_DIR / "score_answer_criteria_fixed.txt"
    with open(path, "r") as fh:
        return fh.read()


def _build_criteria_string(criteria_list: list) -> str:
    """format the criteria list into the string injected into the prompt."""
    parts = []
    for criterion in criteria_list:
        scores_str = "\n    ".join(
            f"{s}: {d}" for s, d in criterion["scores"].items()
        )
        part = "- {}\n   {}\n    {}".format(
            criterion["criteria"], criterion["desc"], scores_str,
        )
        parts.append(part)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------

# system prompt for the judge – kept functionally equivalent to the original
_SYSTEM_PROMPT = (
    "\nYou are an expert Earth Observation (EO) scientist and evaluation specialist.\n"
    "You assess AI-generated answer to a scientific EO question based on several "
    "criteria, using a 1-5 scale for each criterion. \n"
    "Your evaluation must follow the provided rubrics. Respond with valid JSON format.\n"
)


class DLREvaluator:
    """
    dlr-compatible evaluator.

    parameters
    ----------
    model_name : str
        openai model used for judging (default: gpt-4.1-mini).
    use_placeholders : bool
        when True, failed evaluations return -1 placeholder scores;
        when False they return neutral 3.0 scores.
    api_key : str | None
        explicit openai api key; falls back to OPENAI_API_KEY env var.
    criteria_path / prompt_path : Path | None
        override locations for the criteria json and prompt template.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        use_placeholders: bool = True,
        api_key: Optional[str] = None,
        criteria_path: Optional[Path] = None,
        prompt_path: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.use_placeholders = use_placeholders

        self.criteria_list = _load_criteria(criteria_path)
        self.score_prompt = _load_score_prompt(prompt_path)

        # all criteria weighted equally (matching dlr)
        self.weights = {c["criteria"].lower(): 1 for c in self.criteria_list}
        self.criteria_str = _build_criteria_string(self.criteria_list)

        import os
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.openai_client = OpenAI(api_key=key)

        # anomaly detection history
        self.score_history: Dict[str, list] = defaultdict(list)
        self.anomaly_thresholds = {
            "range": (1, 5),
            "std_dev": 1.5,
            "uniformity": 0.3,
        }

    # ------------------------------------------------------------------
    # score validation (unchanged semantics)
    # ------------------------------------------------------------------

    def _validate_scores(self, scores: Dict[str, int]) -> Dict[str, Any]:
        """
        validate that scores follow expected statistical patterns.

        returns dict with 'valid' bool, 'message' str, 'severity' str.
        """
        try:
            score_values = list(scores.values())
            valid_scores = [
                s for s in score_values if isinstance(s, int) and 1 <= s <= 5
            ]

            if len(valid_scores) != 5:
                return {
                    "valid": False,
                    "message": f"Expected 5 valid scores, got {len(valid_scores)}",
                    "severity": "high",
                }

            out_of_range = [s for s in valid_scores if s < 1 or s > 5]
            if out_of_range:
                return {
                    "valid": False,
                    "message": f"Scores out of range (1-5): {out_of_range}",
                    "severity": "high",
                }

            std_dev = (
                statistics.stdev(valid_scores)
                if len(set(valid_scores)) > 1
                else 0
            )

            if std_dev > self.anomaly_thresholds["std_dev"]:
                return {
                    "valid": False,
                    "message": (
                        f"High standard deviation: {std_dev:.2f} > "
                        f"{self.anomaly_thresholds['std_dev']}"
                    ),
                    "severity": "medium",
                }

            if std_dev < 0.5:
                if all(score == 5 for score in valid_scores):
                    mean_score = statistics.mean(valid_scores)
                    return {
                        "valid": True,
                        "message": (
                            f"Perfect score achieved: {mean_score:.1f}/5.0 - "
                            "All criteria scored maximum points"
                        ),
                        "severity": "info",
                    }
                return {
                    "valid": False,
                    "message": (
                        f"Unusually uniform scores (std_dev: {std_dev:.2f}), "
                        "may indicate lack of discrimination"
                    ),
                    "severity": "low",
                }

            mean_score = statistics.mean(valid_scores)
            if abs(mean_score - 3) > 1.5:
                return {
                    "valid": False,
                    "message": (
                        f"Extreme skewness: mean {mean_score:.1f} deviates "
                        "significantly from expected center (3.0)"
                    ),
                    "severity": "medium",
                }

            for criterion, score in scores.items():
                self.score_history[criterion].append(score)

            return {"valid": True, "message": "Scores appear normal", "severity": "none"}

        except Exception as e:
            logger.error("Score validation error: %s", e)
            return {
                "valid": False,
                "message": f"Validation error: {e}",
                "severity": "critical",
            }

    # ------------------------------------------------------------------
    # main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate_answer(
        self, question: str, answer: str, context: str = "",
    ) -> Dict[str, Any]:
        """
        evaluate a single answer using the dlr structured evaluation.

        returns dict with overall_score, criteria_scores, feedback, and
        optional validation_anomaly / anomaly_details.
        """
        instr = self.score_prompt.format(
            question=question,
            answer=answer,
            criteria=self.criteria_str,
            context=context,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": instr},
        ]

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                response_content = response.choices[0].message.content

                try:
                    response_data = json.loads(response_content)

                    # normalise capitalised keys returned by the model
                    if "Feedback" in response_data and "Scores" in response_data:
                        scores_data = response_data["Scores"]
                        response_data = {
                            "feedback": response_data["Feedback"],
                            "factuality": scores_data.get("Factuality", -1),
                            "relevance": scores_data.get("Relevance", -1),
                            "groundedness": scores_data.get("Groundedness", -1),
                            "helpfulness": scores_data.get("Helpfulness", -1),
                            "depth": scores_data.get("Depth", -1),
                        }

                    valid_scores = [
                        v
                        for k, v in response_data.items()
                        if k.lower() in self.weights and isinstance(v, int)
                    ]
                    overall_score = (
                        sum(valid_scores) / len(valid_scores)
                        if valid_scores
                        else 1.0
                    )

                    criteria_scores = {
                        "factuality": response_data.get("factuality", -1),
                        "relevance": response_data.get("relevance", -1),
                        "groundedness": response_data.get("groundedness", -1),
                        "helpfulness": response_data.get("helpfulness", -1),
                        "depth": response_data.get("depth", -1),
                    }

                    validation = self._validate_scores(criteria_scores)

                    if validation["valid"] and "Perfect score achieved" in validation["message"]:
                        logger.info("🎯 %s", validation["message"])

                    elif not validation["valid"]:
                        logger.warning(
                            "Score validation anomaly: %s", validation["message"],
                        )
                        return {
                            "overall_score": overall_score,
                            "criteria_scores": criteria_scores,
                            "feedback": (
                                response_data.get("feedback", "")
                                + f" [ANOMALY: {validation['message']}]"
                            ),
                            "validation_anomaly": True,
                            "anomaly_details": validation,
                        }

                    return {
                        "overall_score": overall_score,
                        "criteria_scores": criteria_scores,
                        "feedback": response_data.get("feedback", ""),
                        "validation_anomaly": False,
                    }

                except Exception as e:
                    logger.warning("Parsing error (attempt %d): %s", attempt + 1, e)
                    if attempt < max_retries - 1:
                        continue
                    return self._failure_result(
                        "GPT-4.1-Mini did not return valid JSON format"
                    )

            except Exception as e:
                logger.warning("API error (attempt %d): %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    continue
                return self._failure_result(
                    f"Failed after {max_retries} attempts - {e}"
                )

        # unreachable, but satisfies type checkers
        return self._failure_result("Unexpected evaluation exit")

    def _failure_result(self, reason: str) -> Dict[str, Any]:
        """build a placeholder or neutral fallback result."""
        if self.use_placeholders:
            return {
                "overall_score": -1,
                "criteria_scores": {c: -1 for c in self.weights},
                "feedback": f"Evaluation failed - {reason}. PENDING EVALUATION (-1).",
            }
        return {
            "overall_score": 3.0,
            "criteria_scores": {c: 3 for c in self.weights},
            "feedback": (
                f"Evaluation failed - {reason}. Using neutral fallback scores."
            ),
        }
