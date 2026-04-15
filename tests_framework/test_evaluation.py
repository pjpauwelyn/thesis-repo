"""
end-to-end tests for the dlr evaluation layer.

exercises DLREvaluator.evaluate_answer() with a mock openai client
to verify scoring logic, validation, anomaly detection, and edge cases
without api calls.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

from tests_framework.config import (
    PROJECT_ROOT,
    TEST_QUESTIONS,
    Thresholds,
)
from tests_framework.fixtures import MockOpenAIClient
from tests_framework.checks import (
    CheckResult,
    check_eval_score_range,
    check_eval_required_criteria,
    check_eval_overall_score,
    check_eval_feedback_present,
    check_no_failure_markers,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class EvalTestResult:
    """aggregated outcome of a single evaluation test scenario."""
    scenario: str
    question_id: str
    question: str
    passed: bool = True
    checks: List[CheckResult] = field(default_factory=list)
    overall_score: float = 0.0
    criteria_scores: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    has_anomaly: bool = False
    error: Optional[str] = None

    def add_check(self, cr: CheckResult):
        self.checks.append(cr)
        if not cr.passed and cr.severity == "error":
            self.passed = False


# ---------------------------------------------------------------------------
# helpers to build mock openai responses
# ---------------------------------------------------------------------------


def _make_mock_openai_response(scores: Dict[str, int], feedback: str) -> MagicMock:
    """build a MagicMock that behaves like openai chat completion."""
    payload = json.dumps({**scores, "feedback": feedback})
    choice = MagicMock()
    choice.message.content = payload
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# evaluation test suite
# ---------------------------------------------------------------------------


class EvaluationTestSuite:
    """runs end-to-end evaluation scenarios with mocked openai."""

    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds

    def run_all(self) -> List[EvalTestResult]:
        results: List[EvalTestResult] = []
        results.extend(self._test_normal_evaluation())
        results.extend(self._test_perfect_scores())
        results.extend(self._test_failure_placeholder_scores())
        results.extend(self._test_anomaly_detection())
        results.extend(self._test_empty_answer())
        return results

    # ---- scenario: normal evaluation ------------------------------------

    def _test_normal_evaluation(self) -> List[EvalTestResult]:
        """evaluate representative questions with typical scores."""
        scenario = "normal_evaluation"
        results: List[EvalTestResult] = []

        for q in TEST_QUESTIONS:
            tr = self._run_single_eval(
                scenario=scenario,
                question=q["question"],
                question_id=q["question_id"],
                answer=(
                    "Sentinel-3 with SLSTR provides 1 km SST data. "
                    "VIIRS on Suomi NPP achieves 750 m resolution."
                ),
                context=q["structured_context"],
                mock_scores={
                    "factuality": 4,
                    "relevance": 4,
                    "groundedness": 3,
                    "helpfulness": 4,
                    "depth": 3,
                },
                mock_feedback="Good answer with accurate mission details.",
            )
            results.append(tr)
        return results

    # ---- scenario: perfect scores ---------------------------------------

    def _test_perfect_scores(self) -> List[EvalTestResult]:
        """verify evaluator handles all-5 scores (triggers info-level note)."""
        tr = self._run_single_eval(
            scenario="perfect_scores",
            question=TEST_QUESTIONS[0]["question"],
            question_id="901",
            answer="Comprehensive answer covering all aspects.",
            context=TEST_QUESTIONS[0]["structured_context"],
            mock_scores={
                "factuality": 5,
                "relevance": 5,
                "groundedness": 5,
                "helpfulness": 5,
                "depth": 5,
            },
            mock_feedback="Perfect score achieved across all criteria.",
        )
        # extra check: overall should be exactly 5.0
        tr.add_check(CheckResult(
            name="perfect_overall",
            passed=abs(tr.overall_score - 5.0) < 0.01,
            message=f"overall={tr.overall_score:.2f}, expected 5.0",
            severity="error",
        ))
        return [tr]

    # ---- scenario: failure / placeholder scores -------------------------

    def _test_failure_placeholder_scores(self) -> List[EvalTestResult]:
        """verify evaluator returns -1 placeholders when api fails."""
        scenario = "failure_placeholders"
        tr = EvalTestResult(
            scenario=scenario,
            question_id="901",
            question=TEST_QUESTIONS[0]["question"][:80],
        )

        try:
            import sys
            if str(PROJECT_ROOT.parent) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT.parent))

            # patch openai client to raise an exception
            with patch("evaluation.dlr_evaluator.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create.side_effect = Exception(
                    "API connection error (mock)"
                )
                mock_cls.return_value = mock_client

                # also need to bypass the api key check
                with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                    from evaluation.dlr_evaluator import DLREvaluator

                    evaluator = DLREvaluator(
                        use_placeholders=True,
                        api_key="test-key",
                    )
                    # override the client after init
                    evaluator.openai_client = mock_client

                    result = evaluator.evaluate_answer(
                        question=TEST_QUESTIONS[0]["question"],
                        answer="some answer",
                        context="some context",
                    )

            # the evaluator should return -1 placeholders
            tr.overall_score = result.get("overall_score", 0)
            tr.criteria_scores = result.get("criteria_scores", {})
            tr.feedback = result.get("feedback", "")

            tr.add_check(CheckResult(
                name="placeholder_overall",
                passed=tr.overall_score == -1,
                message=f"overall={tr.overall_score}, expected -1",
                severity="error",
            ))
            for criterion, score in tr.criteria_scores.items():
                tr.add_check(CheckResult(
                    name=f"placeholder_{criterion}",
                    passed=score == -1,
                    message=f"{criterion}={score}, expected -1",
                    severity="error",
                ))

        except Exception as exc:
            tr.passed = False
            tr.error = str(exc)
            tr.add_check(CheckResult(
                name="scenario_execution",
                passed=False,
                message=f"scenario failed: {exc}",
                severity="error",
            ))

        return [tr]

    # ---- scenario: anomaly detection ------------------------------------

    def _test_anomaly_detection(self) -> List[EvalTestResult]:
        """verify _validate_scores flags suspicious patterns."""
        scenario = "anomaly_detection"
        results: List[EvalTestResult] = []

        # uniform low scores (should flag low std-dev)
        tr = self._run_single_eval(
            scenario=scenario + "_uniform",
            question=TEST_QUESTIONS[0]["question"],
            question_id="901",
            answer="Generic answer.",
            context="",
            mock_scores={
                "factuality": 3,
                "relevance": 3,
                "groundedness": 3,
                "helpfulness": 3,
                "depth": 3,
            },
            mock_feedback="All scores are identical.",
        )
        # the validator should flag this as anomalous (uniform, low std-dev)
        tr.add_check(CheckResult(
            name="anomaly_flagged",
            passed=tr.has_anomaly,
            message=(
                "anomaly detected as expected"
                if tr.has_anomaly
                else "expected anomaly flag for uniform scores"
            ),
            severity="warning",
        ))
        results.append(tr)

        return results

    # ---- scenario: empty answer -----------------------------------------

    def _test_empty_answer(self) -> List[EvalTestResult]:
        """verify evaluator handles an empty answer string gracefully."""
        tr = self._run_single_eval(
            scenario="empty_answer",
            question=TEST_QUESTIONS[0]["question"],
            question_id="901",
            answer="",
            context=TEST_QUESTIONS[0]["structured_context"],
            mock_scores={
                "factuality": 1,
                "relevance": 1,
                "groundedness": 1,
                "helpfulness": 1,
                "depth": 1,
            },
            mock_feedback="The answer is empty.",
        )
        return [tr]

    # ---- core execution logic -------------------------------------------

    def _run_single_eval(
        self,
        scenario: str,
        question: str,
        question_id: str,
        answer: str,
        context: str,
        mock_scores: Dict[str, int],
        mock_feedback: str,
    ) -> EvalTestResult:
        tr = EvalTestResult(
            scenario=scenario,
            question_id=question_id,
            question=question[:80],
        )

        try:
            import sys
            if str(PROJECT_ROOT.parent) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT.parent))

            mock_response = _make_mock_openai_response(mock_scores, mock_feedback)

            with patch("evaluation.dlr_evaluator.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_cls.return_value = mock_client

                with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                    from evaluation.dlr_evaluator import DLREvaluator

                    evaluator = DLREvaluator(
                        use_placeholders=True,
                        api_key="test-key",
                    )
                    evaluator.openai_client = mock_client

                    result = evaluator.evaluate_answer(
                        question=question,
                        answer=answer,
                        context=context,
                    )

            tr.overall_score = result.get("overall_score", 0)
            tr.criteria_scores = result.get("criteria_scores", {})
            tr.feedback = result.get("feedback", "")
            tr.has_anomaly = result.get("validation_anomaly", False)

            # standard checks
            tr.add_check(check_eval_required_criteria(
                tr.criteria_scores,
                self.thresholds.required_criteria,
            ))
            tr.add_check(check_eval_score_range(
                tr.criteria_scores,
                self.thresholds.valid_score_range,
            ))
            tr.add_check(check_eval_overall_score(
                tr.overall_score,
                self.thresholds.min_overall_score,
            ))
            tr.add_check(check_eval_feedback_present(tr.feedback))

        except Exception as exc:
            tr.passed = False
            tr.error = str(exc)
            tr.add_check(CheckResult(
                name="scenario_execution",
                passed=False,
                message=f"scenario '{scenario}' raised: {exc}",
                severity="error",
            ))

        return tr
