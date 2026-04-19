"""
deepeval-based evaluator.

provides hallucination, answer relevancy, faithfulness, contextual relevancy
metrics (and optionally coherence, toxicity, bias when deepeval >= 3.8.8).

"""

import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# conditional deepeval imports
# ---------------------------------------------------------------------------

DEEPEVAL_AVAILABLE = False
HAS_NEW_METRICS = False

try:
    from deepeval import evaluate as deepeval_evaluate  # noqa: F401
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase

    try:
        from deepeval.metrics import BiasMetric, CoherenceMetric, ToxicityMetric

        HAS_NEW_METRICS = True
    except ImportError:
        HAS_NEW_METRICS = False

    DEEPEVAL_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# metric name normalisation
# ---------------------------------------------------------------------------

_METRIC_NAME_MAP = {
    "answerrelevancy": "relevancy",
    "contextualrelevancy": "contextual_relevancy",
}


def _normalise_metric_name(cls_name: str) -> str:
    raw = cls_name.replace("Metric", "").lower()
    return _METRIC_NAME_MAP.get(raw, raw)


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------


class DeepEvalEvaluator:
    """
    robust deepeval evaluator with optional fallback and resumption support.

    parameters
    ----------
    use_fallbacks : bool
        when True, failed metrics get a historical-average fallback score;
        when False they get -1.
    faithfulness_enabled : bool
        set False to skip the often-slow faithfulness metric.
    timeout_seconds : int
        per-evaluation timeout (covers all metrics for one answer).
    """

    # base metrics that are always attempted
    BASE_METRICS = ["hallucination", "relevancy", "faithfulness", "contextual_relevancy"]
    # extra metrics only when deepeval >= 3.8.8
    EXTRA_METRICS = ["coherence", "toxicity", "bias"]

    def __init__(
        self,
        use_fallbacks: bool = False,
        faithfulness_enabled: bool = True,
        timeout_seconds: int = 300,
    ):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "deepeval is not installed. Run: pip install deepeval"
            )

        self.use_fallbacks = use_fallbacks
        self.faithfulness_enabled = faithfulness_enabled
        self.timeout_seconds = timeout_seconds

        # metric score history for dynamic fallbacks
        self.metric_histories: Dict[str, List[float]] = {
            m: [] for m in self.BASE_METRICS
        }
        if HAS_NEW_METRICS:
            self.metric_histories.update({m: [] for m in self.EXTRA_METRICS})

        # adaptive faithfulness disabling
        self._faithfulness_checks = 0
        self._faithfulness_failures = 0

        self._setup_openai()

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_openai() -> None:
        """ensure openai key is available and configure deepeval for reliability."""
        key = os.getenv("OPENAI_API_KEY") or os.getenv("MISTRAL_API_KEY")
        if key:
            os.environ["OPENAI_API_KEY"] = key
        os.environ["DEEPVAL_ASYNC_MODE"] = "False"
        os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "60"

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def create_test_case(
        self, question: str, answer: str, context: str,
    ) -> "LLMTestCase":
        """build a deepeval test case from raw strings."""
        return LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=answer,
            context=[context] if context else [],
            retrieval_context=[context] if context else [],
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str = "",
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        evaluate a single answer with the requested (or default) metrics.

        returns a dict mapping metric names to float scores (0-1), plus an
        ``overall_quality`` key.  failed metrics receive -1 when
        *use_fallbacks* is False, or the historical average otherwise.
        """
        test_case = self.create_test_case(question, answer, context)

        if metrics is None:
            metrics = list(self.BASE_METRICS)
            if HAS_NEW_METRICS:
                metrics.extend(self.EXTRA_METRICS)

        scores: Dict[str, float] = {}

        for metric_name in metrics:
            if metric_name == "faithfulness" and not self.faithfulness_enabled:
                scores[metric_name] = -1
                continue
            score = self._evaluate_single_metric(metric_name, test_case)
            scores[metric_name] = score

        # overall quality from valid scores
        valid = [s for s in scores.values() if s >= 0]
        scores["overall_quality"] = sum(valid) / len(valid) if valid else -1
        return scores

    def evaluate_relevancy_only(
        self, question: str, answer: str,
    ) -> float:
        """
        lightweight relevancy-only evaluation (replaces robust_deepeval_relevancy).

        returns a float score (0-1) or -1 on failure.
        """
        try:
            test_case = LLMTestCase(input=question, actual_output=answer)
            metric = AnswerRelevancyMetric(threshold=0.5)
            result = deepeval_evaluate([test_case], [metric])

            if hasattr(result, "test_results") and result.test_results:
                tr = result.test_results[0]
                if hasattr(tr, "metrics_data") and tr.metrics_data:
                    md = tr.metrics_data[0]
                    if hasattr(md, "score"):
                        return float(md.score)
                    if getattr(md, "success", False):
                        return 1.0
                    return 0.0
            if hasattr(result, "score"):
                return float(result.score)
            return -1.0
        except Exception:
            return -1.0

    # ------------------------------------------------------------------
    # fallback helpers
    # ------------------------------------------------------------------

    def _fallback_score(self, metric_name: str) -> float:
        """return -1 or the historical average depending on config."""
        if not self.use_fallbacks:
            return -1
        history = self.metric_histories.get(metric_name, [])
        if not history:
            return 0.5
        return sum(history) / len(history)

    def load_histories_from_csv(self, filepath: str) -> None:
        """pre-populate metric histories from an existing results csv."""
        import pandas as pd

        if not os.path.exists(filepath):
            return
        df = pd.read_csv(filepath)
        for name in self.metric_histories:
            if name not in df.columns:
                continue
            valid = df[(df[name] > 0) & (df[name] != 0.5) & (df[name] <= 1.0)][name].tolist()
            if valid:
                self.metric_histories[name] = valid

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _evaluate_single_metric(
        self, metric_name: str, test_case: "LLMTestCase",
    ) -> float:
        """evaluate one metric, returning a float score or fallback."""
        metric_obj = self._build_metric(metric_name)
        if metric_obj is None:
            return self._fallback_score(metric_name)

        try:
            metric_obj.measure(test_case)
            score = getattr(metric_obj, "score", None)
            if score is None:
                score = getattr(metric_obj, "get_score", lambda: -1)()

            if isinstance(score, (int, float)) and 0 <= score <= 1.0:
                self.metric_histories.setdefault(metric_name, []).append(score)
                return float(score)
            return self._fallback_score(metric_name)

        except Exception:
            if metric_name == "faithfulness":
                self._faithfulness_checks += 1
                self._faithfulness_failures += 1
                if (
                    self._faithfulness_checks >= 3
                    and self._faithfulness_failures / self._faithfulness_checks > 0.5
                ):
                    self.faithfulness_enabled = False
            return self._fallback_score(metric_name)

    @staticmethod
    def _build_metric(name: str):
        """instantiate a deepeval metric object by name."""
        mapping = {
            "hallucination": lambda: HallucinationMetric(threshold=0.5),
            "relevancy": lambda: AnswerRelevancyMetric(threshold=0.5),
            "faithfulness": lambda: FaithfulnessMetric(threshold=0.5),
            "contextual_relevancy": lambda: ContextualRelevancyMetric(threshold=0.5),
        }
        if HAS_NEW_METRICS:
            mapping.update({
                "coherence": lambda: CoherenceMetric(threshold=0.5),
                "toxicity": lambda: ToxicityMetric(threshold=0.5),
                "bias": lambda: BiasMetric(threshold=0.5),
            })
        factory = mapping.get(name)
        return factory() if factory else None
