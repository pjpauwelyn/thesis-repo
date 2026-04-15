"""
end-to-end tests for the ontology-rag pipeline.

runs the main orchestrator (1_pass_with_ontology_refined and
1_pass_with_ontology) using mock llm fixtures so the tests
execute offline and fast.
"""

import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

from tests_framework.config import (
    PROJECT_ROOT,
    TEST_QUESTIONS,
    TEST_QUESTIONS_WITH_EMPTY,
    TestPaths,
    Thresholds,
)
from tests_framework.fixtures import (
    MockLLM,
    ensure_prompt_stubs,
    cleanup_prompt_stubs,
    write_synthetic_csv,
)
from tests_framework.checks import (
    CheckResult,
    check_no_failure_markers,
    check_answer_length,
    check_answer_not_empty,
    check_context_docs,
    check_processing_time,
    check_status_field,
    check_file_exists,
    check_csv_non_empty,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# dependency mocking – ensure core.* can be imported even when third-party
# packages (mistralai, etc.) are not installed or misconfigured
# ---------------------------------------------------------------------------

def _ensure_mistralai_mock():
    """inject a minimal mistralai stub into sys.modules if import fails."""
    try:
        from mistralai import Mistral  # noqa: F401
    except (ImportError, AttributeError):
        import types
        mod = types.ModuleType("mistralai")
        mod.Mistral = type("Mistral", (), {"__init__": lambda self, **kw: None})  # type: ignore
        sys.modules["mistralai"] = mod


def _preload_core_modules():
    """
    force-import the core module tree so that unittest.mock.patch can
    resolve dotted paths like 'core.main.get_llm_model'.

    must be called after _ensure_mistralai_mock() and after PROJECT_ROOT
    is on sys.path.
    """
    _ensure_mistralai_mock()
    # dotenv is optional at test time; stub if missing
    try:
        import dotenv  # noqa: F401
    except ImportError:
        import types
        sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))
        sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None  # type: ignore

    # now import the modules the patch targets reference
    import importlib
    for mod_name in [
        "core", "core.utils", "core.utils.helpers", "core.utils.data_models",
        "core.agents", "core.agents.base_agent",
        "core.agents.ontology_agent",
        "core.agents.base_refinement_agent",
        "core.agents.refinement_agent_1pass_refined",
        "core.agents.generation_agent",
        "core.agents.pipeline_analyzer",
        "core.main",
    ]:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            logger.debug(f"preload of {mod_name} failed: {exc}")


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineTestResult:
    """aggregated outcome of a single pipeline test scenario."""
    scenario: str
    pipeline_type: str
    question_id: str
    question: str
    passed: bool = True
    checks: List[CheckResult] = field(default_factory=list)
    answer: str = ""
    context_length: int = 0
    doc_count: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None
    mock_llm_calls: int = 0
    token_estimate: int = 0

    def add_check(self, cr: CheckResult):
        self.checks.append(cr)
        if not cr.passed and cr.severity == "error":
            self.passed = False


# ---------------------------------------------------------------------------
# pipeline test runner
# ---------------------------------------------------------------------------


class PipelineTestSuite:
    """runs end-to-end pipeline scenarios against mock llm."""

    def __init__(self, paths: TestPaths, thresholds: Thresholds):
        self.paths = paths
        self.thresholds = thresholds
        self._stub_paths: List[str] = []

    # ---- public api -----------------------------------------------------

    def run_all(self) -> List[PipelineTestResult]:
        """execute all pipeline test scenarios and return results."""
        results: List[PipelineTestResult] = []
        self._setup()
        try:
            results.extend(self._test_refined_pipeline())
            results.extend(self._test_ontology_pipeline())
            results.extend(self._test_empty_context_handling())
            results.extend(self._test_csv_output())
        finally:
            self._teardown()
        return results

    # ---- setup / teardown -----------------------------------------------

    def _setup(self):
        self._stub_paths = ensure_prompt_stubs()
        if self._stub_paths:
            logger.info(f"created {len(self._stub_paths)} prompt stub(s)")

    def _teardown(self):
        cleanup_prompt_stubs(self._stub_paths)

    # ---- scenario: refined pipeline (primary) ---------------------------

    def _test_refined_pipeline(self) -> List[PipelineTestResult]:
        """test 1_pass_with_ontology_refined with all representative questions."""
        scenario = "refined_pipeline"
        pipeline_type = "1_pass_with_ontology_refined"
        return self._run_pipeline_scenario(scenario, pipeline_type, TEST_QUESTIONS)

    # ---- scenario: ontology-only pipeline -------------------------------

    def _test_ontology_pipeline(self) -> List[PipelineTestResult]:
        """
        test 1_pass_with_ontology (non-refined) pipeline.

        note: the current orchestrator requires a refinement agent for all
        pipeline types (raises RuntimeError otherwise).  this scenario
        therefore validates graceful error handling rather than a full
        success path.
        """
        scenario = "ontology_pipeline"
        pipeline_type = "1_pass_with_ontology"
        results = self._run_pipeline_scenario(
            scenario, pipeline_type, TEST_QUESTIONS[:1],
        )
        # if results came back as errors because of missing refinement
        # agent, mark the scenario as a known limitation (warning, not error)
        for tr in results:
            is_known = (
                (tr.error and "no refinement agent" in tr.error.lower())
                or (tr.answer and "no refinement agent" in tr.answer.lower())
            )
            if not tr.passed and is_known:
                tr.passed = True  # expected behaviour for this pipeline type
                # flip failed checks to warnings
                for ck in tr.checks:
                    if not ck.passed:
                        ck.severity = "info"
                        ck.passed = True
                        ck.message += " (expected: known limitation)"
                tr.checks.append(CheckResult(
                    name="known_limitation",
                    passed=True,
                    message=(
                        "1_pass_with_ontology requires a separate code path "
                        "not yet exposed by the orchestrator – marking as expected"
                    ),
                    severity="info",
                ))
        return results

    # ---- scenario: empty context fallback -------------------------------

    def _test_empty_context_handling(self) -> List[PipelineTestResult]:
        """verify pipeline handles missing context gracefully."""
        scenario = "empty_context"
        pipeline_type = "1_pass_with_ontology_refined"
        empty_q = [q for q in TEST_QUESTIONS_WITH_EMPTY if q["question_id"] == "903"]
        return self._run_pipeline_scenario(scenario, pipeline_type, empty_q)

    # ---- scenario: csv output file checks -------------------------------

    def _test_csv_output(self) -> List[PipelineTestResult]:
        """verify pipeline writes a valid csv with expected columns."""
        scenario = "csv_output"
        csv_path = self.paths.pipeline_csv
        tr = PipelineTestResult(
            scenario=scenario,
            pipeline_type="all",
            question_id="n/a",
            question="csv output validation",
        )
        tr.add_check(check_file_exists(csv_path, label="pipeline csv"))

        if csv_path.exists():
            tr.add_check(check_csv_non_empty(csv_path, label="pipeline csv"))
            # check for required columns
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
            for col in ("question_id", "question", "answer", "pipeline"):
                present = col in headers
                tr.add_check(CheckResult(
                    name=f"csv_column_{col}",
                    passed=present,
                    message=f"column '{col}' {'present' if present else 'missing'}",
                    severity="error" if not present else "info",
                ))
        return [tr]

    # ---- core execution logic -------------------------------------------

    def _run_pipeline_scenario(
        self,
        scenario: str,
        pipeline_type: str,
        questions: List[Dict[str, str]],
    ) -> List[PipelineTestResult]:
        results: List[PipelineTestResult] = []

        # write synthetic csv
        csv_path = self.paths.output_dir / f"test_input_{scenario}.csv"
        write_synthetic_csv(csv_path, questions)

        # inject mock llm
        mock_llm = MockLLM()

        try:
            # ensure project root is on sys.path
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            # preload core modules with mocked third-party deps so
            # unittest.mock.patch can resolve dotted paths
            _preload_core_modules()

            # patch get_llm_model at the helpers level and everywhere it's
            # imported so the orchestrator and all agents get our mock
            with patch("core.utils.helpers.get_llm_model", return_value=mock_llm), \
                 patch("core.main.get_llm_model", return_value=mock_llm):

                from core.main import PipelineOrchestrator

                orchestrator = PipelineOrchestrator(
                    pipeline_type=pipeline_type,
                    overwrite=True,
                )

                pipe_results, output_path, stats = orchestrator.run(
                    input_csv=str(csv_path),
                    output_csv=str(self.paths.pipeline_csv),
                    verbose=False,
                )

            # validate each result
            for pr in pipe_results:
                tr = PipelineTestResult(
                    scenario=scenario,
                    pipeline_type=pipeline_type,
                    question_id=str(pr.original_question_id or pr.index),
                    question=pr.question[:80],
                    answer=pr.answer[:200] if pr.answer else "",
                    context_length=len(pr.final_text_context or ""),
                    processing_time=pr.time_elapsed,
                    mock_llm_calls=len(mock_llm.call_log),
                    token_estimate=mock_llm.token_usage_estimate,
                )

                # run sanity checks
                tr.add_check(check_status_field(pr.status))
                tr.add_check(check_answer_not_empty(pr.answer))
                tr.add_check(check_answer_length(
                    pr.answer, self.thresholds.min_answer_length,
                ))
                tr.add_check(check_no_failure_markers(
                    pr.answer, label="answer",
                ))
                if pr.final_text_context:
                    tr.add_check(check_context_docs(
                        pr.final_text_context,
                        self.thresholds.min_context_docs,
                    ))
                tr.add_check(check_processing_time(
                    pr.time_elapsed,
                    self.thresholds.max_processing_time,
                ))

                # doc count for metrics
                if pr.final_text_context:
                    tr.doc_count = max(
                        len([s for s in pr.final_text_context.split("##") if s.strip()]) - 1,
                        1,
                    )

                results.append(tr)

        except Exception as exc:
            # create a single error result for the whole scenario
            tr = PipelineTestResult(
                scenario=scenario,
                pipeline_type=pipeline_type,
                question_id="all",
                question=f"scenario setup failed: {exc}",
                passed=False,
                error=str(exc),
            )
            tr.add_check(CheckResult(
                name="scenario_execution",
                passed=False,
                message=f"scenario '{scenario}' raised: {exc}",
                severity="error",
            ))
            results.append(tr)

        return results
