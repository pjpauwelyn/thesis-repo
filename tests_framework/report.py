"""
summary report builder for the testing framework.

collects pipeline and evaluation test results and writes:
- a json summary (test_summary.json)
- a csv summary (test_summary.csv)
- a human-readable console summary
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests_framework.test_pipeline import PipelineTestResult
from tests_framework.test_evaluation import EvalTestResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# aggregation helpers
# ---------------------------------------------------------------------------


def _count_checks(checks) -> Dict[str, int]:
    """count passed/failed/warning across a list of CheckResult."""
    total = len(checks)
    passed = sum(1 for c in checks if c.passed)
    errors = sum(1 for c in checks if not c.passed and c.severity == "error")
    warnings = sum(1 for c in checks if not c.passed and c.severity == "warning")
    return {
        "total": total,
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# json summary builder
# ---------------------------------------------------------------------------


def build_summary(
    pipeline_results: List[PipelineTestResult],
    eval_results: List[EvalTestResult],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """build the full summary dict."""
    all_checks = []
    for pr in pipeline_results:
        all_checks.extend(pr.checks)
    for er in eval_results:
        all_checks.extend(er.checks)

    check_counts = _count_checks(all_checks)

    # pipeline breakdown
    pipe_scenarios = {}
    for pr in pipeline_results:
        key = pr.scenario
        if key not in pipe_scenarios:
            pipe_scenarios[key] = {"passed": 0, "failed": 0, "questions": []}
        bucket = "passed" if pr.passed else "failed"
        pipe_scenarios[key][bucket] += 1
        pipe_scenarios[key]["questions"].append({
            "question_id": pr.question_id,
            "passed": pr.passed,
            "answer_preview": pr.answer[:120] if pr.answer else "",
            "processing_time": round(pr.processing_time, 2),
            "context_length": pr.context_length,
            "doc_count": pr.doc_count,
            "mock_llm_calls": pr.mock_llm_calls,
            "token_estimate": pr.token_estimate,
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message}
                for c in pr.checks
            ],
            "error": pr.error,
        })

    # evaluation breakdown
    eval_scenarios = {}
    for er in eval_results:
        key = er.scenario
        if key not in eval_scenarios:
            eval_scenarios[key] = {"passed": 0, "failed": 0, "evaluations": []}
        bucket = "passed" if er.passed else "failed"
        eval_scenarios[key][bucket] += 1
        eval_scenarios[key]["evaluations"].append({
            "question_id": er.question_id,
            "passed": er.passed,
            "overall_score": er.overall_score,
            "criteria_scores": er.criteria_scores,
            "feedback_preview": er.feedback[:120] if er.feedback else "",
            "has_anomaly": er.has_anomaly,
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message}
                for c in er.checks
            ],
            "error": er.error,
        })

    # aggregate metrics
    total_tests = len(pipeline_results) + len(eval_results)
    total_passed = sum(1 for p in pipeline_results if p.passed) + \
                   sum(1 for e in eval_results if e.passed)
    total_failed = total_tests - total_passed

    # token usage estimate (from pipeline mocks)
    total_token_estimate = sum(pr.token_estimate for pr in pipeline_results)

    # avg processing time (pipeline only)
    times = [pr.processing_time for pr in pipeline_results if pr.processing_time > 0]
    avg_time = sum(times) / len(times) if times else 0.0

    # avg doc count
    docs = [pr.doc_count for pr in pipeline_results if pr.doc_count > 0]
    avg_docs = sum(docs) / len(docs) if docs else 0.0

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "total_elapsed_seconds": round(elapsed_seconds, 2),
            "framework_version": "1.0.0",
        },
        "summary": {
            "total_test_cases": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": round(total_passed / total_tests * 100, 1) if total_tests else 0,
        },
        "checks": check_counts,
        "metrics": {
            "avg_processing_time_seconds": round(avg_time, 2),
            "avg_documents_in_context": round(avg_docs, 1),
            "total_token_estimate": total_token_estimate,
        },
        "pipeline_scenarios": pipe_scenarios,
        "evaluation_scenarios": eval_scenarios,
    }


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------


def write_summary_json(summary: Dict, path: Path) -> None:
    """write the full summary to a json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"json summary written to {path}")


def write_summary_csv(
    pipeline_results: List[PipelineTestResult],
    eval_results: List[EvalTestResult],
    path: Path,
) -> None:
    """write a flat csv with one row per test case."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "type", "scenario", "question_id", "passed", "overall_score",
        "answer_preview", "processing_time", "context_length", "doc_count",
        "mock_llm_calls", "token_estimate", "checks_total", "checks_passed",
        "checks_errors", "checks_warnings", "error",
    ]

    rows = []
    for pr in pipeline_results:
        cc = _count_checks(pr.checks)
        rows.append({
            "type": "pipeline",
            "scenario": pr.scenario,
            "question_id": pr.question_id,
            "passed": pr.passed,
            "overall_score": "",
            "answer_preview": (pr.answer[:80] if pr.answer else ""),
            "processing_time": round(pr.processing_time, 2),
            "context_length": pr.context_length,
            "doc_count": pr.doc_count,
            "mock_llm_calls": pr.mock_llm_calls,
            "token_estimate": pr.token_estimate,
            "checks_total": cc["total"],
            "checks_passed": cc["passed"],
            "checks_errors": cc["errors"],
            "checks_warnings": cc["warnings"],
            "error": pr.error or "",
        })

    for er in eval_results:
        cc = _count_checks(er.checks)
        rows.append({
            "type": "evaluation",
            "scenario": er.scenario,
            "question_id": er.question_id,
            "passed": er.passed,
            "overall_score": er.overall_score,
            "answer_preview": "",
            "processing_time": "",
            "context_length": "",
            "doc_count": "",
            "mock_llm_calls": "",
            "token_estimate": "",
            "checks_total": cc["total"],
            "checks_passed": cc["passed"],
            "checks_errors": cc["errors"],
            "checks_warnings": cc["warnings"],
            "error": er.error or "",
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"csv summary written to {path}")


# ---------------------------------------------------------------------------
# console report
# ---------------------------------------------------------------------------


def print_console_report(
    summary: Dict,
    pipeline_results: List[PipelineTestResult],
    eval_results: List[EvalTestResult],
) -> None:
    """pretty-print a human-readable summary to stdout."""
    s = summary["summary"]
    m = summary["metrics"]
    c = summary["checks"]

    print()
    print("=" * 72)
    print("  TEST FRAMEWORK SUMMARY")
    print("=" * 72)
    print()
    print(f"  Total test cases:  {s['total_test_cases']}")
    print(f"  Passed:            {s['passed']}")
    print(f"  Failed:            {s['failed']}")
    print(f"  Pass rate:         {s['pass_rate']}%")
    print()
    print(f"  Sanity checks:     {c['total']} total, "
          f"{c['passed']} passed, {c['errors']} errors, {c['warnings']} warnings")
    print()
    print(f"  Avg processing time: {m['avg_processing_time_seconds']}s")
    print(f"  Avg docs in context: {m['avg_documents_in_context']}")
    print(f"  Token usage (est.):  {m['total_token_estimate']}")
    print()

    # pipeline scenarios
    print("  PIPELINE SCENARIOS")
    print("  " + "-" * 68)
    for name, data in summary.get("pipeline_scenarios", {}).items():
        status = "PASS" if data["failed"] == 0 else "FAIL"
        print(f"    [{status}] {name}: "
              f"{data['passed']} passed, {data['failed']} failed")
        for q in data["questions"]:
            sym = "✓" if q["passed"] else "✗"
            print(f"      {sym} Q{q['question_id']}: "
                  f"{q['processing_time']}s, {q['doc_count']} doc(s)")
            if q.get("error"):
                print(f"        error: {q['error'][:80]}")
            for ck in q["checks"]:
                if not ck["passed"]:
                    print(f"        ✗ {ck['name']}: {ck['message']}")
    print()

    # evaluation scenarios
    print("  EVALUATION SCENARIOS")
    print("  " + "-" * 68)
    for name, data in summary.get("evaluation_scenarios", {}).items():
        status = "PASS" if data["failed"] == 0 else "FAIL"
        print(f"    [{status}] {name}: "
              f"{data['passed']} passed, {data['failed']} failed")
        for ev in data["evaluations"]:
            sym = "✓" if ev["passed"] else "✗"
            anomaly = " [ANOMALY]" if ev.get("has_anomaly") else ""
            print(f"      {sym} Q{ev['question_id']}: "
                  f"score={ev['overall_score']:.2f}{anomaly}")
            if ev.get("error"):
                print(f"        error: {ev['error'][:80]}")
            for ck in ev["checks"]:
                if not ck["passed"]:
                    print(f"        ✗ {ck['name']}: {ck['message']}")
    print()

    print("  " + "-" * 68)
    elapsed = summary["meta"]["total_elapsed_seconds"]
    print(f"  Completed in {elapsed}s")
    print("=" * 72)
    print()
