"""
reusable sanity-check predicates for pipeline and evaluation outputs.

each function returns a CheckResult namedtuple so the runner can
aggregate pass/fail counts and rich diagnostic messages.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tests_framework.config import FAILURE_MARKERS, FAILURE_MARKER_EXCEPTIONS, Thresholds

# ---------------------------------------------------------------------------
# check result
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """outcome of a single sanity check."""
    name: str
    passed: bool
    message: str
    severity: str = "error"        # error | warning | info
    details: Optional[Dict] = None


# ---------------------------------------------------------------------------
# pipeline output checks
# ---------------------------------------------------------------------------


def check_no_failure_markers(text: str, label: str = "output") -> CheckResult:
    """detect known failure markers in text (answer, logs, context)."""
    for marker in FAILURE_MARKERS:
        # skip markers that appear inside whitelisted tokens
        if any(exc in marker.lower() for exc in FAILURE_MARKER_EXCEPTIONS):
            continue
        if marker in text:
            # check it isn't just the csv header name
            stripped = text.strip()
            if stripped == marker or marker not in FAILURE_MARKER_EXCEPTIONS:
                return CheckResult(
                    name=f"no_failure_markers ({label})",
                    passed=False,
                    message=f"found failure marker '{marker}' in {label}",
                    severity="error",
                    details={"marker": marker, "source": label},
                )
    return CheckResult(
        name=f"no_failure_markers ({label})",
        passed=True,
        message=f"no failure markers detected in {label}",
    )


def check_answer_length(answer: str, threshold: int = 30) -> CheckResult:
    """verify the answer exceeds a minimum character count."""
    length = len(answer.strip())
    passed = length >= threshold
    return CheckResult(
        name="answer_length",
        passed=passed,
        message=(
            f"answer length {length} chars"
            + ("" if passed else f" (below threshold {threshold})")
        ),
        severity="error" if not passed else "info",
        details={"length": length, "threshold": threshold},
    )


def check_answer_not_empty(answer: str) -> CheckResult:
    """verify the answer is not blank or whitespace-only."""
    passed = bool(answer and answer.strip())
    return CheckResult(
        name="answer_not_empty",
        passed=passed,
        message="answer is non-empty" if passed else "answer is empty or whitespace",
        severity="error" if not passed else "info",
    )


def check_context_docs(context: str, min_docs: int = 1) -> CheckResult:
    """
    count documents in the final text context.
    heuristic: split on '## ' or '---' section markers.
    """
    if not context:
        return CheckResult(
            name="context_docs",
            passed=False,
            message="context is empty (0 documents)",
            severity="warning",
            details={"doc_count": 0, "min_required": min_docs},
        )

    # count section-like markers as proxy for documents
    section_count = max(
        len([s for s in context.split("##") if s.strip()]) - 1,
        len([s for s in context.split("---") if s.strip()]) - 1,
        1 if len(context.strip()) > 20 else 0,
    )
    passed = section_count >= min_docs
    return CheckResult(
        name="context_docs",
        passed=passed,
        message=f"{section_count} document(s) in context"
                + ("" if passed else f" (need >= {min_docs})"),
        severity="warning" if not passed else "info",
        details={"doc_count": section_count, "min_required": min_docs},
    )


def check_processing_time(elapsed: float, max_seconds: float = 120.0) -> CheckResult:
    """flag questions that took longer than the allowed ceiling."""
    passed = elapsed <= max_seconds
    return CheckResult(
        name="processing_time",
        passed=passed,
        message=f"processing time {elapsed:.2f}s"
                + ("" if passed else f" (exceeds {max_seconds}s limit)"),
        severity="warning" if not passed else "info",
        details={"elapsed": elapsed, "max_seconds": max_seconds},
    )


def check_status_field(status: str) -> CheckResult:
    """verify PipelineResult.status == 'success'."""
    passed = status == "success"
    return CheckResult(
        name="status_field",
        passed=passed,
        message=f"status='{status}'" + ("" if passed else " (expected 'success')"),
        severity="error" if not passed else "info",
    )


# ---------------------------------------------------------------------------
# evaluation output checks
# ---------------------------------------------------------------------------


def check_eval_score_range(
    scores: Dict[str, Any], valid_range: tuple = (1, 5),
) -> CheckResult:
    """verify each criteria score falls within the valid range."""
    out_of_range = {}
    for criterion, score in scores.items():
        if not isinstance(score, (int, float)):
            out_of_range[criterion] = score
        elif score < valid_range[0] or score > valid_range[1]:
            out_of_range[criterion] = score

    passed = len(out_of_range) == 0
    return CheckResult(
        name="eval_score_range",
        passed=passed,
        message=(
            "all scores in valid range"
            if passed
            else f"scores out of range: {out_of_range}"
        ),
        severity="error" if not passed else "info",
        details={"out_of_range": out_of_range, "valid_range": valid_range},
    )


def check_eval_required_criteria(
    scores: Dict[str, Any],
    required: tuple = ("factuality", "relevance", "groundedness", "helpfulness", "depth"),
) -> CheckResult:
    """verify all required criteria are present and non-null."""
    missing = [c for c in required if c not in scores or scores[c] is None]
    passed = len(missing) == 0
    return CheckResult(
        name="eval_required_criteria",
        passed=passed,
        message=(
            "all required criteria present"
            if passed
            else f"missing criteria: {missing}"
        ),
        severity="error" if not passed else "info",
        details={"missing": missing},
    )


def check_eval_overall_score(
    overall: float, min_score: float = 1.0,
) -> CheckResult:
    """verify the overall score is above the failure floor (-1 = failed eval)."""
    passed = overall >= min_score
    return CheckResult(
        name="eval_overall_score",
        passed=passed,
        message=(
            f"overall score {overall:.2f}"
            + ("" if passed else f" (below minimum {min_score})")
        ),
        severity="error" if not passed else "info",
        details={"overall": overall, "min_score": min_score},
    )


def check_eval_feedback_present(feedback: str) -> CheckResult:
    """verify the evaluator returned non-empty feedback text."""
    passed = bool(feedback and feedback.strip())
    return CheckResult(
        name="eval_feedback_present",
        passed=passed,
        message="feedback present" if passed else "feedback is empty",
        severity="warning" if not passed else "info",
    )


# ---------------------------------------------------------------------------
# file-level checks
# ---------------------------------------------------------------------------


def check_file_exists(path, label: str = "file") -> CheckResult:
    """verify that an expected output file was created."""
    from pathlib import Path as _P
    exists = _P(path).exists()
    return CheckResult(
        name=f"file_exists ({label})",
        passed=exists,
        message=f"{label} exists at {path}" if exists else f"{label} missing: {path}",
        severity="error" if not exists else "info",
    )


def check_csv_non_empty(path, label: str = "csv") -> CheckResult:
    """verify that a csv file has at least one data row."""
    import csv as _csv
    from pathlib import Path as _P

    p = _P(path)
    if not p.exists():
        return CheckResult(
            name=f"csv_non_empty ({label})",
            passed=False,
            message=f"{label} does not exist",
            severity="error",
        )
    with open(p, "r", encoding="utf-8") as f:
        reader = _csv.reader(f)
        rows = list(reader)
    data_rows = len(rows) - 1 if rows else 0
    passed = data_rows > 0
    return CheckResult(
        name=f"csv_non_empty ({label})",
        passed=passed,
        message=f"{label} has {data_rows} data row(s)",
        severity="error" if not passed else "info",
    )
