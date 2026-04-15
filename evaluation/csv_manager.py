"""
shared csv i/o for all evaluation runners.

handles initialisation, loading, saving, updating, and duplicate/skip
detection so that individual runners never re-implement this logic.
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

# handle large context fields without truncation
csv.field_size_limit(int(1e8))


# ---------------------------------------------------------------------------
# header presets used by different evaluation modes
# ---------------------------------------------------------------------------

DLR_FIELDNAMES = [
    "question_id", "question", "pipeline", "context", "answer",
    "factuality", "relevance", "groundedness", "helpfulness", "depth",
    "overall_score", "processing_time", "documents_parsed", "timestamp",
    "validation_anomaly", "anomaly_details", "error",
]

EXPERIMENT_FIELDNAMES = [
    "question_id", "question", "pipeline", "experiment_type",
    "context", "answer", "references",
    "factuality", "relevance", "groundedness", "helpfulness", "depth",
    "overall_score", "processing_time", "timestamp", "evaluation_status",
    "ontology_attributes", "ontology_relationships",
]

DEEPEVAL_BASE_FIELDNAMES = [
    "question_id", "question", "pipeline", "answer",
    "hallucination", "relevancy", "faithfulness", "contextual_relevancy",
    "overall_quality", "timestamp",
]

DEEPEVAL_EXTRA_FIELDNAMES = ["coherence", "toxicity", "bias"]


def deepeval_fieldnames(has_new_metrics: bool = False) -> List[str]:
    """return the full deepeval header list, optionally including v3.8.8+ columns."""
    fns = list(DEEPEVAL_BASE_FIELDNAMES)
    if has_new_metrics:
        fns.extend(DEEPEVAL_EXTRA_FIELDNAMES)
    return fns


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

def initialise_csv(filepath: str, fieldnames: List[str]) -> None:
    """create *filepath* with *fieldnames* as header if it does not exist."""
    if os.path.exists(filepath):
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()


def load_rows(filepath: str, limit: Optional[int] = None) -> List[Dict]:
    """read all rows from *filepath*, optionally capped at *limit*."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[:limit] if limit else rows


def append_row(filepath: str, fieldnames: List[str], row: Dict) -> None:
    """append a single *row* to *filepath*."""
    with open(filepath, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writerow(row)


def upsert_row(
    filepath: str,
    fieldnames: List[str],
    row: Dict,
    key_columns: List[str],
    sort_key: Optional[str] = None,
) -> None:
    """
    insert or replace a row matched by *key_columns*.

    if a row with the same key values exists it is replaced; otherwise the
    new row is appended.  optionally re-sorts the file by *sort_key*
    (treated as int).
    """
    existing = load_rows(filepath)

    def matches(existing_row: Dict) -> bool:
        return all(
            str(existing_row.get(k, "")) == str(row.get(k, ""))
            for k in key_columns
        )

    existing = [r for r in existing if not matches(r)]
    existing.append(row)

    if sort_key:
        existing.sort(key=lambda r: int(r.get(sort_key, 0)))

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def row_exists(
    filepath: str, match: Dict[str, str],
) -> bool:
    """return True if *filepath* contains a row where every k/v in *match* agrees."""
    for row in load_rows(filepath):
        if all(str(row.get(k, "")) == str(v) for k, v in match.items()):
            return True
    return False


def needs_evaluation(
    filepath: str,
    match: Dict[str, str],
    score_column: str = "overall_score",
    criteria_columns: Optional[List[str]] = None,
) -> bool:
    """
    return True when the result identified by *match* should be (re-)evaluated.

    a result needs evaluation when:
    - it is absent from the file, or
    - its *score_column* equals -1, or
    - any of *criteria_columns* equals -1 (if provided).
    """
    for row in load_rows(filepath):
        if not all(str(row.get(k, "")) == str(v) for k, v in match.items()):
            continue
        # row found – check scores
        try:
            if float(row.get(score_column, -1)) == -1:
                return True
        except (ValueError, TypeError):
            return True
        if criteria_columns:
            for col in criteria_columns:
                try:
                    if float(row.get(col, -1)) == -1:
                        return True
                except (ValueError, TypeError):
                    return True
        return False  # row exists with valid scores
    return True  # row not found
