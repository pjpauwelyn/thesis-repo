"""parallel pipeline runner. see --help for usage."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as _futures_wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipelines.pipeline import Pipeline

csv.field_size_limit(int(1e8))
log = logging.getLogger("run_pipeline")

# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------
_BAD_ESCAPE_RE = re.compile(r'\\u(?![0-9a-fA-F]{4})')

def _sanitise(s: str) -> str:
    """Escape bare \\u not followed by 4 hex digits so json.dumps is safe.

    The LLM occasionally emits strings like \\units or \\upward which are
    not valid JSON Unicode escapes and cause json.JSONDecodeError when the
    JSONL file is read back.
    """
    if not s:
        return s
    return _BAD_ESCAPE_RE.sub(r'\\\\u', s)


_CSV_CANDIDATES = [
    "data/dlr/questions.csv",
    "data/dlr/dlr_questions.csv",
    "data/questions.csv",
    "data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv",
]


def _find_dlr_csv() -> Optional[str]:
    for p in _CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    data_root = Path("data")
    if data_root.exists():
        for csv_path in sorted(data_root.rglob("*.csv")):
            try:
                header = csv_path.open("r", encoding="utf-8").readline()
            except Exception:
                continue
            if "aql_results" in header or "question" in header.lower():
                return str(c