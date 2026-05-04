"""Routing smoke-test: profile N questions and print their assigned tier.

Invoked by `make test-profile N=<n>` and also usable directly:
    python scripts/_test_profile.py 20
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipelines.pipeline import Pipeline


def main(n: int = 10) -> None:
    csv_candidates = [
        "data/dlr/questions.csv",
        "data/dlr/dlr_questions.csv",
        "data/questions.csv",
        "data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv",
    ]
    csv_path = next((p for p in csv_candidates if Path(p).exists()), None)
    if csv_path is None:
        print("ERROR: could not find questions CSV", file=sys.stderr)
        sys.exit(1)

    pipeline = Pipeline()
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for row in rows[:n]:
        q = (row.get("question") or row.get("Question") or "").strip()
        if not q:
            continue
        _, _, cfg = pipeline.profile_and_route(q)
        print(f"[{cfg.rule_hit:<14}] {q[:90]}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n)
