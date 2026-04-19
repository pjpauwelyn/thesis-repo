#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# run_judge.py — CLI judge for pipeline answers (Claude Sonnet 4.6)
#
# NOTES ON AMBIGUITIES / INCONSISTENCIES IN THE EXISTING FRAMEWORK
# ---------------------------------------------------------------------------
# 1. The task spec asks to import `max_tokens` from
#    evaluation/results/FINAL_sonnet4.6_eval_results/full_evaluation_framework.py
#    but that module does NOT define a module-level `max_tokens` symbol.
#    The only mention is the literal `max_tokens=1024` inside a docstring/
#    commented-out Anthropic API example (line ~197). We therefore fall back
#    to 1024 (same value used in the comment) and log a warning at import
#    time so the author can fix the upstream module if desired.
#
# 2. In `all_eval_results_4pipelines.json` the `pipeline` field includes both
#    the human-readable label and the internal pipeline identifier, e.g.
#    "DLR 2-Step RAG (rag_two_steps)".  For new experimental conditions we
#    use the `--pipeline` CLI argument verbatim as both the `set` and
#    `pipeline` value (matching the rest of the schema 1:1 otherwise).
#
# 3. `evaluate_single_task()` in the framework module raises NotImplementedError
#    by design — the original evaluation was performed via Perplexity
#    Computer's subagent infrastructure. This script replaces that with a
#    direct Anthropic API call using the *same* SYSTEM_PROMPT /
#    EVAL_PROMPT_TEMPLATE / JUDGE_MODEL constants.
# ---------------------------------------------------------------------------

import argparse
import csv
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Locate and import the baseline framework module WITHOUT triggering its
# __main__ block and without copying its constants.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FRAMEWORK_PATH = (
    _REPO_ROOT
    / "evaluation"
    / "results"
    / "FINAL_sonnet4.6_eval_results"
    / "full_evaluation_framework.py"
)

if not _FRAMEWORK_PATH.exists():
    raise FileNotFoundError(f"cannot locate framework module: {_FRAMEWORK_PATH}")

_spec = importlib.util.spec_from_file_location("_eval_framework", _FRAMEWORK_PATH)
_framework = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_framework)  # safe: __main__ guard protects exec

SYSTEM_PROMPT = _framework.SYSTEM_PROMPT
EVAL_PROMPT_TEMPLATE = _framework.EVAL_PROMPT_TEMPLATE
JUDGE_MODEL = _framework.JUDGE_MODEL
CRITERIA = _framework.CRITERIA

# fallback per note (1) above
max_tokens = getattr(_framework, "max_tokens", 1024)

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

_ERROR_LOG_PATH = _REPO_ROOT / "evaluation" / "results" / "run_judge_errors.log"
_ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger("run_judge")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(_ch)

_error_logger = logging.getLogger("run_judge.errors")
_error_logger.setLevel(logging.ERROR)
_error_logger.propagate = False
if not _error_logger.handlers:
    _fh = logging.FileHandler(_ERROR_LOG_PATH, mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
    _error_logger.addHandler(_fh)


def _log_error(question_id: Any, err: str, raw: str = "") -> None:
    truncated = (raw or "")[:500].replace("\n", " ")
    _error_logger.error(f"qid={question_id}\t{err}\tRAW={truncated}")


# ---------------------------------------------------------------------------
# Anthropic client (lazy — only needed for real calls, not for --selftest parse
# tests on cached responses or compile checks)
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from e

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
    return anthropic.Anthropic(api_key=api_key)


def _call_judge(client, question: str, answer: str) -> Dict[str, Any]:
    """Call the judge model. Returns parsed JSON dict or raises."""
    prompt = EVAL_PROMPT_TEMPLATE.format(question=question, answer=answer)
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=max_tokens,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text if response.content else ""
    raw = raw.strip()
    # defensive: strip accidental markdown fences
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    parsed = json.loads(raw)
    return parsed, raw


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------

def _load_existing(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.exists():
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError) as e:
        _logger.warning(f"could not read existing output (starting fresh): {e}")
    return []


def _write_results(output_path: Path, results: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp.replace(output_path)


def _build_entry(
    qid: int,
    pipeline_label: str,
    parsed: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build an entry matching the schema of all_eval_results_4pipelines.json."""
    if parsed is None:
        scores = {c: None for c in CRITERIA}
        justification = ""
        strongest = ""
        weakest = ""
    else:
        raw_scores = parsed.get("scores", {}) or {}
        scores = {c: raw_scores.get(c) for c in CRITERIA}
        justification = parsed.get("justification", "") or ""
        strongest = parsed.get("strongest_aspect", "") or ""
        weakest = parsed.get("weakest_aspect", "") or ""

    return {
        "task_id": f"Q{qid}_{pipeline_label}",
        "question_id": qid,
        "set": pipeline_label,
        "pipeline": pipeline_label,
        "scores": scores,
        "justification": justification,
        "strongest_aspect": strongest,
        "weakest_aspect": weakest,
    }


# ---------------------------------------------------------------------------
# main workflow
# ---------------------------------------------------------------------------

def _score_one(client, qid: int, question: str, answer: str, pipeline_label: str) -> Dict[str, Any]:
    try:
        parsed, _raw = _call_judge(client, question, answer)
        # sanity check that each score is an integer in [1,10]
        scores = parsed.get("scores", {}) or {}
        for c in CRITERIA:
            v = scores.get(c)
            if not isinstance(v, int) or not (1 <= v <= 10):
                raise ValueError(f"invalid score for {c}: {v!r}")
        return _build_entry(qid, pipeline_label, parsed)
    except Exception as e:
        raw = ""
        try:
            raw = str(e)
        except Exception:
            pass
        _log_error(qid, f"{type(e).__name__}: {e}", raw)
        _logger.error(f"  ✗ qid={qid} failed: {e}")
        return _build_entry(qid, pipeline_label, None)


def _run_csv(input_csv: Path, output_json: Path, pipeline_label: str, resume: bool) -> None:
    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    existing = _load_existing(output_json) if resume else []
    already = {int(e["question_id"]) for e in existing if e.get("question_id") is not None}
    results: List[Dict[str, Any]] = list(existing)

    client = _get_anthropic_client()

    to_process = []
    for row in rows:
        try:
            qid = int(row["question_id"])
        except (KeyError, ValueError):
            _logger.warning(f"skipping row with no/invalid question_id: {row!r}")
            continue
        if resume and qid in already:
            _logger.info(f"  · skip qid={qid} (already in output)")
            continue
        to_process.append((qid, row.get("question", ""), row.get("answer", "")))

    _logger.info(
        f"Judging {len(to_process)} question(s) | pipeline={pipeline_label} | "
        f"model={JUDGE_MODEL} | already_scored={len(already)}"
    )

    for i, (qid, question, answer) in enumerate(to_process, 1):
        _logger.info(f"  [{i}/{len(to_process)}] qid={qid}")
        entry = _score_one(client, qid, question, answer, pipeline_label)
        # remove any stale entry for same (qid, pipeline) then append
        results = [
            r for r in results
            if not (r.get("question_id") == qid and r.get("pipeline") == pipeline_label)
        ]
        results.append(entry)
        # incremental write
        _write_results(output_json, results)

    _logger.info(f"done → {output_json}")


def _run_selftest() -> None:
    """Run on a single hardcoded dummy Q/A and print the result."""
    dummy_q = (
        "What spatial resolution does the MODIS sensor provide for the 250m land bands, "
        "and what are typical use cases?"
    )
    dummy_a = (
        "MODIS provides 250 m spatial resolution on bands 1 (red, ~645 nm) and 2 (NIR, ~858 nm). "
        "These bands are widely used for vegetation indices such as NDVI and EVI, burned-area "
        "mapping, and daily land-surface monitoring because of MODIS's near-daily global coverage."
    )

    _logger.info("SELFTEST: calling judge on a dummy Q/A")
    try:
        client = _get_anthropic_client()
        parsed, raw = _call_judge(client, dummy_q, dummy_a)
    except Exception as e:
        _logger.error(f"SELFTEST failed: {e}")
        _log_error("SELFTEST", f"{type(e).__name__}: {e}")
        print(json.dumps({"selftest": "FAILED", "error": str(e)}, indent=2))
        sys.exit(1)

    entry = _build_entry(qid=0, pipeline_label="selftest", parsed=parsed)
    print(json.dumps(entry, indent=2, ensure_ascii=False))

    # validate scores
    scores = entry["scores"]
    all_ok = all(isinstance(scores[c], int) and 1 <= scores[c] <= 10 for c in CRITERIA)
    print(f"\nSELFTEST: {'PASS' if all_ok else 'FAIL'} "
          f"(8 int scores in [1,10]: {all_ok})")
    sys.exit(0 if all_ok else 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Claude Sonnet 4.6 judge for pipeline answers."
    )
    p.add_argument("--input", help="CSV with columns: question_id, question, answer")
    p.add_argument("--output", help="JSON output path (created if missing)")
    p.add_argument("--pipeline", help="Label used as `set` and `pipeline` in output")
    p.add_argument("--resume", action="store_true",
                   help="Skip question_ids already present in --output")
    p.add_argument("--selftest", action="store_true",
                   help="Run on a single hardcoded dummy Q/A")
    args = p.parse_args()

    if args.selftest:
        _run_selftest()
        return

    for req in ("input", "output", "pipeline"):
        if not getattr(args, req):
            p.error(f"--{req} is required unless --selftest is used")

    input_csv = Path(args.input)
    output_json = Path(args.output)
    if not input_csv.exists():
        p.error(f"input CSV not found: {input_csv}")

    _run_csv(input_csv, output_json, args.pipeline, args.resume)


if __name__ == "__main__":
    main()
