"""
main runner – executes the full test suite and writes reports.

usage:
    python -m tests_framework.run_full_scan
    python tests_framework/run_full_scan.py
    python tests_framework/run_full_scan.py --output-dir /tmp/test_output
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# ensure project root is on sys.path so `core.*` and `evaluation.*` resolve
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# also add the parent of project root for `evaluation.*` imports
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from tests_framework.config import TestPaths, Thresholds
from tests_framework.test_pipeline import PipelineTestSuite
from tests_framework.test_evaluation import EvaluationTestSuite
from tests_framework.report import (
    build_summary,
    write_summary_json,
    write_summary_csv,
    print_console_report,
)

# ---------------------------------------------------------------------------
# logging setup
# ---------------------------------------------------------------------------


def _setup_logging(log_file: Path, verbose: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    # file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def run(output_dir: Optional[str] = None, verbose: bool = False) -> int:
    """
    run the full test scan.

    returns 0 if all tests pass, 1 otherwise.
    """
    paths = TestPaths(output_dir=Path(output_dir)) if output_dir else TestPaths()
    thresholds = Thresholds()

    _setup_logging(paths.log_file, verbose=verbose)
    logger = logging.getLogger("tests_framework")

    logger.info("=" * 60)
    logger.info("  ONTOLOGY-RAG TESTING FRAMEWORK")
    logger.info("=" * 60)
    logger.info(f"output directory: {paths.output_dir}")
    logger.info("")

    start = time.time()

    # ---- pipeline tests -------------------------------------------------
    logger.info("running pipeline tests ...")
    pipe_suite = PipelineTestSuite(paths, thresholds)
    pipeline_results = pipe_suite.run_all()
    pipe_passed = sum(1 for r in pipeline_results if r.passed)
    pipe_total = len(pipeline_results)
    logger.info(f"pipeline tests: {pipe_passed}/{pipe_total} passed")
    logger.info("")

    # ---- evaluation tests -----------------------------------------------
    logger.info("running evaluation tests ...")
    eval_suite = EvaluationTestSuite(thresholds)
    eval_results = eval_suite.run_all()
    eval_passed = sum(1 for r in eval_results if r.passed)
    eval_total = len(eval_results)
    logger.info(f"evaluation tests: {eval_passed}/{eval_total} passed")
    logger.info("")

    elapsed = time.time() - start

    # ---- reports --------------------------------------------------------
    summary = build_summary(pipeline_results, eval_results, elapsed)
    write_summary_json(summary, paths.summary_json)
    write_summary_csv(pipeline_results, eval_results, paths.summary_csv)
    print_console_report(summary, pipeline_results, eval_results)

    # ---- exit code ------------------------------------------------------
    all_passed = summary["summary"]["failed"] == 0
    if all_passed:
        logger.info("all tests passed")
    else:
        logger.warning(
            f"{summary['summary']['failed']} test(s) failed – "
            "see reports for details"
        )

    return 0 if all_passed else 1


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run the full ontology-RAG test scan.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for test output files (default: <project>/test_output)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="enable debug-level logging",
    )
    args = parser.parse_args()
    sys.exit(run(output_dir=args.output_dir, verbose=args.verbose))


if __name__ == "__main__":
    main()
