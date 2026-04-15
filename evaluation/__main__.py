#!/usr/bin/env python3
"""
unified evaluation cli.

entry point for all evaluation modes:
  python -m evaluation dlr          – dlr llm-as-a-judge evaluation
  python -m evaluation experiment   – experiment evaluation (dlr-based)
  python -m evaluation deepeval     – deepeval metric evaluation
  python -m evaluation relevancy    – deepeval relevancy-only evaluation
"""

import argparse
import logging
import os
import sys

# add parent directory to path for backward compatibility
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dlr_evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ===================================================================
# sub-command: dlr
# ===================================================================


def cmd_dlr(args: argparse.Namespace) -> None:
    """run the dlr llm-as-a-judge evaluation."""
    from .runners import DLRRunner

    runner = DLRRunner(pipeline_type=args.pipeline)
    logger.info("Results → %s", runner.results_file)

    if args.evaluate_only:
        runner.evaluate_existing_answers(args.questions, args.question_id)
    elif args.validate_only:
        runner.validate_existing_results()
    else:
        questions = runner._load_questions(args.questions, args.question_id)
        if not questions:
            logger.warning("No questions found")
            return
        results = runner.evaluate_batch(
            questions, batch_size=args.batch_size, max_workers=args.workers,
        )
        for r in results:
            runner.save_result(r, overwrite=False)
        logger.info("Evaluated %d questions", len(results))


# ===================================================================
# sub-command: experiment
# ===================================================================

# default experiment definitions (input → output mappings)
_DEFAULT_EXPERIMENTS = {
    "scores_only": {
        "input": "../results_to_be_processed/scores-only_results.csv",
        "output": "../results_final/scores_only_evaluation_results.csv",
    },
    "slim_ontology": {
        "input": "../results_to_be_processed/slim-ontology_results.csv",
        "output": "../results_final/slim_ontology_evaluation_results.csv",
    },
    "refined_context": {
        "input": "../results_to_be_processed/refined-context_results.csv",
        "output": "../results_final/experiment_3_evaluation_results.csv",
    },
}


def cmd_experiment(args: argparse.Namespace) -> None:
    """run experiment evaluation (dlr-based scoring)."""
    from .runners import ExperimentRunner

    if args.experiment == "all":
        experiments = list(_DEFAULT_EXPERIMENTS.keys())
    else:
        experiments = [args.experiment]

    for name in experiments:
        cfg = _DEFAULT_EXPERIMENTS.get(name)
        if not cfg:
            logger.warning("Unknown experiment: %s", name)
            continue

        input_file = args.input or cfg["input"]
        output_file = args.output or cfg["output"]

        runner = ExperimentRunner(
            input_file=input_file,
            output_file=output_file,
            experiment_type=name,
        )
        results = runner.run(
            num_questions=args.questions, force=args.force,
        )
        logger.info("%s: %d evaluated", name, len(results))


# ===================================================================
# sub-command: deepeval
# ===================================================================


def cmd_deepeval(args: argparse.Namespace) -> None:
    """run deepeval metric evaluation."""
    from .runners import DeepEvalRunner

    runner = DeepEvalRunner(
        results_file=args.output or "deepeval_results.csv",
        use_fallbacks=args.fallbacks,
        faithfulness_enabled=not args.no_faithfulness,
    )

    # load rows from the provided input csv
    from . import csv_manager

    rows = csv_manager.load_rows(args.input, limit=args.questions)
    if not rows:
        logger.warning("No rows in %s", args.input)
        return

    # normalise column names to what evaluate_rows expects
    normalised = []
    for row in rows:
        normalised.append({
            "question_id": row.get("question_id", ""),
            "question": row.get("question", ""),
            "pipeline": row.get("pipeline", "unknown"),
            "answer": row.get("answer") or row.get("generation", ""),
            "context": row.get("context") or row.get("final_context", ""),
        })

    count = runner.evaluate_rows(normalised)
    logger.info("DeepEval: evaluated %d rows", count)


# ===================================================================
# sub-command: relevancy
# ===================================================================


def cmd_relevancy(args: argparse.Namespace) -> None:
    """run deepeval relevancy-only evaluation."""
    import pandas as pd

    from .runners import DeepEvalRunner

    runner = DeepEvalRunner(
        results_file=args.output or "deepeval_relevancy_scores.csv",
        use_fallbacks=False,
    )

    from . import csv_manager

    rows = csv_manager.load_rows(args.input, limit=args.questions)
    if not rows:
        logger.warning("No rows in %s", args.input)
        return

    normalised = [
        {
            "question_id": r.get("question_id", ""),
            "question": r.get("question", ""),
            "pipeline": r.get("pipeline", ""),
            "answer": r.get("answer") or r.get("generation", ""),
        }
        for r in rows
    ]

    results = runner.evaluate_relevancy_only(normalised)
    df = pd.DataFrame(results)
    df.to_csv(args.output or "deepeval_relevancy_scores.csv", index=False)
    logger.info("Relevancy: evaluated %d rows, avg %.3f",
                len(df), df["relevancy_score"].mean())


# ===================================================================
# argument parser
# ===================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified DLR / DeepEval Evaluation Framework",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # -- dlr --
    p_dlr = subs.add_parser("dlr", help="DLR LLM-as-a-judge evaluation")
    p_dlr.add_argument("--questions", type=int, default=None)
    p_dlr.add_argument("--question-id", type=int, default=None)
    p_dlr.add_argument("--evaluate-only", action="store_true",
                        help="Re-evaluate existing placeholder scores")
    p_dlr.add_argument("--validate-only", action="store_true",
                        help="Validate existing score distributions")
    p_dlr.add_argument("--batch-size", type=int, default=5)
    p_dlr.add_argument("--workers", type=int, default=1)
    p_dlr.add_argument("--pipeline", default="rag_two_steps",
                        choices=["zero_shot", "rag", "rag_two_steps"])
    p_dlr.set_defaults(func=cmd_dlr)

    # -- experiment --
    p_exp = subs.add_parser("experiment", help="Experiment evaluation (DLR-based)")
    p_exp.add_argument("--experiment", default="all",
                        choices=["scores_only", "slim_ontology", "refined_context", "all"])
    p_exp.add_argument("--input", default=None,
                        help="Override input CSV path")
    p_exp.add_argument("--output", default=None,
                        help="Override output CSV path")
    p_exp.add_argument("--questions", type=int, default=None)
    p_exp.add_argument("--force", action="store_true",
                        help="Force re-evaluation")
    p_exp.set_defaults(func=cmd_experiment)

    # -- deepeval --
    p_de = subs.add_parser("deepeval", help="DeepEval metric evaluation")
    p_de.add_argument("--input", required=True, help="Input CSV path")
    p_de.add_argument("--output", default=None, help="Output CSV path")
    p_de.add_argument("--questions", type=int, default=None)
    p_de.add_argument("--fallbacks", action="store_true",
                       help="Use historical-average fallback scores")
    p_de.add_argument("--no-faithfulness", action="store_true",
                       help="Skip faithfulness metric")
    p_de.set_defaults(func=cmd_deepeval)

    # -- relevancy --
    p_rel = subs.add_parser("relevancy", help="DeepEval relevancy-only evaluation")
    p_rel.add_argument("--input", required=True, help="Input CSV path")
    p_rel.add_argument("--output", default=None, help="Output CSV path")
    p_rel.add_argument("--questions", type=int, default=None)
    p_rel.set_defaults(func=cmd_relevancy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
