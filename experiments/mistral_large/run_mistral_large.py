#!/usr/bin/env python3
"""experiment b: run the standard ontology pipeline with mistral large/medium.

this experiment reuses the existing pipeline code unchanged.  the only
difference is the model name passed to every llm wrapper.  results are
written to a dedicated csv (default: results_mistral_large.csv) so they
do not interfere with the baseline runs.

the model can also be set via the PIPELINE_MODEL environment variable
(see core/utils/helpers.py) but the --model flag takes precedence.

usage:
    python experiments/mistral_large/run_mistral_large.py \\
        --csv data.csv --num 5

    # or with medium:
    python experiments/mistral_large/run_mistral_large.py \\
        --csv data.csv --model mistral-medium-latest
"""

import argparse
import os
import sys

# ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.main import PipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: run pipeline with Mistral large / medium"
    )
    parser.add_argument("--csv", required=True, help="input csv")
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument(
        "--model",
        default="mistral-large-latest",
        help="model name (default: mistral-large-latest)",
    )
    parser.add_argument(
        "--output-csv",
        default="results_mistral_large.csv",
        help="output csv (default: results_mistral_large.csv)",
    )
    parser.add_argument("--context-filter", default="full", choices=["full", "slim", "scores_only"])
    parser.add_argument("--include-relationships", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refinement-temp", type=float, default=0.1)
    parser.add_argument("--generation-temp", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # the only thing that changes: model_name and experiment_type
    orchestrator = PipelineOrchestrator(
        pipeline_type="1_pass_with_ontology_refined",
        model_name=args.model,
        context_filter=args.context_filter,
        include_ontology_relationships=args.include_relationships,
        experiment_type=f"mistral_large_{args.model}",
        overwrite=args.overwrite,
        refinement_temperature=args.refinement_temp,
        generation_temperature=args.generation_temp,
    )

    results, path, stats = orchestrator.run(
        input_csv=args.csv,
        num_questions=args.num,
        output_csv=args.output_csv,
        verbose=not args.quiet,
    )

    ok = stats.successful
    fail = stats.failed
    print(f"\nexperiment b done: {ok} ok, {fail} failed -> {path}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
