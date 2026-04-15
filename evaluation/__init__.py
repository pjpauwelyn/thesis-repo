"""
unified dlr / deepeval evaluation framework.

modules
-------
csv_manager        – shared csv i/o (init, load, save, upsert, duplicate checks)
dlr_evaluator      – llm-as-a-judge evaluator (5 criteria, 1-5 scale)
deepeval_evaluator – deepeval-based metric evaluator (hallucination, relevancy, etc.)
runners            – batch runners for dlr, experiment, and deepeval evaluation modes
__main__           – unified cli entry point
"""

from .dlr_evaluator import DLREvaluator
from .runners import DLRRunner, DeepEvalRunner, ExperimentRunner

__all__ = [
    "DLREvaluator",
    "DLRRunner",
    "ExperimentRunner",
    "DeepEvalRunner",
]
