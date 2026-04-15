"""csv-based execution analytics for pipeline runs."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class PipelineAnalyzer:
    """logs per-question pipeline metrics to a timestamped csv."""

    FIELDNAMES = [
        "timestamp", "question_id", "question", "pipeline_type",
        "ontology_attributes_count", "ontology_relationships_count",
        "aql_results_length", "structured_context_length",
        "refinement_prompt_length", "refined_context_length",
        "refined_context_tokens", "context_within_target_range",
        "generation_input_length", "final_answer_length",
        "processing_time", "ontology_time", "refinement_time", "generation_time",
        "success_status", "error_message", "notes",
    ]

    def __init__(self, log_dir: str = "analysis_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.analysis_file = self.log_dir / f"pipeline_analysis_{ts}.csv"

        with open(self.analysis_file, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def log_pipeline_execution(self, **kwargs) -> None:
        row = {
            "timestamp": datetime.now().isoformat(),
            "question_id": kwargs.get("question_id"),
            "question": (kwargs.get("question", "")[:100] + "..."),
            "pipeline_type": kwargs.get("pipeline_type"),
            "ontology_attributes_count": kwargs.get("ontology_attrs", 0),
            "ontology_relationships_count": kwargs.get("ontology_rels", 0),
            "aql_results_length": kwargs.get("aql_length", 0),
            "structured_context_length": kwargs.get("structured_context_length", 0),
            "refinement_prompt_length": kwargs.get("refinement_prompt_length", 0),
            "refined_context_length": kwargs.get("refined_context_length", 0),
            "refined_context_tokens": kwargs.get("refined_context_tokens", 0),
            "context_within_target_range": kwargs.get("context_in_range", False),
            "generation_input_length": kwargs.get("generation_input_length", 0),
            "final_answer_length": kwargs.get("final_answer_length", 0),
            "processing_time": f"{kwargs.get('processing_time', 0):.2f}",
            "ontology_time": f"{kwargs.get('ontology_time', 0):.2f}",
            "refinement_time": f"{kwargs.get('refinement_time', 0):.2f}",
            "generation_time": f"{kwargs.get('generation_time', 0):.2f}",
            "success_status": "success" if kwargs.get("success") else "failed",
            "error_message": (kwargs.get("error_msg") or "")[:200],
            "notes": (kwargs.get("notes") or "")[:100],
        }

        with open(self.analysis_file, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)

    def log_detailed_context_analysis(
        self,
        question_id: int,
        original_context: str,
        refined_context: str,
        ontology_summary: str,
    ) -> None:
        detail_file = self.log_dir / f"question_{question_id}_detailed_analysis.txt"
        with open(detail_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\nDETAILED PIPELINE ANALYSIS\n" + "=" * 80 + "\n\n")
            f.write(f"1. ORIGINAL CONTEXT ({len(original_context)} chars)\n")
            f.write(f"   {original_context[:200]}...\n\n")
            f.write(f"2. ONTOLOGY SUMMARY\n   {ontology_summary}\n\n")
            f.write(f"3. REFINED CONTEXT ({len(refined_context)} chars, ~{len(refined_context)//4} tokens)\n")
            f.write(f"{refined_context}\n")
