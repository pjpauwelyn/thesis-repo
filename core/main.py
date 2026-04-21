"""pipeline orchestrator – runs ontology-rag experiments from a csv of questions.

usage examples:
    python3 core/main.py run --type 1_pass_with_ontology_refined --csv data.csv --num 5
    python3 core/main.py run --type 1_pass_with_ontology_refined --csv data.csv --model mistral-large-latest
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

csv.field_size_limit(int(1e8))

# ---------------------------------------------------------------------------
# logging setup
# ---------------------------------------------------------------------------

class _DuplicateFilter(logging.Filter):
    """suppress consecutive identical log messages."""
    def filter(self, record):
        current = (record.name, record.levelno, record.getMessage())
        if current == getattr(self, "_last", None):
            return False
        self._last = current
        return True


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(message)s"))
_console.addFilter(_DuplicateFilter())

_file = logging.FileHandler("pipeline_debug.log", mode="w")
_file.setLevel(logging.DEBUG)
_file.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_file.addFilter(_DuplicateFilter())

logger.addHandler(_console)
logger.addHandler(_file)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# separate file for raw prompts
_raw = logging.getLogger("raw_prompts")
_raw_handler = logging.FileHandler("raw_prompts_debug.log", mode="w")
_raw_handler.setLevel(logging.DEBUG)
_raw_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
_raw.addHandler(_raw_handler)
_raw.setLevel(logging.DEBUG)
_raw.propagate = False

# ---------------------------------------------------------------------------
# imports (after logging so agents inherit config)
# ---------------------------------------------------------------------------

from core.utils.helpers import get_llm_model, DEFAULT_MODEL
from core.agents.ontology_agent import OntologyConstructionAgent
from core.agents.refinement_agent_1pass_refined import RefinementAgent1PassRefined
from core.agents.pipeline_analyzer import PipelineAnalyzer
from core.agents.generation_agent import GenerationAgent

# ---------------------------------------------------------------------------
# result data classes
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    index: int
    question: str
    answer: str = ""
    references: List[str] = field(default_factory=list)
    formatted_references: List[str] = field(default_factory=list)
    ontology: Optional[Any] = None
    refined_context_summary: Optional[str] = None
    final_text_context: Optional[str] = None
    time_elapsed: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # set during processing
    original_question_id: Optional[int] = None
    clean_aql_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineStats:
    total: int = 0
    successful: int = 0
    failed: int = 0
    avg_time: float = 0.0

    def update(self, results: List[PipelineResult]):
        self.total = len(results)
        self.successful = sum(1 for r in results if r.status == "success")
        self.failed = self.total - self.successful
        ok = [r for r in results if r.status == "success"]
        self.avg_time = (sum(r.time_elapsed for r in ok) / len(ok)) if ok else 0.0

# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

# csv columns written for every experiment (order matters)
_CSV_FIELDS = [
    "question_id", "question", "pipeline", "context", "clean_aql_results",
    "answer", "references", "formatted_references", "processing_time",
    "documents_parsed", "timestamp", "experiment_type",
    "factuality", "relevance", "groundedness", "helpfulness", "depth",
    "overall_score", "evaluation_feedback", "validation_anomaly", "anomaly_details",
]


class PipelineOrchestrator:
    PIPELINE_TYPES = {
        "1_pass_with_ontology": "1-Pass with Ontology",
        "1_pass_without_ontology": "1-Pass without Ontology",
        "1_pass_with_ontology_refined": "1-Pass with Ontology Refined",
        "adaptive": "Adaptive (set5) - profile + route per question",
    }

    def __init__(
        self,
        pipeline_type: str,
        model_name: str = DEFAULT_MODEL,
        prompt_dir: str = "prompts",
        context_filter: str = "full",
        include_ontology_relationships: bool = False,
        experiment_type: str = "default",
        overwrite: bool = False,
        refinement_temperature: float = 0.1,
        generation_temperature: float = 0.2,
    ):
        if pipeline_type not in self.PIPELINE_TYPES:
            raise ValueError(f"unknown pipeline type: {pipeline_type}")

        self.pipeline_type = pipeline_type
        self.prompt_dir = prompt_dir
        self.context_filter = context_filter
        self.include_ontology_relationships = include_ontology_relationships
        self.overwrite = overwrite

        if "1_pass_with_ontology_refined" in pipeline_type and experiment_type == "default":
            self.experiment_type = "refined_context"
        else:
            self.experiment_type = experiment_type

        # llm instances
        self.ontology_refinement_llm = get_llm_model(model=model_name, temperature=refinement_temperature)
        self.generation_llm = get_llm_model(model=model_name, temperature=generation_temperature)

        # agents
        self.ontology_agent = OntologyConstructionAgent(
            self.ontology_refinement_llm, prompt_dir=f"{prompt_dir}/ontology"
        )
        self.refinement_agent: Optional[RefinementAgent1PassRefined] = None
        if "1_pass_with_ontology_refined" in pipeline_type:
            ref_llm = get_llm_model(model=model_name, temperature=refinement_temperature)
            self.refinement_agent = RefinementAgent1PassRefined(
                ref_llm, prompt_dir=f"{prompt_dir}/refinement"
            )

        self.generation_agent = GenerationAgent(self.generation_llm, prompt_dir=f"{prompt_dir}/generation")
        self.pipeline_analyzer = PipelineAnalyzer()

        # adaptive pipeline instance (lazy, only used when pipeline_type == "adaptive")
        self._adaptive_pipeline = None

        logger.info(
            f"orchestrator ready: {self.PIPELINE_TYPES[pipeline_type]} | model={model_name}"
        )

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        input_csv: str,
        num_questions: Optional[int] = None,
        output_csv: Optional[str] = None,
        verbose: bool = True,
        indices: Optional[List[int]] = None,
    ) -> Tuple[List[PipelineResult], str, PipelineStats]:

        data = self._load_csv(input_csv)
        if not data:
            return [], "", PipelineStats()

        # preserve original 0-based row index so qid stays stable across --indices subsets
        for i, row in enumerate(data):
            row["_row_index"] = i

        if num_questions:
            data = data[:num_questions]
        if indices:
            idx_set = set(indices)
            data = [row for row in data if row["_row_index"] in idx_set]

        # resolve output path
        output_path = self._resolve_output_path(output_csv)

        # skip already-processed questions
        existing = self._load_existing_results(output_path)
        data, skipped = self._filter_already_processed(data, existing, verbose)

        if verbose:
            self._print_header(data)

        results: List[PipelineResult] = []
        for idx, row in enumerate(data):
            orig_idx = row.get("_row_index")
            qid = int(row.get("question_id") or (int(orig_idx) + 1 if orig_idx is not None else (idx + 1)))
            result = self._process_question(idx, row, verbose=verbose, question_id=qid)
            results.append(result)

        stats = PipelineStats()
        stats.update(results)

        self._write_results_csv(output_path, results)

        if verbose:
            self._print_summary(stats, output_path)

        return results, str(output_path), stats

    # ------------------------------------------------------------------
    # single-question processing
    # ------------------------------------------------------------------

    def _process_question(
        self, idx: int, row: Dict, verbose: bool = True, question_id: int = 0
    ) -> PipelineResult:
        question = row.get("question", "")
        if verbose:
            print(f"\n[{idx + 1}] {question[:70]}...")

        result = PipelineResult(index=idx, question=question)
        result.original_question_id = question_id

        aql_results = row.get("aql_results", "")
        if aql_results:
            from core.utils.aql_parser import parse_aql_results
            result.clean_aql_used = parse_aql_results(aql_results)

        structured_context = row.get("structured_context", "")
        start = time.time()
        ontology_time = refinement_time = generation_time = 0.0
        ontology = None
        final_text_context = ""

        try:
            # --- adaptive pipeline shortcut ---
            if self.pipeline_type == "adaptive":
                from core.pipelines.adaptive_pipeline import AdaptivePipeline
                if self._adaptive_pipeline is None:
                    self._adaptive_pipeline = AdaptivePipeline(
                        prompts_root=self.prompt_dir
                    )
                adaptive_result = self._adaptive_pipeline.run(
                    question=question,
                    aql_results_str=aql_results or "",
                )
                result.answer = adaptive_result.answer
                result.references = adaptive_result.references
                result.formatted_references = adaptive_result.formatted_references
                result.final_text_context = adaptive_result.enriched_context
                result.status = "success"
                result.time_elapsed = time.time() - start
                if verbose:
                    rule = getattr(adaptive_result.pipeline_config, "rule_hit", "?")
                    model = getattr(adaptive_result.pipeline_config, "model_name", "?")
                    self._progress("adaptive", "success", f"(rule={rule} model={model} {result.time_elapsed:.1f}s)")
                return result

            # --- ontology ---
            if "without_ontology" not in self.pipeline_type:
                self._progress("building ontology", "in_progress")
                t0 = time.time()
                ontology = self.ontology_agent.process(
                    question, include_relationships=self.include_ontology_relationships
                )
                ontology_time = time.time() - t0
                if ontology and not ontology.should_use_ontology:
                    ontology = None
                if ontology:
                    result.ontology = ontology
                    self._progress(
                        "building ontology", "success",
                        f"({len(ontology.attribute_value_pairs)} attrs)",
                    )
                else:
                    self._progress("building ontology", "success", "(skipped)")

            # --- refinement ---
            self._progress("refining context", "in_progress")
            t0 = time.time()

            include_ontology = "without_ontology" not in self.pipeline_type
            parsed_aql = result.clean_aql_used or aql_results or None

            if self.refinement_agent:
                refined = self.refinement_agent.process_context(
                    question=question,
                    structured_context=structured_context,
                    ontology=ontology,
                    include_ontology=include_ontology,
                    aql_results_str=parsed_aql,
                    context_filter=self.context_filter,
                )
            else:
                raise RuntimeError("no refinement agent configured for this pipeline type")

            refinement_time = time.time() - t0
            result.refined_context_summary = refined.summary
            final_text_context = refined.enriched_context or structured_context or ""

            # --- generation ---
            self._progress("generating answer", "in_progress")
            t0 = time.time()
            answer_obj = self.generation_agent.generate(
                question=question,
                text_context=final_text_context,
                ontology=ontology,
            )
            generation_time = time.time() - t0

            result.answer = answer_obj.answer
            result.references = answer_obj.references
            result.formatted_references = answer_obj.formatted_references
            result.final_text_context = final_text_context
            result.status = "success"
            self._progress("generating answer", "success", f"({generation_time:.1f}s)")

        except Exception as exc:
            result.status = "error"
            result.error_message = str(exc)
            result.answer = f"ERROR: {str(exc)[:100]}"
            logger.error(f"question {idx} failed: {exc}")
            if verbose:
                traceback.print_exc()

        finally:
            result.time_elapsed = time.time() - start
            # log analytics
            self._log_analytics(
                result, question, structured_context, aql_results,
                final_text_context, ontology, ontology_time,
                refinement_time, generation_time,
            )

        return result

    # ------------------------------------------------------------------
    # csv i/o
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: str) -> List[Dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except FileNotFoundError:
            logger.error(f"csv not found: {path}")
            return []

    def _resolve_output_path(self, output_csv: Optional[str]) -> Path:
        if output_csv:
            p = Path(output_csv)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        base = Path("results_to_be_processed")
        base.mkdir(parents=True, exist_ok=True)
        if "1_pass_with_ontology_refined" in self.pipeline_type:
            return base / "refined-context_results.csv"
        name = self.pipeline_type.replace("_", "-")
        return base / f"{name}_results.csv"

    def _load_existing_results(self, path: Path) -> Dict[str, bool]:
        existing: Dict[str, bool] = {}
        if not path.exists():
            return existing
        try:
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    qid = row.get("question_id", "")
                    pipe = row.get("pipeline", "")
                    exp = row.get("experiment_type", "default")
                    answer = row.get("answer", "").strip()
                    ok = not (answer.startswith("Error:") or answer.startswith("ERROR:"))
                    key = f"{qid}_{pipe}" if exp == "default" else f"{qid}_{pipe}_{exp}"
                    existing[key] = ok
        except Exception as exc:
            logger.warning(f"could not read existing results: {exc}")
        return existing

    def _tracking_key(self, question_id) -> str:
        if self.experiment_type == "default":
            return f"{question_id}_{self.pipeline_type}"
        return f"{question_id}_{self.pipeline_type}_{self.experiment_type}"

    def _filter_already_processed(
        self, data: List[Dict], existing: Dict[str, bool], verbose: bool
    ) -> Tuple[List[Dict], int]:
        filtered = []
        skipped = 0
        for idx, row in enumerate(data):
            qid = row.get("question_id") or str(idx + 1)
            key = self._tracking_key(qid)
            if not self.overwrite and key in existing and existing[key]:
                skipped += 1
                if verbose:
                    logger.info(f"skipping question {qid} (already processed)")
            else:
                filtered.append(row)
        if verbose and skipped:
            logger.info(f"skipped {skipped} already-processed questions")
        return filtered, skipped

    def _write_results_csv(self, path: Path, results: List[PipelineResult]) -> None:
        """merge new results into the csv, preserving existing rows."""
        existing_rows: List[Dict] = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))

        for r in results:
            row = self._result_to_row(r)
            # remove any matching old row
            existing_rows = [
                er for er in existing_rows
                if not (
                    str(er.get("question_id")) == str(row["question_id"])
                    and er.get("pipeline") == row["pipeline"]
                    and er.get("experiment_type") == row["experiment_type"]
                )
            ]
            existing_rows.append(row)

        existing_rows.sort(key=lambda x: int(x.get("question_id", 0)))

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(existing_rows)

        logger.info(f"wrote {len(results)} results to {path}")

    def _result_to_row(self, r: PipelineResult) -> Dict[str, Any]:
        ctx = r.final_text_context or ""
        return {
            "question_id": r.original_question_id,
            "question": r.question,
            "pipeline": self.pipeline_type,
            "context": ctx,
            "clean_aql_results": r.clean_aql_used,
            "answer": r.answer,
            "references": " | ".join(r.references) if r.references else "",
            "formatted_references": " | ".join(r.formatted_references) if r.formatted_references else "",
            "processing_time": f"{r.time_elapsed:.2f}",
            "documents_parsed": len([s for s in ctx.split("##") if s.strip()]) if ctx else 0,
            "timestamp": datetime.now().isoformat(),
            "experiment_type": self.experiment_type,
            # evaluation placeholders
            "factuality": "", "relevance": "", "groundedness": "",
            "helpfulness": "", "depth": "", "overall_score": "",
            "evaluation_feedback": "", "validation_anomaly": False,
            "anomaly_details": "",
        }

    # ------------------------------------------------------------------
    # analytics
    # ------------------------------------------------------------------

    def _log_analytics(self, result, question, structured_context, aql_results,
                       final_text_context, ontology, ont_time, ref_time, gen_time):
        try:
            self.pipeline_analyzer.log_pipeline_execution(
                question_id=result.original_question_id,
                question=question,
                pipeline_type=self.pipeline_type,
                ontology_attrs=len(ontology.attribute_value_pairs) if ontology else 0,
                ontology_rels=len(ontology.logical_relationships) if ontology else 0,
                aql_length=len(aql_results or ""),
                structured_context_length=len(structured_context or ""),
                refined_context_length=len(final_text_context or ""),
                refined_context_tokens=len(final_text_context or "") // 4,
                generation_input_length=len(final_text_context or ""),
                final_answer_length=len(result.answer or ""),
                processing_time=result.time_elapsed,
                ontology_time=ont_time,
                refinement_time=ref_time,
                generation_time=gen_time,
                success=(result.status == "success"),
                error_msg=result.error_message,
            )
        except Exception as exc:
            logger.debug(f"analytics logging failed: {exc}")

    # ------------------------------------------------------------------
    # display helpers
    # ------------------------------------------------------------------

    def _print_header(self, data):
        print("=" * 80)
        print("ORCHESTRATOR")
        print("=" * 80)
        print(f"Pipeline: {self.PIPELINE_TYPES[self.pipeline_type]}")
        print(f"Questions: {len(data)}")
        print("=" * 80)

    @staticmethod
    def _progress(step, status, detail=""):
        symbols = {"in_progress": "•", "success": "✓", "error": "✗", "warning": "⚠"}
        sym = symbols.get(status, "•")
        indent = "  " if status == "in_progress" else "    "
        print(f"{indent}{sym} {step} {detail}".rstrip())

    @staticmethod
    def _print_summary(stats, path):
        print("=" * 80)
        print(f"Total: {stats.total} | OK: {stats.successful} | Failed: {stats.failed}")
        print(f"Avg time: {stats.avg_time:.2f}s")
        print(f"Output: {path}")
        print("=" * 80)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ontology-RAG Pipeline Orchestrator")
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--type", required=True, choices=list(PipelineOrchestrator.PIPELINE_TYPES))
    parser.add_argument("--csv", required=True, help="input csv path")
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--output-csv", default=None, help="custom output csv path")
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="mistral model name")
    parser.add_argument("--context-filter", default="full", choices=["full", "slim", "scores_only"])
    parser.add_argument("--experiment-type", default="default")
    parser.add_argument("--include-relationships", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--refinement-temp", type=float, default=0.1)
    parser.add_argument("--generation-temp", type=float, default=0.2)

    args = parser.parse_args()

    if args.verbose:
        _console.setLevel(logging.DEBUG)

    orchestrator = PipelineOrchestrator(
        pipeline_type=args.type,
        model_name=args.model,
        context_filter=args.context_filter,
        include_ontology_relationships=args.include_relationships,
        experiment_type=args.experiment_type,
        overwrite=args.overwrite,
        refinement_temperature=args.refinement_temp,
        generation_temperature=args.generation_temp,
    )

    _results, _path, stats = orchestrator.run(
        input_csv=args.csv,
        num_questions=args.num,
        output_csv=args.output_csv,
        verbose=not args.quiet,
        indices=args.indices,
    )

    sys.exit(0 if stats.failed == 0 else 1)


if __name__ == "__main__":
    main()
