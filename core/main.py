"""pipeline orchestrator - runs ontology-rag experiments from a csv of questions.

usage examples:
    python3 core/main.py run --type 1_pass_with_ontology_refined --csv data.csv --num 5
    python3 core/main.py run --type 1_pass_with_ontology_refined --csv data.csv --model mistral-large-latest
"""

import argparse
import csv
import json
import threading
import asyncio
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

csv.field_size_limit(int(1e8))

# pipeline_analyzer is optional — it may not be present in all checkouts.
try:
    from core.agents.pipeline_analyzer import PipelineAnalyzer as _PipelineAnalyzer
except ImportError:
    _PipelineAnalyzer = None  # type: ignore[assignment,misc]

from core.utils.logger import (
    configure_pipeline_logging,
    get_logger,
    log_question_end,
    log_question_start,
    log_run_summary,
)

# ── logging bootstrap ──────────────────────────────────────────────────────────────
configure_pipeline_logging(log_file="logs/pipeline.log")
logger = get_logger(__name__)

from core.utils.helpers import get_llm_model, DEFAULT_MODEL
from core.agents.ontology_agent import OntologyAgent, OntologyConstructionAgent
from core.agents.refinement_agent_abstracts import RefinementAgentAbstracts, RefinementAgent1PassRefined
from core.agents.generation_agent import GenerationAgent


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
    original_question_id: Optional[int] = None
    clean_aql_used: str = ""
    kg_docs_used: int = 0
    rule_hit: str = ""
    model_name: str = ""
    evidence_mode: str = ""
    kg_source: str = ""

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


_CSV_FIELDS = [
    "question_id", "question", "pipeline", "context", "clean_aql_results",
    "answer", "references", "formatted_references", "processing_time",
    "documents_parsed", "timestamp", "experiment_type",
    "rule_hit", "model_name", "evidence_mode", "kg_source",
    "factuality", "relevance", "groundedness", "helpfulness", "depth",
    "overall_score", "evaluation_feedback", "validation_anomaly", "anomaly_details",
]


class PipelineOrchestrator:
    PIPELINE_TYPES = {
        "1_pass_with_ontology": "1-Pass with Ontology",
        "1_pass_without_ontology": "1-Pass without Ontology",
        "1_pass_with_ontology_refined": "1-Pass with Ontology Refined",
        "adaptive": "Adaptive - profile + route per question",
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

        self.ontology_refinement_llm = get_llm_model(model=model_name, temperature=refinement_temperature)
        self.generation_llm = get_llm_model(model=model_name, temperature=generation_temperature)

        self.ontology_agent = OntologyAgent(
            self.ontology_refinement_llm, prompt_dir=f"{prompt_dir}/ontology"
        )
        self.refinement_agent: Optional[RefinementAgentAbstracts] = None
        if "1_pass_with_ontology_refined" in pipeline_type:
            ref_llm = get_llm_model(model=model_name, temperature=refinement_temperature)
            self.refinement_agent = RefinementAgentAbstracts(
                ref_llm, prompt_dir=f"{prompt_dir}/refinement"
            )

        self.generation_agent = GenerationAgent(self.generation_llm, prompt_dir=f"{prompt_dir}/generation")

        self.pipeline_analyzer = _PipelineAnalyzer() if _PipelineAnalyzer is not None else None
        self._adaptive_pipeline = None

        logger.info(
            "orchestrator ready: %s | model=%s",
            self.PIPELINE_TYPES[pipeline_type], model_name,
        )

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

        for i, row in enumerate(data):
            row["_row_index"] = i

        if num_questions:
            data = data[:num_questions]
        if indices:
            idx_set = set(indices)
            data = [row for row in data if row["_row_index"] in idx_set]

        output_path = self._resolve_output_path(output_csv)
        existing = self._load_existing_results(output_path)
        data, skipped = self._filter_already_processed(data, existing, verbose)

        if verbose:
            self._print_header(data)

        # Fix 8: reset adaptive pipeline session state at the start of each run
        # so that _session_full_doc_uris does not bleed across separate run() calls.
        if self.pipeline_type == "adaptive" and self._adaptive_pipeline is not None:
            self._adaptive_pipeline.reset_session_state()

        results: List[PipelineResult] = []
        # Fix 1: result_lock guards appends to the shared results list; the
        # semaphore is now acquired *inside* each coroutine so it truly limits
        # the number of concurrently executing to_thread calls.
        result_lock = threading.Lock()
        self._csv_lock = threading.Lock()

        async def process_question(semaphore, idx, row):
            orig_idx = row.get("_row_index")
            qid = int(row.get("question_id") or (int(orig_idx) + 1 if orig_idx is not None else (idx + 1)))
            # Fix 1: acquire semaphore here, wrapping the blocking I/O call,
            # not around task creation.
            async with semaphore:
                result = await asyncio.to_thread(self._process_question, idx, row, verbose=verbose, question_id=qid)
            await asyncio.sleep(0.05)
            with result_lock:
                results.append(result)

        async def main():
            semaphore = asyncio.Semaphore(8)
            tasks = [process_question(semaphore, idx, row) for idx, row in enumerate(data)]
            await asyncio.gather(*tasks)

        asyncio.run(main())

        stats = PipelineStats()
        stats.update(results)
        self._write_results_csv(output_path, results)

        if verbose:
            log_run_summary(logger, stats)
            logger.info("  output: %s", output_path)

        return results, str(output_path), stats

    def _process_question(
        self, idx: int, row: Dict, verbose: bool = True, question_id: int = 0
    ) -> PipelineResult:
        question = row.get("question", "")
        if verbose:
            log_question_start(logger, idx + 1, question)

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
            if self.pipeline_type == "adaptive":
                from core.pipelines.pipeline import Pipeline
                if self._adaptive_pipeline is None:
                    self._adaptive_pipeline = Pipeline(
                        prompts_root=self.prompt_dir
                    )
                adaptive_result = self._adaptive_pipeline.run(
                    question=question,
                    aql_results_str=aql_results or "",
                )
                result.answer = adaptive_result.answer
                result.references = adaptive_result.references or []
                result.formatted_references = adaptive_result.formatted_references or []
                result.final_text_context = adaptive_result.enriched_context
                result.kg_docs_used = adaptive_result.kg_docs_used
                result.rule_hit = adaptive_result.rule_hit or ""
                result.kg_source = adaptive_result.kg_source or ""
                cfg = adaptive_result.pipeline_config
                if cfg is not None:
                    result.model_name = getattr(cfg, "model_name", "")
                    result.evidence_mode = getattr(cfg, "evidence_mode", "")
                result.status = "success"
                result.time_elapsed = time.time() - start

            else:
                from core.utils.logger import log_ontology
                if "without_ontology" not in self.pipeline_type:
                    t0 = time.time()
                    ontology = self.ontology_agent.process(
                        question, include_relationships=self.include_ontology_relationships
                    )
                    ontology_time = time.time() - t0
                    if ontology and not ontology.should_use_ontology:
                        ontology = None
                    result.ontology = ontology
                    log_ontology(logger, ontology, elapsed=ontology_time)

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

                from core.utils.logger import log_refinement, log_generation
                log_refinement(logger, final_text_context, elapsed=refinement_time)

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
                log_generation(logger, answer_obj, elapsed=generation_time)

        except Exception as exc:
            result.status = "error"
            result.error_message = str(exc)
            result.answer = f"ERROR: {str(exc)[:100]}"
            logger.error("question %d failed: %s", idx, exc)
            if verbose:
                traceback.print_exc()

        finally:
            result.time_elapsed = time.time() - start
            log_question_end(
                logger, idx + 1, result.status, result.time_elapsed, result.error_message
            )
            self._log_analytics(
                result, question, structured_context, aql_results,
                final_text_context, ontology, ontology_time,
                refinement_time, generation_time,
            )

        return result

    @staticmethod
    def _load_csv(path: str) -> List[Dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except FileNotFoundError:
            logger.error("csv not found: %s", path)
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
            logger.warning("could not read existing results: %s", exc)
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
                    logger.info("skipping Q%s (already processed)", qid)
            else:
                filtered.append(row)
        if verbose and skipped:
            logger.info("skipped %d already-processed questions", skipped)
        return filtered, skipped

    def _write_results_csv(self, path: Path, results: List[PipelineResult]) -> None:
        existing_rows: List[Dict] = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))

        for r in results:
            row = self._result_to_row(r)
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

        logger.info("wrote %d results to %s", len(results), path)

    def _result_to_row(self, r: PipelineResult) -> Dict[str, Any]:
        ctx = r.final_text_context or ""
        refs = r.references or []
        fmt_refs = r.formatted_references or []
        return {
            "question_id": r.original_question_id,
            "question": r.question,
            "pipeline": self.pipeline_type,
            "context": ctx,
            "clean_aql_results": r.clean_aql_used,
            "answer": r.answer,
            "references": " | ".join(str(x) for x in refs) if refs else "",
            "formatted_references": " | ".join(str(x) for x in fmt_refs) if fmt_refs else "",
            "processing_time": f"{r.time_elapsed:.2f}",
            "documents_parsed": (
                r.kg_docs_used if r.kg_docs_used > 0
                else (len([s for s in ctx.split("##") if s.strip()]) if ctx else 0)
            ),
            "timestamp": datetime.now().isoformat(),
            "experiment_type": self.experiment_type,
            "rule_hit": r.rule_hit,
            "model_name": r.model_name,
            "evidence_mode": r.evidence_mode,
            "kg_source": r.kg_source,
            "factuality": "", "relevance": "", "groundedness": "",
            "helpfulness": "", "depth": "", "overall_score": "",
            "evaluation_feedback": "", "validation_anomaly": False,
            "anomaly_details": "",
        }

    def _log_analytics(self, result, question, structured_context, aql_results,
                       final_text_context, ontology, ont_time, ref_time, gen_time):
        if self.pipeline_analyzer is None:
            return
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
            logger.debug("analytics logging failed: %s", exc)

    def _print_header(self, data):
        logger.info("\n%s", "=" * 78)
        logger.info("ORCHESTRATOR  pipeline=%s  questions=%d",
                    self.PIPELINE_TYPES[self.pipeline_type], len(data))
        logger.info("%s", "=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Ontology-RAG Pipeline Orchestrator")
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--type", required=True, choices=list(PipelineOrchestrator.PIPELINE_TYPES))
    parser.add_argument("--csv", required=True, help="input csv path")
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-filter", default="full", choices=["full", "slim", "scores_only"])
    parser.add_argument("--experiment-type", default="default")
    parser.add_argument("--include-relationships", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--refinement-temp", type=float, default=0.1)
    parser.add_argument("--generation-temp", type=float, default=0.2)
    parser.add_argument("--log-file", default="logs/pipeline.log", help="path for debug log file")

    args = parser.parse_args()

    import logging as _logging
    configure_pipeline_logging(
        log_file=args.log_file,
        console_level=_logging.DEBUG if args.verbose else _logging.INFO,
    )

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
