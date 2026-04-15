"""
evaluation runners.

provides three runner classes that share csv_manager for i/o and delegate
actual scoring to dlr_evaluator or deepeval_evaluator:

- DLRRunner          – evaluates answers from the dlr results csv
- ExperimentRunner   – evaluates experiment csv files (scores-only,
                       slim-ontology, refined-context, etc.)
- DeepEvalRunner     – evaluates answers with deepeval metrics
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import csv_manager
from .dlr_evaluator import DLREvaluator

logger = logging.getLogger(__name__)


# ===================================================================
# DLRRunner – replaces the DLRRunner class from dlr_evaluation-3.py
# ===================================================================


class DLRRunner:
    """
    batch evaluator for the dlr dataset using the llm-as-a-judge approach.

    handles csv discovery, batch/parallel execution, result saving,
    re-evaluation of placeholder scores, and score validation.
    """

    # possible locations for the source data file (tried in order)
    _DATA_FILE_CANDIDATES = [
        "dlr_data/dlr-results.csv",
        "../dlr_data/dlr-results.csv",
        "dlr-results.csv",
        "dlr_data/DARES25_EarthObsertvation_QA_RAG_results_v1.csv",
        "DARES25_EarthObsertvation_QA_RAG_results_v1.csv",
    ]

    def __init__(self, pipeline_type: str = "rag_two_steps"):
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.results_dir = os.path.join(base_dir, "results_to_be_processed")
        os.makedirs(self.results_dir, exist_ok=True)

        self.pipeline_type = pipeline_type
        self.data_file = self._find_data_file()
        self.results_file = os.path.join(
            self.results_dir, "dlr_evaluation_results.csv"
        )
        self.evaluator = DLREvaluator(use_placeholders=True)
        csv_manager.initialise_csv(self.results_file, csv_manager.DLR_FIELDNAMES)

    # ------------------------------------------------------------------
    # data discovery
    # ------------------------------------------------------------------

    def _find_data_file(self) -> str:
        for candidate in self._DATA_FILE_CANDIDATES:
            if os.path.exists(candidate):
                return candidate
        # return last candidate even if missing; will fail at load time with
        # a clear error rather than silently
        return self._DATA_FILE_CANDIDATES[-1]

    def _load_questions(
        self, num_questions: Optional[int] = None, question_id: Optional[int] = None,
    ) -> List[Dict]:
        rows = csv_manager.load_rows(self.data_file)
        for i, row in enumerate(rows):
            row["question_id"] = i + 1
        if question_id:
            return [r for r in rows if r["question_id"] == question_id]
        return rows[:num_questions] if num_questions else rows

    # ------------------------------------------------------------------
    # single-question evaluation
    # ------------------------------------------------------------------

    def _answer_column(self) -> str:
        return {
            "zero_shot": "zero_shot_answer",
            "rag": "rag_answer",
        }.get(self.pipeline_type, "rag_two_steps_answer")

    def _evaluate_single(self, qdata: Dict) -> Dict:
        """evaluate one question row and return a result dict."""
        question_id = qdata["question_id"]
        question = qdata["question"]
        answer = qdata.get(self._answer_column(), "")
        if not answer or not str(answer).strip():
            raise ValueError(
                f"No valid answer for question {question_id}. "
                f"Expected non-empty '{self._answer_column()}' column."
            )

        context = qdata.get("structured_context", "") or qdata.get("context", "")

        start = datetime.now()
        evaluation = self.evaluator.evaluate_answer(question, answer, context)
        proc_time = (datetime.now() - start).total_seconds()

        result: Dict[str, Any] = {
            "question_id": question_id,
            "question": question,
            "pipeline": self.pipeline_type,
            "context": context,
            "answer": answer,
            "factuality": evaluation["criteria_scores"]["factuality"],
            "relevance": evaluation["criteria_scores"]["relevance"],
            "groundedness": evaluation["criteria_scores"]["groundedness"],
            "helpfulness": evaluation["criteria_scores"]["helpfulness"],
            "depth": evaluation["criteria_scores"]["depth"],
            "overall_score": evaluation["overall_score"],
            "processing_time": round(proc_time, 2),
            "documents_parsed": (
                len([d for d in context.split("##") if d.strip()]) if context else 0
            ),
            "timestamp": datetime.now().isoformat(),
        }

        if evaluation.get("validation_anomaly"):
            result["validation_anomaly"] = True
            result["anomaly_details"] = json.dumps(evaluation["anomaly_details"])
            msg = evaluation["anomaly_details"]["message"]
            if "Perfect score achieved" in msg:
                logger.info("Q%s: 🎯 %s", question_id, msg)
            else:
                logger.warning("Q%s: anomaly - %s", question_id, msg)

        return result

    def _error_result(self, qdata: Dict, error: str) -> Dict:
        return {
            "question_id": qdata["question_id"],
            "question": qdata.get("question", ""),
            "pipeline": self.pipeline_type,
            "context": qdata.get("structured_context", ""),
            "answer": qdata.get(self._answer_column(), ""),
            "factuality": -1, "relevance": -1, "groundedness": -1,
            "helpfulness": -1, "depth": -1, "overall_score": -1,
            "processing_time": 0, "documents_parsed": 0,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "validation_anomaly": True,
            "anomaly_details": json.dumps({
                "valid": False, "message": f"Evaluation failed: {error}",
                "severity": "critical",
            }),
        }

    # ------------------------------------------------------------------
    # batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        questions: List[Dict],
        batch_size: int = 5,
        max_workers: int = 1,
    ) -> List[Dict]:
        """evaluate *questions* in batches, optionally in parallel."""
        import concurrent.futures

        results: List[Dict] = []
        total = len(questions)
        logger.info("Starting batch evaluation of %d questions", total)

        for start in range(0, total, batch_size):
            batch = questions[start : start + batch_size]
            end = min(start + batch_size, total)
            logger.info(
                "Batch %d: questions %d-%d",
                start // batch_size + 1, start + 1, end,
            )

            if max_workers <= 1:
                for qd in batch:
                    try:
                        results.append(self._evaluate_single(qd))
                    except Exception as e:
                        logger.error("Q%s error: %s", qd["question_id"], e)
                        results.append(self._error_result(qd, str(e)))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futs = {pool.submit(self._evaluate_single, qd): qd for qd in batch}
                    for fut in concurrent.futures.as_completed(futs):
                        qd = futs[fut]
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            logger.error("Q%s error: %s", qd["question_id"], e)
                            results.append(self._error_result(qd, str(e)))

            logger.info("%.1f%% done (%d/%d)", end / total * 100, end, total)

        return results

    def save_result(self, result: Dict, overwrite: bool = False) -> None:
        """persist a single result row."""
        if overwrite:
            csv_manager.upsert_row(
                self.results_file,
                csv_manager.DLR_FIELDNAMES,
                result,
                key_columns=["question_id", "pipeline"],
            )
        else:
            csv_manager.append_row(
                self.results_file, csv_manager.DLR_FIELDNAMES, result,
            )

    # ------------------------------------------------------------------
    # re-evaluation of placeholder (-1) scores
    # ------------------------------------------------------------------

    def evaluate_existing_answers(
        self,
        num_questions: Optional[int] = None,
        question_id: Optional[int] = None,
    ) -> None:
        """re-evaluate rows that still carry placeholder scores."""
        existing = csv_manager.load_rows(self.results_file)
        to_eval = [
            r for r in existing
            if r.get("answer", "").strip()
            and float(r.get("overall_score", 0)) == -1.0
            and (question_id is None or int(r["question_id"]) == question_id)
            and (num_questions is None or int(r["question_id"]) <= num_questions)
        ]
        logger.info("Found %d existing answers to re-evaluate", len(to_eval))

        for row in to_eval:
            q_id = int(row["question_id"])
            questions = self._load_questions(question_id=q_id)
            qdata = next((q for q in questions if q["question_id"] == q_id), None)
            if not qdata:
                logger.warning("Q%d not in source data, skipping", q_id)
                continue

            answer = row.get(self._answer_column()) or row.get("answer", "")
            if not answer or not str(answer).strip():
                logger.warning("Q%d has no answer, skipping", q_id)
                continue

            context = qdata.get("structured_context", "") or qdata.get("context", "")
            start = time.time()
            evaluation = self.evaluator.evaluate_answer(
                qdata["question"], answer, context,
            )

            cs = evaluation.get("criteria_scores", {})
            row.update({
                "question": qdata["question"],
                "context": context,
                "factuality": cs.get("factuality", -1),
                "relevance": cs.get("relevance", -1),
                "groundedness": cs.get("groundedness", -1),
                "helpfulness": cs.get("helpfulness", -1),
                "depth": cs.get("depth", -1),
                "overall_score": evaluation.get("overall_score", -1),
                "processing_time": round(time.time() - start, 2),
                "timestamp": datetime.now().isoformat(),
            })
            self.save_result(row, overwrite=True)

    # ------------------------------------------------------------------
    # score validation
    # ------------------------------------------------------------------

    def validate_existing_results(self) -> None:
        """check all existing scores for statistical anomalies."""
        rows = csv_manager.load_rows(self.results_file)
        if not rows:
            logger.warning("No results to validate")
            return

        summary: Dict[str, int] = defaultdict(int)
        for row in rows:
            try:
                scores = {
                    k: int(float(row[k]))
                    for k in ("factuality", "relevance", "groundedness", "helpfulness", "depth")
                }
                val = self.evaluator._validate_scores(scores)
                if val["valid"] and "Perfect score achieved" in val["message"]:
                    logger.info("Q%s (%s): 🎯 %s", row["question_id"], row["pipeline"], val["message"])
                elif not val["valid"]:
                    summary[val["severity"]] += 1
                    logger.warning("Q%s (%s): %s", row["question_id"], row["pipeline"], val["message"])
                    row["validation_anomaly"] = True
                    row["anomaly_details"] = json.dumps(val)
                    self.save_result(row, overwrite=True)
                else:
                    summary["normal"] += 1
            except Exception as e:
                logger.error("Validation error Q%s: %s", row.get("question_id"), e)
                summary["errors"] += 1

        logger.info("Validation: %d rows, %s", len(rows), dict(summary))


# ===================================================================
# ExperimentRunner – replaces ExperimentEvaluator + Experiment3Evaluator
# ===================================================================


class ExperimentRunner:
    """
    generic experiment evaluator using the dlr llm-as-a-judge approach.

    works for any experiment type (scores-only, slim-ontology, refined-context,
    etc.) by accepting the input file, output file, and experiment name at
    construction time.
    """

    _CRITERIA_COLUMNS = [
        "factuality", "relevance", "groundedness", "helpfulness", "depth",
    ]

    def __init__(
        self,
        input_file: str,
        output_file: str,
        experiment_type: str,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.experiment_type = experiment_type
        self.evaluator = DLREvaluator(use_placeholders=True)

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        csv_manager.initialise_csv(output_file, csv_manager.EXPERIMENT_FIELDNAMES)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def run(
        self,
        num_questions: Optional[int] = None,
        force: bool = False,
    ) -> List[Dict]:
        """evaluate up to *num_questions* rows from the input csv."""
        rows = csv_manager.load_rows(self.input_file, limit=num_questions)
        if not rows:
            logger.warning("No rows in %s", self.input_file)
            return []

        results: List[Dict] = []
        skipped = 0
        for row in rows:
            match = {
                "question_id": row["question_id"],
                "pipeline": row.get("pipeline", ""),
                "experiment_type": row.get("experiment_type", self.experiment_type),
            }
            if not force and not csv_manager.needs_evaluation(
                self.output_file, match,
                criteria_columns=self._CRITERIA_COLUMNS,
            ):
                skipped += 1
                continue

            evaluated = self._evaluate_row(row)
            results.append(evaluated)

            csv_manager.upsert_row(
                self.output_file,
                csv_manager.EXPERIMENT_FIELDNAMES,
                evaluated,
                key_columns=["question_id", "pipeline", "experiment_type"],
                sort_key="question_id",
            )

        logger.info(
            "%s: evaluated %d, skipped %d", self.experiment_type, len(results), skipped,
        )
        return results

    def _evaluate_row(self, row: Dict) -> Dict:
        question = row["question"]
        answer = row["answer"]
        context = row.get("context", "")

        start = time.time()
        evaluation = self.evaluator.evaluate_answer(question, answer, context)
        proc_time = time.time() - start

        cs = evaluation.get("criteria_scores", {})
        overall = evaluation.get("overall_score", -1)

        return {
            "question_id": row["question_id"],
            "question": question,
            "pipeline": row.get("pipeline", ""),
            "experiment_type": row.get("experiment_type", self.experiment_type),
            "context": context,
            "answer": answer,
            "references": row.get("references", ""),
            "factuality": cs.get("factuality", -1),
            "relevance": cs.get("relevance", -1),
            "groundedness": cs.get("groundedness", -1),
            "helpfulness": cs.get("helpfulness", -1),
            "depth": cs.get("depth", -1),
            "overall_score": overall,
            "processing_time": round(proc_time, 2),
            "timestamp": datetime.now().isoformat(),
            "evaluation_status": "success" if overall != -1 else "failed",
            "ontology_attributes": row.get("ontology_attributes", ""),
            "ontology_relationships": row.get("ontology_relationships", ""),
        }


# ===================================================================
# DeepEvalRunner – replaces robust_deepeval-5 + evaluate_experiments_only-6
# ===================================================================


class DeepEvalRunner:
    """
    batch runner for deepeval-based evaluation.

    supports full evaluation (all pipelines), experiment-only mode,
    relevancy-only mode, and csv resumption.
    """

    def __init__(
        self,
        results_file: str = "deepeval_results.csv",
        use_fallbacks: bool = False,
        faithfulness_enabled: bool = True,
    ):
        from .deepeval_evaluator import DeepEvalEvaluator, HAS_NEW_METRICS

        self.results_file = results_file
        self.has_new_metrics = HAS_NEW_METRICS

        fieldnames = csv_manager.deepeval_fieldnames(HAS_NEW_METRICS)
        csv_manager.initialise_csv(results_file, fieldnames)

        self.evaluator = DeepEvalEvaluator(
            use_fallbacks=use_fallbacks,
            faithfulness_enabled=faithfulness_enabled,
        )
        self.evaluator.load_histories_from_csv(results_file)

    # ------------------------------------------------------------------
    # result i/o
    # ------------------------------------------------------------------

    def _fieldnames(self) -> List[str]:
        return csv_manager.deepeval_fieldnames(self.has_new_metrics)

    def _save(self, data: Dict) -> bool:
        """append row if no duplicate exists."""
        match = {
            "question_id": str(data["question_id"]),
            "pipeline": data["pipeline"],
        }
        if csv_manager.row_exists(self.results_file, match):
            return False
        csv_manager.append_row(self.results_file, self._fieldnames(), data)
        return True

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate_rows(
        self,
        rows: List[Dict],
        rate_limit_every: int = 20,
    ) -> int:
        """
        evaluate a list of result dicts (must contain question, answer,
        context, question_id, pipeline).  returns the number of successfully
        evaluated rows.
        """
        success = 0
        start = time.time()

        for i, row in enumerate(rows):
            # basic rate-limiting
            if i > 0 and i % rate_limit_every == 0:
                elapsed = time.time() - start
                if elapsed < 60:
                    time.sleep(60 - elapsed)
                start = time.time()

            scores = self.evaluator.evaluate(
                question=row["question"],
                answer=row["answer"],
                context=row.get("context", ""),
            )

            data: Dict[str, Any] = {
                "question_id": row["question_id"],
                "question": row["question"],
                "pipeline": row["pipeline"],
                "answer": row["answer"],
                "hallucination": scores.get("hallucination", -1),
                "relevancy": scores.get("relevancy", -1),
                "faithfulness": scores.get("faithfulness", -1),
                "contextual_relevancy": scores.get("contextual_relevancy", -1),
                "overall_quality": scores.get("overall_quality", -1),
                "timestamp": datetime.now().isoformat(),
            }
            if self.has_new_metrics:
                data.update({
                    "coherence": scores.get("coherence", -1),
                    "toxicity": scores.get("toxicity", -1),
                    "bias": scores.get("bias", -1),
                })

            self._save(data)
            success += 1

        return success

    def evaluate_relevancy_only(
        self,
        rows: List[Dict],
    ) -> List[Dict]:
        """
        lightweight relevancy-only evaluation (replaces robust_deepeval_relevancy).
        """
        results = []
        for row in rows:
            score = self.evaluator.evaluate_relevancy_only(
                row["question"], row["answer"],
            )
            results.append({
                "question_id": row["question_id"],
                "pipeline": row.get("pipeline", ""),
                "question": row["question"],
                "answer": row["answer"],
                "relevancy_score": score,
            })
        return results
