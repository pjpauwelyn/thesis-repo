# tests_framework – lightweight end-to-end testing for the ontology-rag pipeline

## overview

this framework runs a small suite of representative questions through the core
pipeline components (retrieval, refinement, generation) and the dlr evaluation
layer, then validates outputs with sanity checks and writes an aggregate report.

**design principles:**
- **offline by default** – uses mock llm / mock openai so tests run without api
  keys and complete in seconds.
- **non-invasive** – no modifications to `core/` or `evaluation/` source code;
  tests call existing entrypoints and patch only the llm client at runtime.
- **lightweight** – 1-3 questions per scenario; full scan finishes in < 30s.
- **single command** – `python -m tests_framework` from the project root.

## folder layout

```
tests_framework/
├── __init__.py           # package marker
├── __main__.py           # enables `python -m tests_framework`
├── config.py             # test questions, paths, thresholds, failure markers
├── fixtures.py           # MockLLM, MockOpenAIClient, prompt stubs, synthetic csv
├── checks.py             # reusable sanity-check predicates (CheckResult)
├── test_pipeline.py      # pipeline scenarios (refined, ontology, empty context, csv)
├── test_evaluation.py    # evaluation scenarios (normal, perfect, failure, anomaly)
├── report.py             # json/csv/console summary builder
├── run_full_scan.py      # main runner script
└── README.md             # this file
```

## quick start

```bash
cd refactored/
python -m tests_framework
```

or equivalently:

```bash
python tests_framework/run_full_scan.py
```

### options

| flag               | description                          |
|--------------------|--------------------------------------|
| `--output-dir DIR` | custom output directory               |
| `--verbose`, `-v`  | enable debug-level logging            |

## what is tested

### pipeline scenarios

| scenario              | pipeline type                      | questions | what it checks                              |
|-----------------------|------------------------------------|-----------|---------------------------------------------|
| `refined_pipeline`    | `1_pass_with_ontology_refined`     | 2         | full refined path: ontology → refinement → generation |
| `ontology_pipeline`   | `1_pass_with_ontology`             | 1         | ontology-only path (no refinement agent)    |
| `empty_context`       | `1_pass_with_ontology_refined`     | 1         | graceful handling of missing context/aql    |
| `csv_output`          | all                                | –         | output csv exists, has data, correct columns |

### evaluation scenarios

| scenario               | what it checks                                                |
|------------------------|---------------------------------------------------------------|
| `normal_evaluation`    | typical 3-4 scores across all 5 criteria                     |
| `perfect_scores`       | all-5 scores, overall == 5.0, info-level note                |
| `failure_placeholders` | api failure returns -1 placeholders (use_placeholders=True)   |
| `anomaly_detection`    | uniform scores flagged by _validate_scores                   |
| `empty_answer`         | evaluator returns valid (low) scores for empty input         |

### sanity checks per test case

**pipeline checks:**
- `status_field` – PipelineResult.status == "success"
- `answer_not_empty` – non-blank answer
- `answer_length` – answer exceeds minimum character threshold
- `no_failure_markers` – no "No documents available", "ERROR:", etc.
- `context_docs` – at least 1 document in final context
- `processing_time` – within time budget (120s default)

**evaluation checks:**
- `eval_required_criteria` – all 5 criteria present
- `eval_score_range` – scores between 1-5
- `eval_overall_score` – overall >= 1.0 (not -1 failure)
- `eval_feedback_present` – non-empty feedback text

## output files

all written to `test_output/` (or `--output-dir`):

| file                        | format | description                                |
|-----------------------------|--------|--------------------------------------------|
| `test_summary.json`         | json   | full structured report with all details    |
| `test_summary.csv`          | csv    | one row per test case, flat format         |
| `test_pipeline_results.csv` | csv    | pipeline output csv (standard format)      |
| `test_run.log`              | text   | detailed debug log                         |

## metrics collected

- total test cases, pass/fail counts, pass rate
- processing time per question (average, per-scenario)
- documents in final context (count)
- mock llm call count and estimated token usage
- evaluation scores (per-criteria and overall)
- check-level pass/fail/warning breakdown

## mock architecture

the framework never calls external apis. instead:

- **MockLLM** (`fixtures.py`) – replaces `MistralLLMWrapper`. returns
  deterministic json (for ontology/refinement) or text (for generation)
  based on prompt keyword matching. tracks call count and token estimates.

- **MockOpenAIClient** (`fixtures.py`) – replaces `openai.OpenAI` inside
  `DLREvaluator`. returns a fixed evaluation json payload so the evaluator's
  parsing, validation, and anomaly detection code runs end-to-end.

- **prompt stubs** (`fixtures.py`) – minimal prompt templates are written to
  disk before tests if the real files are missing, then cleaned up afterward.

## extending

- add new questions to `TEST_QUESTIONS` in `config.py`
- add new check predicates to `checks.py`
- add new scenarios as methods on `PipelineTestSuite` / `EvaluationTestSuite`
- adjust thresholds in `config.py` → `Thresholds` dataclass
