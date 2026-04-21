# AGENTS.md — thesis-repo living instructions for any coding agent
# Branch: feature/adaptive-pipeline-impl
# Last deep-read: 2026-04-21

## Session protocol

### On "brief yourself":
1. Read PROJECT.md in full (this file is AGENTS.md, always loaded first).
2. Reply in 4 bullets max: current state / active blocker / what was last done / today's task.
3. Do not ask clarifying questions unless PROJECT.md is genuinely ambiguous.

### On "session done":
1. Draft PROJECT.md update: what changed, new blockers, where we stopped.
2. Show the diff. Wait for PJ to confirm before committing.
3. Commit: docs: update PROJECT.md [YYYY-MM-DD] on feature/adaptive-pipeline-impl.
4. Do NOT push unless explicitly told.

---

## What this repo is
Bachelor's thesis: multi-agent RAG pipeline with dynamic ontologies for Earth
Observation QA. Compares 5 pipeline variants (sets 1-5) on 70 domain questions
from the DLR DARES25 dataset, scored by LLM-as-judge (Claude Sonnet 4.6).
Author: Pieterjan Pauwelyn, Radboud University CS, graduating 2026.

---

## Repo layout (exhaustive)

thesis-repo-clean/
  core/
    main.py                                  CLI runner for sets 1-4
    agents/
      base_agent.py                          ABC: __init__(name, llm), self.logger
      base_refinement_agent.py               shared refinement logic (see below)
      ontology_agent.py                      OntologyConstructionAgent
      refinement_agent_1pass.py              set1-3 refinement (1-pass, structured context)
      refinement_agent_1pass_refined.py      set4/set5 refinement (1-pass, abstracts path)
      refinement_agent_fulltext.py           set5 refinement (excerpts path)
      generation_agent.py                    GenerationAgent
      pipeline_analyzer.py                   telemetry logger
    pipelines/
      adaptive_pipeline.py                   AdaptivePipeline (set5 only)
    policy/
      router.py                              Router: reads rules.yaml, returns PipelineConfig
      rules.yaml                             tier routing rules (tier-1/2/3/safety)
    utils/
      data_models.py                         ALL pydantic models (source of truth)
      helpers.py                             MistralLLMWrapper, get_llm_model()
      aql_parser.py                          parse_aql_results()
      fulltext_indexer.py                    FullTextIndexer (PDF cache + excerpt selection)
      openalex_client.py                     OpenAlex DOI resolver
  evaluation/
    runners.py                               set1-3 generators
    run_fulltext_pipeline.py                 set4 generator
    run_adaptive_pipeline.py                 SET5 generator (current work)
    run_judge.py                             LLM-as-judge (Claude Sonnet 4.6) — run by Perplexity Computer
    dlr_evaluator.py                         DLR evaluation metrics
    deepeval_evaluator.py                    DeepEval metrics
    run_analysis.py                          aggregate analysis across all sets
    csv_manager.py                           CSV read/write helpers
    generate_manual_analysis_csv.py          helper for manual annotation
    results/
      set1_*/answers_*.csv
      set2_*/answers_*.csv
      set3_*/answers_*.csv
      set4_*/answers_*.csv
      set5_adaptive/answers_adaptive.csv     SET5 output (in progress)
  prompts/
    ontology/
      extract_attributes_slim.txt            set1-4 av-pair extraction
      extract_profile.txt                    set5 fused av-pair + QuestionProfile extraction
      extract_relationships.txt              logical relationships (not used in set5)
    refinement/
      refinement_1pass_refined_exp4.txt      set5 abstracts path
    generation/
      zero_shot.txt                          step-1 draft (all pipelines)
      generation_prompt_exp4.txt             set4/set5 base generation prompt
      generation_structured.txt             set5 structured prompt ({answer_shape_directives}
                                             and {synthesis_mode_directives} placeholders)
  data/
    dlr/
      DARES25_EarthObsertvation_QA_RAG_results_v1.csv   70 questions + pre-fetched AQL results
  tests/
    [pytest suite]
  cache/fulltext/                            prebuild cache for FullTextIndexer

---

## Data models (core/utils/data_models.py — source of truth)

### QuestionProfile (pydantic BaseModel)
Fields emitted by OntologyConstructionAgent.process_with_profile():

  identity:               str               question text
  one_line_summary:       str = ""
  question_type:          Literal["definition","mechanism","comparison",
                                  "quantitative","method_eval","application","continuous"]
  complexity:             float [0,1] = 0.5
  quantitativity:         float [0,1] = 0.3
  spatial_specificity:    float [0,1] = 0.1
  temporal_specificity:   float [0,1] = 0.1
  methodological_depth:   float [0,1] = 0.1
  scope:                  Literal["global","regional","local"] = "global"
  needs_numeric_emphasis: bool = False
  answer_shape:           Literal["direct_paragraph","short_explainer","structured_long",
                                  "comparison_table","mechanism_walkthrough","raw"]
  confidence:             Optional[float] = None   (None triggers safety-tier3 in router)

IMPORTANT: QuestionProfile does NOT have expected_length.
run_adaptive_pipeline.py references prof.expected_length — this is the active bug.
Fix: remove expected_length from _FIELDNAMES and all _write_row calls (recommended),
or use getattr(prof, "expected_length", "") as a one-liner patch.

### PipelineConfig (pydantic BaseModel)
Fields set by Router.select():

  model_name:             str = "mistral-small-latest"
  evidence_mode:          Literal["abstracts","excerpts_narrow","excerpts_full"]
  top_k_per_doc:          int = 6
  per_doc_budget:         int = 6000
  global_budget:          int = 30000
  refinement_prompt:      str = "refinement_1pass_refined_exp4.txt"
  generation_prompt:      str = "generation_prompt_exp4.txt"
  scope_filter:           bool = False
  synthesis_mode:         Literal["homogeneous","focused"]
  temperature_refine:     float = 0.1
  temperature_generate:   float = 0.2
  rule_hit:               str = "fallback"
  reason:                 str = ""
  gen_context_cap:        int = 60_000
  max_output_tokens:      int = 700
  system_prompt_modifier: str = ""
  doc_filter_min_keep:    int = 6

### DynamicOntology (pydantic BaseModel)

  attribute_value_pairs:  List[AttributeValuePair]
  logical_relationships:  List[LogicalRelationship]   (not used in set5)
  source_query:           str
  created_at:             datetime
  should_use_ontology:    bool = True
  include_relationships:  bool = True

Methods: get_critical_attributes(threshold=0.8), get_contextual_attributes(threshold=0.4)

### AttributeValuePair

  attribute:    str
  value:        str
  description:  str
  value_type:   Literal["numeric","categorical","temporal","spatial"] = "categorical"
  constraints:  List[str]
  centrality:   float [0,1]

### AdaptiveResult (dataclass, returned by AdaptivePipeline.run())

  answer:               str
  references:           List[str]
  formatted_references: List[str]
  profile:              Optional[QuestionProfile]
  pipeline_config:      Optional[PipelineConfig]
  enriched_context:     str
  rule_hit:             str
  excerpt_stats:        Dict[str,Any]

### RefinedContext (pydantic BaseModel)

  original_context:   str
  question:           str
  ontology_summary:   str
  assessments:        List[DocumentAssessment]
  summary:            str
  enriched_context:   Optional[str]
  enriched_documents: Optional[List[Dict]]

---

## Agent internals

### OntologyConstructionAgent (core/agents/ontology_agent.py)
- process(question) -> DynamicOntology  [sets 1-4]
  calls extract_attributes_slim.txt, then optionally extract_relationships.txt
- process_with_profile(question) -> (DynamicOntology, QuestionProfile)  [set5]
  single LLM call with extract_profile.txt (fused av-pairs + profile JSON)
  falls back to process() + null-confidence profile on any parse error
  null confidence triggers router safety-tier3 fallback
- filter_documents(docs, ontology, profile, question, min_keep=6) -> (full, abstract, drop)
  called from FullTextIndexer to classify doc relevance before excerpt selection
  guardrail: n_full >= max(3, n//4), n_total >= max(min_keep, n//2)
  on guardrail violation: logs WARNING, returns (all_docs, [], [])

### BaseRefinementAgent (core/agents/base_refinement_agent.py)
Abstract base for all refinement agents. Provides:
- process_context(question, structured_context, ontology, include_ontology,
                  aql_results_str, context_filter) -> RefinedContext
- _parse_documents(): parses RELATED DOCUMENTS section from structured_context,
  attaches parsed AQL data to first doc
- _reorder_by_ontology(): sorts docs by keyword overlap with critical AV-pairs
- _build_enriched_context(): builds Refined Context markdown block
  context_filter=full: all fields (metrics, geo/temporal scope, refs)
  context_filter=slim: compact version
  context_filter=scores_only: scores + guidance only
- Abstract: _assess_documents() — implemented by each subclass

### RefinementAgent1PassRefined (core/agents/refinement_agent_1pass_refined.py)
Set4/Set5 abstracts path. Implements _assess_documents():
- One LLM call per batch with refinement_1pass_refined_exp4.txt
- Returns List[DocumentAssessment]

### RefinementAgent1PassFullText (core/agents/refinement_agent_fulltext.py)
Set5 excerpts path (evidence_mode != abstracts).
- Requires set_excerpts(text) call before process_context()
- Passes raw excerpt text through fulltext refinement prompt
- Optional set_scope_filter(bool) for focused synthesis mode

### GenerationAgent (core/agents/generation_agent.py)
Two-step generation: zero-shot draft -> context-grounded refinement.
- generate(question, text_context, ontology) -> Answer
- step 1: zero_shot.txt (question only, no context)
- step 2: refinement_template (default: generation_prompt_exp4.txt)
  replaces {question}, {draft_answer}, {context}, {ontology}
  in set5, AdaptivePipeline pre-fills {answer_shape_directives} and
  {synthesis_mode_directives} BEFORE calling generate(), using
  generation_structured.txt as the template
- Context capped at 8000 chars (_MAX_CONTEXT_CHARS) before LLM call
- Output capped at 800 tokens (_MAX_OUTPUT_TOKENS)
- Reference extraction: looks for References/REFERENCES section header,
  extracts [bracket] citations and plain-text refs

---

## LLM layer (core/utils/helpers.py)

### MistralLLMWrapper
- invoke(prompt_text, force_json=False) -> dict | str | None
- streaming by default (MISTRAL_STREAM=1); avoids idle-connection kills
- 8-attempt retry with exponential backoff + jitter (max 60s delay)
- retryable: 429, 5xx, disconnected, timeout, broken pipe, EOF, overloaded, etc.
- force_json=True: strips markdown fences, JSON-parses, normalises bare lists -> dict
- timeout: 300000ms (5 min) via MISTRAL_TIMEOUT_MS env var

### get_llm_model(model, temperature) -> MistralLLMWrapper
- reads MISTRAL_API_KEY from .env
- DEFAULT_MODEL = env PIPELINE_MODEL or mistral-small-24b-instruct-2503
- AdaptivePipeline caches LLM instances by (model, temperature) key

### Models used

  Set5 profiling:           mistral-small-latest  temp=0.0
  Set5 tier-1/2 refine:     mistral-small-latest  temp=0.1
  Set5 tier-1/2 generate:   mistral-small-latest  temp=0.2
  Set5 tier-3 refine:       mistral-large-latest  temp=0.1
  Set5 tier-3 generate:     mistral-large-latest  temp=0.2
  Sets 1-4:                 mistral-small-24b-instruct-2503 (or env override) temp=0.3
  Judge:                    Claude Sonnet 4.6 (Anthropic, separate key)

---

## Adaptive pipeline flow (set5) — end to end

  INPUT: question + aql_results_str + docs[]
    |
  OntologyConstructionAgent.process_with_profile()
    LLM: mistral-small-latest, temp=0.0
    prompt: prompts/ontology/extract_profile.txt
    -> DynamicOntology (av_pairs), QuestionProfile (complexity, answer_shape, etc.)
    |
  Router.select(profile)
    reads: core/policy/rules.yaml
    -> PipelineConfig (model_name, evidence_mode, rule_hit, ...)
    |
  AdaptivePipeline._build_evidence(cfg, question, ontology, docs)
    if evidence_mode == abstracts:
      -> empty string (use AQL abstracts from aql_results_str directly)
    else:
      -> FullTextIndexer.select_excerpts_for_question(...)
         uses OntologyConstructionAgent.filter_documents() internally
         -> excerpts text block
    |
  Refinement
    if evidence_mode == abstracts:
      RefinementAgent1PassRefined.process_context(...)
    else:
      RefinementAgent1PassFullText.set_excerpts(excerpts)
      RefinementAgent1PassFullText.process_context(...)
    -> RefinedContext.enriched_context
    |
  GenerationAgent.generate(question, enriched_context, ontology)
    template pre-filled with answer_shape + synthesis_mode directives
    step 1: zero_shot.txt -> draft
    step 2: generation_structured.txt -> final answer
    -> Answer.answer
    |
  OUTPUT: AdaptiveResult (answer, profile, pipeline_config, ...)

---

## run_adaptive_pipeline.py — full specification

Entry: python3 evaluation/run_adaptive_pipeline.py --workers 4

CLI flags:
  --input         default: data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv
  --output        default: evaluation/results/set5_adaptive/answers_adaptive.csv
  --num           limit question count (e.g. --num 5 for quick test)
  --indices       run specific 0-based row indices only
  --cache-dir     default: cache/fulltext
  --rules         default: core/policy/rules.yaml
  --prompts-root  default: prompts
  --overwrite     re-run already-done questions (default: skip done)
  --dry-run       profile+route only, no generation, prints tier histogram
  --workers       parallel threads (default: 4)
  --verbose       DEBUG logging

Output CSV columns (_FIELDNAMES):
  question_id, question, answer, rule_hit, model_name, evidence_mode,
  expected_length, answer_shape, complexity, quantitativity,
  spatial_specificity, temporal_specificity, profile_confidence, profile_json

  NOTE: expected_length is in _FIELDNAMES but NOT in QuestionProfile.
  This is the active bug — crashes every row during _write_row telemetry.
  Fix: remove expected_length from _FIELDNAMES and all _write_row calls.

Resume logic:
  Loads already-done qids from output CSV on startup.
  Skips if answer length >= 200 chars AND answer does not start with ERROR.
  Auto-resume happens on re-run without --overwrite.

Concurrency model (post cfff7b9):
  ThreadPoolExecutor with --workers threads
  large-concurrency = 1 (mistral-large serialized via semaphore)
  small-concurrency = 4 (mistral-small parallel)
  --job-timeout-s 600 watchdog (kills stuck tier-3 calls)
  End-of-run retry pass for failed tier-3 jobs
  Small-model jobs submitted first

---

## AQL / input data

File: data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv
Columns: question, question_id, aql_results, [other DLR metadata]
aql_results: JSON string of retrieved documents (title, abstract, fulltext_url, etc.)
parse_aql_results() in core/utils/aql_parser.py cleans and normalises this JSON
FullTextIndexer: downloads and caches PDFs from fulltext_url, chunks them,
                 selects relevant excerpts using ontology-guided scoring

---

## Tier routing (core/policy/rules.yaml)

  tier-1:       complexity<0.4, quant<0.2          model=mistral-small  evidence=abstracts        ~14 questions
  tier-2:       complexity<0.7 or default           model=mistral-small  evidence=excerpts_narrow  ~42 questions
  tier-3:       complexity>=0.7 OR quant>=0.6
                OR spatial>=0.6 OR temporal>=0.6    model=mistral-large  evidence=excerpts_full    ~14 questions
  safety-tier3: confidence==None (parse failed)     model=mistral-large  evidence=excerpts_full    rare

---

## Pipeline variants compared

  Set 1  1_pass_with_ontology          ontology + abstract context              runner: core/main.py
  Set 2  1_pass_without_ontology       no ontology, abstract context            runner: core/main.py
  Set 3  1_pass_with_ontology_refined  ontology + document-assessed context     runner: core/main.py
  Set 4  fulltext_1pass                ontology + fulltext excerpts, fixed model runner: evaluation/run_fulltext_pipeline.py
  Set 5  adaptive                      per-question routing tier-1/2/3          runner: evaluation/run_adaptive_pipeline.py

---

## Run commands

  source venv/bin/activate

  # Prebuild fulltext index cache (run once)
  python3 -m core.utils.fulltext_indexer prebuild \
      --input data/dlr/DARES25_EarthObsertvation_QA_RAG_results_v1.csv

  # Dry-run: tier routing only, no LLM generation
  python3 evaluation/run_adaptive_pipeline.py --dry-run

  # Quick smoke test
  python3 evaluation/run_adaptive_pipeline.py --num 3 --workers 1

  # Full set5 run (auto-resumes)
  python3 evaluation/run_adaptive_pipeline.py --workers 4

  # Re-run from scratch
  python3 evaluation/run_adaptive_pipeline.py --workers 4 --overwrite

  # Specific indices
  python3 evaluation/run_adaptive_pipeline.py --indices 1 5 10

---

## Environment (.env)

  MISTRAL_API_KEY=...
  MISTRAL_STREAM=1               1=streaming (default), 0=non-streaming
  MISTRAL_TIMEOUT_MS=300000      5 min timeout, prevents idle-connection kills
  PIPELINE_MODEL=mistral-small-24b-instruct-2503   default for sets 1-4
  ANTHROPIC_API_KEY=...          for run_judge.py only

---

## Coding rules
- Python 3.8.17. pip + venv. No ruff/mypy — do not add them.
- Read the file before proposing any edit. Never edit blindly.
- Minimal diffs — do not reformat untouched code.
- After any code change: python3 -m pytest tests/ -v. Report all failures.
- Conventional commits on feature/adaptive-pipeline-impl.
- Never: git reset --hard, push --force, clean -fd without typed confirmation.

## Reasoning rule
Trace the logic before claiming confidence.
If uncertain: say so explicitly before proposing any change.
