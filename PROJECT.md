# PROJECT.md — living context, updated each session
# Do not edit mid-session. Updated by agent after PJ confirms at session end.

## What this project is
Multi-agent RAG pipeline with dynamic ontologies, evaluated on 70 DLR Earth
Observation QA questions. Compares 5 pipeline variants using LLM-as-judge
(Claude Sonnet 4.6). Bachelor's thesis, Radboud University CS, PJ Pauwelyn, 2026.

---

## Overall status

  Sets 1-4 generation:   COMPLETE
  Set5 generation:       BLOCKED (see below)
  Judge + evaluation:    NOT STARTED — run by Perplexity Computer after generation
  Thesis writing:        IN PARALLEL

---

## Active blocker — MUST FIX BEFORE SET5 CAN COMPLETE

Error: 'QuestionProfile' object has no attribute 'expected_length'
Where: evaluation/run_adaptive_pipeline.py, function _process_row

Affects: every question that completes generation — the answer is generated but
NOT written to the CSV because the crash fires during _write_row telemetry.

Root cause (confirmed by reading data_models.py):
QuestionProfile in core/utils/data_models.py does NOT define expected_length.
run_adaptive_pipeline.py has expected_length in _FIELDNAMES and accesses
prof.expected_length unconditionally in both the dry-run and full-run branches.

Exact broken lines in run_adaptive_pipeline.py:
  dry-run branch (~line 197):   "expected_length": profile.expected_length,
  full-run branch (~line 240):  "expected_length": prof.expected_length if prof else "",

Fix (recommended — Option 2):
  Remove expected_length from _FIELDNAMES and all _write_row calls entirely.
  The field does not exist on the model and has no downstream use.

---

## Secondary issues (not crashes, monitor only)

1. filter_documents guardrail warnings firing frequently:
   "filter_documents failed (guardrail violated) — falling back to full set"
   Working as designed. Not a crash. Monitor for quality impact.

2. mistral-large-latest disconnects — up to 8 retries, ~4 min per tier-3 question.
   Per-model semaphore (concurrency=1) in place since cfff7b9.
   Retry-pass and watchdog handle remaining failures once expected_length is fixed.

---

## Completed work

  [x] Sets 1-4 generated and results CSVs saved
  [x] Adaptive pipeline architecture: QuestionProfile, PipelineConfig, Router,
      tier-1/2/3 routing via rules.yaml
  [x] FullTextIndexer: PDF download, caching, excerpt selection, doc filtering
  [x] Parallel execution redesigned (cfff7b9): per-model semaphores (large=1,
      small=4), watchdog timeout 600s, end-of-run tier-3 retry pass
  [x] AGENTS.md + PROJECT.md: exhaustive living context added to repo

---

## Open tasks

  [ ] BLOCKER: Fix expected_length AttributeError in run_adaptive_pipeline.py
  [ ] Verify full 70-question run completes with 0 errors
  [ ] Confirm output CSV has all 70 rows, answer >= 200 chars, no ERROR rows
  [ ] Hand off to Perplexity Computer: run_judge.py, run_analysis.py, graphs, report

---

## Decisions and standing constraints

- evaluation/ contains both generators AND judges — do not reorganize
- Do NOT suggest architectural changes unless PJ explicitly asks
- mistral-large-latest is flaky: tier-3 concurrency MUST stay at 1
- Fulltext cache must be prebuilt before any run with evidence_mode != abstracts
- Judge uses Anthropic (Claude Sonnet 4.6), separate key from Mistral
- Perplexity Computer handles: judge runs, aggregation, graphs, reports
  — do not build evaluation tooling, that is not this agent's job

---

## Last session summary (2026-04-21)

Previous session (2026-04-20, Perplexity Computer):
- Redesigned parallel execution: per-model semaphores, watchdog, retry (cfff7b9)
- Fixed Mistral large endpoint hammering
- Did NOT fix expected_length bug (pre-existing)

This session (2026-04-21, Perplexity):
- Deep read of all agents, data models, pipeline, helpers, runners
- AGENTS.md and PROJECT.md rewritten with full exhaustive detail
- expected_length bug confirmed: QuestionProfile has no such field

Next: fix expected_length (remove from _FIELDNAMES + _write_row), run full set5.
