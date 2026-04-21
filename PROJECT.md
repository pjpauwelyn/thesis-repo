# PROJECT.md — living context, updated each session
# _Do not edit mid-session. Updated by agent after PJ confirms at session end._

## What this project is
Multi-agent RAG pipeline with dynamic ontologies, evaluated on 70 DLR Earth
Observation QA questions. Compares 5 pipeline variants using LLM-as-judge
(Claude Sonnet 4.6). Bachelor's thesis, Radboud University CS, PJ Pauwelyn, 2026.

---

## Overall status

  Sets 1-4 generation:   COMPLETE
  Set5 generation:       BLOCKED (see below)
  Judge + evaluation:    NOT STARTED — run by Perplexity Computer after generation completes
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

Fix options (pick one):
  Option 1 — one-liner: replace both with getattr(prof, "expected_length", "")
  Option 2 — cleanest:  delete "expected_length" from _FIELDNAMES and all _write_row calls
  Option 3 — add field: add expected_length: str = "" to QuestionProfile in data_models.py

Recommended: Option 2. The field has no value since it does not exist in the model.

---

## Secondary issues visible in logs (not crashes, just monitor)

1. filter_documents guardrail warnings firing frequently:
   "filter_documents failed (guardrail violated: n_full=2 (min 2), n_total=5 (min 4))
   — falling back to full set"
   This is working as designed (fail-safe keeps all docs). Not a crash.
   Worth monitoring whether it degrades answer quality.

2. mistral-large-latest disconnects — retrying up to 8 times, up to ~4 min per
   tier-3 question. Per-model semaphore (concurrency=1) is in place since cfff7b9.
   With the expected_length fix, the retry-pass and watchdog handle remaining failures.

---

## Completed work

  [x] Sets 1-4 generated and results CSVs saved
  [x] Adaptive pipeline architecture: QuestionProfile, PipelineConfig, Router,
      tier-1/2/3 routing via rules.yaml
  [x] FullTextIndexer: PDF download, caching, excerpt selection, doc filtering
  [x] Parallel execution redesigned (cfff7b9): per-model semaphores (large=1,
      small=4), watchdog timeout (600s), end-of-run tier-3 retry pass
  [x] AGENTS.md + PROJECT.md: exhaustive living context added to repo

---

## Open tasks

  [ ] BLOCKER: Fix expected_length AttributeError in run_adaptive_pipeline.py
  [ ] Verify full 70-question run completes with 0 errors
  [ ] Confirm output CSV has all 70 rows, answer >= 200 chars, no ERROR rows
  [ ] Hand off to Perplexity Computer for: run_judge.py, run_analysis.py,
      graphs, evaluation report

---

## Decisions and standing constraints

- evaluation/ contains both generators AND judges — established convention, do not reorganize
- Do NOT suggest architectural changes without PJ explicitly asking for them
- mistral-large-latest endpoint is flaky: tier-3 concurrency MUST stay at 1
- Fulltext cache must be prebuilt before any run that uses evidence_mode != "abstracts"
- Judge uses Anthropic (Claude Sonnet 4.6), separate API key from Mistral key
- Perplexity Computer handles: judge runs, result aggregation, graphs, reports
  — do not build evaluation tooling, that is not this agent's job

---

## Last session summary (2026-04-21)

Previous session (2026-04-20, Perplexity Computer):
- Redesigned parallel execution: per-model semaphores, pre-profile stage,
  watchdog timeout, retry pass (commit cfff7b9)
- Fixed Mistral large endpoint hammering problem
- Did NOT fix: expected_length AttributeError (pre-existing bug)

This session (2026-04-21, Perplexity):
- Deep repo read: all core agents, data models, pipeline, helpers, runners
- AGENTS.md and PROJECT.md rewritten with full exhaustive detail end-to-end
- expected_length bug confirmed: QuestionProfile has no such field

Next: fix expected_length (Option 2), run full set5, verify 70/70 rows complete.
