# Runbook — Adaptive RAG Pipeline

This document is the single reference for running and debugging the
pipeline. All commands are run from the **repo root**.

Answer evaluation is done manually — feed `phase3_answers_readable.txt`
to your evaluation prompt of choice.

---

## Prerequisites

```bash
pip install -r requirements.txt
export MISTRAL_API_KEY=<your-key>
```

The live ArangoDB KG is optional. If `ARANGO_ROOT_PASSWORD` is not set the
pipeline falls back to the pre-parsed CSV documents automatically.

---

## Command reference

### 1 — Sanity check after a code change

```bash
make smoke
```

Runs **5 questions** (1 per tier) in parallel. Takes ~5 minutes. Use this
to confirm the pipeline is healthy before committing or before a long run.

```bash
# Override number per tier or worker count
make smoke TIER_MIX=2,2,2,2,2 WORKERS=8
```

Output files written to `tests/output/`:
- `phase3_answers.jsonl` — machine-readable results
- `phase3_answers_readable.txt` — human-readable answers + references

---

### 2 — Full generation run (~70 questions)

```bash
make eval
```

Processes the full DLR question set (~70 questions) using the production
tier mix `5,15,10,10,30`. Takes **30–90 minutes** depending on API latency.

```bash
# More workers for faster throughput (watch rate limits)
make eval WORKERS=8

# Custom tier mix
make eval TIER_MIX=5,15,10,10,30

# Save enriched_context in JSONL for post-run debugging
python scripts/run_pipeline.py --tier-mix 5,15,10,10,30 --workers 4 --save-context
```

Same output files as `make smoke`.

---

### 3 — Single question

```bash
# 1-based CSV row index
make run-one IDX=7

# Equivalent direct command
python scripts/run_pipeline.py --indices 7 --workers 1

# Multiple questions
python scripts/run_pipeline.py --indices 1 5 12 --workers 3
```

---

### 4 — Routing inspection (no generation)

```bash
# Profile 20 questions and print their assigned tier
make test-profile N=20
```

No API calls to the generation model are made. Only the ontology/profiler
LLM call (mistral-small-latest) runs. Useful for checking routing rules.

---

### 5 — Diagnostics

```bash
make diag         # full: cache audit + filter probe
make diag-cache   # phase 1: PDF cache status only
make diag-filter  # phase 2: document filter + refinement handoff
```

---

### 6 — Test suite

```bash
make test-all      # entire tests/ suite (offline, no real API calls)
make test-gen      # tests/test_adaptive_v2.py::test_phase3_generation
make test-dist     # tier distribution tests
make test-filter   # document filter phase
make test-router   # routing rule tests
make test-fixes    # phase 3 fix validation
```

All test targets set `OPENALEX_OFFLINE=1` and `PYTHONPATH=.` automatically.
Logs are written to `logs/<target>_<timestamp>.log`.

---

## Tier reference

| Tier | Evidence mode | Model | Typical question type |
|------|--------------|-------|-----------------------|
| `tier-1` / `tier-1-def` | `abstracts` | mistral-small | Simple factual / definition |
| `tier-m` | `abstracts` | mistral-small | Mechanism, moderate complexity |
| `tier-2a` | `excerpts_narrow` | mistral-small | Quantitative, narrow scope |
| `tier-2b` | `excerpts_narrow` | mistral-medium | Quantitative, broader scope |
| `tier-3` / `safety-tier3` | `excerpts_full` | mistral-large | Comparison, high complexity |
| `fallback` | `abstracts` | mistral-small | Profiler parse failure |

Routing rules are defined in `core/policy/rules.yaml`.

---

## `run_pipeline.py` CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--tier-mix` | `1,1,1,1,1` | Questions per tier: 5-int `(t1,tm,t2a,t2b,t3)` or 3-int legacy `(t1,t2b,t3)` |
| `--workers` | `4` | Thread pool size |
| `--indices` | — | Space-separated 1-based row indices; bypasses tier-mix |
| `--n` | — | Total question count; weights from `--tier-mix` |
| `--output-dir` | `tests/output` | Directory for JSONL + TXT output |
| `--save-context` | off | Include `enriched_context` in JSONL (large; for debugging) |
| `--max-retries` | `3` | Retry count per question on transient errors |
| `--small-concurrency` | `4` | Max concurrent tier-1/fallback jobs |
| `--medium-concurrency` | `2` | Max concurrent tier-m/2a/2b jobs |
| `--large-concurrency` | `1` | Max concurrent tier-3 jobs |
| `--min-gap-ms` | `0` | Minimum ms between job starts (rate-limit guard) |
| `--verbose` / `-v` | off | DEBUG-level logging |
| `--csv` | auto-detect | Path to questions CSV |

---

## Typical workflow

```bash
# 1. After any code change — quick health check
make smoke

# 2. Full generation run
make eval

# 3. Hand `tests/output/phase3_answers_readable.txt` to your evaluation
#    prompt / external AI for scoring.

# 4. If a specific question failed, re-run it alone
make run-one IDX=12
```

---

## Output files

| File | Description |
|------|-------------|
| `tests/output/phase3_answers.jsonl` | One JSON record per question; primary output |
| `tests/output/phase3_answers_readable.txt` | Human-readable answers + formatted references |
| `logs/*.log` | Timestamped logs from test-suite targets |
