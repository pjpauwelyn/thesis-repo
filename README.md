# Thesis Repo

Bachelor's thesis — *Dynamic Ontologies in Multi-Agent RAG Systems*.
Radboud University, 2025.

---

## What this is

An adaptive retrieval-augmented generation (RAG) pipeline that routes each
incoming question to the right evidence mode and model tier at runtime, guided
by a dynamically constructed ontology and a lightweight question profile.

Core ideas:
- **Dynamic ontology** — attribute-value pairs + logical relationships are
  extracted per question by `OntologyAgent`.
- **Question profiling** — `OntologyAgent.process_with_profile()` fuses ontology
  extraction and profile scoring (complexity, quantitativity, …) into one LLM
  call.
- **Policy routing** — `core/policy/router.py` reads `rules.yaml` and maps a
  profile to a `PipelineConfig` (model, evidence mode, budgets).
- **Evidence modes** — `abstracts` (single-pass, low latency) or
  `excerpts_narrow` / `excerpts_full` (PDF full-text, higher recall).
- **Refinement** — `RefinementAgentAbstracts` (abstracts mode) or
  `RefinementAgent1PassFullText` (full-text mode) compress retrieved docs into
  a grounded context block before generation.

Answer evaluation is **manual** — hand
`tests/output/full_gen_attempt-N/answers_readable.txt` to your evaluation
prompt of choice.

---

## Repo layout

```
core/
  agents/
    ontology_agent.py            # OntologyAgent (profile + filter)
    refinement_agent_abstracts.py # RefinementAgentAbstracts
    refinement_agent_fulltext.py  # RefinementAgent1PassFullText
    generation_agent.py
    base_agent.py / base_refinement_agent.py
  pipelines/
    pipeline.py                  # Pipeline  (AdaptivePipeline alias kept)
    adaptive_pipeline.py         # compat shim -> re-exports Pipeline
  policy/
    router.py / rules.yaml
  utils/
    data_models.py               # Pydantic models (PipelineConfig, RouteConfig, …)
    fulltext_indexer.py
    aql_parser.py
    openalex_client.py
    helpers.py
scripts/
  run_pipeline.py                # parallel batch runner
  diag.py                        # cache + filter diagnostics
  _test_profile.py               # routing smoke-test helper (used by make test-profile)
prompts/
  ontology/
  refinement/
  generation/
tests/
data/
  dlr/                           # DLR EO question set + AQL result cache
cache/
  fulltext/                      # PDF + text extraction cache
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Provide API keys -- either in a `.env` at the repo root (auto-loaded)
#    or via shell exports.  Tier-2b / tier-3 / safety-tier3 also need
#    OPENROUTER_API_KEY.
export MISTRAL_API_KEY=...
export OPENROUTER_API_KEY=...

# 3. Sanity check — 5 questions (1 per tier), ~5 min
make smoke

# 4. Representative 20-question generation — ~10-25 min, mirrors the
#    production tier mix.  Use as a quality gate before the full run.
make eval-20

# 5. Full generation run — ~70 questions, 30-90 min
make eval

# Run a single question by 1-based CSV row index
make run-one IDX=5

# Diagnostics
make diag         # full: cache audit + filter probe
make diag-cache   # cache audit only
make diag-filter  # filter + refinement handoff only

# Routing smoke-test (no generation)
make test-profile N=20
```

For the full operator guide including all CLI flags, tier reference, and
typical workflows, see **[RUNBOOK.md](RUNBOOK.md)**.

---

## Routing tiers

| Tier | Evidence mode | Model |
|------|---------------|-------|
| `tier-1` / `tier-1-def` | `abstracts` | mistral-small |
| `tier-m` | `abstracts` | mistral-small |
| `tier-2a` | `excerpts_narrow` | mistral-small |
| `tier-2b` | `excerpts_narrow` | mistral-medium |
| `tier-3` / `safety-tier3` | `excerpts_full` | mistral-large |
| `fallback` | `abstracts` | mistral-small |

Routing rules are defined in `core/policy/rules.yaml`.

---

## Key backward-compat aliases

| Old name | New name | Location |
|----------|----------|----------|
| `AdaptivePipeline` | `Pipeline` | `core/pipelines/pipeline.py` |
| `AdaptiveResult` | `PipelineResult` | same |
| `OntologyConstructionAgent` | `OntologyAgent` | `core/agents/ontology_agent.py` |
| `RefinementAgent1PassRefined` | `RefinementAgentAbstracts` | `core/agents/refinement_agent_abstracts.py` |
| `PipelineConfig` | `RouteConfig` | `core/utils/data_models.py` |
| `DocumentAssessment` | `DocScore` | same |

All old names still work — they are kept as module-level aliases.
