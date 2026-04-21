# AGENTS.md — thesis-repo

## What this repo is
Bachelor's thesis: multi-agent RAG system with dynamic ontologies.
Radboud University CS, PJ Pauwelyn, graduating 2026.
Active branch: feature/adaptive-pipeline-v2

## Session start protocol
When PJ says "brief yourself":
1. Read PROJECT.md in full.
2. Summarise in max 3 bullets: current state / open blocker / what we're doing today.
3. Do not ask clarifying questions unless PROJECT.md is genuinely unclear.

## Session end protocol
When PJ says "session done":
1. Draft a PROJECT.md update (what changed, what's open, where we left off).
2. Show the diff to PJ and wait for confirmation.
3. On confirmation: commit PROJECT.md with "docs: update PROJECT.md [YYYY-MM-DD]"
4. Do not push unless PJ explicitly says so.

## Entry point
python3 core/main.py run --type <type> --csv <file>
Pipeline types: 1_pass_with_ontology | 1_pass_without_ontology | 1_pass_with_ontology_refined | adaptive

## Key directories
- core/           main pipeline logic, agents, orchestrator
- core/agents/    ontology, refinement, generation, pipeline_analyzer agents
- core/pipelines/ pipeline implementations
- core/policy/    routing/policy logic
- core/utils/     helpers, aql parser, model loader
- prompts/        prompt templates (ontology/, refinement/, generation/)
- evaluation/     evaluation scripts
- data/           input CSVs
- results_to_be_processed/  output CSVs
- tests/          pytest tests

## Coding rules
- Python 3.8.17. pip + venv. No ruff/mypy config exists.
- Read before editing. Minimal diffs.
- Run tests after changes: python3 -m pytest tests/ -v
- Conventional commits on feature/adaptive-pipeline-v2

## Reasoning rule
Trace reasoning before stating a solution. Confidence follows logic, not the other way around.
