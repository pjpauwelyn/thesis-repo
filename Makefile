.PHONY: run diag test-profile lint format clean

# -----------------------------------------------------------------------
# run the parallel pipeline (defaults: 1 question per tier, 4 workers)
# override: make run TIER_MIX=2,2,2,2,2 WORKERS=8
# -----------------------------------------------------------------------
TIER_MIX  ?= 1,1,1,1,1
WORKERS   ?= 4
OUTPUT_DIR ?= tests/output

run:
	python scripts/run_pipeline.py \
		--tier-mix $(TIER_MIX) \
		--workers $(WORKERS) \
		--output-dir $(OUTPUT_DIR)

# run a single question by 1-based CSV index
# usage: make run-one IDX=7
IDX ?= 1
run-one:
	python scripts/run_pipeline.py --indices $(IDX) --workers 1

# -----------------------------------------------------------------------
# diagnostics
# -----------------------------------------------------------------------
diag:
	python scripts/diag.py

diag-cache:
	python scripts/diag.py --phase 1

diag-filter:
	python scripts/diag.py --phase 2

# -----------------------------------------------------------------------
# quick routing smoke-test (no generation, prints tier for N questions)
# -----------------------------------------------------------------------
N ?= 10
test-profile:
	python -c "
import csv, sys
sys.path.insert(0, '.')
from core.pipelines.pipeline import Pipeline
p = Pipeline()
rows = list(csv.DictReader(open('data/dlr/questions.csv')))
for r in rows[:int('$(N)')]:
    q = r.get('question', '').strip()
    if not q: continue
    _, _, cfg = p.profile_and_route(q)
    print(f'[{cfg.rule_hit:<14}] {q[:90]}')
"

# -----------------------------------------------------------------------
# code quality
# -----------------------------------------------------------------------
lint:
	ruff check core/ scripts/ tests/

format:
	ruff format core/ scripts/ tests/

# -----------------------------------------------------------------------
# clean generated artefacts
# -----------------------------------------------------------------------
clean:
	rm -rf tests/output/*.jsonl tests/output/*.txt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true
