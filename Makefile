.PHONY: run run-one diag diag-cache diag-filter test-profile \
        test-gen test-dist test-filter test-router test-fixes test-all \
        lint format clean

# -----------------------------------------------------------------------
# run the parallel pipeline (defaults: 1 question per tier, 4 workers)
# override: make run TIER_MIX=2,2,2,2,2 WORKERS=8
# -----------------------------------------------------------------------
TIER_MIX   ?= 1,1,1,1,1
WORKERS    ?= 4
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
	python -c "\
import csv, sys;\
sys.path.insert(0, '.');\
from core.pipelines.pipeline import Pipeline;\
p = Pipeline();\
rows = list(csv.DictReader(open('data/dlr/questions.csv')));\
[print(f'[{cfg.rule_hit:<14}] {q[:90]}') for r in rows[:int('$(N)')] for q in [r.get('question','').strip()] if q for _,_,cfg in [p.profile_and_route(q)]]\
"

# -----------------------------------------------------------------------
# test runs  (OPENALEX_OFFLINE=1 = no live API calls)
#
# usage:
#   make test-gen       <- generation phase (adaptive v2)
#   make test-dist      <- tier distribution
#   make test-filter    <- document filter phase 2
#   make test-router    <- routing rules
#   make test-fixes     <- phase 3 fix validation
#   make test-all       <- full test suite
# -----------------------------------------------------------------------
TEST_ENV = OPENALEX_OFFLINE=1 PYTHONPATH=.
PYTEST   = python -m pytest -q -s --tb=short

test-gen:
	mkdir -p logs tests/output
	$(TEST_ENV) $(PYTEST) \
		tests/test_adaptive_v2.py::test_phase3_generation \
		2>&1 | tee logs/gen_$(shell date +%Y%m%d_%H%M%S).log

test-dist:
	mkdir -p logs tests/output
	$(TEST_ENV) $(PYTEST) \
		tests/test_adaptive_distribution.py \
		2>&1 | tee logs/dist_$(shell date +%Y%m%d_%H%M%S).log

test-filter:
	mkdir -p logs tests/output
	$(TEST_ENV) $(PYTEST) \
		tests/test_filter_phase2.py \
		2>&1 | tee logs/filter_$(shell date +%Y%m%d_%H%M%S).log

test-router:
	mkdir -p logs tests/output
	$(TEST_ENV) $(PYTEST) \
		tests/test_router_rules.py \
		2>&1 | tee logs/router_$(shell date +%Y%m%d_%H%M%S).log

test-fixes:
	mkdir -p logs tests/output
	$(TEST_ENV) $(PYTEST) \
		tests/test_fixes_phase3.py \
		2>&1 | tee logs/fixes_$(shell date +%Y%m%d_%H%M%S).log

test-all:
	mkdir -p logs tests/output
	$(TEST_ENV) $(PYTEST) tests/ \
		2>&1 | tee logs/all_$(shell date +%Y%m%d_%H%M%S).log

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
