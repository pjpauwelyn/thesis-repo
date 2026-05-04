.PHONY: smoke run run-one eval eval-full \
        diag diag-cache diag-filter \
        test-profile \
        audit score check-retraction \
        test-gen test-dist test-filter test-router test-fixes test-all \
        lint format clean

# ============================================================
# Configuration
# ============================================================
TIER_MIX   ?= 1,1,1,1,1
WORKERS    ?= 4
OUTPUT_DIR ?= tests/output
TEST_ENV    = OPENALEX_OFFLINE=1 PYTHONPATH=.
PYTEST      = python -m pytest -q -s --tb=short

# ============================================================
# Quick sanity check  (1 question per tier = 5 questions total)
# Use this to verify the pipeline is healthy after a code change.
# Takes ~5 min.  Output: tests/output/phase3_answers.jsonl
# ============================================================
smoke:
	python scripts/run_pipeline.py \
		--tier-mix $(TIER_MIX) \
		--workers $(WORKERS) \
		--output-dir $(OUTPUT_DIR)

# Alias kept for muscle memory
run: smoke

# ============================================================
# Full evaluation run  (all ~70 questions, 4 parallel workers)
# This is the main research output.  Takes 30-90 min.
# Output: tests/output/phase3_answers.jsonl
#         tests/output/phase3_answers_readable.txt
# Override workers:   make eval WORKERS=8
# Override tier mix:  make eval TIER_MIX=5,15,10,10,30
# ============================================================
eval:
	python scripts/run_pipeline.py \
		--tier-mix 5,15,10,10,30 \
		--workers $(WORKERS) \
		--output-dir $(OUTPUT_DIR)

eval-full: eval

# ============================================================
# Run a single question by 1-based CSV row index
# Example:  make run-one IDX=7
# ============================================================
IDX ?= 1
run-one:
	python scripts/run_pipeline.py --indices $(IDX) --workers 1

# ============================================================
# Diagnostics  (cache audit + document filter inspection)
# ============================================================
diag:
	python scripts/diag.py

diag-cache:
	python scripts/diag.py --phase 1

diag-filter:
	python scripts/diag.py --phase 2

# ============================================================
# Routing smoke-test  (profile N questions, no generation)
# Example:  make test-profile N=20
# ============================================================
N ?= 10
test-profile:
	python scripts/_test_profile.py $(N)

# ============================================================
# Post-run analysis
# audit: structural scan of a JSONL output for bad references,
#        truncation, citation gaps.  Writes phase3_audit.txt.
# score: 9-dimension scorecard for all answers.
#        Writes phase3_scorecard_full.txt.
# check-retraction: scan retrieved docs for retracted papers.
# ============================================================
audit:
	python scripts/audit_phase3.py

score:
	python scripts/score_phase3_full.py

check-retraction:
	python scripts/check_retraction.py

# ============================================================
# Test suite  (OPENALEX_OFFLINE=1 = no live API calls)
# ============================================================
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

# ============================================================
# Code quality
# ============================================================
lint:
	ruff check core/ scripts/ tests/

format:
	ruff format core/ scripts/ tests/

# ============================================================
# Clean generated artefacts
# ============================================================
clean:
	rm -rf tests/output/*.jsonl tests/output/*.txt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true
