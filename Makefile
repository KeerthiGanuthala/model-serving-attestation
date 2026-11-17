.PHONY: verify metrics test check-numbers demo clean

PYTHON ?= python3

verify: metrics test check-numbers demo
	@echo "===================================================================="
	@echo "ALL VERIFICATION CHECKS PASSED (offline, deterministic, 0 drift)"
	@echo "===================================================================="

metrics:
	PYTHONPATH=src $(PYTHON) tools/collect_metrics.py

test:
	PYTHONPATH=src $(PYTHON) tools/run_tests_with_coverage.py

check-numbers:
	PYTHONPATH=src $(PYTHON) tools/check_numbers.py --self-test

demo:
	PYTHONPATH=src $(PYTHON) -m attestation.cli --demo

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
