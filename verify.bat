@echo off
setlocal
set PYTHONPATH=src
echo [1/4] Collecting deterministic experiment metrics...
python tools\collect_metrics.py
if errorlevel 1 exit /b 1

echo [2/4] Running property-named test suite with line coverage...
python tools\run_tests_with_coverage.py
if errorlevel 1 exit /b 1

echo [3/4] Checking published numbers against docs/metrics.json...
python tools\check_numbers.py --self-test
if errorlevel 1 exit /b 1

echo [4/4] Running in-process container self-attestation demo...
python -m attestation.cli --demo
if errorlevel 1 exit /b 1

echo ====================================================================
echo ALL VERIFICATION CHECKS PASSED (offline, deterministic, 0 drift)
echo ====================================================================
