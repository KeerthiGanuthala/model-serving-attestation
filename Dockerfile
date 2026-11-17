FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY pyproject.toml Makefile README.md ./
COPY src/ ./src/
COPY tools/ ./tools/
COPY tests/ ./tests/
COPY docs/ ./docs/

# Verify tests, metrics, and number consistency at build time so the image proves its own claims
RUN python3 tools/collect_metrics.py --quiet \
    && python3 tools/run_tests_with_coverage.py \
    && python3 tools/check_numbers.py --self-test

CMD ["python3", "-m", "attestation.cli", "--demo"]
