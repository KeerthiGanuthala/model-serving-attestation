#!/usr/bin/env python3
"""Runs the full property-named unittest suite under Python's stdlib `trace` module and enforces line coverage."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import trace
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TESTS_DIR))


def executable_lines_in_file(py_file: Path) -> set[int]:
    """Compute executable statement line numbers in `py_file` using `ast`."""
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    lines: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.Return,
                ast.Raise,
                ast.Assert,
                ast.If,
                ast.For,
                ast.While,
                ast.With,
                ast.Try,
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            # Skip standalone docstrings and Protocol stub methods (`...`)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and node.body[0].value.value is Ellipsis
            ):
                continue
            lines.add(node.lineno)
        elif isinstance(node, ast.Expr):
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, (str, type(Ellipsis)))):
                lines.add(node.lineno)

    return lines


def run_suite() -> None:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main(min_coverage_pct: float = 88.0) -> int:
    tracer = trace.Trace(count=1, trace=0, ignoredirs=[sys.prefix, sys.exec_prefix])
    tracer.runfunc(run_suite)
    results = tracer.results()

    executed_by_file: dict[Path, set[int]] = {}
    for (filename, lineno), count in results.counts.items():
        if count > 0:
            p = Path(filename).resolve()
            executed_by_file.setdefault(p, set()).add(lineno)

    print("\nLine Coverage Report (src/attestation):")
    print("-" * 68)
    print(f"{'Module':<38} {'Exec':>6} {'Hit':>6} {'Coverage':>12}")
    print("-" * 68)

    total_exec = 0
    total_hit = 0
    for py_file in sorted((SRC_DIR / "attestation").glob("*.py")):
        resolved = py_file.resolve()
        exec_lines = executable_lines_in_file(resolved)
        hit_lines = exec_lines & executed_by_file.get(resolved, set())
        pct = (len(hit_lines) / len(exec_lines) * 100.0) if exec_lines else 100.0
        total_exec += len(exec_lines)
        total_hit += len(hit_lines)
        rel_name = str(py_file.relative_to(REPO_ROOT))
        print(f"{rel_name:<38} {len(exec_lines):>6} {len(hit_lines):>6} {pct:>11.1f}%")

    overall_pct = (total_hit / total_exec * 100.0) if total_exec else 100.0
    print("-" * 68)
    print(f"{'TOTAL':<38} {total_exec:>6} {total_hit:>6} {overall_pct:>11.1f}%")
    print("-" * 68)

    if overall_pct < min_coverage_pct:
        print(
            f"FAIL: Overall coverage {overall_pct:.1f}% is below threshold {min_coverage_pct:.1f}%",
            file=sys.stderr,
        )
        return 1
    print(f"Coverage threshold passed ({overall_pct:.1f}% >= {min_coverage_pct:.1f}%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
