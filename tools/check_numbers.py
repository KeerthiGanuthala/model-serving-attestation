#!/usr/bin/env python3
"""Verifies that every anchor-phrased figure in documentation matches `docs/metrics.json`.

Matches explicit semantic anchor phrases rather than bare digits so prose changes never cause
false positives while any stale published metric fails the build immediately.
Includes `--self-test` to prove the checker fails when a figure is deliberately broken.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

ANCHOR_SPECS: list[tuple[str, str, str]] = [
    (
        "README.md",
        "exp1_wrong_weights_replica_steps",
        "served wrong weights for {value} of 3,344 replica-steps",
    ),
    (
        "README.md",
        "exp1_wrong_weights_requests",
        "({value:,} of 33,440 logical requests",
    ),
    (
        "README.md",
        "exp1_wrong_weights_share_pct",
        "or {value} of all traffic)",
    ),
    (
        "README.md",
        "exp1_stale_no_reload_steps",
        "| `stale_no_reload` | {value} |",
    ),
    (
        "README.md",
        "exp1_cached_path_steps",
        "| `cached_path_after_alias_move` | {value} |",
    ),
    (
        "README.md",
        "exp1_mutated_artifact_steps",
        "| `mutated_artifact_under_label` | {value} |",
    ),
    (
        "README.md",
        "exp1_stalled_rollout_steps",
        "| `stalled_rollout` | {value} |",
    ),
    (
        "README.md",
        "exp2_in_process_caught_count",
        "in-process attestation catches {value} of 3 failure modes",
    ),
    (
        "README.md",
        "exp3_stalled_first_detected_step",
        "first flags the stalled rollout at step {value}",
    ),
    (
        "README.md",
        "exp4_knee_interval_steps",
        "knee of the curve sits at an audit interval of {value} steps",
    ),
    (
        "README.md",
        "exp4_knee_reduction_pct",
        "eliminating {value} of wrong-weights request exposure",
    ),
    (
        "README.md",
        "exp4_knee_audit_calls",
        "at a cost of {value} replica audit calls",
    ),
    (
        "README.md",
        "exp5_canary_50_capacity_cost_pct",
        "50% canary attestation routing cuts wrong-weights exposure in half with an {value} rollout capacity cost",
    ),
    (
        "docs/adr/0001-in-process-attestation-vs-orchestrator-spec.md",
        "exp1_wrong_weights_replica_steps",
        "accumulate {value} wrong-weight replica-steps across 20 rollouts",
    ),
    (
        "docs/adr/0004-spec-derived-mix-window-vs-tolerance-constant.md",
        "exp3_stalled_first_detected_step",
        "detects the stalled 8-replica rollout at step {value}",
    ),
]


def format_phrase(template: str, raw_val: object) -> str:
    if isinstance(raw_val, int) and "{value:,}" in template:
        return template.format(value=raw_val)
    return template.format(value=raw_val)


def verify_documents(
    repo_root: Path,
    metrics_payload: dict[str, object],
    doc_overrides: dict[str, str] | None = None,
) -> list[str]:
    anchors: dict[str, object] = metrics_payload["anchors"]  # type: ignore[assignment]
    errors: list[str] = []

    for rel_path, anchor_key, template in ANCHOR_SPECS:
        if anchor_key not in anchors:
            errors.append(f"Missing anchor key '{anchor_key}' in docs/metrics.json")
            continue
        expected_phrase = format_phrase(template, anchors[anchor_key])
        if doc_overrides and rel_path in doc_overrides:
            content = doc_overrides[rel_path]
        else:
            full_path = repo_root / rel_path
            if not full_path.exists():
                errors.append(f"Missing target document: {rel_path}")
                continue
            content = full_path.read_text(encoding="utf-8")

        if expected_phrase not in content:
            errors.append(
                f"[{rel_path}] Missing or drifted anchor phrase for '{anchor_key}': "
                f"expected to find exact substring {expected_phrase!r}"
            )

    return errors


def run_self_test(repo_root: Path, metrics_payload: dict[str, object]) -> bool:
    """Break one figure in memory and prove the checker catches the drift."""
    broken_metrics = copy.deepcopy(metrics_payload)
    broken_metrics["anchors"]["exp1_wrong_weights_replica_steps"] = 999999  # type: ignore[index]
    errors = verify_documents(repo_root, broken_metrics)
    if not errors:
        print(
            "SELF-TEST FAILED: check_numbers.py did not fail when exp1_wrong_weights_replica_steps was mutated!",
            file=sys.stderr,
        )
        return False
    print(
        f"Self-test passed: mutating 'exp1_wrong_weights_replica_steps' triggered expected error -> {errors[0]}"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify that breaking one figure causes the checker to fail, in addition to checking real docs.",
    )
    args = parser.parse_args()

    metrics_path = REPO_ROOT / "docs" / "metrics.json"
    if not metrics_path.exists():
        print("Error: docs/metrics.json not found. Run tools/collect_metrics.py first.", file=sys.stderr)
        return 1

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    errors = verify_documents(REPO_ROOT, metrics_payload)
    if errors:
        for err in errors:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1

    print(f"All {len(ANCHOR_SPECS)} anchor phrases verified against docs/metrics.json.")

    if args.self_test:
        if not run_self_test(REPO_ROOT, metrics_payload):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
