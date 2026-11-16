#!/usr/bin/env python3
"""Re-measures every published figure and writes `docs/experiments/*.json` and `docs/metrics.json`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from attestation.experiments import format_experiment_tables, run_all_experiments  # noqa: E402


def build_consolidated_metrics(all_results: dict[str, object]) -> dict[str, object]:
    exp1 = all_results["exp1_green_dashboard"]  # type: ignore[index]
    exp2 = all_results["exp2_label_vs_bytes"]  # type: ignore[index]
    exp3 = all_results["exp3_rollout_mix_vs_stuck"]  # type: ignore[index]
    exp4 = all_results["exp4_polling_cost_vs_exposure"]  # type: ignore[index]
    exp5 = all_results["exp5_canary_attestation"]  # type: ignore[index]

    h1 = exp1["headline"]  # type: ignore[index]
    fb_steps = exp1["fault_breakdown_replica_steps"]  # type: ignore[index]
    fb_reqs = exp1["fault_breakdown_requests"]  # type: ignore[index]
    s2 = exp2["summary"]  # type: ignore[index]
    s3 = exp3["stalled_vs_slow_comparison"]  # type: ignore[index]
    k4 = exp4["knee_of_curve"]  # type: ignore[index]
    s4_rows = {row["audit_interval_steps"]: row for row in exp4["sweep_table"]}  # type: ignore[index]
    s5_rows = {
        f"{row['attested_routing_fraction']:.2f}": row
        for row in exp5["canary_routing_table"]  # type: ignore[index]
    }

    return {
        "schema_version": 1,
        "seed": 20261117,
        "anchors": {
            "exp1_total_replica_steps": h1["total_replica_steps"],
            "exp1_green_dashboard_replica_steps": h1["green_dashboard_replica_steps"],
            "exp1_wrong_weights_replica_steps": h1["wrong_weights_replica_steps"],
            "exp1_wrong_weights_requests": h1["wrong_weights_requests"],
            "exp1_wrong_weights_share_pct": f"{h1['wrong_weights_share_pct']:.2f}%",
            "exp1_honest_false_alarms": h1["honest_rollout_false_alarms"],
            "exp1_stale_no_reload_steps": fb_steps["stale_no_reload"],
            "exp1_cached_path_steps": fb_steps["cached_path_after_alias_move"],
            "exp1_mutated_artifact_steps": fb_steps["mutated_artifact_under_label"],
            "exp1_stalled_rollout_steps": fb_steps["stalled_rollout"],
            "exp1_stale_no_reload_requests": fb_reqs["stale_no_reload"],
            "exp1_cached_path_requests": fb_reqs["cached_path_after_alias_move"],
            "exp1_mutated_artifact_requests": fb_reqs["mutated_artifact_under_label"],
            "exp1_stalled_rollout_requests": fb_reqs["stalled_rollout"],
            "exp2_registry_api_caught_count": s2["registry_api_caught_count"],
            "exp2_in_process_caught_count": s2["in_process_attestation_caught_count"],
            "exp3_stalled_spec_window_steps": s3["stalled_rollout_spec_window_steps"],
            "exp3_stalled_first_detected_step": s3["stalled_first_detected_step"],
            "exp3_slow_rollout_spec_window_steps": s3["slow_rollout_spec_window_steps"],
            "exp3_slow_rollout_false_alarms": s3["slow_rollout_false_alarms_at_step_17"],
            "exp4_knee_interval_steps": k4["recommended_audit_interval_steps"],
            "exp4_knee_audit_calls": k4["audit_calls_made"],
            "exp4_knee_wrong_requests": k4["wrong_weights_requests"],
            "exp4_knee_reduction_pct": f"{k4['exposure_reduction_pct']:.2f}%",
            "exp4_interval_1_calls": s4_rows[1]["audit_calls_made"],
            "exp4_interval_1_wrong_requests": s4_rows[1]["wrong_weights_requests"],
            "exp4_interval_1_reduction_pct": f"{s4_rows[1]['exposure_reduction_pct']:.2f}%",
            "exp5_canary_50_wrong_requests": s5_rows["0.50"]["wrong_weights_requests"],
            "exp5_canary_50_capacity_cost_pct": f"{s5_rows['0.50']['rollout_window_capacity_cost_pct']:.2f}%",
            "exp5_canary_100_wrong_requests": s5_rows["1.00"]["wrong_weights_requests"],
            "exp5_canary_100_capacity_cost_pct": f"{s5_rows['1.00']['rollout_window_capacity_cost_pct']:.2f}%",
        },
        "experiments": all_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write JSON files without printing tables to stdout.",
    )
    args = parser.parse_args()

    experiments_dir = REPO_ROOT / "docs" / "experiments"
    metrics_path = REPO_ROOT / "docs" / "metrics.json"

    all_results = run_all_experiments(output_dir=experiments_dir, seed=20261117)
    consolidated = build_consolidated_metrics(all_results)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(consolidated, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(format_experiment_tables(all_results))
        print(f"\nWrote 5 experiment files to {experiments_dir} and {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
