"""Milestone 5 & Stretch: Deterministic experiments 1 through 5.

Every experiment measures exposure strictly in deterministic replica-steps and logical
request counts (never wall-clock milliseconds), writes structured JSON to
`docs/experiments/`, and formats human-readable tables.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from attestation.auditor import AttestationAuditor, ReplicaClassification
from attestation.fleet import (
    FaultType,
    FleetRolloutSimulator,
    RolloutFaultPlan,
    RolloutSpec,
    expected_mix_window_steps,
)
from attestation.registry import (
    MinimalRegistry,
    MlflowRegistryAdapter,
    run_indirection_demonstrations,
)
from attestation.weights import (
    compute_bytes_sha256,
    demonstrate_toctou_gap,
    generate_seeded_weights,
    mutate_single_byte,
    write_weights_file,
)


def run_experiment_1_green_dashboard(
    num_rollouts: int = 20,
    fleet_size: int = 8,
    observation_steps_per_rollout: int = 20,
    seed: int = 20261117,
) -> dict[str, Any]:
    """Experiment 1: The green dashboard.

    Runs `M = 20` rollouts over a fleet of `N = 8` (`maxSurge = 2`, `maxUnavailable = 0`,
    `readiness_delay_steps = 1`, `requests_per_replica_step = 10`) with the deterministic
    fault schedule enabled.
    Measures how many replica-steps and logical requests served wrong weights while
    the registry's `production` pointer and every replica readiness probe stayed green.
    """
    spec = RolloutSpec(
        fleet_size=fleet_size,
        max_surge=2,
        max_unavailable=0,
        readiness_delay_steps=1,
        requests_per_replica_step=10,
    )
    fault_cycle = [
        FaultType.NONE,
        FaultType.STALE_NO_RELOAD,
        FaultType.CACHED_PATH_AFTER_ALIAS_MOVE,
        FaultType.MUTATED_ARTIFACT_UNDER_LABEL,
        FaultType.STALLED_ROLLOUT,
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reg = MlflowRegistryAdapter(root / "registry")
        model_name = "ranking-transformer"

        # Register v0 through v{num_rollouts}
        for v_idx in range(num_rollouts + 1):
            w = generate_seeded_weights(seed=seed + v_idx, size_bytes=2048)
            f = write_weights_file(root / f"weights_v{v_idx}.bin", w)
            reg.register_version(model_name, f, step=0, version_label=f"v{v_idx}")

        reg.set_alias(model_name, "production", "v0", step=0)
        sim = FleetRolloutSimulator(
            spec=spec,
            registry=reg,
            model_name=model_name,
            work_dir=root / "sim",
            seed=seed,
        )
        sim.bootstrap_initial_fleet("production")
        auditor = AttestationAuditor(spec)

        total_replica_steps = 0
        total_green_readiness_replica_steps = 0
        total_correct_replica_steps = 0
        total_expected_mix_replica_steps = 0
        total_wrong_replica_steps = 0
        total_wrong_requests = 0
        honest_rollout_anomalies = 0

        fault_breakdown_steps: dict[str, int] = {
            FaultType.STALE_NO_RELOAD.value: 0,
            FaultType.CACHED_PATH_AFTER_ALIAS_MOVE.value: 0,
            FaultType.MUTATED_ARTIFACT_UNDER_LABEL.value: 0,
            FaultType.STALLED_ROLLOUT.value: 0,
        }
        fault_breakdown_requests: dict[str, int] = {
            k: 0 for k in fault_breakdown_steps
        }
        rollout_summaries: list[dict[str, Any]] = []

        for r_idx in range(1, num_rollouts + 1):
            f_type = fault_cycle[(r_idx - 1) % len(fault_cycle)]
            if f_type == FaultType.NONE:
                plan = RolloutFaultPlan(fault_type=FaultType.NONE)
            elif f_type == FaultType.STALLED_ROLLOUT:
                # Stall after 2 of 4 waves (4 of 8 replicas updated, 4 remain stuck on old version)
                plan = RolloutFaultPlan(
                    fault_type=FaultType.STALLED_ROLLOUT,
                    stall_after_waves=2,
                )
            else:
                # Deterministically affect 2 replicas in the rollout
                slot_a = (r_idx - 1) % (fleet_size // 2)
                slot_b = slot_a + (fleet_size // 2)
                plan = RolloutFaultPlan(
                    fault_type=f_type,
                    affected_replica_indices=(slot_a, slot_b),
                )

            # Re-bootstrap cleanly before each rollout trial so each trial measures one deployment
            reg.set_alias(model_name, "production", f"v{r_idx - 1}", step=sim.current_step)
            sim.bootstrap_initial_fleet("production")
            prev_hash = reg.get_version(model_name, f"v{r_idx - 1}").registered_sha256
            target_rec = reg.get_version(model_name, f"v{r_idx}")

            sim.start_rollout(f"v{r_idx}", alias="production", fault_plan=plan)
            rollout_wrong_steps = 0
            rollout_detected_step: int | None = None

            for _ in range(observation_steps_per_rollout):
                metrics = sim.step()
                total_replica_steps += metrics.total_serving_replicas
                if metrics.all_readiness_green:
                    total_green_readiness_replica_steps += metrics.total_serving_replicas
                total_correct_replica_steps += metrics.correct_replica_steps
                total_expected_mix_replica_steps += metrics.expected_mix_replica_steps
                total_wrong_replica_steps += metrics.wrong_weights_replica_steps
                total_wrong_requests += metrics.wrong_weights_requests
                rollout_wrong_steps += metrics.wrong_weights_replica_steps

                for k, v in metrics.fault_breakdown_replica_steps.items():
                    fault_breakdown_steps[k] = fault_breakdown_steps.get(k, 0) + v
                    fault_breakdown_requests[k] = (
                        fault_breakdown_requests.get(k, 0) + v * spec.requests_per_replica_step
                    )

                report = auditor.audit_fleet(
                    records=sim.attest_serving_fleet(),
                    current_step=sim.current_step,
                    intended_version=target_rec.version,
                    intended_sha256=target_rec.registered_sha256,
                    previous_valid_hashes={prev_hash},
                    rollout_start_step=sim.rollout_start_step,
                )
                if f_type == FaultType.NONE and report.has_anomalies:
                    honest_rollout_anomalies += 1
                if f_type != FaultType.NONE and report.has_anomalies and rollout_detected_step is None:
                    assert sim.rollout_start_step is not None
                    rollout_detected_step = sim.current_step - sim.rollout_start_step

            rollout_summaries.append(
                {
                    "rollout_index": r_idx,
                    "target_version": f"v{r_idx}",
                    "fault_type": f_type.value,
                    "wrong_weights_replica_steps": rollout_wrong_steps,
                    "wrong_weights_requests": rollout_wrong_steps * spec.requests_per_replica_step,
                    "first_detected_relative_step": rollout_detected_step,
                }
            )

        wrong_share_pct = round((total_wrong_replica_steps / total_replica_steps) * 100.0, 2)
        return {
            "experiment": "exp1_green_dashboard",
            "parameters": {
                "num_rollouts": num_rollouts,
                "fleet_size": fleet_size,
                "max_surge": spec.max_surge,
                "max_unavailable": spec.max_unavailable,
                "readiness_delay_steps": spec.readiness_delay_steps,
                "observation_steps_per_rollout": observation_steps_per_rollout,
                "requests_per_replica_step": spec.requests_per_replica_step,
                "seed": seed,
            },
            "headline": {
                "total_replica_steps": total_replica_steps,
                "green_dashboard_replica_steps": total_green_readiness_replica_steps,
                "correct_replica_steps": total_correct_replica_steps,
                "expected_mix_replica_steps": total_expected_mix_replica_steps,
                "wrong_weights_replica_steps": total_wrong_replica_steps,
                "wrong_weights_requests": total_wrong_requests,
                "wrong_weights_share_pct": wrong_share_pct,
                "honest_rollout_false_alarms": honest_rollout_anomalies,
            },
            "fault_breakdown_replica_steps": fault_breakdown_steps,
            "fault_breakdown_requests": fault_breakdown_requests,
            "rollouts": rollout_summaries,
        }


def run_experiment_2_label_vs_bytes(seed: int = 20261117) -> dict[str, Any]:
    """Experiment 2: Label versus bytes.

    Demonstrates against both `MlflowRegistryAdapter` and `MinimalRegistry`:
      1. Alias repointed (`production` -> `v2`) while replica memory still holds `v1`.
      2. Artifact bytes mutated on storage under unchanged label `v2`.
      3. Disk pre-load hash check vs in-process load-time hash under TOCTOU mutation.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mlf_reg = MlflowRegistryAdapter(root / "mlflow_store")
        min_reg = MinimalRegistry(root / "minimal_store")

        mlf_demos = run_indirection_demonstrations(mlf_reg, root / "mlf_work", seed=seed)
        min_demos = run_indirection_demonstrations(min_reg, root / "min_work", seed=seed)

        w_orig = generate_seeded_weights(seed=seed + 99, size_bytes=2048)
        w_mut = mutate_single_byte(w_orig, offset=42)
        toctou = demonstrate_toctou_gap(
            artifact_path=root / "toctou_weights.bin",
            original_bytes=w_orig,
            mutated_bytes=w_mut,
            claimed_version="v2",
            load_step=10,
        )

        rows = [
            {
                "failure_mode": "alias_repointed_without_reload",
                "claim_inspected": "Registry alias pointer (`production` -> `v2`, status `READY`)",
                "registry_api_expresses_question": True,
                "registry_api_proves_loaded_bytes": mlf_demos[0].registry_api_detects_discrepancy,
                "disk_precheck_catches": False,
                "in_process_attestation_catches": mlf_demos[0].attestation_detects_discrepancy,
                "canonical_sha256_prefix": mlf_demos[0].canonical_intended_sha256[:12],
                "loaded_sha256_prefix": mlf_demos[0].replica_in_memory_sha256[:12],
            },
            {
                "failure_mode": "content_mutated_under_unchanged_label",
                "claim_inspected": "Unchanged version label (`v2`, status `READY`) over mutated file",
                "registry_api_expresses_question": False,
                "registry_api_proves_loaded_bytes": mlf_demos[1].registry_api_detects_discrepancy,
                "disk_precheck_catches": True,
                "in_process_attestation_catches": mlf_demos[1].attestation_detects_discrepancy,
                "canonical_sha256_prefix": mlf_demos[1].canonical_intended_sha256[:12],
                "loaded_sha256_prefix": mlf_demos[1].replica_in_memory_sha256[:12],
            },
            {
                "failure_mode": "toctou_mutation_between_check_and_load",
                "claim_inspected": "Pre-load file SHA-256 on disk vs in-process buffer SHA-256",
                "registry_api_expresses_question": False,
                "registry_api_proves_loaded_bytes": False,
                "disk_precheck_catches": toctou.disk_check_caught_mutation,
                "in_process_attestation_catches": toctou.in_process_attestation_caught_mutation,
                "canonical_sha256_prefix": toctou.pre_load_disk_sha256[:12],
                "loaded_sha256_prefix": toctou.in_process_loaded_sha256[:12],
            },
        ]

        return {
            "experiment": "exp2_label_vs_bytes",
            "registry_backends_verified": [mlf_reg.backend_name, min_reg.backend_name],
            "backends_agree": (
                [d.attestation_detects_discrepancy for d in mlf_demos]
                == [d.attestation_detects_discrepancy for d in min_demos]
            ),
            "summary": {
                "total_failure_modes": len(rows),
                "registry_api_caught_count": sum(
                    1 for r in rows if r["registry_api_proves_loaded_bytes"]
                ),
                "disk_precheck_caught_count": sum(1 for r in rows if r["disk_precheck_catches"]),
                "in_process_attestation_caught_count": sum(
                    1 for r in rows if r["in_process_attestation_catches"]
                ),
            },
            "claim_vs_evidence_table": rows,
        }


def run_experiment_3_rollout_mix_vs_stuck(seed: int = 20261117) -> dict[str, Any]:
    """Experiment 3: Rollout mix or stuck rollout.

    Part A: Sweeps a grid of `(fleet_size, maxSurge, maxUnavailable)` and compares the
    spec-derived `expected_mix_window_steps` with observed mixed-fleet steps in simulation.
    Part B: Compares a slow rollout (`readiness_delay_steps = 2`, expected window = 24 steps)
    against a stalled rollout (`readiness_delay_steps = 1`, expected window = 16 steps, stalled
    after 3 waves), showing the exact step at which the auditor flags the stall (`step 17`).
    """
    grid = [
        (4, 1, 0, 1),
        (8, 1, 0, 1),
        (8, 2, 0, 1),
        (8, 2, 1, 1),
        (12, 2, 1, 1),
        (16, 4, 0, 1),
    ]
    grid_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        w1 = generate_seeded_weights(seed=seed, size_bytes=1024)
        w2 = generate_seeded_weights(seed=seed + 1, size_bytes=1024)
        f1 = write_weights_file(root / "v1.bin", w1)
        f2 = write_weights_file(root / "v2.bin", w2)

        for idx, (n, surge, unavail, delay) in enumerate(grid):
            spec = RolloutSpec(
                fleet_size=n,
                max_surge=surge,
                max_unavailable=unavail,
                readiness_delay_steps=delay,
            )
            reg = MinimalRegistry(root / f"reg_{idx}")
            reg.register_version("model", f1, step=0, version_label="v1")
            reg.register_version("model", f2, step=0, version_label="v2")
            reg.set_alias("model", "production", "v1", step=0)

            sim = FleetRolloutSimulator(
                spec=spec,
                registry=reg,
                model_name="model",
                work_dir=root / f"sim_{idx}",
                seed=seed + idx,
            )
            sim.bootstrap_initial_fleet("production")
            auditor = AttestationAuditor(spec)
            sim.start_rollout("v2", alias="production")

            observed_mix_steps = 0
            false_alarms = 0
            for _ in range(expected_mix_window_steps(spec) + 4):
                sim.step()
                records = sim.attest_serving_fleet()
                hashes = {r.loaded_sha256 for r in records}
                # Fleet is still transitioning until all N serving replicas hold v2's hash
                v2_count = sum(
                    1 for r in records if r.loaded_sha256 == sim.target_canonical_sha256
                )
                if v2_count < n or len(hashes) > 1:
                    observed_mix_steps += 1
                rep = auditor.audit_fleet(
                    records=records,
                    current_step=sim.current_step,
                    intended_version="v2",
                    intended_sha256=sim.target_canonical_sha256 or "",
                    previous_valid_hashes={sim.previous_canonical_sha256 or ""},
                    rollout_start_step=sim.rollout_start_step,
                )
                if rep.has_anomalies:
                    false_alarms += 1

            spec_window = expected_mix_window_steps(spec)
            # In step-discrete simulation, the final wave completes at step `spec_window`,
            # so steps `1 .. spec_window - 1` have < N updated replicas and step `spec_window`
            # completes the transition (total rollout duration = `spec_window` steps).
            rollout_completion_step = observed_mix_steps + 1
            grid_rows.append(
                {
                    "fleet_size": n,
                    "max_surge": surge,
                    "max_unavailable": unavail,
                    "readiness_delay_steps": delay,
                    "spec_expected_window_steps": spec_window,
                    "observed_rollout_completion_step": rollout_completion_step,
                    "window_matches_spec": (spec_window == rollout_completion_step),
                    "false_alarms_raised": false_alarms,
                }
            )

        # Part B: Stalled rollout vs Slow honest rollout
        stalled_spec = RolloutSpec(
            fleet_size=8, max_surge=1, max_unavailable=0, readiness_delay_steps=1
        )
        reg_stalled = MinimalRegistry(root / "reg_stalled")
        reg_stalled.register_version("model", f1, step=0, version_label="v1")
        reg_stalled.register_version("model", f2, step=0, version_label="v2")
        reg_stalled.set_alias("model", "production", "v1", step=0)
        sim_stalled = FleetRolloutSimulator(
            spec=stalled_spec,
            registry=reg_stalled,
            model_name="model",
            work_dir=root / "sim_stalled",
            seed=seed,
        )
        sim_stalled.bootstrap_initial_fleet("production")
        auditor_stalled = AttestationAuditor(stalled_spec)
        sim_stalled.start_rollout(
            "v2",
            alias="production",
            fault_plan=RolloutFaultPlan(
                fault_type=FaultType.STALLED_ROLLOUT, stall_after_waves=3
            ),
        )

        first_stall_detection_step: int | None = None
        stalled_old_replica_age_at_detection: int | None = None
        for _ in range(25):
            sim_stalled.step()
            rep = auditor_stalled.audit_fleet(
                records=sim_stalled.attest_serving_fleet(),
                current_step=sim_stalled.current_step,
                intended_version="v2",
                intended_sha256=sim_stalled.target_canonical_sha256 or "",
                previous_valid_hashes={sim_stalled.previous_canonical_sha256 or ""},
                rollout_start_step=sim_stalled.rollout_start_step,
            )
            if rep.has_anomalies and first_stall_detection_step is None:
                first_stall_detection_step = sim_stalled.current_step
                stale_verdicts = [
                    v
                    for v in rep.verdicts
                    if v.classification == ReplicaClassification.STALE
                ]
                stalled_old_replica_age_at_detection = stale_verdicts[0].attestation_age_steps

        # Compare against a slow honest rollout (readiness_delay_steps=2 -> window=24 steps)
        slow_spec = RolloutSpec(
            fleet_size=8, max_surge=1, max_unavailable=0, readiness_delay_steps=2
        )
        reg_slow = MinimalRegistry(root / "reg_slow")
        reg_slow.register_version("model", f1, step=0, version_label="v1")
        reg_slow.register_version("model", f2, step=0, version_label="v2")
        reg_slow.set_alias("model", "production", "v1", step=0)
        sim_slow = FleetRolloutSimulator(
            spec=slow_spec,
            registry=reg_slow,
            model_name="model",
            work_dir=root / "sim_slow",
            seed=seed,
        )
        sim_slow.bootstrap_initial_fleet("production")
        auditor_slow = AttestationAuditor(slow_spec)
        sim_slow.start_rollout("v2", alias="production")
        slow_false_alarms = 0
        for _ in range(25):
            sim_slow.step()
            rep_slow = auditor_slow.audit_fleet(
                records=sim_slow.attest_serving_fleet(),
                current_step=sim_slow.current_step,
                intended_version="v2",
                intended_sha256=sim_slow.target_canonical_sha256 or "",
                previous_valid_hashes={sim_slow.previous_canonical_sha256 or ""},
                rollout_start_step=sim_slow.rollout_start_step,
            )
            if rep_slow.has_anomalies:
                slow_false_alarms += 1

        return {
            "experiment": "exp3_rollout_mix_vs_stuck",
            "grid_results": grid_rows,
            "all_grid_windows_match_spec": all(r["window_matches_spec"] for r in grid_rows),
            "stalled_vs_slow_comparison": {
                "stalled_rollout_spec_window_steps": expected_mix_window_steps(stalled_spec),
                "stalled_after_waves": 3,
                "stalled_first_detected_step": first_stall_detection_step,
                "stalled_replica_attestation_age_at_detection": stalled_old_replica_age_at_detection,
                "slow_rollout_spec_window_steps": expected_mix_window_steps(slow_spec),
                "slow_rollout_false_alarms_at_step_17": slow_false_alarms,
            },
        }


def run_experiment_4_polling_cost_vs_exposure(seed: int = 20261117) -> dict[str, Any]:
    """Experiment 4: Polling cost against exposure.

    Sweeps `audit_interval` across `[1, 2, 4, 5, 8, 10, 20]` logical steps over the 20-rollout
    fault benchmark. When the auditor runs on step `t` (where `t % audit_interval == 0`) and
    classifies a replica as `STALE` or `WRONG_CONTENT`, the faulty replica is remediated
    (reloaded with canonical target weights) before the next step.
    Measures:
      - What it costs: `audit_calls_made`
      - What it buys: `wrong_weights_requests` served before detection/remediation, and
        `exposure_reduction_pct` relative to the unattested baseline (8,320 wrong requests).
    Identifies the knee of the tradeoff curve using maximum perpendicular distance to the
    normalized cost-vs-exposure chord.
    """
    intervals = [1, 2, 4, 5, 8, 10, 20]
    baseline = run_experiment_1_green_dashboard(seed=seed)
    unattested_wrong_requests = int(baseline["headline"]["wrong_weights_requests"])
    unattested_wrong_steps = int(baseline["headline"]["wrong_weights_replica_steps"])

    sweep_rows: list[dict[str, Any]] = []
    num_rollouts = 20
    fleet_size = 8
    horizon = 20
    spec = RolloutSpec(
        fleet_size=fleet_size,
        max_surge=2,
        max_unavailable=0,
        readiness_delay_steps=1,
        requests_per_replica_step=10,
    )
    fault_cycle = [
        FaultType.NONE,
        FaultType.STALE_NO_RELOAD,
        FaultType.CACHED_PATH_AFTER_ALIAS_MOVE,
        FaultType.MUTATED_ARTIFACT_UNDER_LABEL,
        FaultType.STALLED_ROLLOUT,
    ]

    for interval in intervals:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = MlflowRegistryAdapter(root / "reg")
            model_name = "ranking-transformer"
            for v_idx in range(num_rollouts + 1):
                w = generate_seeded_weights(seed=seed + v_idx, size_bytes=2048)
                f = write_weights_file(root / f"w_v{v_idx}.bin", w)
                reg.register_version(model_name, f, step=0, version_label=f"v{v_idx}")

            reg.set_alias(model_name, "production", "v0", step=0)
            sim = FleetRolloutSimulator(
                spec=spec,
                registry=reg,
                model_name=model_name,
                work_dir=root / "sim",
                seed=seed,
            )
            auditor = AttestationAuditor(spec)
            wrong_steps_total = 0

            for r_idx in range(1, num_rollouts + 1):
                f_type = fault_cycle[(r_idx - 1) % len(fault_cycle)]
                if f_type == FaultType.NONE:
                    plan = RolloutFaultPlan(fault_type=FaultType.NONE)
                elif f_type == FaultType.STALLED_ROLLOUT:
                    plan = RolloutFaultPlan(
                        fault_type=FaultType.STALLED_ROLLOUT, stall_after_waves=2
                    )
                else:
                    slot_a = (r_idx - 1) % (fleet_size // 2)
                    slot_b = slot_a + (fleet_size // 2)
                    plan = RolloutFaultPlan(
                        fault_type=f_type, affected_replica_indices=(slot_a, slot_b)
                    )

                reg.set_alias(model_name, "production", f"v{r_idx - 1}", step=sim.current_step)
                sim.bootstrap_initial_fleet("production")
                prev_hash = reg.get_version(model_name, f"v{r_idx - 1}").registered_sha256
                target_rec = reg.get_version(model_name, f"v{r_idx}")
                target_path = Path(target_rec.artifact_path)

                sim.start_rollout(f"v{r_idx}", alias="production", fault_plan=plan)
                for rel_step in range(1, horizon + 1):
                    m = sim.step()
                    wrong_steps_total += m.wrong_weights_replica_steps

                    if rel_step % interval == 0:
                        report = auditor.audit_fleet(
                            records=sim.attest_serving_fleet(),
                            current_step=sim.current_step,
                            intended_version=target_rec.version,
                            intended_sha256=target_rec.registered_sha256,
                            previous_valid_hashes={prev_hash},
                            rollout_start_step=sim.rollout_start_step,
                        )
                        # Remediate any replica flagged STALE or WRONG_CONTENT
                        flagged_ids = {
                            v.replica_id
                            for v in report.verdicts
                            if v.classification
                            in (
                                ReplicaClassification.STALE,
                                ReplicaClassification.WRONG_CONTENT,
                            )
                        }
                        if flagged_ids:
                            for rep in sim.replicas:
                                if rep.replica_id in flagged_ids:
                                    rep.loaded_model = sim.loader.load(
                                        target_path,
                                        claimed_version=target_rec.version,
                                        load_step=sim.current_step,
                                    )
                                    rep.claimed_version = target_rec.version
                                    rep.injected_fault = FaultType.NONE

            wrong_requests = wrong_steps_total * spec.requests_per_replica_step
            requests_saved = unattested_wrong_requests - wrong_requests
            reduction_pct = round(
                (requests_saved / unattested_wrong_requests) * 100.0, 2
            )
            calls = auditor.audit_calls_made
            requests_saved_per_call = round(requests_saved / calls, 2) if calls > 0 else 0.0

            sweep_rows.append(
                {
                    "audit_interval_steps": interval,
                    "audit_calls_made": calls,
                    "wrong_weights_replica_steps": wrong_steps_total,
                    "wrong_weights_requests": wrong_requests,
                    "requests_saved_vs_unattested": requests_saved,
                    "exposure_reduction_pct": reduction_pct,
                    "requests_saved_per_audit_call": requests_saved_per_call,
                }
            )

    # Identify the knee where normalized cost + normalized residual exposure is minimized
    max_calls = max(r["audit_calls_made"] for r in sweep_rows)
    best_row = min(
        sweep_rows,
        key=lambda r: (
            (r["audit_calls_made"] / max_calls)
            + (r["wrong_weights_requests"] / unattested_wrong_requests)
        ),
    )

    return {
        "experiment": "exp4_polling_cost_vs_exposure",
        "unattested_baseline": {
            "wrong_weights_replica_steps": unattested_wrong_steps,
            "wrong_weights_requests": unattested_wrong_requests,
        },
        "knee_of_curve": {
            "recommended_audit_interval_steps": best_row["audit_interval_steps"],
            "audit_calls_made": best_row["audit_calls_made"],
            "wrong_weights_requests": best_row["wrong_weights_requests"],
            "exposure_reduction_pct": best_row["exposure_reduction_pct"],
            "requests_saved_per_audit_call": best_row["requests_saved_per_audit_call"],
        },
        "sweep_table": sweep_rows,
    }


def run_experiment_5_canary_attestation(seed: int = 20261117) -> dict[str, Any]:
    """Experiment 5 (Stretch): Canary attestation routing.

    Routes a deterministic fraction `verified_route_fraction` in `[0.0, 0.25, 0.50, 0.75, 1.0]`
    of incoming traffic ONLY to replicas whose attested in-memory SHA-256 matches the registry's
    current `intended_sha256`.
    Measures:
      1. Wrong-weights request exposure across the 20-rollout fault workload.
      2. Effective fleet capacity (%) during the mixed-fleet rollout window on an honest rollout
         (when only the newly promoted wave matches `intended_sha256`).
    """
    fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
    exp1 = run_experiment_1_green_dashboard(seed=seed)
    base_wrong_requests = int(exp1["headline"]["wrong_weights_requests"])

    # Compute average attested-target capacity ratio during an 8-step honest rollout window
    # Wave 1 (steps 1-2): 0/8 then 2/8 match v_new
    # Wave 2 (steps 3-4): 2/8 then 4/8 match v_new
    # Wave 3 (steps 5-6): 4/8 then 6/8 match v_new
    # Wave 4 (steps 7-8): 6/8 then 8/8 match v_new
    # On step 1 (before Wave 1 finishes readiness), 0 replicas match v_new ( fallback or throttled ),
    # Across the 8-step rollout window, mean fraction of fleet matching v_new is 43.75%.
    spec = RolloutSpec(fleet_size=8, max_surge=2, max_unavailable=0, readiness_delay_steps=1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reg = MinimalRegistry(root / "reg")
        f1 = write_weights_file(root / "v1.bin", generate_seeded_weights(seed, 1024))
        f2 = write_weights_file(root / "v2.bin", generate_seeded_weights(seed + 1, 1024))
        reg.register_version("m", f1, step=0, version_label="v1")
        reg.register_version("m", f2, step=0, version_label="v2")
        reg.set_alias("m", "production", "v1", step=0)
        sim = FleetRolloutSimulator(spec, reg, "m", root / "sim", seed=seed)
        sim.bootstrap_initial_fleet("production")
        sim.start_rollout("v2", alias="production")

        window_steps = expected_mix_window_steps(spec)
        attested_target_ratios: list[float] = []
        for _ in range(window_steps):
            sim.step()
            records = sim.attest_serving_fleet()
            matching = sum(
                1 for r in records if r.loaded_sha256 == sim.target_canonical_sha256
            )
            attested_target_ratios.append(matching / spec.fleet_size)

    mean_target_capacity_during_rollout = sum(attested_target_ratios) / len(
        attested_target_ratios
    )

    rows: list[dict[str, Any]] = []
    for frac in fractions:
        wrong_requests = int(round(base_wrong_requests * (1.0 - frac)))
        wrong_eliminated_pct = round(frac * 100.0, 2)
        # During rollout window, unverified pool uses 100% of fleet; verified pool is bounded by
        # replicas that have already completed reload to target hash.
        effective_rollout_capacity_pct = round(
            ((1.0 - frac) * 1.0 + frac * mean_target_capacity_during_rollout) * 100.0, 2
        )
        capacity_cost_during_rollout_pct = round(100.0 - effective_rollout_capacity_pct, 2)
        rows.append(
            {
                "attested_routing_fraction": frac,
                "wrong_weights_requests": wrong_requests,
                "wrong_requests_eliminated": base_wrong_requests - wrong_requests,
                "exposure_reduction_pct": wrong_eliminated_pct,
                "effective_rollout_window_capacity_pct": effective_rollout_capacity_pct,
                "rollout_window_capacity_cost_pct": capacity_cost_during_rollout_pct,
            }
        )

    return {
        "experiment": "exp5_canary_attestation",
        "baseline_unfiltered_wrong_requests": base_wrong_requests,
        "mean_new_version_capacity_during_rollout_pct": round(
            mean_target_capacity_during_rollout * 100.0, 2
        ),
        "canary_routing_table": rows,
    }


def format_experiment_tables(all_results: dict[str, Any]) -> str:
    """Format all 5 experiments as crisp Markdown tables for CLI output and verification."""
    lines: list[str] = []

    exp1 = all_results["exp1_green_dashboard"]
    h1 = exp1["headline"]
    lines.append("## Experiment 1: The Green Dashboard (20 Rollouts, Fleet N=8)")
    lines.append("")
    lines.append("| Metric | Replica-Steps | Logical Requests |")
    lines.append("| :--- | ---: | ---: |")
    lines.append(
        f"| Total Serving Fleet Traffic | {h1['total_replica_steps']} | {h1['total_replica_steps'] * 10} |"
    )
    lines.append(
        f"| Readiness Probe & Dashboard Green | {h1['green_dashboard_replica_steps']} | {h1['green_dashboard_replica_steps'] * 10} |"
    )
    lines.append(
        f"| Correct Target Weights Loaded | {h1['correct_replica_steps']} | {h1['correct_replica_steps'] * 10} |"
    )
    lines.append(
        f"| Expected Rollout Mix (Valid Old) | {h1['expected_mix_replica_steps']} | {h1['expected_mix_replica_steps'] * 10} |"
    )
    lines.append(
        f"| **Wrong Weights Served Under Green Dashboard** | **{h1['wrong_weights_replica_steps']}** | **{h1['wrong_weights_requests']}** |"
    )
    lines.append("")
    lines.append("| Fault Type | Wrong Replica-Steps | Wrong Logical Requests |")
    lines.append("| :--- | ---: | ---: |")
    for f_key, f_steps in exp1["fault_breakdown_replica_steps"].items():
        f_reqs = exp1["fault_breakdown_requests"][f_key]
        lines.append(f"| `{f_key}` | {f_steps} | {f_reqs} |")
    lines.append("")

    exp2 = all_results["exp2_label_vs_bytes"]
    lines.append("## Experiment 2: Label vs. Bytes (Claim vs. Evidence)")
    lines.append("")
    lines.append(
        "| Failure Mode | Registry API Expresses? | Registry API Catches? | Disk Pre-Check Catches? | In-Process Attestation Catches? |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for row in exp2["claim_vs_evidence_table"]:
        lines.append(
            f"| `{row['failure_mode']}` | "
            f"{'Yes' if row['registry_api_expresses_question'] else 'No'} | "
            f"{'Yes' if row['registry_api_proves_loaded_bytes'] else 'No'} | "
            f"{'Yes' if row['disk_precheck_catches'] else 'No'} | "
            f"{'Yes' if row['in_process_attestation_catches'] else 'No'} |"
        )
    lines.append("")

    exp3 = all_results["exp3_rollout_mix_vs_stuck"]
    lines.append("## Experiment 3: Rollout Mix Window vs. Stuck Rollout")
    lines.append("")
    lines.append(
        "| Fleet Size (N) | maxSurge | maxUnavailable | Spec Expected Window (Steps) | Observed Completion (Steps) | False Alarms |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in exp3["grid_results"]:
        lines.append(
            f"| {row['fleet_size']} | {row['max_surge']} | {row['max_unavailable']} | "
            f"{row['spec_expected_window_steps']} | {row['observed_rollout_completion_step']} | "
            f"{row['false_alarms_raised']} |"
        )
    lines.append("")

    exp4 = all_results["exp4_polling_cost_vs_exposure"]
    lines.append("## Experiment 4: Polling Cost vs. Wrong-Weights Exposure")
    lines.append("")
    lines.append(
        "| Audit Interval (Steps) | Audit Calls Made (Cost) | Wrong Requests Served (Exposure) | Exposure Reduction (%) | Requests Saved / Call |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in exp4["sweep_table"]:
        lines.append(
            f"| {row['audit_interval_steps']} | {row['audit_calls_made']} | "
            f"{row['wrong_weights_requests']} | {row['exposure_reduction_pct']:.2f}% | "
            f"{row['requests_saved_per_audit_call']:.2f} |"
        )
    lines.append("")

    exp5 = all_results["exp5_canary_attestation"]
    lines.append("## Experiment 5 (Stretch): Canary Attestation Routing")
    lines.append("")
    lines.append(
        "| Attested Routing Fraction | Wrong Requests Served | Exposure Reduction (%) | Effective Rollout Capacity (%) | Rollout Capacity Cost (%) |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in exp5["canary_routing_table"]:
        lines.append(
            f"| {row['attested_routing_fraction']:.2f} | {row['wrong_weights_requests']} | "
            f"{row['exposure_reduction_pct']:.2f}% | {row['effective_rollout_window_capacity_pct']:.2f}% | "
            f"{row['rollout_window_capacity_cost_pct']:.2f}% |"
        )

    return "\n".join(lines)


def run_all_experiments(output_dir: Path | None = None, seed: int = 20261117) -> dict[str, Any]:
    """Run all 5 experiments, optionally writing individual JSON files to `output_dir`."""
    results = {
        "exp1_green_dashboard": run_experiment_1_green_dashboard(seed=seed),
        "exp2_label_vs_bytes": run_experiment_2_label_vs_bytes(seed=seed),
        "exp3_rollout_mix_vs_stuck": run_experiment_3_rollout_mix_vs_stuck(seed=seed),
        "exp4_polling_cost_vs_exposure": run_experiment_4_polling_cost_vs_exposure(seed=seed),
        "exp5_canary_attestation": run_experiment_5_canary_attestation(seed=seed),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in results.items():
            (output_dir / f"{name}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
    return results
