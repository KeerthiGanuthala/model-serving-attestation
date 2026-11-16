"""Property-named test suite for model-serving-attestation (M1 through M5 + Pinned Bug)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from attestation import (
    AttestRecord,
    AttestationAuditor,
    FaultType,
    FleetRolloutSimulator,
    InProcessLoader,
    MinimalRegistry,
    MlflowRegistryAdapter,
    ReplicaClassification,
    ReplicaLifecycle,
    RolloutFaultPlan,
    RolloutSpec,
    compute_bytes_sha256,
    demonstrate_toctou_gap,
    expected_mix_window_steps,
    generate_seeded_weights,
    mutate_single_byte,
    run_indirection_demonstrations,
    write_weights_file,
)
from attestation.experiments import (
    format_experiment_tables,
    run_all_experiments,
    run_experiment_1_green_dashboard,
    run_experiment_2_label_vs_bytes,
    run_experiment_3_rollout_mix_vs_stuck,
    run_experiment_4_polling_cost_vs_exposure,
    run_experiment_5_canary_attestation,
)


class TestM1WeightsAndInProcessHashing(unittest.TestCase):
    """M1 property tests: seeded weights, loader, and TOCTOU verification."""

    def test_seeded_weights_generation_is_deterministic_and_respects_requested_byte_size(self) -> None:
        w_a = generate_seeded_weights(seed=101, size_bytes=4096)
        w_b = generate_seeded_weights(seed=101, size_bytes=4096)
        w_c = generate_seeded_weights(seed=102, size_bytes=4096)
        self.assertEqual(len(w_a), 4096)
        self.assertEqual(w_a, w_b)
        self.assertNotEqual(w_a, w_c)
        with self.assertRaises(ValueError):
            generate_seeded_weights(seed=1, size_bytes=0)

    def test_in_process_loader_preserves_sha256_identity_and_detects_single_byte_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            raw = generate_seeded_weights(seed=7, size_bytes=1024)
            write_weights_file(path, raw)
            loader = InProcessLoader()
            loaded = loader.load(path, claimed_version="v1", load_step=3)
            self.assertTrue(loaded.verify_in_memory_hash())
            self.assertEqual(loaded.loaded_sha256, compute_bytes_sha256(raw))

            mutated = mutate_single_byte(raw, offset=512)
            write_weights_file(path, mutated)
            loaded_mutated = loader.load(path, claimed_version="v1", load_step=4)
            self.assertNotEqual(loaded.loaded_sha256, loaded_mutated.loaded_sha256)

            with self.assertRaises(ValueError):
                mutate_single_byte(b"")

    def test_in_process_load_time_hashing_catches_toctou_file_mutation_that_disk_precheck_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toctou.bin"
            orig = generate_seeded_weights(seed=11, size_bytes=512)
            mut = mutate_single_byte(orig, offset=9)
            res = demonstrate_toctou_gap(path, orig, mut, claimed_version="v1", load_step=2)
            self.assertFalse(res.disk_check_caught_mutation)
            self.assertTrue(res.in_process_attestation_caught_mutation)
            self.assertNotEqual(res.pre_load_disk_sha256, res.in_process_loaded_sha256)
            self.assertEqual(res.in_process_loaded_sha256, res.post_mutation_disk_sha256)

            res_hook = demonstrate_toctou_gap(
                path,
                orig,
                mut,
                claimed_version="v1",
                load_step=3,
                between_check_and_use_hook=lambda p: write_weights_file(p, mut),
            )
            self.assertTrue(res_hook.in_process_attestation_caught_mutation)


class TestM2SharedRegistrySuite(unittest.TestCase):
    """M2 property tests: shared behavioral suite across MinimalRegistry and MlflowRegistryAdapter."""

    def _run_shared_registry_assertions(self, reg_factory) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = reg_factory(root / "store")
            self.assertEqual(reg.list_versions("nonexistent"), [])
            f1 = write_weights_file(root / "v1.bin", generate_seeded_weights(1, 512))
            f2 = write_weights_file(root / "v2.bin", generate_seeded_weights(2, 512))

            r1 = reg.register_version("recommender", f1, step=1, version_label="v1")
            r2 = reg.register_version("recommender", f2, step=2)
            self.assertEqual(r1.version, "v1")
            self.assertEqual(r2.version, "v2")
            self.assertEqual(len(reg.list_versions("recommender")), 2)

            t1 = reg.set_alias("recommender", "production", "v1", step=3)
            self.assertIsNone(t1.from_version)
            self.assertEqual(t1.to_version, "v1")
            self.assertEqual(reg.resolve_alias("recommender", "production").version, "v1")

            t2 = reg.set_alias("recommender", "production", "v2", step=5)
            self.assertEqual(t2.from_version, "v1")
            self.assertEqual(t2.to_version, "v2")
            self.assertEqual(reg.resolve_alias("recommender", "production").version, "v2")
            self.assertEqual(len(reg.transition_log("recommender")), 2)
            self.assertEqual(len(reg.transition_log()), 2)

            with self.assertRaises(KeyError):
                reg.get_version("recommender", "v99")
            with self.assertRaises(KeyError):
                reg.set_alias("recommender", "production", "v99", step=6)
            with self.assertRaises(KeyError):
                reg.resolve_alias("recommender", "nonexistent_alias")

            demos = run_indirection_demonstrations(reg, root / "demos", seed=55)
            self.assertEqual(len(demos), 2)
            for demo in demos:
                self.assertFalse(demo.registry_api_detects_discrepancy)
                self.assertTrue(demo.attestation_detects_discrepancy)

    def test_minimal_registry_passes_shared_version_alias_and_indirection_suite(self) -> None:
        self._run_shared_registry_assertions(MinimalRegistry)

    def test_mlflow_registry_adapter_passes_shared_version_alias_and_indirection_suite(self) -> None:
        self._run_shared_registry_assertions(MlflowRegistryAdapter)


class TestM3FleetAndRolloutStateMachine(unittest.TestCase):
    """M3 property tests: deterministic rollout waves, spec validation, and pinned bug regression."""

    def test_expected_mix_window_matches_enumerated_small_cases_and_rejects_invalid_specs(self) -> None:
        cases = [
            (RolloutSpec(fleet_size=4, max_surge=1, max_unavailable=0, readiness_delay_steps=1), 8),
            (RolloutSpec(fleet_size=4, max_surge=1, max_unavailable=1, readiness_delay_steps=1), 4),
            (RolloutSpec(fleet_size=6, max_surge=2, max_unavailable=1, readiness_delay_steps=1), 4),
            (RolloutSpec(fleet_size=8, max_surge=2, max_unavailable=0, readiness_delay_steps=1), 8),
            (RolloutSpec(fleet_size=8, max_surge=1, max_unavailable=0, readiness_delay_steps=2), 24),
        ]
        for spec, expected in cases:
            self.assertEqual(expected_mix_window_steps(spec), expected)

        with self.assertRaises(ValueError):
            RolloutSpec(fleet_size=0, max_surge=1, max_unavailable=0)
        with self.assertRaises(ValueError):
            RolloutSpec(fleet_size=4, max_surge=0, max_unavailable=0)
        with self.assertRaises(ValueError):
            RolloutSpec(fleet_size=4, max_surge=-1, max_unavailable=1)
        with self.assertRaises(ValueError):
            RolloutSpec(fleet_size=4, max_surge=1, max_unavailable=0, readiness_delay_steps=-1)

    def test_pinned_bug_wave_ceiling_arithmetic_rounds_up_partial_final_wave(self) -> None:
        """Pinned bug story:
        In our initial prototype of `expected_mix_window_steps`, wave count was computed with
        integer floor division `fleet_size // (max_surge + max_unavailable)` instead of
        `math.ceil(fleet_size / (max_surge + max_unavailable))`.
        When `fleet_size = 8`, `max_surge = 2`, `max_unavailable = 1` (wave capacity = 3),
        waves 1 and 2 replace 6 replicas (4 steps), leaving 2 replicas for a 3rd partial wave
        (steps 5 and 6). Floor division predicted a 4-step window and caused the auditor to
        raise false-positive `STALE` alarms on steps 5 and 6 of every honest `(8, 2, 1)` rollout!
        """
        spec = RolloutSpec(fleet_size=8, max_surge=2, max_unavailable=1, readiness_delay_steps=1)
        buggy_floor_window = (spec.fleet_size // (spec.max_surge + spec.max_unavailable)) * (
            spec.readiness_delay_steps + 1
        )
        self.assertEqual(buggy_floor_window, 4)
        self.assertEqual(expected_mix_window_steps(spec), 6)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = MinimalRegistry(root / "reg")
            f1 = write_weights_file(root / "v1.bin", generate_seeded_weights(1, 512))
            f2 = write_weights_file(root / "v2.bin", generate_seeded_weights(2, 512))
            reg.register_version("m", f1, step=0, version_label="v1")
            reg.register_version("m", f2, step=0, version_label="v2")
            reg.set_alias("m", "production", "v1", step=0)

            sim = FleetRolloutSimulator(spec, reg, "m", root / "sim", seed=42)
            sim.bootstrap_initial_fleet("production")
            auditor = AttestationAuditor(spec)
            sim.start_rollout("v2", alias="production")

            for _ in range(8):
                sim.step()
                report = auditor.audit_fleet(
                    records=sim.attest_serving_fleet(),
                    current_step=sim.current_step,
                    intended_version="v2",
                    intended_sha256=sim.target_canonical_sha256 or "",
                    previous_valid_hashes={sim.previous_canonical_sha256 or ""},
                    rollout_start_step=sim.rollout_start_step,
                )
                self.assertFalse(
                    report.has_anomalies,
                    f"False alarm raised at step {sim.current_step}: {report.verdicts}",
                )


class TestM4FaultInjectionAndAuditorClassification(unittest.TestCase):
    """M4 property tests: every injected fault is detected and classified accurately."""

    def test_auditor_classifies_all_four_fault_types_and_unknown_records_with_zero_false_positives(self) -> None:
        spec = RolloutSpec(fleet_size=4, max_surge=2, max_unavailable=0, readiness_delay_steps=1)
        auditor_direct = AttestationAuditor(spec)

        # Verify UNKNOWN classification when loaded_sha256 is None
        unknown_verdict = auditor_direct.classify_replica(
            record=AttestRecord(
                replica_id="rep-missing",
                lifecycle=ReplicaLifecycle.SERVING,
                claimed_version="v2",
                loaded_sha256=None,
                load_step=None,
                readiness_green=True,
            ),
            current_step=5,
            intended_version="v2",
            intended_sha256="abc123",
            previous_valid_hashes={"old123"},
            rollout_start_step=1,
        )
        self.assertEqual(unknown_verdict.classification, ReplicaClassification.UNKNOWN)

        # Verify STALE classification when rollout_start_step is None and old hash is served
        stale_no_rollout = auditor_direct.classify_replica(
            record=AttestRecord(
                replica_id="rep-old",
                lifecycle=ReplicaLifecycle.SERVING,
                claimed_version="v1",
                loaded_sha256="old123",
                load_step=1,
                readiness_green=True,
            ),
            current_step=5,
            intended_version="v2",
            intended_sha256="abc123",
            previous_valid_hashes={"old123"},
            rollout_start_step=None,
        )
        self.assertEqual(stale_no_rollout.classification, ReplicaClassification.STALE)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = MinimalRegistry(root / "reg")
            f1 = write_weights_file(root / "v1.bin", generate_seeded_weights(10, 512))
            f2 = write_weights_file(root / "v2.bin", generate_seeded_weights(20, 512))
            reg.register_version("m", f1, step=0, version_label="v1")
            reg.register_version("m", f2, step=0, version_label="v2")

            for fault, expected_class in [
                (FaultType.STALE_NO_RELOAD, ReplicaClassification.STALE),
                (FaultType.CACHED_PATH_AFTER_ALIAS_MOVE, ReplicaClassification.STALE),
                (FaultType.MUTATED_ARTIFACT_UNDER_LABEL, ReplicaClassification.WRONG_CONTENT),
                (FaultType.STALLED_ROLLOUT, ReplicaClassification.STALE),
            ]:
                reg.set_alias("m", "production", "v1", step=0)
                sim = FleetRolloutSimulator(spec, reg, "m", root / f"sim_{fault.value}", seed=9)
                sim.bootstrap_initial_fleet("production")
                auditor = AttestationAuditor(spec)
                plan = (
                    RolloutFaultPlan(fault_type=fault, stall_after_waves=1)
                    if fault == FaultType.STALLED_ROLLOUT
                    else RolloutFaultPlan(fault_type=fault, affected_replica_indices=(0,))
                )
                sim.start_rollout("v2", alias="production", fault_plan=plan)
                for _ in range(expected_mix_window_steps(spec) + 2):
                    sim.step()
                report = auditor.audit_fleet(
                    records=sim.attest_serving_fleet(),
                    current_step=sim.current_step,
                    intended_version="v2",
                    intended_sha256=sim.target_canonical_sha256 or "",
                    previous_valid_hashes={sim.previous_canonical_sha256 or ""},
                    rollout_start_step=sim.rollout_start_step,
                )
                self.assertTrue(report.has_anomalies)
                self.assertGreater(report.counts_by_classification[expected_class.value], 0)


class TestM5ExperimentsDeterminism(unittest.TestCase):
    """M5 property tests: experiments 1 through 5 produce deterministic figures."""

    def test_experiments_one_through_five_reproduce_identical_numbers_across_runs(self) -> None:
        r1 = run_experiment_1_green_dashboard(seed=20261117)
        r1_repeat = run_experiment_1_green_dashboard(seed=20261117)
        self.assertEqual(r1, r1_repeat)
        self.assertEqual(r1["headline"]["honest_rollout_false_alarms"], 0)
        self.assertEqual(r1["headline"]["wrong_weights_replica_steps"], 600)

        r2 = run_experiment_2_label_vs_bytes(seed=20261117)
        self.assertTrue(r2["backends_agree"])
        self.assertEqual(r2["summary"]["in_process_attestation_caught_count"], 3)

        r3 = run_experiment_3_rollout_mix_vs_stuck(seed=20261117)
        self.assertTrue(r3["all_grid_windows_match_spec"])
        self.assertEqual(
            r3["stalled_vs_slow_comparison"]["stalled_first_detected_step"], 17
        )

        r4 = run_experiment_4_polling_cost_vs_exposure(seed=20261117)
        self.assertEqual(r4["knee_of_curve"]["recommended_audit_interval_steps"], 5)

        r5 = run_experiment_5_canary_attestation(seed=20261117)
        self.assertEqual(r5["canary_routing_table"][-1]["wrong_weights_requests"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            all_res = run_all_experiments(output_dir=Path(tmp), seed=20261117)
            formatted = format_experiment_tables(all_res)
            self.assertIn("Experiment 1: The Green Dashboard", formatted)


if __name__ == "__main__":
    unittest.main()
