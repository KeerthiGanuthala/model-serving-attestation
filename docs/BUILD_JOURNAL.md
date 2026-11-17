# Build Journal: `model-serving-attestation`

This journal records the design decisions, bugs caught, and measurements collected while building `model-serving-attestation` from M1 through M6.

## Milestone 1 (M1): Weights, Loader, and In-Process Hashing

- Implemented `generate_seeded_weights(seed, size_bytes)` using a deterministic SHA-256 counter-mode byte stream so weight files reproduce bit-for-bit across Linux, macOS, and Windows without relying on NumPy RNG state versions.
- Built `InProcessLoader.load()` to read bytes into process memory and immediately compute `hashlib.sha256(raw_bytes).hexdigest()` on the in-memory buffer.
- Wrote `demonstrate_toctou_gap()` and [ADR 0002](adr/0002-hash-at-load-time-vs-hash-on-disk.md) showing how mutating an artifact on disk after a pre-load file check causes disk-level verification to report clean while the process loads mutated bytes.

## Milestone 2 (M2): Minimal Registry and MLflow File-Store Adapter

- Defined `RegistryProtocol` and implemented `MinimalRegistry` alongside `MlflowRegistryAdapter`.
- Designed `MlflowRegistryAdapter` to use `mlflow.tracking.MlflowClient(tracking_uri="file://...")` when `mlflow` is installed in the environment, and to read/write the exact MLflow local file-store directory layout (`models/<name>/meta.yaml`, `models/<name>/version-<N>/meta.yaml`, `models/<name>/aliases/<alias>`) when running in an offline stdlib-only container.
- Verified that both backends pass the identical test suite (`TestM2SharedRegistrySuite`) and demonstrate both directions of registry indirection failure.

## Milestone 3 (M3): Fleet State Machine and the Pinned Wave-Ceiling Bug

- Implemented `FleetRolloutSimulator` with replica lifecycles (`starting`, `ready`, `serving`, `terminating`) and wave-based `RollingUpdate` scheduling (`maxSurge`, `maxUnavailable`, `readiness_delay_steps`).
- **Pinned Bug Story (`test_pinned_bug_wave_ceiling_arithmetic_rounds_up_partial_final_wave`):**
  In the first draft of `expected_mix_window_steps(spec)`, I computed the number of rollout waves using integer floor division:
  `num_waves = spec.fleet_size // (spec.max_surge + spec.max_unavailable)`.
  When testing `(fleet_size=8, max_surge=2, max_unavailable=1, readiness_delay_steps=1)`, each wave replaces `2 + 1 = 3` replicas in `2` steps. Waves 1 and 2 replace 6 replicas in 4 steps, leaving 2 replicas for a 3rd partial wave that finishes at step 6. Floor division computed `8 // 3 = 2` waves (`4` steps), causing the auditor to falsely classify the last 2 replicas as `STALE` on steps 5 and 6 of an honest rollout! Changing the formula to `math.ceil(spec.fleet_size / (spec.max_surge + spec.max_unavailable))` fixed the bug, and I pinned the regression in `TestM3FleetAndRolloutStateMachine`.

## Milestone 4 (M4): Fault Injection and the Attestation Auditor

- Added four deterministic fault modes: `stale_no_reload`, `cached_path_after_alias_move`, `mutated_artifact_under_label`, and `stalled_rollout`.
- Built `AttestationAuditor.classify_replica()` to classify every replica into `correct`, `expected_mix`, `stale`, `wrong_content`, or `unknown`.
- Crucial refinement: if a replica already claims `intended_version` (`claimed_version == intended_version`) while its `loaded_sha256` belongs to a previous version, the auditor classifies it as `STALE` immediately on step 1 rather than waiting for the rollout window to finish.

## Milestone 5 (M5) & Stretch: Experiments 1 through 5

- Ran the 20-rollout benchmark (`seed=20261117`): measured `600` wrong-weight replica-steps (`6,000` logical requests out of `33,440`, or `17.94%`) under a 100% green readiness dashboard.
- Discovered an interesting non-monotonic step effect in Experiment 4: because the 8-replica (`maxSurge=2, maxUnavailable=0`) rollout window is `8` steps long, polling at `audit_interval=4` checks at step `8` (when `rollout_elapsed == 8` is still valid) and does not catch a stalled rollout until step `12`. Polling at `audit_interval=5` checks at step `10` (`rollout_elapsed == 10 > 8`), catching the stalled rollout two steps earlier with fewer total calls (`672` calls vs `800` calls, reducing wrong requests to `1,100` vs `1,360`).

## Milestone 6 (M6): Verification Machinery, ADRs, and Packaging

- Built `tools/collect_metrics.py`, `tools/check_numbers.py` (with `--self-test` proving that mutating a metric fails the build), and `tools/run_tests_with_coverage.py` (achieving `97.8%` line coverage across `src/attestation/`).
- Authored all 5 ADRs, `Dockerfile`, `Makefile`, and `.github/workflows/ci.yml`.
