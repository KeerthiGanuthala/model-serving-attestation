# model-serving-attestation

A registry stage label or alias (`production -> v7`) is a mutable pointer, so it cannot prove which weights a serving replica actually loaded into memory: the label names an intention, and only a cryptographic hash computed inside the serving process at load time names a fact.

In our 20-rollout benchmark over an 8-replica fleet (`maxSurge=2, maxUnavailable=0`), replicas with realistic deployment faults **served wrong weights for 600 of 3,344 replica-steps (6,000 of 33,440 logical requests, or 17.94% of all traffic)** while the registry pointer read `production == target` and every replica readiness probe stayed 100% green.

This repository implements the in-process SHA-256 attestation loop and spec-derived rollout auditor that closes that gap, backed by reproducible experiments and automated number verification.

## Quickstart

Reproduce all five experiments, run the property-named test suite with line coverage enforcement, and verify every figure quoted in this README and the ADRs with zero network access:

```bash
make verify
```

Or run the individual steps directly (works identically on Linux, macOS, and Windows CMD):

```bash
python3 tools/collect_metrics.py
python3 tools/run_tests_with_coverage.py
python3 tools/check_numbers.py --self-test
python3 -m attestation.cli --demo
```

Build and run the self-attesting container image:

```bash
docker build -t model-serving-attestation:latest .
docker run --rm model-serving-attestation:latest
```

## Experiment 1: The Green Dashboard

We run 20 deterministic `RollingUpdate` deployments across an 8-replica fleet (`maxSurge=2, maxUnavailable=0, readiness_delay_steps=1, observation_steps_per_rollout=20, requests_per_replica_step=10`) cycling across five conditions: honest rollouts, `stale_no_reload` (readiness reports green without reloading weights), `cached_path_after_alias_move` (reloads from an un-invalidated local cache path), `mutated_artifact_under_label` (1 byte flipped in storage under an unchanged version label), and `stalled_rollout` (rollout halts after wave 2 of 4).

| Metric | Replica-Steps | Logical Requests |
| :--- | ---: | ---: |
| Total Serving Fleet Traffic | 3,344 | 33,440 |
| Readiness Probe & Dashboard Green | 3,344 | 33,440 |
| Correct Target Weights Loaded | 2,072 | 20,720 |
| Expected Rollout Mix (Valid Old) | 672 | 6,720 |
| **Wrong Weights Served Under Green Dashboard** | **600** | **6,000** |

Breakdown of wrong-weights exposure by fault category (`docs/experiments/exp1_green_dashboard.json`):

| Fault Type | Wrong Replica-Steps | Wrong Logical Requests |
| :--- | ---: | ---: |
| `stale_no_reload` | 136 | 1,360 |
| `cached_path_after_alias_move` | 136 | 1,360 |
| `mutated_artifact_under_label` | 136 | 1,360 |
| `stalled_rollout` | 192 | 1,920 |

Across all 4 honest rollouts, the auditor raises `0` false alarms.

## Experiment 2: Label vs. Bytes (Claim vs. Evidence)

Using both our `MinimalRegistry` and `MlflowRegistryAdapter` (backed by a local `file://` store), we test three indirection failures (`docs/experiments/exp2_label_vs_bytes.json`). The registry API surface can express a question about which version string an alias points to, but it cannot inspect process memory or detect in-place artifact mutations. By contrast, in-process attestation catches 3 of 3 failure modes:

| Failure Mode | Registry API Expresses? | Registry API Catches? | Disk Pre-Check Catches? | In-Process Attestation Catches? |
| :--- | :---: | :---: | :---: | :---: |
| `alias_repointed_without_reload` | Yes | No | No | Yes |
| `content_mutated_under_unchanged_label` | No | No | Yes | Yes |
| `toctou_mutation_between_check_and_load` | No | No | No | Yes |

## Experiment 3: Rollout Mix Window vs. Stuck Rollout

Rather than using a static tolerance constant, the auditor derives the theoretical mixed-fleet rollout window directly from the `RolloutSpec`:

$$\text{expected\_window}(N, S, U, d) = \left\lceil \frac{N}{S + U} \right\rceil \times (d + 1)$$

Across all six grid configurations (`docs/experiments/exp3_rollout_mix_vs_stuck.json`), the spec-derived window matches the observed completion step with zero false alarms:

| Fleet Size (N) | maxSurge | maxUnavailable | Spec Expected Window (Steps) | Observed Completion (Steps) | False Alarms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 0 | 8 | 8 | 0 |
| 8 | 1 | 0 | 16 | 16 | 0 |
| 8 | 2 | 0 | 8 | 8 | 0 |
| 8 | 2 | 1 | 6 | 6 | 0 |
| 12 | 2 | 1 | 8 | 8 | 0 |
| 16 | 4 | 0 | 8 | 8 | 0 |

When an 8-replica rollout (`maxSurge=1, maxUnavailable=0, readiness_delay_steps=1`, expected window = 16 steps) stalls after wave 3, the auditor first flags the stalled rollout at step 17 (when the remaining unreplaced replicas reach an attestation age of 17 steps), while raising `0` false alarms at step 17 on a slow rollout (`readiness_delay_steps=2`, expected window = 24 steps).

## Experiment 4: Polling Cost vs. Wrong-Weights Exposure

We sweep the auditor polling interval from `1` to `20` steps (`docs/experiments/exp4_polling_cost_vs_exposure.json`), remediating any replica classified as `STALE` or `WRONG_CONTENT` upon detection:

| Audit Interval (Steps) | Audit Calls Made (Cost) | Wrong Requests Served (Exposure) | Exposure Reduction (%) | Requests Saved / Call |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3,344 | 400 | 93.33% | 1.67 |
| 2 | 1,600 | 800 | 86.67% | 3.25 |
| 4 | 800 | 1,360 | 77.33% | 5.80 |
| 5 | 672 | 1,100 | 81.67% | 7.29 |
| 8 | 320 | 2,480 | 58.67% | 11.00 |
| 10 | 320 | 2,000 | 66.67% | 12.50 |
| 20 | 160 | 6,000 | 0.00% | 0.00 |

Notice a subtle boundary interaction: for an 8-step rollout window, polling at `interval=4` checks at step 8 (still inside the valid window) and cannot flag a stalled rollout until step 12, whereas `interval=5` checks at step 10 and catches the stall two steps earlier. Consequently, the knee of the curve sits at an audit interval of 5 steps, eliminating 81.67% of wrong-weights request exposure at a cost of 672 replica audit calls (20.1% of every-step polling cost).

## Experiment 5 (Stretch): Canary Attestation Routing

Routing a deterministic fraction of requests exclusively to replicas whose attested in-memory SHA-256 already matches `intended_sha256` (`docs/experiments/exp5_canary_attestation.json`) removes wrong-weight exposure proportionally, at the cost of temporarily restricting capacity during honest rollouts to waves that have finished loading the new weights:

| Attested Routing Fraction | Wrong Requests Served | Exposure Reduction (%) | Effective Rollout Capacity (%) | Rollout Capacity Cost (%) |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 6,000 | 0.00% | 100.00% | 0.00% |
| 0.25 | 4,500 | 25.00% | 90.62% | 9.38% |
| 0.50 | 3,000 | 50.00% | 81.25% | 18.75% |
| 0.75 | 1,500 | 75.00% | 71.88% | 28.12% |
| 1.00 | 0 | 100.00% | 62.50% | 37.50% |

In particular, 50% canary attestation routing cuts wrong-weights exposure in half with an 18.75% rollout capacity cost during the transition window.

## What This Project Cannot Measure

By design (see [ADR 0003](docs/adr/0003-simulated-fleet-vs-real-cluster.md)), our in-process discrete-step simulator measures rollout semantics and exposure arithmetic deterministically, and does not measure:

1. **Real network partitions and asymmetric split-brain:** Cases where the ingress router can reach a stale pod that the attestation auditor cannot poll.
2. **Kubelet restarts and node pressure:** Container runtime eviction, image pull backoffs, and Linux page-cache thrashing.
3. **Multi-gigabyte hashing latency:** Wall-clock time required to stream and hash 10+ GB LLM checkpoint shards from disk into GPU/host memory (see [ADR 0005](docs/adr/0005-full-file-sha256-vs-chunked-hashing.md)).

## Out of Scope on Purpose

- **Cryptographic signing and supply-chain provenance (Sigstore / SLSA):** This system proves *which bytes are loaded in memory* and whether they match the registry's canonical digest, not *who trained or signed those bytes*. **Trigger to add provenance:** the first deployment environment where the model artifact store or registry metadata database is shared across untrusted tenants or external build pipelines.
- **Multi-model ensembles and GPU tensor-parallel sharded serving:** Left for v2, where each replica rank attests an ordered tuple of `(shard_index, shard_sha256)`.

## Questions Answered from Our Measurements (Section 10)

1. **What does `Production=v7` actually guarantee, and what does it merely intend?**
   In MLflow and similar registries, `Production=v7` guarantees only that a metadata pointer named `production` currently resolves to version record `v7`. It merely *intends* that serving processes have loaded the original `v7` bytes. As Experiment 2 shows, it neither guarantees that running replicas reloaded when the alias moved nor that the artifact file behind `v7` remained unmutated.
2. **How would you detect a replica serving last week's weights behind a green readiness probe?**
   By polling an in-process `/attest` record that returns the SHA-256 computed over the in-memory weight buffer at load time alongside the logical load timestamp (`load_step`) and `claimed_version`. Even when `/healthz` returns `200 OK`, the mismatch between `loaded_sha256` and the registry's canonical target digest immediately classifies the replica as `STALE`.
3. **What should a mixed fleet look like during a rollout with `maxSurge 1` and `maxUnavailable 0`, and when is a mix evidence of a problem?**
   For a fleet of $N$ replicas with `maxSurge=1, maxUnavailable=0` and readiness delay $d$, at most 1 surge replica transitions per wave of $d+1$ steps, so the fleet legitimately mixes old and new hashes for up to $N \times (d + 1)$ steps (16 steps for $N=8, d=1$). A mixed fleet is evidence of a problem either immediately if a replica *claims* the new version label while serving the old hash, or at step $N \times (d + 1) + 1$ (step 17 in Experiment 3) when the rollout window expires with old replicas still serving.
4. **Why hash in the process instead of in the pipeline?**
   Hashing in the CI/CD pipeline or an init-container checks the file on disk at $t_{\text{check}}$, leaving a TOCTOU gap before the serving process reads bytes into memory at $t_{\text{use}}$ and providing zero visibility into failed hot-reloads or stale local caches (ADR 0002).
5. **What did my own sweep say about the audit interval that buys the most exposure reduction per call?**
   In Experiment 4, sweeping intervals from 1 to 20 steps showed that `audit_interval = 5` steps is the knee of the curve: it requires only 672 audit calls across 20 rollouts (20.1% of step-1 polling cost) while cutting wrong-weights exposure from 6,000 requests down to 1,100 requests (an 81.67% reduction, saving 7.29 wrong requests per audit call) because step 10 lands just past the 8-step rollout window and catches stalled rollouts two steps earlier than `interval = 4`.

## Architecture Decision Records

- [ADR 0001: In-Process Attestation vs. Trusting the Orchestrator Pod Spec](docs/adr/0001-in-process-attestation-vs-orchestrator-spec.md)
- [ADR 0002: Hash at Load Time in Memory vs. Hash on Disk Before Load](docs/adr/0002-hash-at-load-time-vs-hash-on-disk.md)
- [ADR 0003: Deterministic In-Process Fleet Simulation vs. Live Kubernetes Cluster](docs/adr/0003-simulated-fleet-vs-real-cluster.md)
- [ADR 0004: Spec-Derived Expected-Mix Window vs. Fixed Tolerance Constant](docs/adr/0004-spec-derived-mix-window-vs-tolerance-constant.md)
- [ADR 0005: Full-File SHA-256 vs. Chunked / Merkle Hashing](docs/adr/0005-full-file-sha256-vs-chunked-hashing.md)
