# ADR 0003: Deterministic In-Process Fleet Simulation vs. Live Kubernetes Cluster

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

We need to measure wrong-weights exposure across dozens of rollouts, fault injections, `(maxSurge, maxUnavailable, fleet_size)` grids, and audit polling intervals. Running these experiments against a live multi-node Kubernetes cluster introduces non-deterministic container image pull latency, kernel scheduling jitter, and wall-clock variance that make exact reproduction across developer laptops and CI runners impossible.

## Decision

Model the replica lifecycle (`starting`, `ready`, `serving`, `terminating`), the `RollingUpdate` wave controller (`maxSurge`, `maxUnavailable`, `readiness_delay_steps`), and the request routing loop as a deterministic, single-process discrete-step state machine (`FleetRolloutSimulator`) seeded by a fixed integer. Measure all exposure strictly in `replica-steps` and logical `requests`, never wall-clock milliseconds.

### What Transfers to Production

- The exact batching arithmetic of Kubernetes `RollingUpdate` (`wave_capacity = maxSurge + maxUnavailable`, wave count `ceil(N / (maxSurge + maxUnavailable))`).
- The logical classification rules distinguishing `EXPECTED_MIX` from `STALE` and `WRONG_CONTENT`.
- The polling interval tradeoff structure between audit call frequency and undetected wrong-weight exposure.

### What Does Not Transfer

- Real network partitions, packet drops, or asymmetric routing where the auditor can reach a pod that the ingress controller cannot (or vice versa).
- Kubelet container restarts, OOMKills, and node eviction timing.
- Wall-clock I/O throughput when reading and hashing multi-gigabyte weight files from object storage.

## Consequences

- **Positive:** Every number in `docs/metrics.json` and `README.md` reproduces bit-for-bit across any machine in under 5 seconds with zero network dependencies.
- **Negative:** Wall-clock latency and kernel/network failure modes are explicitly outside the measurement boundary and documented as non-goals.
