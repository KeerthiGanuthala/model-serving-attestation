# ADR 0001: In-Process Attestation vs. Trusting the Orchestrator Pod Spec

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

In standard ML serving stacks, deployment verification checks two control-plane pointers:

1. The model registry alias or stage tag (for example, `production -> v7`).
2. The container orchestrator pod specification and HTTP readiness endpoint (`/healthz` returning `200 OK`).

Both checks inspect intentions rather than facts. A registry alias is a mutable pointer in a metadata store. A pod spec records which environment variable or config map the control plane asked the container to read. Neither proves which weight bytes currently reside inside the serving process's memory space.

When a hot-reload callback silently skips re-reading weights, when a local volume cache serves stale bytes after an alias move, or when an artifact file is overwritten under an unchanged version label, the registry and orchestrator dashboards remain green while the fleet serves the wrong model.

## Decision

Every serving replica computes the SHA-256 digest of its in-memory weight buffer inside the process at load time (`InProcessLoader.load`) and exposes an `AttestRecord` containing:

- `loaded_sha256`: the lowercase hexadecimal SHA-256 of the bytes actually loaded into memory.
- `load_step`: the logical step at which the in-memory buffer was populated.
- `claimed_version`: the registry version label the replica believes it is serving.

An external `AttestationAuditor` polls the fleet and compares `loaded_sha256` against the canonical SHA-256 of the version currently targeted by the registry's `production` alias.

## Consequences

- **Positive:** Closes the pointer-versus-fact gap completely. In our benchmark workload, unattested faults accumulate 600 wrong-weight replica-steps across 20 rollouts while readiness probes report 100% healthy; in-process attestation flags every single divergence.
- **Negative:** Requires each serving replica to expose an attestation endpoint and requires the loader to hash the byte buffer in memory at load time.
