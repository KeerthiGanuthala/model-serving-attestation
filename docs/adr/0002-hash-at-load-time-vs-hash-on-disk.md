# ADR 0002: Hash at Load Time in Memory vs. Hash on Disk Before Load

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

A common shortcut in deployment pipelines is an init-container or pre-flight script that hashes the weight file on disk (`sha256sum model.bin`), verifies that it matches the expected digest, and exits before the model server starts reading the file.

This creates a classic Time-of-Check to Time-of-Use (TOCTOU) window:

1. At time $t_{\text{check}}$, the pre-flight check hashes `/mnt/models/production/weights.bin` on a shared network volume or node-local cache.
2. Between $t_{\text{check}}$ and $t_{\text{use}}$, a background cache syncer, symlink swap, or concurrent writer overwrites `/mnt/models/production/weights.bin`.
3. At time $t_{\text{use}}$, the model server opens the path and reads the mutated bytes into memory.

The pre-flight check passed, yet the process loaded different bytes.

## Decision

Identity must be computed on the in-memory byte buffer already read into the serving process (`raw_bytes = path.read_bytes(); digest = hashlib.sha256(raw_bytes).hexdigest()`), never by hashing a file path on disk in a separate step prior to load. Furthermore, `LoadedModel.verify_in_memory_hash()` can re-verify the retained buffer in process memory at any point.

## Consequences

- **Positive:** Eliminates the filesystem TOCTOU window between verification and model initialization, as demonstrated deterministically in `demonstrate_toctou_gap()` and Experiment 2.
- **Negative:** The serving process must hold or stream the weight bytes through the hash state inside its own address space during model initialization.
