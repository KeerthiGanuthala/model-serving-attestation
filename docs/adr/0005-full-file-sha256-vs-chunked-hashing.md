# ADR 0005: Full-File SHA-256 vs. Chunked / Merkle Hashing

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

To establish content-addressed identity inside the serving process, we must choose a hashing strategy over the loaded model weight bytes. Options include:

1. Full-buffer cryptographic SHA-256 via Python's C-backed `hashlib.sha256`.
2. Sampled / strided hashing (hashing every $k$-th megabyte).
3. Parallel Merkle tree hashing (for example, BLAKE3 or chunked SHA-256 trees across tensor shards).

## Decision

Use standard full-buffer `hashlib.sha256` over the entire loaded weight byte sequence.

Why choose the boring option now:

- Sampled hashing violates the core guarantee of the project: a single mutated weight byte between sampled offsets would go undetected (`test_in_process_loader_preserves_sha256_identity_and_detects_single_byte_mutation`).
- Standard `hashlib.sha256` uses OpenSSL hardware-accelerated SHA-NI instructions (~1.5 to 2.5 GB/s on modern x86_64/ARMv8 CPUs), requires zero third-party dependencies, and matches the digest format recorded by OCI registries, S3 checksums, and MLflow artifact metadata.

### When We Would Revisit This Decision

We would revisit full-file single-threaded SHA-256 when individual replica weight payloads exceed **4 GB to 8 GB** per process (where single-core SHA-256 adds ~2 to 5 seconds to cold-start readiness latency) or when serving multi-node tensor-parallel shards where each rank loads a distinct subset of `safetensors` shards. At that threshold (>5 GB per rank), we would adopt a two-level Merkle tree over 64 MB chunks hashed in parallel across worker threads during `mmap` page-in.

## Consequences

- **Positive:** Unambiguous, dependency-free, single-byte-sensitive identity compatible with every standard SHA-256 tool.
- **Negative:** Linear single-core hashing pass over the loaded bytes during model startup.
