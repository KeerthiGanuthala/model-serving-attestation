"""Milestone 1 (M1): Seeded weights, in-process loader, and load-time SHA-256 attestation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable


def generate_seeded_weights(seed: int, size_bytes: int = 4096) -> bytes:
    """Deterministically generate dummy model weight bytes from an integer seed.

    Uses a SHA-256 counter-mode stream so the exact byte sequence is
    reproducible across machines, architectures, and Python versions.
    """
    if size_bytes <= 0:
        raise ValueError(f"size_bytes must be positive, got {size_bytes}")
    chunks: list[bytes] = []
    produced = 0
    counter = 0
    header = f"model-weights:seed={seed}:size={size_bytes}:".encode("utf-8")
    while produced < size_bytes:
        digest = hashlib.sha256(header + counter.to_bytes(8, "big")).digest()
        take = min(len(digest), size_bytes - produced)
        chunks.append(digest[:take])
        produced += take
        counter += 1
    return b"".join(chunks)


def compute_bytes_sha256(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def write_weights_file(path: Path, data: bytes) -> Path:
    """Write weight bytes atomically-ish to disk and return the resolved path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def mutate_single_byte(data: bytes, offset: int = 0) -> bytes:
    """Flip one byte at `offset` to prove sensitivity of content-addressed identity."""
    if not data:
        raise ValueError("Cannot mutate empty byte buffer")
    idx = offset % len(data)
    mutated = bytearray(data)
    mutated[idx] ^= 0xFF
    return bytes(mutated)


@dataclass(frozen=True)
class LoadedModel:
    """Fact record produced inside the serving process at the moment weights enter memory."""

    claimed_version: str
    loaded_sha256: str
    size_bytes: int
    load_step: int
    source_path: str
    _weights_buffer: bytes

    def verify_in_memory_hash(self) -> bool:
        """Re-hash the in-process memory buffer to confirm no post-load corruption."""
        return compute_bytes_sha256(self._weights_buffer) == self.loaded_sha256


class InProcessLoader:
    """Loads model weights into memory and hashes the in-process byte buffer at load time."""

    def load(
        self,
        path: Path,
        claimed_version: str,
        load_step: int,
    ) -> LoadedModel:
        raw_bytes = path.read_bytes()
        in_memory_hash = compute_bytes_sha256(raw_bytes)
        return LoadedModel(
            claimed_version=claimed_version,
            loaded_sha256=in_memory_hash,
            size_bytes=len(raw_bytes),
            load_step=load_step,
            source_path=str(path),
            _weights_buffer=raw_bytes,
        )


@dataclass(frozen=True)
class ToctouDemonstrationResult:
    """Evidence comparing pre-load disk hashing against in-process load-time hashing."""

    pre_load_disk_sha256: str
    in_process_loaded_sha256: str
    post_mutation_disk_sha256: str
    disk_check_caught_mutation: bool
    in_process_attestation_caught_mutation: bool


def demonstrate_toctou_gap(
    artifact_path: Path,
    original_bytes: bytes,
    mutated_bytes: bytes,
    claimed_version: str = "v1",
    load_step: int = 1,
    between_check_and_use_hook: Callable[[Path], None] | None = None,
) -> ToctouDemonstrationResult:
    """Demonstrate the Time-of-Check to Time-of-Use (TOCTOU) gap (ADR 0002).

    1. Writes `original_bytes` to `artifact_path`.
    2. Computes the pre-load disk hash (the 'time-of-check' claim).
    3. Executes the mutation on disk before the serving process reads the file
       (the TOCTOU window).
    4. Loads the file via `InProcessLoader` (the 'time-of-use' fact).
    """
    write_weights_file(artifact_path, original_bytes)
    expected_original_hash = compute_bytes_sha256(original_bytes)

    # Step 1: External pipeline or pre-flight script checks file on disk
    pre_load_disk_sha256 = compute_bytes_sha256(artifact_path.read_bytes())

    # Step 2: File changes on shared volume / cache between check and load
    if between_check_and_use_hook is not None:
        between_check_and_use_hook(artifact_path)
    else:
        write_weights_file(artifact_path, mutated_bytes)

    # Step 3: Serving process loads bytes into memory and hashes what it actually loaded
    loader = InProcessLoader()
    loaded = loader.load(artifact_path, claimed_version=claimed_version, load_step=load_step)
    post_mutation_disk_sha256 = compute_bytes_sha256(artifact_path.read_bytes())

    return ToctouDemonstrationResult(
        pre_load_disk_sha256=pre_load_disk_sha256,
        in_process_loaded_sha256=loaded.loaded_sha256,
        post_mutation_disk_sha256=post_mutation_disk_sha256,
        disk_check_caught_mutation=(pre_load_disk_sha256 != expected_original_hash),
        in_process_attestation_caught_mutation=(loaded.loaded_sha256 != expected_original_hash),
    )
