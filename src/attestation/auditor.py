"""Milestone 4 (M4): The attestation auditor and replica classification engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from attestation.fleet import AttestRecord, RolloutSpec, expected_mix_window_steps


class ReplicaClassification(str, Enum):
    """Auditor classification of an individual serving replica."""

    CORRECT = "correct"
    EXPECTED_MIX = "expected_mix"
    STALE = "stale"
    WRONG_CONTENT = "wrong_content"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReplicaAuditVerdict:
    """Detailed audit verdict for a single replica at a given logical step."""

    step: int
    replica_id: str
    classification: ReplicaClassification
    claimed_version: str
    intended_version: str
    loaded_sha256: str | None
    intended_sha256: str
    load_step: int | None
    attestation_age_steps: int | None
    reason: str


@dataclass(frozen=True)
class FleetAuditReport:
    """Fleet-wide audit snapshot at a single polling step."""

    step: int
    intended_version: str
    intended_sha256: str
    rollout_elapsed_steps: int | None
    expected_mix_window_steps: int
    verdicts: tuple[ReplicaAuditVerdict, ...]

    @property
    def counts_by_classification(self) -> dict[str, int]:
        counts: dict[str, int] = {c.value: 0 for c in ReplicaClassification}
        for v in self.verdicts:
            counts[v.classification.value] += 1
        return counts

    @property
    def has_anomalies(self) -> bool:
        return any(
            v.classification
            in (
                ReplicaClassification.STALE,
                ReplicaClassification.WRONG_CONTENT,
                ReplicaClassification.UNKNOWN,
            )
            for v in self.verdicts
        )


class AttestationAuditor:
    """Polls serving replicas and compares attested in-memory hashes against registry intention.

    Crucially, the expected-mix window is derived mathematically from the `RolloutSpec`
    (`expected_mix_window_steps(spec)`), never hardcoded as an arbitrary tolerance constant.
    """

    def __init__(self, spec: RolloutSpec) -> None:
        self.spec = spec
        self.expected_window = expected_mix_window_steps(spec)
        self.audit_calls_made: int = 0

    def classify_replica(
        self,
        record: AttestRecord,
        current_step: int,
        intended_version: str,
        intended_sha256: str,
        previous_valid_hashes: set[str],
        rollout_start_step: int | None,
    ) -> ReplicaAuditVerdict:
        """Classify a single replica's attestation record."""
        if record.loaded_sha256 is None or record.load_step is None:
            return ReplicaAuditVerdict(
                step=current_step,
                replica_id=record.replica_id,
                classification=ReplicaClassification.UNKNOWN,
                claimed_version=record.claimed_version,
                intended_version=intended_version,
                loaded_sha256=record.loaded_sha256,
                intended_sha256=intended_sha256,
                load_step=record.load_step,
                attestation_age_steps=None,
                reason="Missing loaded_sha256 or load_step in attestation response",
            )

        age = current_step - record.load_step

        # Case 1: Loaded bytes match the canonical hash of the registry's current target
        if record.loaded_sha256 == intended_sha256:
            return ReplicaAuditVerdict(
                step=current_step,
                replica_id=record.replica_id,
                classification=ReplicaClassification.CORRECT,
                claimed_version=record.claimed_version,
                intended_version=intended_version,
                loaded_sha256=record.loaded_sha256,
                intended_sha256=intended_sha256,
                load_step=record.load_step,
                attestation_age_steps=age,
                reason="In-memory SHA-256 matches registry intention",
            )

        # Case 2: Loaded bytes belong to a previously valid version
        if record.loaded_sha256 in previous_valid_hashes:
            # Subcase 2a: Replica claims it already loaded `intended_version`, but its hash is old!
            # This immediately catches both STALE_NO_RELOAD and CACHED_PATH_AFTER_ALIAS_MOVE
            # even on step 1 of a rollout.
            if record.claimed_version == intended_version:
                sub_reason = (
                    "Replica claims target version but loaded old bytes from stale cache at rollout"
                    if (rollout_start_step is not None and record.load_step >= rollout_start_step)
                    else "Replica claims target version and green readiness without reloading weights"
                )
                return ReplicaAuditVerdict(
                    step=current_step,
                    replica_id=record.replica_id,
                    classification=ReplicaClassification.STALE,
                    claimed_version=record.claimed_version,
                    intended_version=intended_version,
                    loaded_sha256=record.loaded_sha256,
                    intended_sha256=intended_sha256,
                    load_step=record.load_step,
                    attestation_age_steps=age,
                    reason=sub_reason,
                )

            # Subcase 2b: Old replica awaiting replacement during an active rollout
            if rollout_start_step is not None:
                rollout_elapsed = current_step - rollout_start_step
                if rollout_elapsed <= self.expected_window:
                    return ReplicaAuditVerdict(
                        step=current_step,
                        replica_id=record.replica_id,
                        classification=ReplicaClassification.EXPECTED_MIX,
                        claimed_version=record.claimed_version,
                        intended_version=intended_version,
                        loaded_sha256=record.loaded_sha256,
                        intended_sha256=intended_sha256,
                        load_step=record.load_step,
                        attestation_age_steps=age,
                        reason=(
                            f"Old version within spec-derived rollout window "
                            f"({rollout_elapsed}/{self.expected_window} steps)"
                        ),
                    )
                return ReplicaAuditVerdict(
                    step=current_step,
                    replica_id=record.replica_id,
                    classification=ReplicaClassification.STALE,
                    claimed_version=record.claimed_version,
                    intended_version=intended_version,
                    loaded_sha256=record.loaded_sha256,
                    intended_sha256=intended_sha256,
                    load_step=record.load_step,
                    attestation_age_steps=age,
                    reason=(
                        f"Rollout exceeded spec-derived window "
                        f"({rollout_elapsed} > {self.expected_window} steps); stalled rollout"
                    ),
                )

            return ReplicaAuditVerdict(
                step=current_step,
                replica_id=record.replica_id,
                classification=ReplicaClassification.STALE,
                claimed_version=record.claimed_version,
                intended_version=intended_version,
                loaded_sha256=record.loaded_sha256,
                intended_sha256=intended_sha256,
                load_step=record.load_step,
                attestation_age_steps=age,
                reason="Replica serving previous version outside any active rollout",
            )

        # Case 3: Loaded bytes match neither current intention nor any previous valid version
        return ReplicaAuditVerdict(
            step=current_step,
            replica_id=record.replica_id,
            classification=ReplicaClassification.WRONG_CONTENT,
            claimed_version=record.claimed_version,
            intended_version=intended_version,
            loaded_sha256=record.loaded_sha256,
            intended_sha256=intended_sha256,
            load_step=record.load_step,
            attestation_age_steps=age,
            reason="In-memory SHA-256 matches neither intended nor prior registered version (mutated artifact)",
        )

    def audit_fleet(
        self,
        records: list[AttestRecord],
        current_step: int,
        intended_version: str,
        intended_sha256: str,
        previous_valid_hashes: set[str],
        rollout_start_step: int | None,
    ) -> FleetAuditReport:
        """Poll all provided replica records, increment call counters, and return a FleetAuditReport."""
        self.audit_calls_made += len(records)
        verdicts = [
            self.classify_replica(
                record=rec,
                current_step=current_step,
                intended_version=intended_version,
                intended_sha256=intended_sha256,
                previous_valid_hashes=previous_valid_hashes,
                rollout_start_step=rollout_start_step,
            )
            for rec in records
        ]
        elapsed = (
            current_step - rollout_start_step if rollout_start_step is not None else None
        )
        return FleetAuditReport(
            step=current_step,
            intended_version=intended_version,
            intended_sha256=intended_sha256,
            rollout_elapsed_steps=elapsed,
            expected_mix_window_steps=self.expected_window,
            verdicts=tuple(verdicts),
        )
