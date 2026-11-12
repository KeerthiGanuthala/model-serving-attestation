"""model-serving-attestation: Content-addressed in-process attestation for ML model fleets."""

from attestation.auditor import (
    AttestationAuditor,
    FleetAuditReport,
    ReplicaAuditVerdict,
    ReplicaClassification,
)
from attestation.fleet import (
    AttestRecord,
    FaultType,
    FleetRolloutSimulator,
    Replica,
    ReplicaLifecycle,
    RolloutFaultPlan,
    RolloutSpec,
    StepExposureMetrics,
    expected_mix_window_steps,
)
from attestation.registry import (
    IndirectionFailureDemo,
    MinimalRegistry,
    MlflowRegistryAdapter,
    ModelVersionRecord,
    RegistryProtocol,
    TransitionEntry,
    run_indirection_demonstrations,
)
from attestation.weights import (
    InProcessLoader,
    LoadedModel,
    ToctouDemonstrationResult,
    compute_bytes_sha256,
    demonstrate_toctou_gap,
    generate_seeded_weights,
    mutate_single_byte,
    write_weights_file,
)

__all__ = [
    "AttestRecord",
    "AttestationAuditor",
    "FaultType",
    "FleetAuditReport",
    "FleetRolloutSimulator",
    "InProcessLoader",
    "IndirectionFailureDemo",
    "LoadedModel",
    "MinimalRegistry",
    "MlflowRegistryAdapter",
    "ModelVersionRecord",
    "RegistryProtocol",
    "Replica",
    "ReplicaAuditVerdict",
    "ReplicaClassification",
    "ReplicaLifecycle",
    "RolloutFaultPlan",
    "RolloutSpec",
    "StepExposureMetrics",
    "ToctouDemonstrationResult",
    "TransitionEntry",
    "compute_bytes_sha256",
    "demonstrate_toctou_gap",
    "expected_mix_window_steps",
    "generate_seeded_weights",
    "mutate_single_byte",
    "run_indirection_demonstrations",
    "write_weights_file",
]
