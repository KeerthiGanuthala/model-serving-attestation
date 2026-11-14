"""Milestone 3 (M3): Simulated fleet, deterministic RollingUpdate controller, and fault injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import random

from attestation.registry import RegistryProtocol
from attestation.weights import InProcessLoader, LoadedModel, mutate_single_byte, write_weights_file


class ReplicaLifecycle(str, Enum):
    """Lifecycle states of a model-serving replica."""

    STARTING = "starting"
    READY = "ready"
    SERVING = "serving"
    TERMINATING = "terminating"


class FaultType(str, Enum):
    """Injected fault categories in the rollout simulation."""

    NONE = "none"
    STALE_NO_RELOAD = "stale_no_reload"
    CACHED_PATH_AFTER_ALIAS_MOVE = "cached_path_after_alias_move"
    MUTATED_ARTIFACT_UNDER_LABEL = "mutated_artifact_under_label"
    STALLED_ROLLOUT = "stalled_rollout"


@dataclass(frozen=True)
class AttestRecord:
    """In-process attestation payload emitted by a replica when polled."""

    replica_id: str
    lifecycle: ReplicaLifecycle
    claimed_version: str
    loaded_sha256: str | None
    load_step: int | None
    readiness_green: bool
    injected_fault: FaultType = FaultType.NONE


@dataclass(frozen=True)
class RolloutSpec:
    """Kubernetes-style RollingUpdate parameters."""

    fleet_size: int
    max_surge: int = 1
    max_unavailable: int = 0
    readiness_delay_steps: int = 1
    requests_per_replica_step: int = 10

    def __post_init__(self) -> None:
        if self.fleet_size <= 0:
            raise ValueError("fleet_size must be positive")
        if self.max_surge < 0 or self.max_unavailable < 0:
            raise ValueError("max_surge and max_unavailable must be non-negative")
        if self.max_surge == 0 and self.max_unavailable == 0:
            raise ValueError("max_surge and max_unavailable cannot both be zero")
        if self.readiness_delay_steps < 0:
            raise ValueError("readiness_delay_steps must be non-negative")


def expected_mix_window_steps(spec: RolloutSpec) -> int:
    """Compute the theoretical mixed-fleet duration (in logical steps) from the rollout spec.

    In a Kubernetes RollingUpdate over `N = fleet_size` replicas with `maxSurge = S` and
    `maxUnavailable = U`:
      - Each wave can advance at most `B = S + U` replicas concurrently while respecting both
        the total pod ceiling (`N + S`) and the minimum availability floor (`N - U`).
      - Number of waves is `ceil(N / (S + U))`.
      - Each wave requires `readiness_delay_steps` steps in `STARTING` plus 1 step to
        promote to `READY`/`SERVING` and retire the old wave.
    """
    wave_capacity = spec.max_surge + spec.max_unavailable
    num_waves = math.ceil(spec.fleet_size / wave_capacity)
    steps_per_wave = spec.readiness_delay_steps + 1
    return num_waves * steps_per_wave


@dataclass
class Replica:
    """A simulated serving replica with its own lifecycle and in-process loaded model."""

    replica_id: str
    lifecycle: ReplicaLifecycle
    claimed_version: str
    loaded_model: LoadedModel | None
    created_step: int
    ready_at_step: int
    injected_fault: FaultType = FaultType.NONE

    @property
    def is_serving_traffic(self) -> bool:
        return self.lifecycle in (ReplicaLifecycle.READY, ReplicaLifecycle.SERVING)

    @property
    def readiness_green(self) -> bool:
        """Readiness probe checks lifecycle state, NOT content hash of loaded weights."""
        return self.is_serving_traffic

    def attest(self) -> AttestRecord:
        return AttestRecord(
            replica_id=self.replica_id,
            lifecycle=self.lifecycle,
            claimed_version=self.claimed_version,
            loaded_sha256=(
                self.loaded_model.loaded_sha256 if self.loaded_model is not None else None
            ),
            load_step=(self.loaded_model.load_step if self.loaded_model is not None else None),
            readiness_green=self.readiness_green,
            injected_fault=self.injected_fault,
        )


@dataclass(frozen=True)
class RolloutFaultPlan:
    """Deterministic fault schedule for a single rollout."""

    fault_type: FaultType = FaultType.NONE
    affected_replica_indices: tuple[int, ...] = ()
    stall_after_waves: int | None = None


@dataclass
class StepExposureMetrics:
    """Per-step accounting of traffic served by honest vs wrong-weight replicas."""

    step: int
    intended_version: str
    intended_sha256: str
    total_serving_replicas: int
    all_readiness_green: bool
    correct_replica_steps: int
    expected_mix_replica_steps: int
    unintended_Valid_old_replica_steps: int
    wrong_weights_replica_steps: int
    wrong_weights_requests: int
    fault_breakdown_replica_steps: dict[str, int] = field(default_factory=dict)


class FleetRolloutSimulator:
    """Deterministic in-process simulator of a replica fleet and RollingUpdate controller."""

    def __init__(
        self,
        spec: RolloutSpec,
        registry: RegistryProtocol,
        model_name: str,
        work_dir: Path,
        seed: int = 42,
    ) -> None:
        self.spec = spec
        self.registry = registry
        self.model_name = model_name
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.loader = InProcessLoader()

        self.current_step: int = 0
        self.replicas: list[Replica] = []
        self._next_replica_seq: int = 1
        self._cached_artifact_by_replicaSlot: dict[int, Path] = {}

        # Active rollout state
        self.rollout_in_progress: bool = False
        self.rollout_start_step: int | None = None
        self.target_version: str | None = None
        self.target_canonical_sha256: str | None = None
        self.previous_canonical_sha256: str | None = None
        self._old_replicas_queue: list[Replica] = []
        self._starting_wave: list[tuple[Replica, Replica | None]] = []
        self._waves_completed: int = 0
        self._active_fault_plan: RolloutFaultPlan = RolloutFaultPlan()
        self._replaced_count: int = 0

    def bootstrap_initial_fleet(self, alias: str = "production") -> None:
        """Boot initial `N` replicas serving the current `production` version cleanly."""
        rec = self.registry.resolve_alias(self.model_name, alias)
        self.replicas.clear()
        self.target_version = rec.version
        self.target_canonical_sha256 = rec.registered_sha256
        self.previous_canonical_sha256 = rec.registered_sha256

        for idx in range(self.spec.fleet_size):
            loaded = self.loader.load(
                Path(rec.artifact_path),
                claimed_version=rec.version,
                load_step=self.current_step,
            )
            # Populate per-slot local cache with initial version bytes
            cache_path = self.work_dir / f"slot_{idx}_cache.bin"
            write_weights_file(cache_path, Path(rec.artifact_path).read_bytes())
            self._cached_artifact_by_replicaSlot[idx] = cache_path

            rep = Replica(
                replica_id=f"replica-{self._next_replica_seq:04d}",
                lifecycle=ReplicaLifecycle.SERVING,
                claimed_version=rec.version,
                loaded_model=loaded,
                created_step=self.current_step,
                ready_at_step=self.current_step,
                injected_fault=FaultType.NONE,
            )
            self._next_replica_seq += 1
            self.replicas.append(rep)

    def start_rollout(
        self,
        new_version: str,
        alias: str = "production",
        fault_plan: RolloutFaultPlan | None = None,
    ) -> None:
        """Repoint the registry alias to `new_version` and initiate a RollingUpdate."""
        current_rec = self.registry.resolve_alias(self.model_name, alias)
        self.previous_canonical_sha256 = current_rec.registered_sha256

        self.registry.set_alias(self.model_name, alias, new_version, step=self.current_step)
        new_rec = self.registry.resolve_alias(self.model_name, alias)

        self.rollout_in_progress = True
        self.rollout_start_step = self.current_step
        self.target_version = new_rec.version
        self.target_canonical_sha256 = new_rec.registered_sha256
        self._old_replicas_queue = list(self.replicas)
        self._starting_wave = []
        self._waves_completed = 0
        self._replaced_count = 0
        self._active_fault_plan = fault_plan or RolloutFaultPlan()

        # Launch wave 1 immediately at rollout start step
        self._launch_next_wave_if_ready()

    def _create_replacement_replica(
        self,
        slot_idx: int,
        old_replica: Replica,
        target_rec_path: Path,
        target_version: str,
    ) -> Replica:
        fault = FaultType.NONE
        if slot_idx in self._active_fault_plan.affected_replica_indices:
            fault = self._active_fault_plan.fault_type

        if fault == FaultType.STALE_NO_RELOAD:
            # Orchestrator thinks replica updated its label, but in-process loader never ran:
            # keeps old LoadedModel (old hash AND old load_step) while claiming new version.
            assert old_replica.loaded_model is not None
            loaded = LoadedModel(
                claimed_version=target_version,
                loaded_sha256=old_replica.loaded_model.loaded_sha256,
                size_bytes=old_replica.loaded_model.size_bytes,
                load_step=old_replica.loaded_model.load_step,
                source_path=old_replica.loaded_model.source_path,
                _weights_buffer=old_replica.loaded_model._weights_buffer,
            )
        elif fault == FaultType.CACHED_PATH_AFTER_ALIAS_MOVE:
            # Replica reloads at current step, but reads from its un-invalidated local cache path
            cache_path = self._cached_artifact_by_replicaSlot[slot_idx]
            loaded = self.loader.load(
                cache_path,
                claimed_version=target_version,
                load_step=self.current_step,
            )
        elif fault == FaultType.MUTATED_ARTIFACT_UNDER_LABEL:
            # Artifact bytes on disk were mutated under the unchanged target_version label
            mutated_path = self.work_dir / f"mutated_{target_version}_slot_{slot_idx}.bin"
            orig_bytes = target_rec_path.read_bytes()
            write_weights_file(mutated_path, mutate_single_byte(orig_bytes, offset=slot_idx + 3))
            loaded = self.loader.load(
                mutated_path,
                claimed_version=target_version,
                load_step=self.current_step,
            )
        else:
            # Honest load from registry target artifact path
            loaded = self.loader.load(
                target_rec_path,
                claimed_version=target_version,
                load_step=self.current_step,
            )
            # Update local slot cache with fresh bytes
            cache_path = self._cached_artifact_by_replicaSlot[slot_idx]
            write_weights_file(cache_path, target_rec_path.read_bytes())

        rep = Replica(
            replica_id=f"replica-{self._next_replica_seq:04d}",
            lifecycle=ReplicaLifecycle.STARTING,
            claimed_version=target_version,
            loaded_model=loaded,
            created_step=self.current_step,
            ready_at_step=self.current_step + self.spec.readiness_delay_steps,
            injected_fault=fault,
        )
        self._next_replica_seq += 1
        return rep

    def _launch_next_wave_if_ready(self) -> None:
        if not self.rollout_in_progress or self._starting_wave:
            return
        if not self._old_replicas_queue:
            self.rollout_in_progress = False
            return

        # Check if fault plan stalls rollout after K completed waves
        if (
            self._active_fault_plan.fault_type == FaultType.STALLED_ROLLOUT
            and self._active_fault_plan.stall_after_waves is not None
            and self._waves_completed >= self._active_fault_plan.stall_after_waves
        ):
            # Controller hangs: no new wave is scheduled, leaving remaining old replicas serving
            return

        assert self.target_version is not None
        target_rec = self.registry.get_version(self.model_name, self.target_version)
        target_path = Path(target_rec.artifact_path)

        wave_budget = min(
            len(self._old_replicas_queue),
            self.spec.max_surge + self.spec.max_unavailable,
        )

        # Up to max_unavailable old replicas can be terminated immediately when wave begins
        immediate_unavail = min(self.spec.max_unavailable, wave_budget)

        for i in range(wave_budget):
            old_rep = self._old_replicas_queue.pop(0)
            slot_idx = self._replaced_count
            self._replaced_count += 1

            if i < immediate_unavail:
                old_rep.lifecycle = ReplicaLifecycle.TERMINATING
                if old_rep in self.replicas:
                    self.replicas.remove(old_rep)
                paired_old: Replica | None = None
            else:
                paired_old = old_rep

            new_rep = self._create_replacement_replica(
                slot_idx=slot_idx,
                old_replica=old_rep,
                target_rec_path=target_path,
                target_version=target_rec.version,
            )
            self.replicas.append(new_rep)
            self._starting_wave.append((new_rep, paired_old))

    def step(self) -> StepExposureMetrics:
        """Advance the fleet simulation by 1 deterministic logical step and record exposure."""
        self.current_step += 1

        # 1. Advance any STARTING replicas whose readiness gate has elapsed
        if self._starting_wave:
            all_wave_ready = all(
                self.current_step > new_rep.ready_at_step for new_rep, _ in self._starting_wave
            )
            if all_wave_ready:
                for new_rep, paired_old in self._starting_wave:
                    new_rep.lifecycle = ReplicaLifecycle.SERVING
                    if paired_old is not None:
                        paired_old.lifecycle = ReplicaLifecycle.TERMINATING
                        if paired_old in self.replicas:
                            self.replicas.remove(paired_old)
                self._starting_wave.clear()
                self._waves_completed += 1
                self._launch_next_wave_if_ready()
            else:
                for new_rep, _ in self._starting_wave:
                    if (
                        self.current_step == new_rep.ready_at_step
                        and new_rep.lifecycle == ReplicaLifecycle.STARTING
                    ):
                        new_rep.lifecycle = ReplicaLifecycle.READY

        # 2. Measure traffic exposure across replicas currently serving traffic
        serving_replicas = [r for r in self.replicas if r.is_serving_traffic]
        assert self.target_version is not None
        assert self.target_canonical_sha256 is not None

        exp_window = expected_mix_window_steps(self.spec)
        in_expected_window = (
            self.rollout_start_step is not None
            and (self.current_step - self.rollout_start_step) <= exp_window
        )

        correct_steps = 0
        expected_mix_steps = 0
        wrong_steps = 0
        fault_breakdown: dict[str, int] = {}

        for rep in serving_replicas:
            sha = rep.loaded_model.loaded_sha256 if rep.loaded_model else None
            if sha == self.target_canonical_sha256:
                correct_steps += 1
            elif (
                in_expected_window
                and sha == self.previous_canonical_sha256
                and rep.claimed_version != self.target_version
                and rep.injected_fault == FaultType.NONE
            ):
                expected_mix_steps += 1
            else:
                wrong_steps += 1
                f_key = (
                    rep.injected_fault.value
                    if rep.injected_fault != FaultType.NONE
                    else FaultType.STALLED_ROLLOUT.value
                )
                fault_breakdown[f_key] = fault_breakdown.get(f_key, 0) + 1

        return StepExposureMetrics(
            step=self.current_step,
            intended_version=self.target_version,
            intended_sha256=self.target_canonical_sha256,
            total_serving_replicas=len(serving_replicas),
            all_readiness_green=all(r.readiness_green for r in serving_replicas),
            correct_replica_steps=correct_steps,
            expected_mix_replica_steps=expected_mix_steps,
            unintended_Valid_old_replica_steps=0,
            wrong_weights_replica_steps=wrong_steps,
            wrong_weights_requests=wrong_steps * self.spec.requests_per_replica_step,
            fault_breakdown_replica_steps=fault_breakdown,
        )

    def attest_serving_fleet(self) -> list[AttestRecord]:
        """Return attestation records for all replicas currently serving traffic."""
        return [r.attest() for r in self.replicas if r.is_serving_traffic]
