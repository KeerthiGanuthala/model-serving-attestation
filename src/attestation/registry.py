"""Milestone 2 (M2): Minimal registry and MLflow file-store registry adapter.

Both registries satisfy the same `RegistryProtocol` interface and pass a shared test suite.
Demonstrates the two directions of registry indirection failure:
1. Alias repointed (`production` -> new version) without any content change on existing files.
2. Content mutated or re-registered under an unchanged-looking label.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Protocol

from attestation.weights import compute_bytes_sha256


@dataclass(frozen=True)
class ModelVersionRecord:
    """Metadata record stored by a model registry for a registered version."""

    name: str
    version: str
    artifact_path: str
    registered_sha256: str
    created_step: int
    status: str = "READY"


@dataclass(frozen=True)
class TransitionEntry:
    """Audit log entry recording an alias pointer transition in the registry."""

    step: int
    model_name: str
    alias: str
    from_version: str | None
    to_version: str


@dataclass(frozen=True)
class IndirectionFailureDemo:
    """Structured evidence from the label-vs-bytes registry indirection demonstration."""

    scenario: str
    queried_alias: str
    registry_reported_version: str
    registry_reported_status: str
    registry_metadata_changed: bool
    canonical_intended_sha256: str
    actual_bytes_on_storage_sha256: str
    replica_in_memory_sha256: str
    registry_api_detects_discrepancy: bool
    attestation_detects_discrepancy: bool


class RegistryProtocol(Protocol):
    """Common interface implemented by both MinimalRegistry and MlflowRegistryAdapter."""

    @property
    def backend_name(self) -> str: ...

    def register_version(
        self,
        name: str,
        source_artifact: Path,
        step: int,
        version_label: str | None = None,
    ) -> ModelVersionRecord: ...

    def set_alias(
        self,
        name: str,
        alias: str,
        version: str,
        step: int,
    ) -> TransitionEntry: ...

    def resolve_alias(self, name: str, alias: str = "production") -> ModelVersionRecord: ...

    def get_version(self, name: str, version: str) -> ModelVersionRecord: ...

    def list_versions(self, name: str) -> list[ModelVersionRecord]: ...

    def transition_log(self, name: str | None = None) -> list[TransitionEntry]: ...

    def mutate_stored_artifact(self, name: str, version: str, new_bytes: bytes) -> Path: ...


class MinimalRegistry:
    """In-process + local artifact directory registry with versions, aliases, and transition log."""

    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, dict[str, ModelVersionRecord]] = {}
        self._aliases: dict[str, dict[str, str]] = {}
        self._transitions: list[TransitionEntry] = []

    @property
    def backend_name(self) -> str:
        return "minimal"

    def register_version(
        self,
        name: str,
        source_artifact: Path,
        step: int,
        version_label: str | None = None,
    ) -> ModelVersionRecord:
        model_versions = self._versions.setdefault(name, {})
        ver = version_label if version_label is not None else f"v{len(model_versions) + 1}"
        dest_dir = self._storage_dir / name / ver
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "weights.bin"
        raw = source_artifact.read_bytes()
        dest_file.write_bytes(raw)
        digest = compute_bytes_sha256(raw)
        record = ModelVersionRecord(
            name=name,
            version=ver,
            artifact_path=str(dest_file),
            registered_sha256=digest,
            created_step=step,
            status="READY",
        )
        model_versions[ver] = record
        return record

    def set_alias(
        self,
        name: str,
        alias: str,
        version: str,
        step: int,
    ) -> TransitionEntry:
        if name not in self._versions or version not in self._versions[name]:
            raise KeyError(f"Unknown model version {name}:{version}")
        model_aliases = self._aliases.setdefault(name, {})
        previous = model_aliases.get(alias)
        model_aliases[alias] = version
        entry = TransitionEntry(
            step=step,
            model_name=name,
            alias=alias,
            from_version=previous,
            to_version=version,
        )
        self._transitions.append(entry)
        return entry

    def resolve_alias(self, name: str, alias: str = "production") -> ModelVersionRecord:
        model_aliases = self._aliases.get(name, {})
        if alias not in model_aliases:
            raise KeyError(f"Alias '{alias}' not set for model '{name}'")
        target_ver = model_aliases[alias]
        return self._versions[name][target_ver]

    def get_version(self, name: str, version: str) -> ModelVersionRecord:
        if name not in self._versions or version not in self._versions[name]:
            raise KeyError(f"Unknown model version {name}:{version}")
        return self._versions[name][version]

    def list_versions(self, name: str) -> list[ModelVersionRecord]:
        return list(self._versions.get(name, {}).values())

    def transition_log(self, name: str | None = None) -> list[TransitionEntry]:
        if name is None:
            return list(self._transitions)
        return [t for t in self._transitions if t.model_name == name]

    def mutate_stored_artifact(self, name: str, version: str, new_bytes: bytes) -> Path:
        record = self.get_version(name, version)
        artifact_path = Path(record.artifact_path)
        artifact_path.write_bytes(new_bytes)
        return artifact_path


class MlflowRegistryAdapter:
    """MLflow local file-store registry adapter.

    When the optional `mlflow` package is installed, drives `mlflow.tracking.MlflowClient`
    against a local `file://` tracking URI alongside local artifact files.
    When running in an environment without `mlflow` installed, reads and writes the
    exact MLflow file-store directory layout (`models/<name>/meta.yaml`,
    `models/<name>/version-<N>/meta.yaml`, `models/<name>/aliases/<alias>`) so that
    `make verify` works offline with zero network calls and identical semantics.
    """

    def __init__(self, tracking_dir: Path) -> None:
        self._tracking_dir = Path(tracking_dir).resolve()
        self._models_root = self._tracking_dir / "models"
        self._artifacts_root = self._tracking_dir / "artifacts"
        self._transitions_file = self._tracking_dir / "transitions.json"
        self._models_root.mkdir(parents=True, exist_ok=True)
        self._artifacts_root.mkdir(parents=True, exist_ok=True)
        if not self._transitions_file.exists():
            self._transitions_file.write_text("[]", encoding="utf-8")

        self._mlflow_client: Any = None
        try:
            import mlflow.tracking  # type: ignore

            self._mlflow_client = mlflow.tracking.MlflowClient(
                tracking_uri=self._tracking_dir.as_uri()
            )
        except ImportError:
            self._mlflow_client = None

    @property
    def backend_name(self) -> str:
        return "mlflow-native" if self._mlflow_client is not None else "mlflow-filestore"

    @staticmethod
    def _normalize_version(version_str: str) -> str:
        return version_str if version_str.startswith("v") else f"v{version_str}"

    @staticmethod
    def _strip_v(version_str: str) -> str:
        return version_str[1:] if version_str.startswith("v") else version_str

    def _ensure_registered_model(self, name: str) -> Path:
        model_dir = self._models_root / name
        model_dir.mkdir(parents=True, exist_ok=True)
        meta_path = model_dir / "meta.yaml"
        if not meta_path.exists():
            meta_path.write_text(
                f"name: {name}\ncreation_timestamp: 0\nlast_updated_timestamp: 0\n",
                encoding="utf-8",
            )
        if self._mlflow_client is not None:
            try:
                self._mlflow_client.get_registered_model(name)
            except Exception:
                self._mlflow_client.create_registered_model(name)
        return model_dir

    def register_version(
        self,
        name: str,
        source_artifact: Path,
        step: int,
        version_label: str | None = None,
    ) -> ModelVersionRecord:
        model_dir = self._ensure_registered_model(name)
        existing = self.list_versions(name)
        next_num = len(existing) + 1
        ver_str = self._normalize_version(version_label) if version_label else f"v{next_num}"
        ver_num = self._strip_v(ver_str)

        artifact_dest_dir = self._artifacts_root / name / ver_str
        artifact_dest_dir.mkdir(parents=True, exist_ok=True)
        artifact_dest = artifact_dest_dir / "weights.bin"
        shutil.copyfile(source_artifact, artifact_dest)
        raw = artifact_dest.read_bytes()
        digest = compute_bytes_sha256(raw)

        if self._mlflow_client is not None:
            mv = self._mlflow_client.create_model_version(
                name=name,
                source=str(artifact_dest),
                run_id=None,
                tags={
                    "registered_sha256": digest,
                    "created_step": str(step),
                    "version_label": ver_str,
                },
            )
            ver_num = str(mv.version)
            ver_str = f"v{ver_num}"

        ver_dir = model_dir / f"version-{ver_num}"
        ver_dir.mkdir(parents=True, exist_ok=True)
        meta_payload = {
            "name": name,
            "version": ver_str,
            "numeric_version": ver_num,
            "source": str(artifact_dest),
            "status": "READY",
            "registered_sha256": digest,
            "created_step": step,
        }
        (ver_dir / "meta.yaml").write_text(
            "\n".join(f"{k}: {v}" for k, v in meta_payload.items()) + "\n",
            encoding="utf-8",
        )
        (ver_dir / "meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

        return ModelVersionRecord(
            name=name,
            version=ver_str,
            artifact_path=str(artifact_dest),
            registered_sha256=digest,
            created_step=step,
            status="READY",
        )

    def set_alias(
        self,
        name: str,
        alias: str,
        version: str,
        step: int,
    ) -> TransitionEntry:
        # Validate target version exists
        _ = self.get_version(name, version)
        norm_ver = self._normalize_version(version)
        num_ver = self._strip_v(norm_ver)

        model_dir = self._ensure_registered_model(name)
        aliases_dir = model_dir / "aliases"
        aliases_dir.mkdir(parents=True, exist_ok=True)
        alias_file = aliases_dir / alias

        previous: str | None = None
        if alias_file.exists():
            prev_raw = alias_file.read_text(encoding="utf-8").strip()
            if prev_raw:
                previous = self._normalize_version(prev_raw)

        if self._mlflow_client is not None:
            self._mlflow_client.set_registered_model_alias(name, alias, num_ver)

        alias_file.write_text(num_ver, encoding="utf-8")

        entry = TransitionEntry(
            step=step,
            model_name=name,
            alias=alias,
            from_version=previous,
            to_version=norm_ver,
        )
        transitions = json.loads(self._transitions_file.read_text(encoding="utf-8"))
        transitions.append(
            {
                "step": entry.step,
                "model_name": entry.model_name,
                "alias": entry.alias,
                "from_version": entry.from_version,
                "to_version": entry.to_version,
            }
        )
        self._transitions_file.write_text(json.dumps(transitions, indent=2), encoding="utf-8")
        return entry

    def resolve_alias(self, name: str, alias: str = "production") -> ModelVersionRecord:
        model_dir = self._models_root / name
        alias_file = model_dir / "aliases" / alias
        if not alias_file.exists():
            raise KeyError(f"Alias '{alias}' not set for model '{name}'")
        num_ver = alias_file.read_text(encoding="utf-8").strip()
        return self.get_version(name, self._normalize_version(num_ver))

    def get_version(self, name: str, version: str) -> ModelVersionRecord:
        norm_ver = self._normalize_version(version)
        num_ver = self._strip_v(norm_ver)
        ver_meta_json = self._models_root / name / f"version-{num_ver}" / "meta.json"
        if not ver_meta_json.exists():
            raise KeyError(f"Unknown model version {name}:{norm_ver}")
        payload = json.loads(ver_meta_json.read_text(encoding="utf-8"))
        return ModelVersionRecord(
            name=payload["name"],
            version=payload["version"],
            artifact_path=payload["source"],
            registered_sha256=payload["registered_sha256"],
            created_step=int(payload["created_step"]),
            status=payload.get("status", "READY"),
        )

    def list_versions(self, name: str) -> list[ModelVersionRecord]:
        model_dir = self._models_root / name
        if not model_dir.exists():
            return []
        records: list[ModelVersionRecord] = []
        for ver_dir in sorted(model_dir.glob("version-*")):
            meta_json = ver_dir / "meta.json"
            if meta_json.exists():
                payload = json.loads(meta_json.read_text(encoding="utf-8"))
                records.append(
                    ModelVersionRecord(
                        name=payload["name"],
                        version=payload["version"],
                        artifact_path=payload["source"],
                        registered_sha256=payload["registered_sha256"],
                        created_step=int(payload["created_step"]),
                        status=payload.get("status", "READY"),
                    )
                )
        return sorted(records, key=lambda r: int(self._strip_v(r.version)))

    def transition_log(self, name: str | None = None) -> list[TransitionEntry]:
        raw_list = json.loads(self._transitions_file.read_text(encoding="utf-8"))
        entries = [
            TransitionEntry(
                step=int(item["step"]),
                model_name=item["model_name"],
                alias=item["alias"],
                from_version=item["from_version"],
                to_version=item["to_version"],
            )
            for item in raw_list
        ]
        if name is None:
            return entries
        return [e for e in entries if e.model_name == name]

    def mutate_stored_artifact(self, name: str, version: str, new_bytes: bytes) -> Path:
        record = self.get_version(name, version)
        artifact_path = Path(record.artifact_path)
        artifact_path.write_bytes(new_bytes)
        return artifact_path


def run_indirection_demonstrations(
    registry: RegistryProtocol,
    work_dir: Path,
    seed: int = 42,
) -> list[IndirectionFailureDemo]:
    """Run the two registry indirection failure demos on any `RegistryProtocol` backend.

    Demo 1 (Alias Repointed Without Replica Reload):
      - `production` points at `v1`. A serving replica loads `v1`.
      - Registry repoints `production` -> `v2`.
      - The registry's alias metadata changes (`v1` -> `v2`), which the registry API can report,
        yet the registry API has zero visibility into the fact that the running replica still
        holds `v1` bytes in memory.

    Demo 2 (Artifact Mutated Under Unchanged Label):
      - `production` points at `v2`.
      - The underlying artifact file for `v2` is overwritten/mutated in storage (or a replica
        loads the mutated artifact under label `v2`).
      - Querying the registry API for `resolve_alias("fraud-detector", "production")` still
        returns `version="v2"` and `status="READY"`. The registry metadata did not change at all.
      - Only content-addressed attestation comparing the canonical hash against the replica's
        in-process loaded hash detects the mutation.
    """
    from attestation.weights import (
        InProcessLoader,
        generate_seeded_weights,
        mutate_single_byte,
        write_weights_file,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    loader = InProcessLoader()
    model_name = "fraud-detector"

    w_v1 = generate_seeded_weights(seed=seed, size_bytes=2048)
    w_v2 = generate_seeded_weights(seed=seed + 1, size_bytes=2048)
    f_v1 = write_weights_file(work_dir / "src_v1.bin", w_v1)
    f_v2 = write_weights_file(work_dir / "src_v2.bin", w_v2)

    rec_v1 = registry.register_version(model_name, f_v1, step=1, version_label="v1")
    rec_v2 = registry.register_version(model_name, f_v2, step=2, version_label="v2")
    registry.set_alias(model_name, "production", "v1", step=2)

    # Replica loads v1 when production == v1
    replica_loaded_v1 = loader.load(Path(rec_v1.artifact_path), claimed_version="v1", load_step=3)

    # Direction 1: Alias repointed to v2 while replica still serves v1 in memory
    registry.set_alias(model_name, "production", "v2", step=4)
    resolved_after_repoint = registry.resolve_alias(model_name, "production")
    demo_repoint = IndirectionFailureDemo(
        scenario="alias_repointed_without_reload",
        queried_alias="production",
        registry_reported_version=resolved_after_repoint.version,
        registry_reported_status=resolved_after_repoint.status,
        registry_metadata_changed=(resolved_after_repoint.version != rec_v1.version),
        canonical_intended_sha256=rec_v2.registered_sha256,
        actual_bytes_on_storage_sha256=compute_bytes_sha256(
            Path(resolved_after_repoint.artifact_path).read_bytes()
        ),
        replica_in_memory_sha256=replica_loaded_v1.loaded_sha256,
        # Registry API knows the pointer moved to v2, but cannot query replica memory
        registry_api_detects_discrepancy=False,
        attestation_detects_discrepancy=(
            replica_loaded_v1.loaded_sha256 != rec_v2.registered_sha256
        ),
    )

    # Direction 2: Artifact mutated in place under unchanged version label v2
    mutated_v2_bytes = mutate_single_byte(w_v2, offset=17)
    mutated_path = registry.mutate_stored_artifact(model_name, "v2", mutated_v2_bytes)
    resolved_after_mutation = registry.resolve_alias(model_name, "production")
    replica_loaded_mutated = loader.load(mutated_path, claimed_version="v2", load_step=5)

    demo_mutation = IndirectionFailureDemo(
        scenario="content_mutated_under_unchanged_label",
        queried_alias="production",
        registry_reported_version=resolved_after_mutation.version,
        registry_reported_status=resolved_after_mutation.status,
        registry_metadata_changed=(
            resolved_after_mutation.version != resolved_after_repoint.version
            or resolved_after_mutation.status != resolved_after_repoint.status
        ),
        canonical_intended_sha256=rec_v2.registered_sha256,
        actual_bytes_on_storage_sha256=compute_bytes_sha256(mutated_path.read_bytes()),
        replica_in_memory_sha256=replica_loaded_mutated.loaded_sha256,
        # Registry pointer and status still say v2 / READY
        registry_api_detects_discrepancy=False,
        attestation_detects_discrepancy=(
            replica_loaded_mutated.loaded_sha256 != rec_v2.registered_sha256
        ),
    )

    return [demo_repoint, demo_mutation]
