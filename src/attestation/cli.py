"""CLI entrypoint for running the self-attesting container demo and printing experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from attestation.auditor import AttestationAuditor
from attestation.experiments import format_experiment_tables, run_all_experiments
from attestation.fleet import FleetRolloutSimulator, RolloutSpec
from attestation.registry import MlflowRegistryAdapter
from attestation.weights import InProcessLoader, generate_seeded_weights, write_weights_file


def run_self_attestation_demo(seed: int = 20261117) -> dict[str, object]:
    """Boot an in-process serving replica, load seeded weights, and emit its own attestation."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reg = MlflowRegistryAdapter(root / "mlflow_store")
        w_v7 = generate_seeded_weights(seed=seed + 7, size_bytes=4096)
        f_v7 = write_weights_file(root / "v7.bin", w_v7)
        rec = reg.register_version("fraud-classifier", f_v7, step=1, version_label="v7")
        reg.set_alias("fraud-classifier", "production", "v7", step=1)

        spec = RolloutSpec(fleet_size=4, max_surge=1, max_unavailable=0, readiness_delay_steps=1)
        sim = FleetRolloutSimulator(spec, reg, "fraud-classifier", root / "sim", seed=seed)
        sim.bootstrap_initial_fleet("production")
        auditor = AttestationAuditor(spec)
        report = auditor.audit_fleet(
            records=sim.attest_serving_fleet(),
            current_step=sim.current_step,
            intended_version=rec.version,
            intended_sha256=rec.registered_sha256,
            previous_valid_hashes={rec.registered_sha256},
            rollout_start_step=None,
        )

        loader = InProcessLoader()
        self_loaded = loader.load(Path(rec.artifact_path), claimed_version="v7", load_step=1)
        return {
            "service": "model-serving-attestation",
            "registry_backend": reg.backend_name,
            "alias": "production",
            "claimed_version": self_loaded.claimed_version,
            "in_process_loaded_sha256": self_loaded.loaded_sha256,
            "in_memory_integrity_verified": self_loaded.verify_in_memory_hash(),
            "size_bytes": self_loaded.size_bytes,
            "fleet_audit_counts": report.counts_by_classification,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the container self-attestation check and print JSON receipt.",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run all 5 deterministic experiments and print tables.",
    )
    args = parser.parse_args()

    if args.experiments:
        results = run_all_experiments(seed=20261117)
        print(format_experiment_tables(results))
        return 0

    payload = run_self_attestation_demo(seed=20261117)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
