"""Run exactly one 6,000-frame validation batch from a full PCP candidate spec."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from commstudy.experiments.bookkeeping import (
    RunContext,
    atomic_write_json,
    make_run_id,
    read_json,
    retry_context,
)
from commstudy.experiments.protocols import (
    content_sha256,
    expected_model_contract,
    load_protocol,
    protocol_spec,
    runtime_mismatches,
)
from commstudy.experiments.runner import run_managed_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--retry", action="store_true", help="Preserve and retry an existing smoke."
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(args.protocol)
    mismatch = runtime_mismatches(protocol["runtime"])
    if mismatch:
        parser.error(f"Candidate runtime mismatch: {mismatch}")
    seed = protocol["selection"]["confirmation_seeds"][0]
    spec = protocol_spec(protocol, model="pcp_comm_identity", seed=seed)
    spec = dataclasses.replace(spec, experiment={**spec.experiment, "max_n_frames": 6000})
    context = RunContext(
        suite_id=f"{protocol['protocol_id']}_runtime_smoke",
        run_id=make_run_id(
            task=spec.task,
            algorithm=spec.algorithm,
            model=spec.model,
            seed=seed,
            namespace=protocol["protocol_id"],
        ),
        output_root=args.output_root.resolve(),
    )
    if args.retry and context.run_dir.exists():
        context = retry_context(context)
    experiment = run_managed_experiment(
        spec,
        context,
        repo_root=root,
        validation_run=True,
        expected_contract=expected_model_contract(spec),
    )
    try:
        metadata = read_json(context.metadata_path)
        metadata["candidate_reference"] = {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": content_sha256(protocol),
            "intentional_changes": {"experiment.max_n_frames": 6000},
            "purpose": "one-batch runtime validation; not learning confirmation",
        }
        atomic_write_json(context.metadata_path, metadata)
        print(context.run_dir)
    finally:
        experiment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
