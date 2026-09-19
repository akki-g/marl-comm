"""Run one preserved 6k validation batch for each declared D4 condition."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from commstudy.experiments.bookkeeping import (
    RunContext,
    atomic_write_json,
    make_run_id,
    read_json,
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(args.protocol)
    conditions = protocol["launch_scope"]["comparison"]
    if not 0 <= args.index < len(conditions):
        parser.error("Index must select one declared comparison condition.")
    mismatch = runtime_mismatches(protocol["runtime"])
    if mismatch:
        parser.error(f"Runtime mismatch: {mismatch}")
    condition = conditions[args.index]
    seed = 10
    spec = protocol_spec(protocol, seed=seed, **condition)
    spec = dataclasses.replace(spec, experiment={**spec.experiment, "max_n_frames": 6000})
    context = RunContext(
        suite_id=f"{protocol['protocol_id']}_runtime_smoke",
        run_id=make_run_id(
            task=spec.task,
            algorithm=spec.algorithm,
            model=spec.model,
            seed=seed,
            namespace=protocol["protocol_id"],
            ablation=condition["ablation"],
            ablation_value=condition["ablation_value"],
        ),
        output_root=root / "runs",
        ablation=condition["ablation"],
        ablation_value=condition["ablation_value"],
    )
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
            "condition": condition,
            "intentional_changes": {
                "experiment.max_n_frames": 6000,
                "seed": 10,
            },
            "purpose": "one-batch runtime validation; reserved comparison seeds remain unused",
        }
        atomic_write_json(context.metadata_path, metadata)
        print(context.run_dir)
    finally:
        experiment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
