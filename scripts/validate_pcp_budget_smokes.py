"""Audit all eight saved D4 validation rows before comparison launch."""

from __future__ import annotations
import argparse
import dataclasses
import json
from pathlib import Path
import tempfile
from collections.abc import Mapping
import torch
from commstudy.analysis.confirmation import (
    COLLECTION_HEALTH_METRICS,
    TRAINING_HEALTH_METRICS,
    _read_metrics,
    _require_series,
    _validate_online_domain,
    file_sha256,
)
from commstudy.analysis.diagnostics import rebuild_spec, load_frozen_experiment
from commstudy.experiments.bookkeeping import atomic_write_json
from commstudy.experiments.config import resolved_experiment_dict, scientific_config_sha256
from commstudy.experiments.protocols import content_sha256, load_protocol, protocol_spec
from commstudy.experiments.provenance import source_fingerprint


def need(condition, detail):
    if not condition:
        raise ValueError(f"Smoke evidence rejected: {detail}")


def validate_policy_checkpoint(run, metadata, scratch):
    path = run / "checkpoints/policy_state.pt"
    before = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    need(isinstance(checkpoint, Mapping), "Final policy checkpoint must be a mapping.")
    for field, expected in {
        "schema_version": 1, "frames": 6000, "run_id": run.name,
        "suite_id": run.parent.name, "task": "vmas_predator_capture_prey", "algorithm": "mappo",
        "model": metadata["model"], "seed": 10,
    }.items():
        need(type(checkpoint.get(field)) is type(expected) and checkpoint[field] == expected,
             f"Final policy checkpoint identity differs: {field}.")
    policies = checkpoint.get("group_policies")
    need(isinstance(policies, Mapping) and set(policies) == {"adversary", "agent"},
         "Final policy must retain exactly both trained groups.")
    experiment = load_frozen_experiment(run, scratch_root=scratch)
    counts = {}
    try:
        for group, saved in policies.items():
            loaded = experiment.group_policies[group].state_dict()
            need(isinstance(saved, Mapping) and set(saved) == set(loaded),
                 f"Saved/loaded policy state keys differ: {group}.")
            counts[group] = 0
            for key, value in saved.items():
                actual = loaded[key]
                if isinstance(value, torch.Tensor):
                    saved_bytes = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                    loaded_bytes = (actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                                    if isinstance(actual, torch.Tensor) else None)
                    need(isinstance(actual, torch.Tensor) and value.dtype == actual.dtype
                         and value.shape == actual.shape and bool(torch.isfinite(value).all())
                         and torch.equal(saved_bytes, loaded_bytes),
                         f"Saved/loaded tensor is invalid or differs: {group}/{key}.")
                    counts[group] += 1
                else:
                    need((key.endswith(".__batch_size") and type(value) is torch.Size
                          and type(actual) is torch.Size and value == actual)
                         or (key.endswith(".__device") and value is None and actual is None),
                         f"Saved/loaded metadata differs: {group}/{key}.")
            need(counts[group] > 0, f"Final policy contains no tensors for {group}.")
    finally:
        experiment.close()
    need(file_sha256(path) == before, "Final policy checkpoint changed during validation.")
    return {"sha256": file_sha256(path), "strict_reload_verified": True,
            "finite_tensor_counts_by_group": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Refusing to replace a smoke review.")
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(args.protocol)
    source = source_fingerprint(root)
    runs = list(args.suite.glob("*/metadata.json"))
    expected = {(c["model"], c["ablation_value"]) for c in protocol["launch_scope"]["comparison"]}
    actual = [
        (json.loads(p.read_text())["model"], json.loads(p.read_text())["ablation_value"])
        for p in runs
    ]
    if len(runs) != 8 or len(set(actual)) != 8 or set(actual) != expected:
        parser.error("Require exactly the eight declared method/visibility smoke rows.")
    results = []
    for path in sorted(runs):
        run = path.parent
        input_paths = [run / name for name in (
            "metadata.json", "status.json", "resolved_config.yaml", "metrics.csv",
            "checkpoints/policy_state.pt",
        )]
        input_hashes = {str(p.resolve()): file_sha256(p) for p in input_paths}
        meta = json.loads(path.read_text())
        need(
            meta["status"]
            == json.loads((run / "status.json").read_text())["status"]
            == "completed",
            "Both saved status records must be completed.",
        )
        need(
            type(meta["seed"]) is int and meta["seed"] == 10
            and type(meta["frames"]) is int and meta["frames"] == 6000
            and type(meta["iterations"]) is int and meta["iterations"] == 1,
            "meta['seed'] == 10 and meta['frames'] == 6000 and (meta['iterations'] == 1)",
        )
        need(
            meta["execution_purpose"] == "validation" and meta["source_sha256"] == source,
            "meta['execution_purpose'] == 'validation' and meta['source_sha256'] == source",
        )
        need(meta["run_id"] == run.name and meta["suite_id"] == run.parent.name
             and meta["task"] == "vmas_predator_capture_prey" and meta["algorithm"] == "mappo"
             and meta["ablation"] == "visibility",
             "Saved smoke run identity differs from the declared condition.")
        spec = protocol_spec(
            protocol,
            model=meta["model"],
            seed=10,
            ablation="visibility",
            ablation_value=meta["ablation_value"],
        )
        spec = dataclasses.replace(spec, experiment={**spec.experiment, "max_n_frames": 6000})
        actual_spec, expected_spec = (
            resolved_experiment_dict(rebuild_spec(run)),
            resolved_experiment_dict(spec),
        )
        for value in (actual_spec, expected_spec):
            value["experiment"].pop("save_folder", None)
        need(actual_spec == expected_spec, "actual_spec == expected_spec")
        need(
            meta["scientific_config_sha256"] == scientific_config_sha256(spec),
            "meta['scientific_config_sha256'] == scientific_config_sha256(spec)",
        )
        need(
            meta["candidate_reference"]["protocol_sha256"] == content_sha256(protocol),
            "meta['candidate_reference']['protocol_sha256'] == content_sha256(protocol)",
        )
        need(meta["candidate_reference"]["protocol_id"] == protocol["protocol_id"]
             and meta["candidate_reference"]["condition"] == {
                 "model": meta["model"], "ablation": "visibility",
                 "ablation_value": meta["ablation_value"],
             }, "Candidate reference condition differs from the saved row.")
        need(
            meta["versions"]
            == {"python": protocol["runtime"]["python"], **protocol["runtime"]["packages"]},
            "Saved software versions differ from the protocol.",
        )
        need(
            all(
                
                    meta["runtime"][key] == "cpu"
                    for key in ("sampling_device", "train_device", "buffer_device")
                
            ),
            "All execution devices must be CPU.",
        )
        need(meta["runtime"]["torch_num_threads"] == 1, "meta['runtime']['torch_num_threads'] == 1")
        params = meta["parameters"]
        identity = meta["model"] == "pcp_comm_identity"
        need(
            params["group/adversary/actor_total"] == (19332 if identity else 27524),
            "params['group/adversary/actor_total'] == (19332 if identity else 27524)",
        )
        need(
            params["group/adversary/communication_total"] == (0 if identity else 8192),
            "params['group/adversary/communication_total'] == (0 if identity else 8192)",
        )
        need(
            params["group/adversary/critic_total"] == 19585,
            "params['group/adversary/critic_total'] == 19585",
        )
        need(params["group/agent/actor_total"] == 156, "params['group/agent/actor_total'] == 156")
        need(params["group/agent/critic_total"] == 193, "params['group/agent/critic_total'] == 193")
        contract = meta["task_runtime_contract"]
        need(
            contract["actor_observation_shapes"]["adversary"] == [3, 17],
            "contract['actor_observation_shapes']['adversary'] == [3, 17]",
        )
        need(contract["critic_state"]["shape"] == [22], "contract['critic_state']['shape'] == [22]")
        need(
            contract["predator_sensing_radius"]
            == (1.0 if meta["ablation_value"] == "radius1" else None),
            "Visibility radius differs from the declared condition.",
        )
        issues = []
        observations, counts = _read_metrics(run / "metrics.csv", issues)
        for group in ("adversary", "agent"):
            for phase, names in (
                ("training", TRAINING_HEALTH_METRICS),
                ("collection", COLLECTION_HEALTH_METRICS),
            ):
                for name in names:
                    values = _require_series(observations, issues, phase, group, name, [6000])
                    if name.endswith("_finite_fraction"):
                        need(values == {6000: 1.0}, "values == {6000: 1.0}")
        _validate_online_domain(observations, issues, [6000], [6000], 128, 100, 6000)
        need(not issues, issues)
        for phase in ("collection", "training", "evaluation"):
            for metric, full_budget in (
                ("comm_realized_sender_bits_per_step", 3072.0),
                ("comm_realized_bits_per_step", 6144.0),
            ):
                need(
                    observations[phase, "adversary", metric, 6000, ""][0]
                    == (full_budget if meta["model"] == "pcp_broadcast_k3" else 0.0),
                    "Communication payload differs from the declared budget.",
                )
        with tempfile.TemporaryDirectory(prefix="pcp-smoke-checkpoint-") as scratch:
            checkpoint = validate_policy_checkpoint(run, meta, Path(scratch))
        need(input_hashes == {str(p.resolve()): file_sha256(p) for p in input_paths},
             "Smoke artifacts changed during validation.")
        results.append(
            {
                "run_id": run.name,
                "model": meta["model"],
                "visibility": meta["ablation_value"],
                "checks_passed": True,
                "logged_scalars_by_group": counts,
                "parameters": params,
                "episodes": 128,
                "frames": 6000,
                "final_policy_checkpoint": checkpoint,
                "artifacts": input_hashes,
            }
        )
    need(source_fingerprint(root) == source, "Package source changed during validation.")
    need(load_protocol(args.protocol) == protocol, "Protocol changed during validation.")
    atomic_write_json(
        args.out,
        {
            "schema_version": 1,
            "checks_passed": True,
            "source_sha256": source,
            "protocol_sha256": content_sha256(protocol),
            "runs": results,
            "validation_training_frames": 48000,
            "evaluation_transitions": 102400,
            "claim": "Bounded implementation/runtime/replay evidence; no learning confirmation.",
        },
    )
    print(f"All {len(results)} condition smokes passed saved-evidence validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
