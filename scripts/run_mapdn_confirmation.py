"""Prepare or execute the fixed two-seed case33 bounded learning confirmation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback

import numpy as np
import torch

from commstudy.analysis.diagnostics import load_frozen_experiment
from commstudy.experiments.bookkeeping import (
    RunContext, atomic_write_json, read_json, task_runtime_contract, utc_now,
)
from commstudy.experiments.config import (
    ExperimentSpec, load_experiment_spec, resolved_experiment_dict, scientific_config_sha256,
)
from commstudy.experiments.runner import run_managed_experiment
from commstudy.tasks.torchrl.mapdn_data import (
    TrainingNormalizer, build_split_manifest, load_split_manifest, save_split_manifest,
)
from commstudy.tasks.torchrl.power_grids import TaskConfig
from run_mapdn_smoke import (
    SmokeEvidence, _actor_parameters, _actor_state, _roll_bank, _sha,
    _source_provenance, _state_sha,
)


PROTOCOL_ID = "mapdn_case33_bounded_confirmation_v1"
SEEDS = (30, 31)
FRAMES, HORIZON, EPISODES = 8192, 239, 16
PACKAGE_NAMES = ("torch", "torchrl", "tensordict", "benchmarl", "numpy", "pandas",
                 "pandapower", "pettingzoo", "gymnasium")


def _canonical_sha(value):
    content = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(
        content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _runtime():
    threads = {name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    if torch.get_num_threads() != 1 or any(value != "1" for value in threads.values()):
        raise ValueError("Scientific preparation/launch requires every declared thread setting = 1")
    return {"python": sys.version, "executable": sys.executable,
            "torch_num_threads": torch.get_num_threads(),
            "thread_environment": threads,
            "versions": {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}}


def _sources(root):
    import mapdn

    vendor = root / "vendor" / "mapdn-source"
    if Path(mapdn.__file__).resolve().parent != vendor / "mapdn":
        raise ValueError("Confirmation must import the isolated vendor MAPDN package")
    return {**_source_provenance(root), "confirmation_script_sha256": _sha(__file__),
            "mapdn_origin_sha256": _sha(vendor / "COMMSTUDY_SOURCE_ORIGIN.json")}


def _fixed_bank(manifest, split, seed):
    starts = manifest["splits"][split]["starts"]
    if len(starts) < EPISODES:
        raise ValueError("A sixteen-episode bank needs sixteen distinct start rows")
    expected = [starts[i * (len(starts) - 1) // (EPISODES - 1)] for i in range(EPISODES)]
    if any(right < left + HORIZON + 1
           for left, right in zip(expected[:-1], expected[1:], strict=True)):
        raise ValueError("Fixed bank episode footprints overlap")
    return [{"episode_id": f"{split}:{i}", "start_row": row, "seed": seed + i}
            for i, row in enumerate(expected)]


def _spec(root, config, seed):
    initial = load_experiment_spec(root / "configs", [
        "task=mapdn_voltage_control", "algorithm=mappo", "model=comm_identity", f"seed={seed}"])
    spec = replace(initial, task_config={"params": config, "return_groups": ["agents"]},
                   experiment={**initial.experiment,
                       "max_n_frames": FRAMES, "max_n_iters": None,
                       "sampling_device": "cpu", "train_device": "cpu", "buffer_device": "cpu",
                       "parallel_collection": False, "on_policy_n_envs_per_worker": 1,
                       "on_policy_collected_frames_per_batch": 256,
                       "on_policy_minibatch_size": 64, "on_policy_n_minibatch_iters": 4,
                       "lr": 5e-5, "adam_eps": 1e-6, "adam_extra_kwargs": {},
                       "gamma": 0.99, "clip_grad_norm": True, "clip_grad_val": 5.0,
                       "evaluation": False, "evaluation_static": True,
                       "evaluation_episodes": EPISODES, "evaluation_interval": FRAMES,
                       "evaluation_deterministic_actions": True,
                       "checkpoint_interval": 4096, "checkpoint_at_end": True,
                       "keep_checkpoints_num": None, "render": False})
    return ExperimentSpec(**resolved_experiment_dict(spec))


def bank_metrics(summary, expected_bank, n_buses, n_inverters=None):
    """Reject incomplete banks before comparing means; frequency uses integer counts."""
    issues = []
    if summary.get("bank") != expected_bank:
        issues.append("episode bank does not match the declared rows/seeds")
    episodes = summary.get("episodes", [])
    if len(episodes) != EPISODES or summary.get("complete_episodes") != EPISODES:
        issues.append("not exactly sixteen complete episodes")
    transitions = []
    for index, episode in enumerate(episodes):
        steps = episode.get("transitions", [])
        if len(steps) != HORIZON or episode.get("transition_count") != HORIZON:
            issues.append(f"episode {index} did not complete the full horizon")
        if not episode.get("complete"):
            issues.append(f"episode {index} is incomplete")
        if index < len(expected_bank):
            for key in ("episode_id", "start_row", "seed"):
                if episode.get(key) != expected_bank[index][key]:
                    issues.append(f"episode {index} {key} differs from its declaration")
        for step_index, step in enumerate(steps):
            diagnostics = step.get("diagnostics", {})
            healthy = {"feasible_step": 1, "valid": 1, "voltage_metrics_valid": 1,
                       "destroy": 0, "action_solver_failure": 0, "advance_solver_failure": 0}
            if any(diagnostics.get(key) != value for key, value in healthy.items()):
                issues.append(f"episode {index} step {step_index} is infeasible or unobserved")
            row = episode.get("start_row", -1) + step_index
            if (diagnostics.get("reward_row") != row
                    or diagnostics.get("observation_row") != row + 1):
                issues.append(f"episode {index} step {step_index} has incorrect temporal alignment")
            if step.get("step") != step_index or step.get("terminated") or (
                bool(step.get("truncated")) != (step_index == HORIZON - 1)
            ):
                issues.append(f"episode {index} step {step_index} has incorrect terminal flags")
            for name in ("voltage_violation_magnitude", "percentage_of_v_out_of_control", "q_loss"):
                value = diagnostics.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    issues.append(f"episode {index} step {step_index} has invalid {name}")
            transitions.append(step)
    if len(transitions) != EPISODES * HORIZON:
        issues.append("full bank transition denominator is not 3824")
    if issues:
        return {"valid": False, "issues": issues, "transitions": len(transitions)}
    counts, action_values = [], []
    for step in transitions:
        diagnostics = step["diagnostics"]
        count = diagnostics.get("voltage_violating_bus_count")
        if diagnostics.get("voltage_bus_count") != n_buses or not isinstance(count, (int, float)):
            return {"valid": False, "issues": ["explicit bus counts missing or changed"]}
        if not math.isfinite(count) or count != int(count) or not 0 <= count <= n_buses:
            return {"valid": False, "issues": ["violation frequency is not an integer bus count"]}
        frequency = diagnostics["percentage_of_v_out_of_control"]
        if abs(frequency - count / n_buses) > 1e-12:
            return {"valid": False, "issues": ["reported voltage frequency differs from counts"]}
        counts.append(int(count))
        actions = np.asarray(step.get("actions"), dtype=np.float64)
        if n_inverters is None and actions.ndim == 2:
            n_inverters = len(actions)
        if actions.shape != (n_inverters, 1) or not np.isfinite(actions).all():
            return {"valid": False, "issues": ["saved action shape/finiteness differs"]}
        action_values.extend(actions[:, 0].tolist())
    actions = np.abs(np.asarray(action_values))
    saturated = int((actions / 0.8 > 0.999).sum())
    return {"valid": True, "issues": [], "transitions": len(transitions),
            "voltage_magnitude": math.fsum(
                step["diagnostics"]["voltage_violation_magnitude"] for step in transitions
            ) / len(transitions),
            "q_effort": math.fsum(step["diagnostics"]["q_loss"] for step in transitions
                                  ) / len(transitions),
            "violated_bus_count": sum(counts), "bus_transition_count": n_buses * len(transitions),
            "violation_frequency": sum(counts) / (n_buses * len(transitions)),
            "action_diagnostics": {
                "physical_bounds": [-0.8, 0.8], "saturation_rule": "abs(action)/0.8 > 0.999",
                "saturated_action_scalars": saturated, "action_scalar_denominator": len(actions),
                "denominator_formula": f"{len(transitions)} transitions * {n_inverters} inverters",
                "physical_bound_saturation_fraction": saturated / len(actions),
                "mean_abs_action": float(actions.mean()), "max_abs_action": float(actions.max())}}


def domain_gate(initial, final, expected_bank, n_buses, n_inverters=None):
    before, after = (bank_metrics(item, expected_bank, n_buses, n_inverters)
                     for item in (initial, final))
    result = {"passed": False, "branch": None, "initial": before, "final": after}
    if not before["valid"] or not after["valid"]:
        result["reason"] = "Initial and final banks must both be complete and entirely feasible"
    elif before["voltage_magnitude"] > 0:
        result["branch"] = "voltage_reduction"
        result["passed"] = (after["voltage_magnitude"] <= 0.95 * before["voltage_magnitude"]
                            and after["violated_bus_count"] <= before["violated_bus_count"])
        result["reason"] = "Require at least 5% voltage-magnitude reduction and nonworse frequency"
    elif before["violated_bus_count"] == 0 and before["q_effort"] > 0:
        result["branch"] = "safe_reactive_effort_reduction"
        result["passed"] = (after["q_effort"] <= 0.95 * before["q_effort"]
                            and after["voltage_magnitude"] == 0
                            and after["violated_bus_count"] == 0)
        result["reason"] = "Require 5% reactive-effort reduction and continued zero violations"
    else:
        result["reason"] = "No available prespecified positive improvement target"
    return result


def prepare(data_path, output, root):
    import pandapower as pp

    runtime = _runtime()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources = _sources(root)
    plan = root / "docs" / "MAPDN_CONFIRMATION_PLAN.md"
    (output / "frozen_plan.md").write_bytes(plan.read_bytes())
    config = asdict(TaskConfig(data_path=str(data_path),
                              manifest_path=str(output / "split_manifest.json"),
                              max_steps=HORIZON, history=1, measurement_noise=False,
                              reset_action=False))
    network = pp.from_pickle(data_path / "model.p")
    dimensions = {"buses": len(network.bus), "inverters": len(network.sgen),
                  "loads": len(network.load)}
    manifest = build_split_manifest(
        data_path, HORIZON, history=1, expected_n_pv=dimensions["inverters"],
        expected_n_loads=dimensions["loads"],
        source_revision=f"vendor-tree-sha256:{sources['mapdn_tree_sha256']}",
        settings={key: value for key, value in config.items()
                  if key not in {"data_path", "manifest_path", "normalization_path"}},
    )
    save_split_manifest(manifest, output / "split_manifest.json")
    banks = {split: _fixed_bank(manifest, split, seed) for split, seed in
             (("train", 100000), ("validation", 200000), ("test", 300000))}
    _write_new_json(output / "banks.json", banks)
    print("Collecting the fixed sixteen training-only normalization episodes", flush=True)
    collection, observations = _roll_bank(
        config, banks["train"], split="train", retain_observations=True)
    _write_new_json(output / "normalization_collection.json", collection)
    if not bank_metrics(collection, banks["train"], dimensions["buses"], dimensions["inverters"])[
        "valid"
    ]:
        raise RuntimeError("Normalization collection did not complete its healthy fixed bank")
    normalizer = TrainingNormalizer.fit(
        np.stack(observations), split="train", manifest_sha256=manifest["sha256"])
    del observations
    if normalizer.sample_count != EPISODES * (HORIZON + 1) * dimensions["inverters"]:
        raise ValueError("Training-only normalization observation count differs")
    normalizer.save(output / "normalizer.json")
    config["normalization_path"] = str(output / "normalizer.json")
    files = ["frozen_plan.md", "split_manifest.json", "banks.json", "normalizer.json",
             "normalization_collection.json"]
    specifications = {}
    for seed in SEEDS:
        spec = _spec(root, config, seed)
        name = f"spec_seed{seed}.json"
        _write_new_json(output / name, asdict(spec))
        files.append(name)
        specifications[str(seed)] = scientific_config_sha256(spec)
    if sources != _sources(root) or _sha(plan) != _sha(output / "frozen_plan.md"):
        raise ValueError("Source or proposed plan changed during preparation")
    packet = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PREPARED",
        "prepared_utc": utc_now(), "root": str(root), "output": str(output),
        "data_path": str(data_path), "allocation": list(SEEDS), "frames_per_seed": FRAMES,
        "max_steps": HORIZON, "episodes_per_bank": EPISODES,
        "feeder_dimensions": dimensions, "sources": sources, "runtime": runtime,
        "manifest_sha256": manifest["sha256"], "normalizer_sha256": normalizer.sha256,
        "scientific_config_sha256": specifications,
        "files": {name: _sha(output / name) for name in files},
        "preparation_seconds": time.monotonic() - started,
        "download_receipt_sha256": (_sha(data_path / "download_receipt.json")
                                    if (data_path / "download_receipt.json").is_file() else None),
        "test_rollouts_performed": 0, "scientific_learning_confirmed": False,
    }
    packet["sha256"] = _canonical_sha(packet)
    if _runtime() != runtime:
        raise ValueError("Runtime changed during preparation")
    _write_new_json(output / "preparation.json", packet)
    print(json.dumps({"prepared": str(output / "preparation.json"), "sha256": packet["sha256"]}),
          flush=True)
    return 0


def verify_preparation(output, root, expected_sha256):
    packet = read_json(output / "preparation.json")
    if packet.get("sha256") != expected_sha256 or packet.get("sha256") != _canonical_sha(packet):
        raise ValueError("Preparation content hash differs")
    if packet.get("protocol_id") != PROTOCOL_ID or packet.get("allocation") != list(SEEDS):
        raise ValueError("Preparation protocol/allocation differs")
    if packet["root"] != str(root) or packet["output"] != str(output):
        raise ValueError("Frozen source/artifact paths moved; prepare a new explicit protocol")
    if (packet["frames_per_seed"], packet["max_steps"], packet["episodes_per_bank"]) != (
        FRAMES, HORIZON, EPISODES
    ):
        raise ValueError("Frozen allocation dimensions differ from the prescribed protocol")
    if packet["sources"] != _sources(root) or packet["runtime"] != _runtime():
        raise ValueError("Frozen source/runtime differs")
    expected_files = {"frozen_plan.md", "split_manifest.json", "banks.json", "normalizer.json",
                      "normalization_collection.json", *(f"spec_seed{seed}.json" for seed in SEEDS)}
    if set(packet["files"]) != expected_files:
        raise ValueError("Frozen preparation file set differs")
    if packet["files"]["frozen_plan.md"] != _sha(root / "docs" / "MAPDN_CONFIRMATION_PLAN.md"):
        raise ValueError("Current confirmation plan differs from the frozen review packet")
    for name, digest in packet["files"].items():
        if _sha(output / name) != digest:
            raise ValueError(f"Frozen artifact differs: {name}")
    manifest = load_split_manifest(output / "split_manifest.json", packet["data_path"])
    if manifest["sha256"] != packet["manifest_sha256"]:
        raise ValueError("Manifest content hash differs from frozen preparation")
    normalizer = TrainingNormalizer.load(output / "normalizer.json")
    if (normalizer.sha256 != packet["normalizer_sha256"]
            or normalizer.manifest_sha256 != manifest["sha256"]
            or normalizer.sample_count != EPISODES * (HORIZON + 1)
            * packet["feeder_dimensions"]["inverters"]):
        raise ValueError("Frozen normalizer provenance or exact training sample count differs")
    config = asdict(TaskConfig(
        data_path=packet["data_path"], manifest_path=str(output / "split_manifest.json"),
        normalization_path=str(output / "normalizer.json"), max_steps=HORIZON, history=1,
        measurement_noise=False, reset_action=False))
    for seed in SEEDS:
        spec = ExperimentSpec(**read_json(output / f"spec_seed{seed}.json"))
        if scientific_config_sha256(spec) != packet["scientific_config_sha256"][str(seed)]:
            raise ValueError(f"Resolved scientific configuration differs for seed {seed}")
        intended = _spec(root, config, seed)
        if scientific_config_sha256(spec) != scientific_config_sha256(intended):
            raise ValueError(f"Frozen seed {seed} differs from declared optimization settings")
    receipt = Path(packet["data_path"]) / "download_receipt.json"
    if packet["download_receipt_sha256"] != (_sha(receipt) if receipt.is_file() else None):
        raise ValueError("Data download receipt differs")
    return packet


class InitialCheckpointEvidence(SmokeEvidence):
    def __init__(self, config, out_dir, preparation_sha):
        super().__init__(config, [], out_dir)
        self.preparation_sha = preparation_sha
        self.initial_manifest = None

    def on_setup(self):
        self.initial_parameters = _actor_parameters(self.experiment)
        path = self.out_dir / "initial_actor.pt"
        state = {group: {name: value.detach().cpu().clone()
                         for name, value in policy.state_dict().items()}
                 for group, policy in self.experiment.group_policies.items()}
        with path.open("xb") as stream:
            torch.save({"group_policies": state}, stream)
        restored = torch.load(path, map_location="cpu", weights_only=False)["group_policies"]
        if set(restored) != set(state) or any(
            set(restored[group]) != set(values) or any(
                not torch.equal(value, restored[group][name]) for name, value in values.items())
            for group, values in state.items()
        ):
            raise ValueError("Initial actor checkpoint failed exact read-back")
        self.initial_manifest = {
            "seed": self.experiment.seed, "before_training_collection": True,
            "preparation_sha256": self.preparation_sha, "file_sha256": _sha(path),
            "actor_state_sha256": _state_sha(_actor_state(self.experiment)),
            "task_runtime_contract": task_runtime_contract(self.experiment),
        }
        _write_new_json(self.out_dir / "initial_actor.json", self.initial_manifest)


def _inventory(source):
    result = {}
    for path in sorted(source.rglob("*")):
        name = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ValueError(f"Archive refuses symlinks: {name}")
        if path.is_dir():
            result[name] = {"kind": "directory"}
        elif path.is_file():
            result[name] = {"kind": "file", "sha256": _sha(path)}
        else:
            raise ValueError(f"Archive refuses special entries: {name}")
    return result


def _archive(source, destination):
    inventory = _inventory(source)
    with tarfile.open(destination, "x:gz") as archive:
        for name in inventory:
            archive.add(source / name, arcname=name, recursive=False)
    with tarfile.open(destination, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        if set(members) != set(inventory):
            raise ValueError("Archive member set differs")
        for name, entry in inventory.items():
            if entry["kind"] == "directory":
                if not members[name].isdir():
                    raise ValueError(f"Archive directory type differs: {name}")
                continue
            if not members[name].isfile():
                raise ValueError(f"Archive file type differs: {name}")
            stream = archive.extractfile(members[name])
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Archive read-back differs for {name}")
    if _inventory(source) != inventory:
        raise ValueError("Source inventory changed during archive")
    return {"path": str(destination), "sha256": _sha(destination), "inventory": inventory,
            "readback_verified": True}


def _validate_native_state(saved, losses, frames, *, require_equal=False):
    if saved.get("state", {}).get("total_frames") != frames:
        raise ValueError("Native checkpoint frame provenance differs")
    groups = {f"loss_{group}" for group in losses}
    if {key for key in saved if key.startswith("loss_")} != groups:
        raise ValueError("Native checkpoint actor/critic groups differ")
    for group, loss in losses.items():
        expected, actual = loss.state_dict(), saved[f"loss_{group}"]
        if set(actual) != set(expected):
            raise ValueError("Native checkpoint actor/critic state keys differ")
        if not any("actor" in key for key in actual) or not any("critic" in key for key in actual):
            raise ValueError("Native checkpoint does not contain both actor and critic")
        for name, tensor in actual.items():
            if isinstance(expected[name], torch.Tensor):
                if not isinstance(tensor, torch.Tensor) or (
                    tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError(f"Unreadable/nonfinite native actor/critic state {name}")
                if require_equal and not torch.equal(tensor, expected[name].detach().cpu()):
                    raise ValueError(f"Final native checkpoint differs from final model: {name}")
            elif type(tensor) is not type(expected[name]) or tensor != expected[name]:
                raise ValueError(f"Native actor/critic metadata differs: {name}")


def _native_checkpoints(experiment, run_dir):
    result = {}
    for frames in (4096, 8192):
        paths = list(run_dir.glob(f"benchmarl/**/checkpoints/checkpoint_{frames}.pt"))
        if len(paths) != 1:
            raise ValueError(f"Require exactly one retained native checkpoint at {frames} frames")
        saved = torch.load(paths[0], map_location="cpu", weights_only=False)
        _validate_native_state(saved, experiment.losses, frames, require_equal=frames == FRAMES)
        result[str(frames)] = {"path": str(paths[0]), "sha256": _sha(paths[0]),
                               "actor_critic_readable_finite": True,
                               "exact_completed_model": frames == FRAMES}
    return result


def _train_one(output, root, packet, seed):
    spec = ExperimentSpec(**read_json(output / f"spec_seed{seed}.json"))
    evidence = output / "evidence" / f"seed{seed}"
    evidence.mkdir(parents=True, exist_ok=False)
    callback = InitialCheckpointEvidence(spec.task_config["params"], evidence, packet["sha256"])
    context = RunContext(suite_id=PROTOCOL_ID, run_id=f"identity_seed{seed}",
                         output_root=output / "runs", command=tuple(sys.argv))
    started = time.monotonic()
    experiment = run_managed_experiment(spec, context, repo_root=root, callbacks=[callback])
    try:
        final = _actor_parameters(experiment)
        initial = callback.initial_parameters
        if initial is None or set(initial) != set(final):
            raise ValueError("Initial/final actor parameter groups differ")
        changed = any(not torch.equal(initial[name], value) for name, value in final.items())
        finite = all(torch.isfinite(value).all().item() for value in final.values())
        checks = {"exact_frames": callback.collected_frames == FRAMES,
                  "exact_batches": callback.batch_counts == [256] * (FRAMES // 256),
                  "actor_changed": changed, "actor_finite": finite,
                  "initial_actor_retained": (
                      _sha(evidence / "initial_actor.pt")
                      == callback.initial_manifest["file_sha256"])}
        native = _native_checkpoints(experiment, context.run_dir)
        summary = {"seed": seed, "run_dir": str(context.run_dir), "evidence_dir": str(evidence),
                   "checks": checks, "integrity_passed": all(checks.values()),
                   "frames": callback.collected_frames, "initial_actor": callback.initial_manifest,
                   "final_actor_state_sha256": _state_sha(_actor_state(experiment)),
                   "native_checkpoints": native,
                   "elapsed_seconds": time.monotonic() - started,
                   "frames_per_second_including_setup": FRAMES / (time.monotonic() - started)}
    finally:
        experiment.close()
    if not summary["integrity_passed"]:
        _write_new_json(evidence / "training_summary.json", summary)
        raise RuntimeError(f"Training integrity failed for seed {seed}: {checks}")
    metadata = read_json(context.metadata_path)
    metadata.update(execution_purpose="bounded_learning_confirmation",
                    preparation_sha256=packet["sha256"])
    atomic_write_json(context.metadata_path, metadata)
    archive_dir = output / "archives"
    archive_dir.mkdir(exist_ok=True)
    summary["managed_run_archive"] = _archive(
        context.run_dir, archive_dir / f"managed_seed{seed}.tar.gz")
    summary["pretest_initial_and_training_archive"] = _archive(
        evidence, archive_dir / f"initial_and_training_seed{seed}.tar.gz")
    _write_new_json(evidence / "training_summary.json", summary)
    return summary


def _measure_seed(output, row, banks, packet):
    seed = row["seed"]
    run_dir, evidence = Path(row["run_dir"]), Path(row["evidence_dir"])
    config = read_json(output / f"spec_seed{seed}.json")["task_config"]["params"]
    results, provenance = {}, {}
    for label in ("final", "initial"):
        experiment = load_frozen_experiment(
            run_dir, scratch_root=output / "scratch" / f"{seed}_{label}")
        try:
            if _state_sha(_actor_state(experiment)) != row["final_actor_state_sha256"]:
                raise ValueError("Strictly reloaded final actor differs from training completion")
            provenance[label] = dict(experiment.analysis_provenance)
            if label == "initial":
                initial_path = evidence / "initial_actor.pt"
                if _sha(initial_path) != row["initial_actor"]["file_sha256"]:
                    raise ValueError("Registered initial actor checkpoint changed")
                checkpoint = torch.load(initial_path, map_location="cpu", weights_only=False)
                states = checkpoint["group_policies"]
                if set(states) != set(experiment.group_policies):
                    raise ValueError("Initial actor policy groups differ")
                for group, state in states.items():
                    experiment.group_policies[group].load_state_dict(state, strict=True)
                if (_state_sha(_actor_state(experiment))
                        != row["initial_actor"]["actor_state_sha256"]):
                    raise ValueError("Initial actor did not restore exactly")
                provenance[label]["actor_override"] = {
                    "purpose": "registered pretraining actor", "path": str(initial_path),
                    "sha256": row["initial_actor"]["file_sha256"]}
            for split in ("validation", "test"):
                verify_preparation(output, Path(packet["root"]), packet["sha256"])
                result, _ = _roll_bank(config, banks[split], split=split, policy=experiment.policy)
                verify_preparation(output, Path(packet["root"]), packet["sha256"])
                _write_new_json(evidence / f"{label}_{split}.json", result)
                results[f"{label}_{split}"] = result
            if label == "final":
                verify_preparation(output, Path(packet["root"]), packet["sha256"])
                replay, _ = _roll_bank(
                    config, banks["validation"][:2], split="validation", policy=experiment.policy)
                verify_preparation(output, Path(packet["root"]), packet["sha256"])
                _write_new_json(evidence / "final_validation_replay.json", replay)
                replay_equal = replay["episodes"] == results["final_validation"]["episodes"][:2]
        finally:
            experiment.close()
    gate = domain_gate(results["initial_test"], results["final_test"], banks["test"],
                       packet["feeder_dimensions"]["buses"],
                       packet["feeder_dimensions"]["inverters"])
    result = {"seed": seed, "domain_gate": gate, "final_validation_replay_exact": replay_equal,
              "reload_provenance": provenance,
              "passed": gate["passed"] and replay_equal and row["integrity_passed"]}
    _write_new_json(evidence / "domain_decision.json", result)
    return result


def execute(output, root, expected_sha256):
    if (output / "launch.json").exists():
        raise FileExistsError("This allocation already launched; no overwrite or automatic retry")
    packet = verify_preparation(output, root, expected_sha256)
    _write_new_json(output / "launch.json", {"launched_utc": utc_now(),
                    "preparation_sha256": packet["sha256"], "allocation": list(SEEDS)})
    started = time.monotonic()
    rows, outcomes = [], []
    stage = "training"
    try:
        for seed in SEEDS:
            verify_preparation(output, root, expected_sha256)
            print(f"Training fixed seed {seed} for {FRAMES} frames", flush=True)
            rows.append(_train_one(output, root, packet, seed))
            if _sources(root) != packet["sources"]:
                raise ValueError("Source changed during the fixed allocation")
        _write_new_json(output / "training_complete.json", {
            "completed_utc": utc_now(), "rows": rows, "test_rollouts_performed": 0})
        stage = "heldout_evaluation"
        verify_preparation(output, root, expected_sha256)
        banks = read_json(output / "banks.json")
        for row in rows:
            outcomes.append(_measure_seed(output, row, banks, packet))
        verify_preparation(output, root, expected_sha256)
        stage = "retention"
        receipt = _archive(
            output / "evidence", output / "archives" / "confirmation_evidence.tar.gz")
        if _sources(root) != packet["sources"] or _runtime() != packet["runtime"]:
            raise ValueError("Source/runtime changed during evaluation")
        report = {"protocol_id": PROTOCOL_ID, "preparation_sha256": packet["sha256"],
                  "verdict": ("CONFIRMED" if all(row["passed"] for row in outcomes)
                              else "NOT_CONFIRMED"),
                  "scope": "bounded two-seed real-case33 domain-learning confirmation",
                  "training": rows, "outcomes": outcomes, "evidence_archive": receipt,
                  "elapsed_seconds": time.monotonic() - started, "finished_utc": utc_now()}
        report["scientific_learning_confirmed"] = report["verdict"] == "CONFIRMED"
        _write_new_json(output / "report.json", report)
        print(json.dumps({"report": str(output / "report.json"), "verdict": report["verdict"]}),
              flush=True)
        return 0 if report["scientific_learning_confirmed"] else 1
    except BaseException as error:
        report = {"protocol_id": PROTOCOL_ID, "preparation_sha256": packet["sha256"],
                  "verdict": "NOT_CONFIRMED", "scientific_learning_confirmed": False,
                  "failure_stage": stage, "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "training": rows, "outcomes": outcomes,
                  "elapsed_seconds": time.monotonic() - started, "finished_utc": utc_now()}
        _write_new_json(output / "report.json", report)
        print(report["traceback"], file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", type=Path, help="Execute a previously reviewed frozen preparation")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--expected-preparation-sha256",
                        help="Required reviewed preparation digest when launching --run")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    if args.prepare_only:
        if (args.data_path is None or args.out_dir is None
                or args.expected_preparation_sha256 is not None):
            parser.error("--prepare-only requires --data-path and a new --out-dir")
        if not args.data_path.is_dir() or args.out_dir.exists():
            parser.error("Data directory must exist and output directory must be new")
        try:
            return prepare(args.data_path.resolve(), args.out_dir.resolve(), root)
        except BaseException as error:
            if args.out_dir.exists():
                _write_new_json(args.out_dir / "preparation_failure.json", {
                    "error_type": type(error).__name__, "error": str(error),
                    "traceback": traceback.format_exc(), "test_rollouts_performed": 0})
            raise
    if args.data_path is not None or args.out_dir is not None:
        parser.error("--run uses only the paths and settings bound in its frozen preparation")
    if args.expected_preparation_sha256 is None:
        parser.error("--run requires the independently reviewed --expected-preparation-sha256")
    return execute(args.run.resolve(), root, args.expected_preparation_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
