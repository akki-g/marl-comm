"""Run a bounded real-feeder MAPDN engineering smoke with strict checkpoint replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from benchmarl.experiment.callback import Callback
import numpy as np
from tensordict import TensorDict
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from commstudy.analysis.diagnostics import load_frozen_experiment
from commstudy.experiments.bookkeeping import (
    RunContext, atomic_write_json, read_json, utc_now,
)
from commstudy.experiments.config import load_experiment_spec
from commstudy.experiments.provenance import source_fingerprint
from commstudy.experiments.runner import run_managed_experiment
from commstudy.tasks.torchrl.mapdn_data import (
    TrainingNormalizer, build_split_manifest, save_split_manifest,
)
from commstudy.tasks.torchrl.power_grids import METRICS, TaskConfig, _make_adapter
from commstudy.utils.rng import preserve_rng_state


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance(root):
    vendor = root / "vendor" / "mapdn-source"
    files = {str(path.relative_to(vendor)): _sha(path)
             for path in sorted(vendor.rglob("*.py"))}
    tree_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"commstudy_sha256": source_fingerprint(root), "mapdn_tree_sha256": tree_sha,
            "mapdn_python_files": files, "smoke_script_sha256": _sha(__file__)}


def _bank(manifest, split, count, seed):
    rows = manifest["splits"][split]["starts"]
    if count > len(rows):
        raise ValueError(f"Requested {count} episodes but {split} has only {len(rows)} starts")
    positions = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [{"episode_id": f"{split}:{i}", "start_row": rows[int(position)], "seed": seed + i}
            for i, position in enumerate(positions)]


def _actor_parameters(experiment):
    return {f"{group}/{name}": value.detach().cpu().clone()
            for group, policy in experiment.group_policies.items()
            for name, value in policy.named_parameters()}


def _actor_state(experiment):
    return {f"{group}/{name}": value.detach().cpu().clone()
            for group, policy in experiment.group_policies.items()
            for name, value in policy.state_dict().items()}


def _state_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _policy_actions(policy, observations, agents):
    values = torch.as_tensor(np.stack([observations[agent] for agent in agents]))
    td = TensorDict({"agents": TensorDict({"observation": values}, [len(agents)])}, [])
    actions = policy(td).get(("agents", "action")).detach().cpu().numpy()
    if actions.shape != (len(agents), 1) or not np.isfinite(actions).all():
        raise ValueError("Actor must produce one finite scalar action per inverter")
    return {agent: actions[index].copy() for index, agent in enumerate(agents)}


def _domain_summary(transitions):
    nonphysical = {"destroy", "action_solver_failure", "advance_solver_failure", "pv_curtailed_mw",
                   "feasible_step", "reward", "voltage_metrics_valid"}
    observed = [step for step in transitions if step["diagnostics"]["voltage_metrics_valid"] > 0]
    means, denominators = {}, {}
    for name in METRICS:
        if name in {"valid", "reward_row", "observation_row"}:
            continue
        selected = transitions if name in nonphysical else observed
        means[name] = float(np.mean([step["diagnostics"][name] for step in selected])) if (
            selected) else None
        denominators[name] = len(selected)
    return means, denominators, len(observed)


def _roll_bank(config, bank, *, split, policy=None, retain_observations=False):
    """Every row/seed is fixed before rollout; no failed episode is replaced."""
    started = time.monotonic()
    rows = []
    observations_for_fit = []
    mode = None if policy is None else policy.training
    with preserve_rng_state(), torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        env = _make_adapter(config, seed=bank[0]["seed"], split=split)
        try:
            if policy is not None:
                policy.eval()
            for entry in bank:
                observations, _ = env.reset(seed=entry["seed"], options={
                    "start_row": entry["start_row"]})
                agents = list(env.possible_agents)
                if retain_observations:
                    observations_for_fit.append(np.stack([observations[a] for a in agents]))
                transitions = []
                for step in range(config["max_steps"]):
                    actions = ({agent: np.zeros(1, dtype=np.float32) for agent in agents}
                               if policy is None else _policy_actions(policy, observations, agents))
                    observations, rewards, terminated, truncated, _ = env.step(actions)
                    diagnostics = {name: float(value)
                                   for name, value in env.get_diagnostics().items()}
                    if not all(np.isfinite(value) for value in diagnostics.values()):
                        raise ValueError("Nonfinite per-transition diagnostic")
                    reward = float(rewards[agents[0]])
                    if any(value != reward for value in rewards.values()):
                        raise ValueError("MAPDN engineering smoke expects shared team rewards")
                    done = all(terminated[a] or truncated[a] for a in agents)
                    transitions.append({
                        "step": step, "actions": [actions[a].tolist() for a in agents],
                        "reward": reward, "terminated": all(terminated.values()),
                        "truncated": all(truncated.values()), "diagnostics": diagnostics,
                    })
                    if retain_observations:
                        observations_for_fit.append(np.stack([observations[a] for a in agents]))
                    if done:
                        break
                if not transitions or not done:
                    raise RuntimeError("A declared evaluation episode did not complete")
                rows.append({**entry, "complete": done,
                             "return": sum(step["reward"] for step in transitions),
                             "transition_count": len(transitions), "transitions": transitions})
        finally:
            if policy is not None:
                policy.train(mode)
            env.close()
    transitions = [step for row in rows for step in row["transitions"]]
    elapsed = time.monotonic() - started
    means, denominators, observed_count = _domain_summary(transitions)
    summary = {
        "split": split,
        "policy": "zero_reactive_action" if policy is None else "deterministic_actor",
        "bank": bank, "episodes": rows, "complete_episodes": len(rows),
        "transition_count": len(transitions), "mean_episode_return": float(np.mean([
            row["return"] for row in rows])), "elapsed_seconds": elapsed,
        "transitions_per_second_including_env_setup": len(transitions) / elapsed,
        "domain_means": means, "domain_denominators": denominators,
        "voltage_observed_transitions": observed_count,
        "domain_denominator": "team transitions once; physical means need voltage_metrics_valid",
        "solver_failure_count": sum(step["diagnostics"].get("destroy", 0) for step in transitions),
    }
    return summary, observations_for_fit


class SmokeEvidence(Callback):
    def __init__(self, config, bank, out_dir):
        super().__init__()
        self.config, self.bank, self.out_dir = config, bank, out_dir
        self.initial_parameters = None
        self.before = None
        self.collected_frames = 0
        self.validation = []
        self.batch_counts = []
        self.before_seconds = 0.0

    def on_setup(self):
        self.initial_parameters = _actor_parameters(self.experiment)
        self.before, _ = _roll_bank(
            self.config, self.bank, split="validation", policy=self.experiment.policy)
        self.before_seconds = self.before["elapsed_seconds"]
        atomic_write_json(self.out_dir / "heldout_before.json", self.before)

    def on_batch_collected(self, batch):
        valid = batch.get(("next", "agents", "info", "valid"))[..., 0, 0].reshape(-1)
        columns = {name: batch.get(("next", "agents", "info", name))[..., 0, 0].reshape(-1)
                   for name in METRICS}
        with (self.out_dir / "training_transitions.jsonl").open("a", encoding="utf-8") as stream:
            for index in range(len(valid)):
                if float(valid[index]) != 1:
                    raise ValueError("Collected transition is missing valid MAPDN diagnostics")
                self.collected_frames += 1
                record = {"frame": self.collected_frames, "split": "train",
                          "diagnostics": {name: float(values[index])
                                          for name, values in columns.items()}}
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.batch_counts.append(len(valid))

    def on_evaluation_end(self, rollouts):
        episodes = []
        for rollout in rollouts:
            done = rollout.get(("next", "done"))
            episodes.append({"complete": bool(done[-1].all()),
                             "transitions": int(rollout.numel())})
        self.validation.append({"split": "validation", "episodes": episodes})
        atomic_write_json(self.out_dir / "validation_evaluations.json", self.validation)


def _experiment_spec(root, config, args):
    spec = load_experiment_spec(root / "configs", overrides=[
        "task=mapdn_voltage_control", "algorithm=mappo", "model=comm_identity",
        f"seed={args.seed}"])
    return replace(spec, task_config={"params": config, "return_groups": ["agents"]}, experiment={
        **spec.experiment, "max_n_frames": args.frames, "max_n_iters": None,
        "sampling_device": "cpu", "train_device": "cpu", "buffer_device": "cpu",
        "parallel_collection": False, "on_policy_n_envs_per_worker": 1,
        "on_policy_collected_frames_per_batch": 64, "on_policy_minibatch_size": 64,
        "on_policy_n_minibatch_iters": 1, "evaluation": True, "evaluation_static": True,
        "evaluation_episodes": args.evaluation_episodes, "evaluation_interval": args.frames,
        "evaluation_deterministic_actions": True, "checkpoint_interval": args.frames,
        "checkpoint_at_end": True, "keep_checkpoints_num": None, "render": False,
    })


def run(args, root):
    started = time.monotonic()
    torch.set_num_threads(1)
    before_source = _source_provenance(root)
    data_path, out_dir = args.data_path.resolve(), args.out_dir.resolve()
    manifest_path, normalizer_path = out_dir / "split_manifest.json", out_dir / "normalizer.json"
    config = asdict(TaskConfig(data_path=str(data_path), manifest_path=str(manifest_path),
                              max_steps=args.max_steps, measurement_noise=False,
                              reset_action=False))
    print("Validating full feeder data and creating immutable temporal split manifest", flush=True)
    manifest = build_split_manifest(
        data_path, args.max_steps, config["history"],
        source_revision=f"vendor-tree-sha256:{before_source['mapdn_tree_sha256']}",
        settings={key: value for key, value in config.items()
                  if key not in {"data_path", "manifest_path", "normalization_path"}},
    )
    save_split_manifest(manifest, manifest_path)
    train_bank = _bank(manifest, "train", args.normalization_episodes, args.seed + 10000)
    validation_bank = _bank(manifest, "validation", args.evaluation_episodes, args.seed + 20000)
    atomic_write_json(out_dir / "declared_banks.json", {
        "normalization": train_bank, "heldout_before_after_reload": validation_bank,
        "selection": "equally spaced manifest starts; fixed before fitting; test split unused",
        "normalization_policy": "zero reactive action", "measurement_noise": False,
    })
    print("Collecting declared training-only normalization episodes", flush=True)
    normalization, observations = _roll_bank(
        config, train_bank, split="train", retain_observations=True)
    atomic_write_json(out_dir / "normalization_collection.json", normalization)
    normalizer = TrainingNormalizer.fit(
        np.stack(observations), split="train", manifest_sha256=manifest["sha256"])
    normalizer.save(normalizer_path)
    del observations
    config["normalization_path"] = str(normalizer_path)
    spec = _experiment_spec(root, config, args)
    context = RunContext(suite_id="mapdn_engineering_smoke", run_id=f"identity_seed{args.seed:03d}",
                         output_root=out_dir / "runs", command=tuple(sys.argv))
    callback = SmokeEvidence(config, validation_bank, out_dir)
    print("Running managed CPU MAPPO with before/after and strict reload checks", flush=True)
    training_started = time.monotonic()
    experiment = run_managed_experiment(
        spec, context, repo_root=root, callbacks=[callback], validation_run=True)
    managed_seconds = time.monotonic() - training_started
    try:
        final_parameters = _actor_parameters(experiment)
        initial = callback.initial_parameters
        if initial is None or set(initial) != set(final_parameters):
            raise ValueError("Initial/final actor parameter sets differ")
        if not all(torch.isfinite(value).all() for value in final_parameters.values()):
            raise ValueError("Training produced nonfinite actor parameters")
        changed = [name for name in initial
                   if not torch.equal(initial[name], final_parameters[name])]
        parameter_delta = max(float((final_parameters[name] - initial[name]).abs().max())
                              for name in initial)
        final_state = _actor_state(experiment)
        after, _ = _roll_bank(
            config, validation_bank, split="validation", policy=experiment.policy)
        atomic_write_json(out_dir / "heldout_after.json", after)
    finally:
        experiment.close()
    frozen = load_frozen_experiment(context.run_dir, scratch_root=out_dir / "reload_scratch")
    try:
        restored_state = _actor_state(frozen)
        state_equal = set(final_state) == set(restored_state) and all(
            torch.equal(value, restored_state[name]) for name, value in final_state.items())
        replay, _ = _roll_bank(config, validation_bank, split="validation", policy=frozen.policy)
        atomic_write_json(out_dir / "heldout_reloaded.json", replay)
        replay_equal = after["episodes"] == replay["episodes"]
        reload_provenance = frozen.analysis_provenance
    finally:
        frozen.close()
    source_unchanged = before_source == _source_provenance(root)
    metadata = read_json(context.metadata_path)
    metadata["execution_purpose"] = "engineering_smoke"
    metadata["scientific_learning_confirmed"] = False
    atomic_write_json(context.metadata_path, metadata)
    versions = {}
    for name in ("torch", "torchrl", "tensordict", "benchmarl", "numpy", "pandas",
                 "pandapower", "pettingzoo", "gymnasium"):
        versions[name] = importlib.metadata.version(name)
    validation_complete = bool(callback.validation) and all(
        len(event["episodes"]) == args.evaluation_episodes
        and all(episode["complete"] for episode in event["episodes"])
        for event in callback.validation)
    checks = {
        "actor_parameters_changed": bool(changed), "actor_parameters_finite": True,
        "exact_requested_frames_collected": callback.collected_frames == args.frames,
        "all_validation_episodes_complete": validation_complete,
        "strict_reloaded_actor_state_equal": state_equal,
        "exact_heldout_actions_rewards_diagnostics_replayed": replay_equal,
        "source_unchanged_during_run": source_unchanged,
    }
    receipt = data_path / "download_receipt.json"
    before = callback.before
    training_seconds = managed_seconds - callback.before_seconds
    report = {
        "schema_version": 1, "execution_purpose": "engineering_smoke", "finished_utc": utc_now(),
        "engineering_checks_passed": all(checks.values()), "checks": checks,
        "scientific_learning_confirmed": False,
        "interpretation": "Bounded integration/replay evidence; learning confirmation pending.",
        "run_dir": str(context.run_dir), "source": before_source,
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "torch_num_threads": torch.get_num_threads(),
                    "versions": versions, "pythonpath": os.environ.get("PYTHONPATH")},
        "data": {"path": str(data_path), "manifest_sha256": manifest["sha256"],
                 "normalizer_sha256": normalizer.sha256,
                 "download_receipt_sha256": _sha(receipt) if receipt.is_file() else None},
        "training": {"requested_frames": args.frames, "collected_frames": callback.collected_frames,
                     "batch_frame_counts": callback.batch_counts,
                     "changed_parameter_tensors": changed,
                     "initial_parameters_sha256": _state_sha(initial),
                     "final_parameters_sha256": _state_sha(final_parameters),
                     "max_abs_parameter_change": parameter_delta,
                     "managed_seconds_excluding_before_validation_bank": training_seconds,
                     "frames_per_second_including_setup_and_validation": (
                         args.frames / training_seconds)},
        "heldout": {"split": "validation", "bank": validation_bank,
                    "before_mean_return": before["mean_episode_return"],
                    "after_mean_return": after["mean_episode_return"],
                    "mean_return_change": (after["mean_episode_return"]
                                           - before["mean_episode_return"]),
                    "return_improved_on_smoke_bank": (
                        after["mean_episode_return"] > before["mean_episode_return"]),
                    "domain_mean_changes_after_minus_before": {
                        name: (None if value is None or before["domain_means"][name] is None
                               else value - before["domain_means"][name])
                        for name, value in after["domain_means"].items()},
                    "before_solver_failures": before["solver_failure_count"],
                    "after_solver_failures": after["solver_failure_count"]},
        "reload": {"provenance": reload_provenance, "final_state_sha256": _state_sha(final_state),
                   "restored_state_sha256": _state_sha(restored_state)},
        "total_seconds": time.monotonic() - started,
    }
    atomic_write_json(out_dir / "report.json", report)
    print(json.dumps({"report": str(out_dir / "report.json"), "checks": checks}, indent=2),
          flush=True)
    return 0 if report["engineering_checks_passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--normalization-episodes", type=int, default=2)
    parser.add_argument("--evaluation-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for name in ("max_steps", "frames", "normalization_episodes", "evaluation_episodes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.frames % 64:
        parser.error("--frames must be a multiple of the fixed 64-frame collector batch")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.out_dir.exists():
        parser.error("--out-dir must not exist; previous engineering evidence is preserved")
    if not args.data_path.is_dir():
        parser.error("--data-path must name the real feeder directory")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    try:
        return run(args, root)
    except BaseException as error:
        atomic_write_json(args.out_dir / "failure.json", {
            "execution_purpose": "engineering_smoke", "scientific_learning_confirmed": False,
            "finished_utc": utc_now(), "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
