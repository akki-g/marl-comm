"""Held-out MAPDN channel interventions with immutable, completely identified banks.

An episode is a split, absolute profile window, and independent environment seed.
The same actor is evaluated on every arm. Only the sender channel changes;
diagnostics and audit state never enter the actor. Physical outcomes are explicitly
conditional on successful solves, alongside failure counts over *all* attempts.
This evaluator does not choose checkpoints, budgets, or favorable episode subsets.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
from tensordict import TensorDict
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from commstudy.communication.base import CommModule
from commstudy.communication.channel import (
    CommChannel, channel_intervention, channel_phase, preserve_channel_rng,
)
from commstudy.experiments.provenance import source_fingerprint
from commstudy.tasks.torchrl.mapdn_data import TrainingNormalizer, load_split_manifest
from commstudy.tasks.torchrl.power_grids import METRICS, TaskConfig, _make_adapter
from commstudy.utils.rng import preserve_rng_state


SCHEMA = "mapdn_paired_heldout_v1"
PACKAGES = ("torch", "torchrl", "tensordict", "benchmarl", "numpy", "pandas",
            "pandapower", "pettingzoo", "gymnasium")
NONPHYSICAL = frozenset({
    "destroy", "action_solver_failure", "advance_solver_failure", "pv_curtailed_mw",
    "feasible_step", "reward", "voltage_metrics_valid",
})


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _contract(config: dict, source_root: Path, *, manifest: dict) -> dict:
    import mapdn

    config = asdict(TaskConfig(**config))
    if config["max_steps"] != manifest["max_steps"] or config["history"] != manifest["history"]:
        raise ValueError("Bank horizon/history differs from the declared manifest")
    if not config["normalization_path"]:
        raise ValueError("Held-out scientific evaluation requires a training-fitted normalizer")
    normalizer = TrainingNormalizer.load(config["normalization_path"])
    if normalizer.to_dict()["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Bank normalizer differs from its split manifest")
    settings = {key: value for key, value in config.items()
                if key not in {"data_path", "manifest_path", "normalization_path"}}
    if manifest["settings"] != settings:
        raise ValueError("Bank task settings differ from training normalization settings")
    source_root = source_root.resolve()
    vendor = source_root / "vendor/mapdn-source"
    if Path(mapdn.__file__).resolve().parent != vendor / "mapdn":
        raise ValueError("Imported MAPDN is not the declared source-root vendor")
    if source_root != _source_root():
        raise ValueError("Imported evaluator is not in the declared source root")
    vendor_files = {p.relative_to(vendor).as_posix(): file_sha(p)
                    for p in sorted((vendor / "mapdn").rglob("*.py"))}
    return {
        "task": "mapdn_voltage_control", "task_config": config,
        "dynamics": "mapdn_aligned_transition_v1", "manifest_sha256": manifest["sha256"],
        "normalizer_sha256": normalizer.sha256, "data": manifest["data"],
        "normalizer_file_sha256": file_sha(config["normalization_path"]),
        "source_root": str(source_root), "source_sha256": source_fingerprint(source_root),
        "vendor_sha256": canonical_sha(vendor_files),
        "vendor_origin_sha256": file_sha(vendor / "COMMSTUDY_SOURCE_ORIGIN.json"),
        "runtime": {"python": platform.python_version(), "system": platform.system(),
                    "machine": platform.machine(),
                    "packages": {name: version(name) for name in PACKAGES}},
        "actor_input_keys": ["observation"], "group": "agents",
        "exploration": "DETERMINISTIC", "actor_diagnostics": "excluded",
        "measurement_noise": config["measurement_noise"],
        "reset_action": config["reset_action"], "history_initialization": "zero_padded",
    }


def _entries(manifest: dict, entries: list[dict], split: str) -> list[dict]:
    if split not in {"validation", "test"}:
        raise ValueError("Paired MAPDN evaluation requires an explicit held-out split")
    if not entries:
        raise ValueError("A held-out bank cannot be empty")
    starts = set(manifest["splits"][split]["starts"])
    lo, hi = manifest["splits"][split]["block"]
    horizon, history = manifest["max_steps"], manifest["history"]
    result = []
    for entry in entries:
        if set(entry) != {"episode_id", "start_row", "seed"}:
            raise ValueError("Episode entry requires exactly episode_id, start_row, seed")
        if not isinstance(entry["episode_id"], str) or not entry["episode_id"].strip():
            raise ValueError("Episode IDs must be nonempty strings")
        for key in ("start_row", "seed"):
            if type(entry[key]) is not int or entry[key] < 0:
                raise ValueError(f"Episode {key} must be a nonnegative integer")
        row = entry["start_row"]
        if row not in starts or not lo <= row - history + 1 < row + horizon + 1 <= hi:
            raise ValueError("Episode footprint is outside its split or permitted start rows")
        result.append({**entry, "split": split, "horizon": horizon, "history": history,
                       "footprint": [row - history + 1, row + horizon + 1]})
    for key in ("episode_id", "start_row", "seed"):
        if len({row[key] for row in result}) != len(result):
            raise ValueError(f"Duplicate episode {key} in bank")
    footprints = sorted(row["footprint"] for row in result)
    if any(left[1] > right[0] for left, right in zip(footprints, footprints[1:], strict=False)):
        raise ValueError("Held-out episode footprints overlap")
    return result


def build_mapdn_bank(config: dict, entries: list[dict], *, split: str, purpose: str,
                     source_root: str | Path | None = None) -> dict:
    """Freeze a bank *before* evaluating it; no environment or policy is run here.

    The caller selects a fresh calibration or locked test bank prospectively.
    This function never reuses/selects historical D5 episode banks by itself.
    """
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("Declare the bank's calibration or locked-test purpose")
    manifest = load_split_manifest(config["manifest_path"], data_path=config["data_path"])
    episodes = _entries(manifest, entries, split)
    contract = _contract(config, Path(source_root) if source_root else _source_root(),
                         manifest=manifest)
    contract_sha = canonical_sha(contract)
    for episode in episodes:
        episode["contract_sha256"] = contract_sha
        episode["identity_sha256"] = canonical_sha(episode)
    value = {"schema": SCHEMA, "purpose": purpose, "split": split,
             "contract": contract, "contract_sha256": contract_sha, "episodes": episodes}
    return {**value, "sha256": canonical_sha(value)}


def validate_mapdn_bank(bank: dict, config: dict) -> None:
    """Reject edits, stale runtimes, wrong task/data/normalizers, or invalid row banks."""
    entries = [{key: row[key] for key in ("episode_id", "start_row", "seed")}
               for row in bank["episodes"]]
    expected = build_mapdn_bank(
        config, entries, split=bank["split"], purpose=bank["purpose"],
        source_root=bank["contract"]["source_root"])
    if expected != bank:
        raise ValueError("Bank identity, source, data, normalizer, task, or runtime changed")


def actor_sha(policy: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("Actor state must contain finite tensors")
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_native_checkpoint(experiment: Any, path: str | Path, *, expected_frames: int) -> dict:
    """Verify the actor actually used by the rollout against the native MAPPO file.

    All saved actor/critic state is validated and restored exactly; the actor
    must already match the managed/in-memory actor, so this cannot silently
    select a different checkpoint. The caller owns checkpoint selection.
    """
    before = actor_sha(experiment.policy)
    digest = file_sha(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("state", {}).get("total_frames") != expected_frames:
        raise ValueError("Native checkpoint does not have the declared frame count")
    if {key for key in saved if key.startswith("loss_")} != {
        f"loss_{group}" for group in experiment.losses
    }:
        raise ValueError("Native checkpoint loss groups differ")
    counts = {}
    for group, policy in experiment.group_policies.items():
        if not any(module is policy for module in experiment.policy.modules()):
            raise ValueError("Native parity policy is not the rollout actor")
        state = saved[f"loss_{group}"]
        current = experiment.losses[group].state_dict()
        if set(state) != set(current) or not any("critic" in key for key in state):
            raise ValueError("Native actor/critic state keys differ")
        for key, expected in current.items():
            actual = state[key]
            if isinstance(expected, torch.Tensor):
                if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or (
                    actual.dtype != expected.dtype or not torch.isfinite(actual).all()
                ):
                    raise ValueError(f"Native tensor is unreadable/nonfinite: {group}/{key}")
            elif type(actual) is not type(expected) or actual != expected:
                raise ValueError("Native metadata differs")
        actor = {key.removeprefix("actor_network_params."): value for key, value in state.items()
                 if key.startswith("actor_network_params.") and isinstance(value, torch.Tensor)}
        if not actor or set(actor) != set(policy.state_dict()):
            raise ValueError("Native actor keys differ from the rollout actor")
        if any(not torch.equal(value.cpu(), policy.state_dict()[key].detach().cpu())
               for key, value in actor.items()):
            raise ValueError("Native actor differs from the selected rollout actor")
        counts[group] = len(actor)
    for group, loss in experiment.losses.items():
        loss.load_state_dict(saved[f"loss_{group}"], strict=True)
        for key, value in loss.state_dict().items():
            expected = saved[f"loss_{group}"][key]
            equal = torch.equal(value.detach().cpu(), expected.cpu()) if isinstance(
                value, torch.Tensor) else value == expected
            if not equal:
                raise ValueError(f"Native state did not restore exactly: {group}/{key}")
    if actor_sha(experiment.policy) != before or file_sha(path) != digest:
        raise ValueError("Native restoration changed the selected actor or checkpoint")
    return {"path": str(Path(path).resolve()), "sha256": digest, "frames": expected_frames,
            "actor_sha256": before, "actor_tensor_counts": counts,
            "actor_critic_restored": True, "native_actor_matches_rollout": True}


@contextmanager
def _isolated_policy(policy, seed):
    modules = [m for m in policy.modules() if isinstance(m, CommModule)]
    channels = [m for m in policy.modules() if isinstance(m, CommChannel)]
    modes = [(m, m.training) for m in policy.modules()]
    stats = [(m, deepcopy(m._stat_sums), deepcopy(m._stat_counts),
              getattr(m, "_stats_transition_weight", None)) for m in modules]
    try:
        with (preserve_rng_state(seed), preserve_channel_rng(channels, seed=seed),
              channel_phase("evaluation"), torch.no_grad(),
              set_exploration_type(ExplorationType.DETERMINISTIC)):
            policy.eval()
            for module in modules:
                module.reset_stats()
                module._stats_transition_weight = 1
            yield channels, modules
    finally:
        for module, sums, counts, weight in stats:
            module._stat_sums, module._stat_counts = sums, counts
            if weight is None:
                delattr(module, "_stats_transition_weight")
            else:
                module._stats_transition_weight = weight
        for module, mode in modes:
            module.training = mode


@contextmanager
def _preserve_observers(experiment):
    """Managed forward hooks must not charge offline evaluation to collection."""
    names = ("sums", "counts", "transitions", "agent_transitions")
    saved = []
    for callback in getattr(experiment, "callbacks", ()):
        for observer in getattr(callback, "_comm_observers", {}).values():
            saved.append((observer, {name: deepcopy(getattr(observer, name)) for name in names}))
    try:
        yield
    finally:
        for observer, state in saved:
            for name, value in state.items():
                setattr(observer, name, value)


def _arrays_sha(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        value = np.asarray(value)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite observation or audited physical input")
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _exogenous(env) -> dict:
    raw = env._env
    net = raw.powergrid
    return {"row": int(raw._row), "inputs_sha256": _arrays_sha([
        net.sgen.p_mw.to_numpy(), net.load.p_mw.to_numpy(), net.load.q_mvar.to_numpy()]),
        "private_rng_sha256": canonical_sha(raw._rng.bit_generator.state)}


def _policy_actions(policy, observations, agents):
    values = np.stack([observations[a] for a in agents])
    _arrays_sha([values])
    parameters = list(policy.parameters())
    device = parameters[0].device if parameters else torch.device("cpu")
    tensor = torch.as_tensor(values, device=device)
    td = TensorDict({"agents": TensorDict({"observation": tensor}, [len(agents)])}, [])
    actions = policy(td).get(("agents", "action")).detach().cpu().numpy()
    if actions.shape != (len(agents), 1) or not np.isfinite(actions).all():
        raise ValueError("Actor must produce one finite scalar per inverter")
    return {a: actions[i].copy() for i, a in enumerate(agents)}


def domain_summary(episodes: list[dict]) -> dict:
    """Counts each team transition once; never turns failed physics into zero violation."""
    transitions = [step for episode in episodes for step in episode["transitions"]]
    observed = [s for s in transitions if s["diagnostics"]["voltage_metrics_valid"] == 1]
    means, denominators = {}, {}
    for name in METRICS:
        if name in {"valid", "reward_row", "observation_row"}:
            continue
        selected = transitions if name in NONPHYSICAL else observed
        means[name] = float(np.mean([s["diagnostics"][name] for s in selected])) if (
            selected) else None
        denominators[name] = len(selected)
    bus_count = sum(s["diagnostics"]["voltage_bus_count"] for s in observed)
    violating = sum(s["diagnostics"]["voltage_violating_bus_count"] for s in observed)
    attempts = len(transitions)
    failures = sum(s["diagnostics"]["destroy"] for s in transitions)
    return {"episode_count": len(episodes), "transition_count": attempts,
            "planned_transition_count": sum(e["horizon"] for e in episodes),
            "unattempted_transition_count": sum(e["horizon"] for e in episodes) - attempts,
            "full_horizon_episodes": sum(len(e["transitions"]) == e["horizon"] and
                                         not e["terminated"] for e in episodes),
            "mean_episode_return": float(np.mean([e["return"] for e in episodes])),
            "domain_means": means, "domain_denominators": denominators,
            "solver_failure_count": failures,
            "solver_failure_fraction_all_attempts": failures / attempts if attempts else None,
            "voltage_observed_transitions": len(observed),
            "bus_transition_count": bus_count, "violated_bus_transition_count": violating,
            "violated_bus_fraction_observed": violating / bus_count if bus_count else None,
            "physical_metric_scope": "conditional on valid action solve; failures kept separately",
            "domain_denominator": "team transition once, never repeated agent copies"}


def _arm_mask(arm: str, n_agents: int):
    if arm == "live":
        return True
    if arm == "severed":
        return False
    if arm.startswith("suppress_sender:"):
        try:
            sender = int(arm.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("Sender intervention requires an integer agent index") from exc
        if not 0 <= sender < n_agents or arm != f"suppress_sender:{sender}":
            raise ValueError("Sender intervention is outside declared agent order")
        mask = torch.ones(n_agents, dtype=torch.bool)
        mask[sender] = False
        return mask
    raise ValueError(f"Unsupported intervention arm {arm!r}")


def _roll_episode(policy, config, entry, arm, *, env=None):
    owns_env = env is None
    with _isolated_policy(policy, entry["seed"]) as (channels, modules):
        if owns_env:
            env = _make_adapter(config, seed=entry["seed"], split=entry["split"])
        try:
            observations, info = env.reset(seed=entry["seed"], options={
                "start_row": entry["start_row"]})
            agents = list(env.possible_agents)
            if any(info.values()):
                raise ValueError("MAPDN reset unexpectedly exposes diagnostics to agents")
            initial_observation_sha = _arrays_sha([observations[a] for a in agents])
            reset_physics_sha = _arrays_sha([
                env._env.powergrid.sgen.q_mvar.to_numpy(),
                env._env.powergrid.res_bus.vm_pu.to_numpy(),
                env._env.powergrid.res_bus.va_degree.to_numpy()])
            transitions = []
            mask = _arm_mask(arm, len(agents))
            with channel_intervention(channels, mask):
                for step in range(entry["horizon"]):
                    exogenous = _exogenous(env)
                    if exogenous["row"] != entry["start_row"] + step:
                        raise ValueError("MAPDN data row differs from the declared episode")
                    actions = _policy_actions(policy, observations, agents)
                    observations, rewards, terminated, truncated, info = env.step(actions)
                    diagnostics = {name: float(value)
                                   for name, value in env.get_diagnostics().items()}
                    if not set(METRICS) <= set(diagnostics) or not all(
                        np.isfinite(v) for v in diagnostics.values()
                    ):
                        raise ValueError("Missing/nonfinite per-transition domain diagnostics")
                    if diagnostics["valid"] != 1 or (
                        diagnostics["voltage_metrics_valid"] not in {0, 1}
                    ):
                        raise ValueError("MAPDN diagnostic validity mask is invalid")
                    reward = float(rewards[agents[0]])
                    if not np.isfinite(reward) or any(
                        value != reward for value in rewards.values()
                    ):
                        raise ValueError("MAPDN requires one finite shared team reward")
                    if any(info.values()) or diagnostics["reward_row"] != exogenous["row"]:
                        raise ValueError("Diagnostics leaked or action/reward rows differ")
                    if len(set(terminated.values())) != 1 or len(set(truncated.values())) != 1:
                        raise ValueError("MAPDN agents disagree about episode termination")
                    term, trunc = all(terminated.values()), all(truncated.values())
                    if term != bool(diagnostics["destroy"]) or (term and trunc):
                        raise ValueError("Solver failure and termination flags disagree")
                    next_exogenous = _exogenous(env)
                    expected_next_row = exogenous["row"] + int(not term)
                    if diagnostics["observation_row"] != expected_next_row or (
                        next_exogenous["row"] != expected_next_row
                    ):
                        raise ValueError("Next observation violates aligned terminal-row contract")
                    transitions.append({"step": step, "exogenous": exogenous,
                                        "next_exogenous": next_exogenous,
                                        "actions": [actions[a].tolist() for a in agents],
                                        "reward": reward, "diagnostics": diagnostics,
                                        "terminated": term, "truncated": trunc})
                    if term or trunc:
                        break
            if not transitions or not (term or trunc):
                raise ValueError("Declared MAPDN evaluation episode did not complete")
            if trunc and len(transitions) != entry["horizon"]:
                raise ValueError("MAPDN truncated before its declared horizon")
            costs = [module.communication_stats() for module in modules]
            return {**entry, "arm": arm, "initial_observation_sha256": initial_observation_sha,
                    "reset_physics_sha256": reset_physics_sha, "agent_order": agents,
                    "terminated": term, "truncated": trunc,
                    "return": sum(s["reward"] for s in transitions),
                    "transition_count": len(transitions), "transitions": transitions,
                    "communication_stats_by_module": costs}
        finally:
            if owns_env:
                env.close()


def _pair(left: dict, right: dict) -> dict:
    for key in ("identity_sha256", "initial_observation_sha256", "reset_physics_sha256",
                "agent_order"):
        if left[key] != right[key]:
            raise ValueError(f"MAPDN paired reset/episode identity differs: {key}")
    shared = min(len(left["transitions"]), len(right["transitions"]))
    if any(a["exogenous"] != b["exogenous"] for a, b in zip(
        left["transitions"][:shared], right["transitions"][:shared], strict=True
    )):
        raise ValueError("MAPDN paired exogenous inputs/private RNG differ")
    if any(a["next_exogenous"] != b["next_exogenous"] for a, b in zip(
        left["transitions"][:shared], right["transitions"][:shared], strict=True
    ) if not a["terminated"] and not b["terminated"]):
        raise ValueError("MAPDN paired next/terminal exogenous inputs differ")
    lm, rm = domain_summary([left]), domain_summary([right])
    metric_deltas = {key: lm["domain_means"][key] - rm["domain_means"][key]
                     if lm["domain_means"][key] is not None and
                     rm["domain_means"][key] is not None else None
                     for key in lm["domain_means"]}
    return {"episode_id": left["episode_id"], "identity_sha256": left["identity_sha256"],
            "contrast": f"live_minus_{right['arm']}", "shared_exogenous_steps": shared,
            "return_delta": left["return"] - right["return"],
            "domain_mean_deltas": metric_deltas,
            "both_full_horizon_without_solver_failure": not left["terminated"] and
                not right["terminated"],
            "exact_action_and_outcome_null": left["transitions"] == right["transitions"]}


def evaluate_mapdn_paired(experiment: Any, bank: dict, *, arms=("live", "severed"),
                          require_exact_local_null: bool = False) -> dict:
    """Evaluate a caller-selected initial/final actor on one explicit held-out bank.

    RNGs, channel interventions/stats, module training modes, and actor weights
    are preserved. A new simulator is created per arm; every episode is reset
    to its declared row/seed with physics/history restored. Training and managed
    evaluation environments are never stepped or reset. A failed
    reset aborts the evidence collection (no retries or replacement rows).
    """
    if not arms or arms[0] != "live" or len(set(arms)) != len(arms):
        raise ValueError("Declare unique arms starting with live")
    if require_exact_local_null and "severed" not in arms:
        raise ValueError("An exact local null requires a severed arm")
    if set(experiment.group_policies) != {"agents"} or set(experiment.group_map) != {"agents"}:
        raise ValueError("MAPDN paired evaluation requires exactly the agents group")
    config = asdict(TaskConfig(**dict(experiment.task.config)))
    validate_mapdn_bank(bank, config)
    count = bank["contract"]["data"]["files"]["pv_active.csv"]["n_columns"]
    if list(experiment.group_map["agents"]) != [f"agent_{i}" for i in range(count)]:
        raise ValueError("MAPDN actor group order differs from the declared inverter order")
    for arm in arms:
        _arm_mask(arm, count)
    policy = experiment.policy
    before = actor_sha(policy)
    outcomes = {arm: [] for arm in arms}
    try:
        with preserve_rng_state(), _preserve_observers(experiment):
            for arm in arms:
                with preserve_rng_state(seed=bank["episodes"][0]["seed"]):
                    env = _make_adapter(config, seed=bank["episodes"][0]["seed"],
                                        split=bank["split"])
                try:
                    for entry in bank["episodes"]:
                        outcomes[arm].append(_roll_episode(policy, config, entry, arm, env=env))
                finally:
                    env.close()
        paired = {arm: [_pair(left, right) for left, right in zip(
            outcomes["live"], rows, strict=True)]
            for arm, rows in outcomes.items() if arm != "live"}
        exact_null = all(pair["exact_action_and_outcome_null"]
                         for rows in paired.values() for pair in rows) if paired else None
        if require_exact_local_null and not exact_null:
            raise ValueError("Local control failed exact paired action/domain null")
        validate_mapdn_bank(bank, config)
        return {"schema": SCHEMA, "bank_sha256": bank["sha256"], "bank": deepcopy(bank),
                "actor_sha256": before, "arms": outcomes,
                "summaries": {arm: domain_summary(rows) for arm, rows in outcomes.items()},
                "paired": paired, "exact_local_null_required": require_exact_local_null,
                "exact_action_and_outcome_null": exact_null,
                "pairing_checks_passed": True,
                "interpretation": "episode pairs within a seed; not independent trained policies",
                "physical_contrast_scope": "conditional on observed physics; check failure counts",
                "relative_return_saliency": "not computed: reward offsets make ratios unstable"}
    finally:
        if actor_sha(policy) != before:
            raise ValueError("Paired evaluation changed actor parameters or buffers")
