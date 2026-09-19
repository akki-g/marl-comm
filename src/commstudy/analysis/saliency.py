"""Frozen-policy channel reliance and same-input behavioural influence.

Reliance compares complete episodes from a declared seed bank under live and
severed channels. Influence instead evaluates two clones of each recorded
policy input, without stepping a simulator. A reliance effect is a property of
this trained policy, not evidence that communication-free control is impossible.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torchrl.envs import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from commstudy.communication.base import CommModule
from commstudy.communication.channel import (
    CommChannel,
    channel_intervention,
    channel_phase,
    preserve_channel_rng,
)
from commstudy.experiments.returns import resolve_return_groups
from commstudy.utils.rng import preserve_rng_state


@dataclass(frozen=True)
class EvaluationEpisode:
    episode_id: str
    seed: int


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_id: str
    seed: int
    return_value: float
    transitions: int
    terminated: bool
    truncated: bool
    complete: bool
    initial_input_sha256: str
    exogenous_episode_ids: tuple[int, ...] = ()
    exogenous_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class PairedEpisodeEvaluation:
    group: str
    exploration: str
    with_comm: tuple[EpisodeOutcome, ...]
    without_comm: tuple[EpisodeOutcome, ...]
    # Explicit trajectory-source bank for a subsequent same-input measurement.
    live_inputs: tuple[Any, ...] = field(repr=False)
    severed_inputs: tuple[Any, ...] = field(repr=False)


@dataclass(frozen=True)
class SameInputInfluence:
    group: str
    input_distribution: str
    input_count: int
    input_sha256: str
    input_keys: tuple[str, ...]
    exploration: str
    action_shift_mean: float
    action_shift_max: float
    policy_kl_mean: float | None
    seed: int


@dataclass(frozen=True)
class SaliencyResult:
    """Compatibility summary, with separately identified measurement contracts."""

    episodes: int
    steps: int
    exploration: str
    return_with_comm: float
    return_without_comm: float
    return_delta: float
    return_delta_fraction: float | None
    per_episode_delta_mean: float | None
    per_episode_delta_std: float | None
    action_shift_mean: float | None
    action_shift_max: float | None
    policy_kl_mean: float | None
    communicating_modules: int
    per_episode_returns_with: tuple[float, ...] = ()
    per_episode_returns_without: tuple[float, ...] = ()
    episode_ids: tuple[str, ...] = ()
    group: str = ""
    input_distribution: str = ""
    input_count: int = 0
    input_sha256: str = ""
    input_keys: tuple[str, ...] = ()
    complete_episodes: bool = False
    outcomes_with: tuple[EpisodeOutcome, ...] = ()
    outcomes_without: tuple[EpisodeOutcome, ...] = ()
    influence_inputs: tuple[Any, ...] = field(default=(), repr=False)

    def as_row(self) -> dict[str, Any]:
        names = (
            "episodes",
            "steps",
            "exploration",
            "return_with_comm",
            "return_without_comm",
            "return_delta",
            "return_delta_fraction",
            "per_episode_delta_mean",
            "per_episode_delta_std",
            "action_shift_mean",
            "action_shift_max",
            "policy_kl_mean",
            "communicating_modules",
            "group",
            "input_distribution",
            "input_count",
            "input_sha256",
            "complete_episodes",
        )
        row = {f"saliency_{name}": getattr(self, name) for name in names}
        row["saliency_schema_version"] = 2
        row["saliency_influence_exploration"] = "DETERMINISTIC"
        return row


def communication_modules(experiment: Any, group: str | None = None) -> list[CommModule]:
    """Distinct actor communication modules, optionally in one explicit group."""
    policies = (
        experiment.group_policies.values() if group is None else [experiment.group_policies[group]]
    )
    found = {}
    for policy in policies:
        for module in policy.modules():
            if isinstance(module, CommModule):
                found.setdefault(id(module), module)
    return list(found.values())


@contextmanager
def severed_communication(modules: Sequence[CommModule]) -> Iterator[None]:
    """Suppress sender contributions even when all-true replay masks are supplied.

    The original channels and their RNG are retained. Intervention masks are
    applied after ordinary availability, and restored even after an exception.
    Fully severed channels consume no random numbers.
    """
    channels = [module.channel for module in modules if hasattr(module, "channel")]
    with channel_intervention(channels, False):
        yield


def _channels(experiment: Any) -> list[CommChannel]:
    return list(
        {
            id(module): module
            for policy in experiment.group_policies.values()
            for module in policy.modules()
            if isinstance(module, CommChannel)
        }.values()
    )


@contextmanager
def _measurement_context(experiment: Any, seed: int):
    modules = communication_modules(experiment)
    stats = [(dict(m._stat_sums), dict(m._stat_counts)) for m in modules]
    try:
        with preserve_rng_state(seed), preserve_channel_rng(_channels(experiment), seed=seed):
            with channel_phase("evaluation"), torch.no_grad():
                yield
    finally:
        for module, (sums, counts) in zip(modules, stats, strict=True):
            module._stat_sums, module._stat_counts = sums, counts


def _measured_group(experiment: Any, groups: Sequence[str] | None) -> str:
    selected = resolve_return_groups(experiment.group_map, groups)
    if len(selected) != 1:
        raise ValueError("Intervention evaluation requires exactly one explicit measured group.")
    return selected[0]


def _episode_returns(
    experiment: Any, rollouts: Sequence[Any], groups: Sequence[str]
) -> list[float]:
    del experiment
    returns = []
    for rollout in rollouts:
        values = []
        for group in groups:
            reward = rollout.get(("next", group, "reward"), None)
            if reward is None:
                raise ValueError(f"Missing reward for measured group {group!r}.")
            if not bool(torch.isfinite(reward).all()):
                raise ValueError(f"Non-finite reward for measured group {group!r}.")
            values.append(float(reward.to(torch.float64).sum(0).mean()))
        if not values:
            raise ValueError("At least one measured group is required.")
        returns.append(sum(values) / len(values))
    return returns


def _hash_inputs(batches: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(str(tuple(batch.batch_size)).encode())
        for key in sorted(batch.keys(include_nested=True, leaves_only=True), key=str):
            value = batch.get(key)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Policy input {key!r} must be a tensor.")
            value = value.detach().cpu().contiguous()
            digest.update(str((key, value.dtype, tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _policy_inputs(experiment: Any, batch: Any, group: str) -> Any:
    policy = experiment.group_policies[group]
    keys = list(policy.in_keys)
    available = set(batch.keys(include_nested=True, leaves_only=True))
    # Context supplied by a task or rollout may not be in the actor's spec.
    # Remove generated-marker/output leaves: this fixed recorded availability
    # is deliberately authoritative in BOTH influence arms.
    for module in policy.modules():
        for key in getattr(module, "comm_context_keys", {}).values():
            if key in available and key not in keys:
                keys.append(key)
    missing = set(policy.in_keys) - available
    if missing:
        raise ValueError(f"Missing declared policy inputs: {sorted(missing, key=str)!r}.")
    selected = batch.select(*keys).detach().clone()
    for key, value in selected.items(include_nested=True, leaves_only=True):
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite policy input at {key!r}.")
    return selected


def _gaussian_kl(location_a, scale_a, location_b, scale_b) -> torch.Tensor:
    """KL(a || b), summed over action coordinates, for matching tanh transforms."""
    for tensor in (location_a, scale_a, location_b, scale_b):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Non-finite policy distribution parameters.")
    if bool((scale_a <= 0).any()) or bool((scale_b <= 0).any()):
        raise ValueError("Policy scales must be positive.")
    return (
        torch.log(scale_b / scale_a)
        + (scale_a.square() + (location_a - location_b).square()) / (2 * scale_b.square())
        - 0.5
    ).sum(dim=-1)


def same_input_influence(
    experiment: Any,
    inputs: Sequence[Any],
    *,
    group: str,
    input_distribution: str,
    seed: int = 0,
) -> SameInputInfluence:
    """Evaluate a frozen feed-forward actor on two identical input clones.

    Actions use deterministic exploration; the shift is L2 across the selected
    group's joint action, averaged over input transitions. KL uses this group's pre-tanh
    parameters. Other groups' outputs never contribute. The dataset may contain
    joint observations from any explicitly named distribution; no env is stepped.
    """
    _measured_group(experiment, (group,))
    if not input_distribution.strip() or not inputs:
        raise ValueError("Declare a nonempty input distribution and observation dataset.")
    if experiment.model_config.is_rnn:
        raise ValueError("Recurrent influence requires an explicit history/state contract.")
    dataset = tuple(_policy_inputs(experiment, batch, group) for batch in inputs)
    fingerprint = _hash_inputs(dataset)
    shifts, kls = [], []
    policy = experiment.group_policies[group]
    modules = communication_modules(experiment, group)
    with (
        _measurement_context(experiment, seed),
        set_exploration_type(ExplorationType.DETERMINISTIC),
    ):
        for index, batch in enumerate(dataset):
            if batch.numel() == 0:
                raise ValueError("Same-input datasets cannot contain empty batches.")
            with (
                preserve_rng_state(seed + index),
                preserve_channel_rng(_channels(experiment), seed=seed + index),
            ):
                left = policy(batch.clone())
            with (
                preserve_rng_state(seed + index),
                preserve_channel_rng(_channels(experiment), seed=seed + index),
            ):
                with severed_communication(modules):
                    right = policy(batch.clone())
            action_left = left.get((group, "action")).to(torch.float64)
            action_right = right.get((group, "action")).to(torch.float64)
            difference = (action_left - action_right).reshape(batch.numel(), -1)
            if not bool(torch.isfinite(difference).all()):
                raise ValueError("Non-finite actions in same-input evaluation.")
            shifts.append(difference.norm(dim=-1))
            params = [
                output.get((group, name), None)
                for output in (left, right)
                for name in ("loc", "scale")
            ]
            if any(value is not None for value in params):
                if any(value is None for value in params):
                    raise ValueError("Incomplete Gaussian policy parameters.")
                kls.append(_gaussian_kl(*(value.to(torch.float64) for value in params)).flatten())
    if _hash_inputs(dataset) != fingerprint:
        raise RuntimeError("Same-input evaluation mutated its recorded dataset.")
    all_shifts = torch.cat(shifts)
    return SameInputInfluence(
        group,
        input_distribution,
        sum(batch.numel() for batch in dataset),
        fingerprint,
        tuple(
            sorted(
                {
                    str(key)
                    for batch in dataset
                    for key in batch.keys(include_nested=True, leaves_only=True)
                }
            )
        ),
        "DETERMINISTIC",
        float(all_shifts.mean()),
        float(all_shifts.max()),
        float(torch.cat(kls).mean()) if kls else None,
        seed,
    )


def paired_return_deltas(
    with_comm: Sequence[EpisodeOutcome],
    without_comm: Sequence[EpisodeOutcome],
) -> tuple[float, ...]:
    """Fail on missing, duplicated, reordered, or falsely paired episode records."""
    left_ids = [record.episode_id for record in with_comm]
    right_ids = [record.episode_id for record in without_comm]
    if not left_ids or len(set(left_ids)) != len(left_ids) or left_ids != right_ids:
        raise ValueError(
            "Paired return episode IDs must be nonempty, unique, and identical in order."
        )
    deltas = []
    for left, right in zip(with_comm, without_comm, strict=True):
        if left.seed != right.seed or left.initial_input_sha256 != right.initial_input_sha256:
            raise ValueError(f"Initial condition mismatch for episode {left.episode_id!r}.")
        if (left.exogenous_episode_ids, left.exogenous_steps) != (
            right.exogenous_episode_ids,
            right.exogenous_steps,
        ):
            raise ValueError(f"Exogenous noise mismatch for episode {left.episode_id!r}.")
        if left.complete != right.complete:
            raise ValueError(f"Episode completion mismatch for {left.episode_id!r}.")
        if not math.isfinite(left.return_value) or not math.isfinite(right.return_value):
            raise ValueError("Paired returns must be finite.")
        deltas.append(left.return_value - right.return_value)
    return tuple(deltas)


def _fresh_env(experiment: Any, seed: int):
    raw = experiment.task.get_env_fun(
        num_envs=1,
        continuous_actions=experiment.continuous_actions,
        seed=seed,
        device=experiment.config.sampling_device,
    )()
    try:
        env = TransformedEnv(raw, Compose(*experiment.task.get_env_transforms(raw))).to(
            experiment.config.sampling_device
        )
        return experiment.algorithm.process_env_fun(lambda: env)()
    except BaseException:
        raw.close()
        raise


def _one_episode(experiment, episode, group, steps, require_complete):
    env = _fresh_env(experiment, episode.seed)
    try:
        initial = env.reset()
        initial_hash = _hash_inputs((initial,))
        raw = env
        while isinstance(raw, TransformedEnv):
            raw = raw.base_env
        scenario = getattr(raw, "scenario", None)
        exogenous = [
            tuple(getattr(scenario, name).reshape(-1).tolist()) if hasattr(scenario, name) else ()
            for name in ("exogenous_episode_ids", "exogenous_steps")
        ]
        rollout = env.rollout(
            max_steps=steps,
            policy=experiment.policy,
            tensordict=initial,
            auto_reset=False,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
        if env.batch_size:
            if env.batch_size != torch.Size([1]):
                raise ValueError("Episode evaluator requires one simulator instance.")
            rollout = rollout[0]
        terminated = bool(rollout.get(("next", "terminated"), torch.tensor([False]))[-1:].any())
        truncated = bool(rollout.get(("next", "truncated"), torch.tensor([False]))[-1:].any())
        done = rollout.get(("next", "done"), None)
        complete = done is not None and bool(done[-1].all())
        if require_complete and not complete:
            raise ValueError(f"Episode {episode.episode_id!r} did not finish within {steps} steps.")
        outcome = EpisodeOutcome(
            episode.episode_id,
            episode.seed,
            _episode_returns(experiment, (rollout,), (group,))[0],
            rollout.shape[0],
            terminated,
            truncated,
            complete,
            initial_hash,
            *exogenous,
        )
        return outcome, _policy_inputs(experiment, rollout, group)
    finally:
        env.close()


def paired_episode_reliance(
    experiment: Any,
    *,
    group: str,
    episodes: int = 32,
    steps: int | None = None,
    exploration: str = "DETERMINISTIC",
    seed: int = 0,
    episode_bank: Sequence[EvaluationEpisode] | None = None,
    require_complete: bool = True,
) -> PairedEpisodeEvaluation:
    """Run exact-count episodes with fresh, matched task instances per arm.

    Corrected PCP task factories isolate disturbances by episode and time.
    Environment factories for future tasks must uphold that same contract.
    Shortened rollouts are allowed only with explicit require_complete=False.
    """
    _measured_group(experiment, (group,))
    if experiment.model_config.is_rnn:
        raise ValueError("Recurrent evaluation requires a validated history/reset contract.")
    max_steps = experiment.max_steps if steps is None else steps
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("steps must be a positive integer.")
    bank = (
        tuple(episode_bank)
        if episode_bank is not None
        else tuple(
            EvaluationEpisode(f"seed:{seed}/episode:{index}", seed + index)
            for index in range(episodes)
        )
    )
    if len(bank) != episodes or len({item.episode_id for item in bank}) != episodes:
        raise ValueError("Episode bank must contain exactly episodes unique IDs.")
    mode = ExplorationType[exploration.upper()]
    modules = communication_modules(experiment, group)
    with_outcomes, without_outcomes, live_inputs, severed_inputs = [], [], [], []
    for episode in bank:
        with _measurement_context(experiment, episode.seed), set_exploration_type(mode):
            left, left_inputs = _one_episode(
                experiment, episode, group, max_steps, require_complete
            )
        with _measurement_context(experiment, episode.seed), set_exploration_type(mode):
            with severed_communication(modules):
                right, right_inputs = _one_episode(
                    experiment, episode, group, max_steps, require_complete
                )
        with_outcomes.append(left)
        without_outcomes.append(right)
        live_inputs.append(left_inputs)
        severed_inputs.append(right_inputs)
    paired_return_deltas(with_outcomes, without_outcomes)
    return PairedEpisodeEvaluation(
        group,
        mode.name,
        tuple(with_outcomes),
        tuple(without_outcomes),
        tuple(live_inputs),
        tuple(severed_inputs),
    )


def communication_saliency(
    experiment: Any,
    *,
    episodes: int = 32,
    steps: int | None = None,
    exploration: str = "DETERMINISTIC",
    seed: int = 0,
    return_groups: Sequence[str] | None = None,
    require_complete: bool = True,
) -> SaliencyResult:
    """Measure episode reliance and influence on a labelled live-policy dataset."""
    group = _measured_group(experiment, return_groups)
    paired = paired_episode_reliance(
        experiment,
        group=group,
        episodes=episodes,
        steps=steps,
        exploration=exploration,
        seed=seed,
        require_complete=require_complete,
    )
    influence = same_input_influence(
        experiment,
        paired.live_inputs,
        group=group,
        input_distribution="live_policy_trajectories",
        seed=seed,
    )
    returns_with = tuple(item.return_value for item in paired.with_comm)
    returns_without = tuple(item.return_value for item in paired.without_comm)
    deltas = torch.tensor(
        paired_return_deltas(paired.with_comm, paired.without_comm), dtype=torch.float64
    )
    mean_with, mean_without = sum(returns_with) / episodes, sum(returns_without) / episodes
    delta = mean_with - mean_without
    return SaliencyResult(
        episodes=episodes,
        steps=experiment.max_steps if steps is None else steps,
        exploration=paired.exploration,
        return_with_comm=mean_with,
        return_without_comm=mean_without,
        return_delta=delta,
        # Legacy descriptive field; origin-dependent, not a cross-task necessity score.
        return_delta_fraction=delta / abs(mean_without) if abs(mean_without) > 1e-12 else None,
        per_episode_delta_mean=float(deltas.mean()),
        per_episode_delta_std=float(deltas.std()) if episodes > 1 else None,
        action_shift_mean=influence.action_shift_mean,
        action_shift_max=influence.action_shift_max,
        policy_kl_mean=influence.policy_kl_mean,
        communicating_modules=len(communication_modules(experiment, group)),
        per_episode_returns_with=returns_with,
        per_episode_returns_without=returns_without,
        episode_ids=tuple(item.episode_id for item in paired.with_comm),
        group=group,
        input_distribution=influence.input_distribution,
        input_count=influence.input_count,
        input_sha256=influence.input_sha256,
        input_keys=influence.input_keys,
        complete_episodes=all(item.complete for item in (*paired.with_comm, *paired.without_comm)),
        outcomes_with=paired.with_comm,
        outcomes_without=paired.without_comm,
        influence_inputs=paired.live_inputs,
    )
