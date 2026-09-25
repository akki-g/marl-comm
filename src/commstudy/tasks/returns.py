"""Task-correct returns: predator-only PCP and shared-team MAPDN.

Collection uses running episode rewards at terminal transitions. Evaluation
sums each rollout over time and averages agents within the measured group.
The spec declares measured groups explicitly, separately from actor/critic groups.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


#: Task-config key declaring the groups whose reward is the study's return.
RETURN_GROUPS_KEY = "return_groups"


def resolve_return_groups(
    group_map: Iterable[str],
    configured: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate the configured measured groups against the real environment.

    An unknown group name is an error rather than an empty measurement: a typo
    would otherwise produce a study whose headline metric is silently missing,
    which is the failure mode this whole module exists to prevent.
    """

    available = tuple(str(group) for group in group_map)
    if configured is None:
        return available

    requested = tuple(str(group) for group in configured)
    if not requested:
        raise ValueError(
            f"'{RETURN_GROUPS_KEY}' is empty. Remove the key to measure every "
            "group, or name the groups the study evaluates."
        )

    unknown = [group for group in requested if group not in available]
    if unknown:
        raise ValueError(
            f"'{RETURN_GROUPS_KEY}' names unknown agent groups {unknown}. "
            f"This environment has {list(available)}."
        )
    return requested


def _scalar(value: Any) -> float:
    return float(value.detach().mean().cpu())


def group_rollout_returns(
    rollout: Any,
    groups: Sequence[str],
) -> dict[str, float]:
    """Summed reward over one evaluation rollout, per group.

    Time is summed and the group's agents are averaged, which is the formula the
    evaluation metric uses.
    """

    returns: dict[str, float] = {}
    for group in groups:
        reward = rollout.get(("next", group, "reward"), None)
        if reward is None:
            reward = rollout.get(("next", "reward"), None)
        if reward is None:
            continue
        returns[group] = _scalar(reward.sum(0))
    return returns


def group_collection_returns(
    batch: Any,
    groups: Sequence[str],
) -> dict[str, float]:
    """Per-group episode return over a collected batch, at the global done.

    Returns an empty mapping when no episode ended in this batch. BenchMARL
    warns and reports NaN in that case; here the metric is simply not written,
    so a NaN never reaches the analysis as if it were a measurement.
    """

    done = batch.get(("next", "done"), None)
    if done is None or not bool(done.any()):
        return {}

    returns: dict[str, float] = {}
    for group in groups:
        episode_reward = batch.get(("next", group, "episode_reward"), None)
        if episode_reward is None:
            shared = batch.get(("next", "episode_reward"), None)
            if shared is None:
                continue
            episode_reward = shared.expand(batch.get(group).shape).unsqueeze(-1)
        # Mean over the group's agents, then keep only the finished episodes.
        finished = episode_reward.mean(-2)[done]
        if finished.numel() == 0:
            continue
        returns[group] = _scalar(finished)
    return returns


def mean_over_groups(returns: Mapping[str, float]) -> float | None:
    """Average per-group returns into the single number the study reports."""

    if not returns:
        return None
    return sum(returns.values()) / len(returns)


def collection_episode_returns(batch: Any, groups: Sequence[str]) -> list[float]:
    """Return each completed episode once, averaging only measured groups."""
    done = batch.get(("next", "done"), None)
    if done is None or not bool(done.any()):
        return []
    values = []
    for group in groups:
        reward = batch.get(("next", group, "episode_reward"), None)
        if reward is None:
            raise ValueError(f"Missing running episode return for {group}")
        values.append(reward.mean(-2)[done])
    return (sum(values) / len(values)).detach().cpu().tolist()
