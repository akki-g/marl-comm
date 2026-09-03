"""Which agent groups define a study's return, and how that return is computed.

A task can train more than one BenchMARL group while only one of them is the
subject of the experiment. Predator-Capture-Prey is the motivating case: its
``agent`` group is a scripted prey whose policy output is discarded every step,
and ``simple_tag``'s rewards are exactly zero-sum between the two groups, so a
return averaged over every group is identically zero no matter how well the
predators do.

The measured groups are therefore declared per task, as ``return_groups`` beside
``params`` in ``configs/tasks/<task>.yaml``. Omitting the key keeps the previous
behaviour of averaging over every group, which is what a single-group task such
as Simple Spread already resolved to.

Both reduction formulas below mirror what they replace, so a single-group task
produces numerically identical values to before:

* collection matches BenchMARL's own ``Logger._log_individual_and_group_rewards``
  -- the running episode reward at the global done, averaged over the group's
  agents -- so ``metrics.csv`` agrees with the ``benchmarl/`` scalars;
* evaluation keeps the summed-reward-over-a-rollout formula the metrics callback
  and the saliency measurement already shared.
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
    evaluation metric and the saliency intervention have always used.
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
