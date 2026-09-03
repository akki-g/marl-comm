"""Communication saliency: how much does the channel actually buy?

A learned communication module can look busy — non-zero messages, structured
attention, active edges — while contributing nothing to the task. Activity
metrics cannot distinguish those cases. Saliency answers the causal question
directly by intervening on a *frozen, already-trained* policy:

```text
return_with_comm    - evaluate normally
return_without_comm - evaluate with every message suppressed
saliency            = return_with_comm - return_without_comm
```

Positive saliency means the channel is load-bearing for the shared objective.
Saliency near zero means the policy achieves its return without the channel,
whatever its messages look like.

## Why severing is exact

Suppression reuses the study's existing channel abstraction: a dropout channel
with ``p=1`` marks every sender unavailable. Because every module decodes
communication through a **bias-free** projection, an empty neighbourhood
produces an exactly zero communication delta, so each learned module collapses
to ``h' = h`` — precisely ``IdentityComm``. Severing therefore removes the
channel while leaving all encoder/decoder weights in place; it does not
perturb the local pathway, and it needs no special-case code per module.

## Two distinct questions

``return_delta`` measures *task benefit*. ``action_shift`` and ``policy_kl``
measure *behavioural influence* on identical states. These come apart in an
informative way: a module whose messages change behaviour a lot but return
little is communicating without coordinating, which is the failure mode that
raw activity metrics hide. This mirrors the "positive listening" distinction
of Lowe et al., *On the Pitfalls of Measuring Emergent Communication* (2019),
though it is measured here by intervention rather than by mutual information.

Evaluation is paired: both arms are run from the same environment seed, so the
difference reflects the intervention rather than episode luck.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from commstudy.communication.base import CommModule
from commstudy.communication.channel import build_channel
from commstudy.experiments.returns import (
    group_rollout_returns,
    mean_over_groups,
    resolve_return_groups,
)


@dataclass(frozen=True)
class SaliencyResult:
    """Paired with/without-communication comparison for one frozen policy."""

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
    per_episode_returns_with: tuple[float, ...] = field(default=())
    per_episode_returns_without: tuple[float, ...] = field(default=())

    def as_row(self) -> dict[str, Any]:
        return {
            "saliency_episodes": self.episodes,
            "saliency_steps": self.steps,
            "saliency_exploration": self.exploration,
            "saliency_return_with_comm": self.return_with_comm,
            "saliency_return_without_comm": self.return_without_comm,
            "saliency_return_delta": self.return_delta,
            "saliency_return_delta_fraction": self.return_delta_fraction,
            "saliency_per_episode_delta_mean": self.per_episode_delta_mean,
            "saliency_per_episode_delta_std": self.per_episode_delta_std,
            "saliency_action_shift_mean": self.action_shift_mean,
            "saliency_action_shift_max": self.action_shift_max,
            "saliency_policy_kl_mean": self.policy_kl_mean,
            "saliency_communicating_modules": self.communicating_modules,
        }


def communication_modules(experiment: Any) -> list[CommModule]:
    """Every distinct communication module inside the actor(s)."""

    found: dict[int, CommModule] = {}
    for policy in experiment.group_policies.values():
        for module in policy.modules():
            if isinstance(module, CommModule):
                found.setdefault(id(module), module)
    return list(found.values())


@contextmanager
def severed_communication(modules: Sequence[CommModule]) -> Iterator[None]:
    """Temporarily suppress every message without touching learned weights.

    Implemented by swapping in a ``p=1`` dropout channel, which marks all
    senders unavailable. The original channels are always restored, including
    on exception, so an audited experiment object stays reusable.
    """

    severing = build_channel({"type": "dropout", "p": 1.0, "mode": "always"})
    originals = [module.channel for module in modules if hasattr(module, "channel")]
    targets = [module for module in modules if hasattr(module, "channel")]
    try:
        for module in targets:
            module.channel = severing
        yield
    finally:
        for module, original in zip(targets, originals, strict=True):
            module.channel = original


def _rollout(experiment: Any, *, steps: int, episodes: int, seed: int | None) -> list[Any]:
    if seed is not None:
        experiment.test_env.set_seed(int(seed))
    if experiment.test_env.batch_size != ():
        batch = experiment.test_env.rollout(
            max_steps=steps,
            policy=experiment.policy,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )
        return list(batch.unbind(0))
    return [
        experiment.test_env.rollout(
            max_steps=steps,
            policy=experiment.policy,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
        for _ in range(episodes)
    ]


def _episode_returns(
    experiment: Any,
    rollouts: Sequence[Any],
    groups: Sequence[str],
) -> list[float]:
    """Summed per-episode reward, averaged over the measured agent groups.

    Saliency is a difference of returns, so it has to measure the same thing the
    study reports. Averaging over every group instead would make the delta
    identically zero on a task whose groups are zero-sum, which is exactly the
    value that marks a non-communicating control.
    """

    del experiment
    returns: list[float] = []
    for rollout in rollouts:
        episode_return = mean_over_groups(group_rollout_returns(rollout, groups))
        if episode_return is not None:
            returns.append(episode_return)
    return returns


def _leaf(batch: Any, name: str) -> torch.Tensor | None:
    for key, value in batch.items(include_nested=True, leaves_only=True):
        parts = key if isinstance(key, tuple) else (key,)
        if parts[0] != "next" and parts[-1] == name and isinstance(value, torch.Tensor):
            return value.detach().to(torch.float64)
    return None


def _gaussian_kl(
    location_a: torch.Tensor,
    scale_a: torch.Tensor,
    location_b: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """KL(a || b) for diagonal Gaussians, elementwise then summed over actions.

    Applied to the pre-tanh parameters BenchMARL exposes. The tanh transform is
    shared and invertible, so this is the KL of the underlying distributions
    and is a monotone stand-in for the divergence of the squashed policies.
    """

    floor = torch.finfo(torch.float64).tiny
    scale_a = scale_a.clamp_min(floor)
    scale_b = scale_b.clamp_min(floor)
    return (
        torch.log(scale_b / scale_a)
        + (scale_a.pow(2) + (location_a - location_b).pow(2)) / (2 * scale_b.pow(2))
        - 0.5
    ).sum(dim=-1)


def _behavioural_divergence(
    with_rollouts: Sequence[Any],
    without_rollouts: Sequence[Any],
) -> tuple[float | None, float | None, float | None]:
    """Compare the two arms state-by-state where the trajectories still align.

    Under DETERMINISTIC evaluation the arms share an initial state and diverge
    only because of the intervention, so early timesteps are directly
    comparable. Later timesteps drift onto different states; that drift is the
    downstream effect and is captured by the return delta instead.
    """

    shifts: list[torch.Tensor] = []
    divergences: list[torch.Tensor] = []
    for left, right in zip(with_rollouts, without_rollouts, strict=False):
        action_left, action_right = _leaf(left, "action"), _leaf(right, "action")
        if action_left is None or action_right is None:
            continue
        steps = min(action_left.shape[0], action_right.shape[0])
        if steps == 0:
            continue
        difference = (action_left[:steps] - action_right[:steps]).flatten(1)
        shifts.append(torch.linalg.vector_norm(difference, dim=-1))

        location_left, location_right = _leaf(left, "loc"), _leaf(right, "loc")
        scale_left, scale_right = _leaf(left, "scale"), _leaf(right, "scale")
        if None in (location_left, location_right, scale_left, scale_right):
            continue
        divergences.append(
            _gaussian_kl(
                location_left[:steps],
                scale_left[:steps],
                location_right[:steps],
                scale_right[:steps],
            ).flatten()
        )

    if not shifts:
        return None, None, None
    all_shifts = torch.cat(shifts)
    finite_shifts = all_shifts[torch.isfinite(all_shifts)]
    shift_mean = float(finite_shifts.mean()) if finite_shifts.numel() else None
    shift_max = float(finite_shifts.max()) if finite_shifts.numel() else None

    kl_mean = None
    if divergences:
        all_kl = torch.cat(divergences)
        finite_kl = all_kl[torch.isfinite(all_kl)]
        if finite_kl.numel():
            kl_mean = float(finite_kl.mean())
    return shift_mean, shift_max, kl_mean


def communication_saliency(
    experiment: Any,
    *,
    episodes: int = 32,
    steps: int | None = None,
    exploration: str = "DETERMINISTIC",
    seed: int = 0,
    return_groups: Sequence[str] | None = None,
) -> SaliencyResult:
    """Measure the task benefit and behavioural influence of the channel.

    The policy is never modified or retrained; only the channel is suppressed.
    A model with no communication module (the MLP reference, or Identity)
    yields exactly zero saliency, which is the correct control value rather
    than a missing measurement.
    """

    modules = communication_modules(experiment)
    max_steps = int(steps if steps is not None else experiment.max_steps)
    mode = ExplorationType[exploration.upper()]
    groups = resolve_return_groups(experiment.group_map, return_groups)

    with torch.no_grad(), set_exploration_type(mode):
        with_rollouts = _rollout(experiment, steps=max_steps, episodes=episodes, seed=seed)
        with severed_communication(modules):
            without_rollouts = _rollout(
                experiment, steps=max_steps, episodes=episodes, seed=seed
            )

    returns_with = _episode_returns(experiment, with_rollouts, groups)
    returns_without = _episode_returns(experiment, without_rollouts, groups)
    mean_with = float(sum(returns_with) / len(returns_with)) if returns_with else math.nan
    mean_without = (
        float(sum(returns_without) / len(returns_without)) if returns_without else math.nan
    )
    delta = mean_with - mean_without

    paired = [
        left - right for left, right in zip(returns_with, returns_without, strict=False)
    ]
    paired_tensor = torch.tensor(paired, dtype=torch.float64) if paired else None
    shift_mean, shift_max, kl_mean = _behavioural_divergence(with_rollouts, without_rollouts)

    # Simple Spread returns are negative costs, so normalise by magnitude to
    # keep the sign of the improvement rather than inverting it.
    scale = abs(mean_without)
    fraction = delta / scale if scale > 1e-12 and math.isfinite(delta) else None

    return SaliencyResult(
        episodes=len(with_rollouts),
        steps=max_steps,
        exploration=mode.name,
        return_with_comm=mean_with,
        return_without_comm=mean_without,
        return_delta=delta,
        return_delta_fraction=fraction,
        per_episode_delta_mean=(
            float(paired_tensor.mean()) if paired_tensor is not None else None
        ),
        per_episode_delta_std=(
            float(paired_tensor.std(unbiased=True))
            if paired_tensor is not None and paired_tensor.numel() > 1
            else None
        ),
        action_shift_mean=shift_mean,
        action_shift_max=shift_max,
        policy_kl_mean=kl_mean,
        communicating_modules=len(modules),
        per_episode_returns_with=tuple(returns_with),
        per_episode_returns_without=tuple(returns_without),
    )
