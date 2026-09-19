"""Episode-level PCP contact and information-access measurements.

Info is read from the post-transition state that generated the reward. Team
statistics are repeated per predator by VMAS; they are checked for agreement,
then read once, never summed across those duplicate copies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from commstudy.analysis.saliency import _fresh_env, _measurement_context
from commstudy.tasks.vmas.domain_metrics import PCP_DOMAIN_PROTOCOL, pcp_metric_specs


def domain_transition_series(rollout, group: str = "adversary") -> dict[str, torch.Tensor]:
    if group != "adversary":
        raise ValueError("PCP domain measurements require the predator group 'adversary'.")
    reward = rollout.get(("next", group, "reward"))
    predators = reward.shape[-2]
    result = {}
    for name, definition in pcp_metric_specs(predators).items():
        value = rollout.get(("next", group, "info", name), None)
        if value is None:
            raise ValueError(f"Missing PCP domain field: {name}.")
        value = value.detach()
        if value.shape == (*rollout.batch_size, predators, 1):
            value = value.squeeze(-1)
        if value.shape != (*rollout.batch_size, predators):
            raise ValueError(f"Unexpected PCP domain shape for {name}: {value.shape}.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite PCP domain field: {name}.")
        first = value[..., 0]
        if not torch.equal(value, first.unsqueeze(-1).expand_as(value)):
            raise ValueError(f"Duplicated team statistics disagree for {name}.")
        kind = definition["kind"]
        if kind != "signed_reward" and bool((first < 0).any()):
            raise ValueError(f"Negative PCP domain field: {name}.")
        if kind in {"count", "indicator"} and not torch.equal(first, first.round()):
            raise ValueError(f"Non-integer PCP domain field: {name}.")
        if kind in {"indicator", "fraction"} and bool((first > 1).any()):
            raise ValueError(f"PCP domain field outside [0,1]: {name}.")
        result[name] = first.to(dtype=torch.float64)
    result["predator_reward"] = reward.detach().mean(dim=(-2, -1)).to(torch.float64)
    return result


def domain_episode_metrics(rollout, *, require_complete: bool = True) -> dict[str, float]:
    """Summarize one [time] rollout; first contact uses one-based transition IDs.

    No-contact episodes are right-censored at the observed horizon. Both the
    observed flag and the censored time are exported; conditional contact times
    must never silently omit the no-contact probability.
    """
    if len(rollout.batch_size) != 1 or not rollout.batch_size[0]:
        raise ValueError("Domain episode input must have a nonempty [time] batch.")
    values = domain_transition_series(rollout)
    steps = int(rollout.batch_size[0])
    done = rollout.get(("next", "done"), None)
    complete = done is not None and bool(done[-1].all())
    if require_complete and not complete:
        raise ValueError("PCP episode is incomplete; domain confirmation requires full episodes.")
    if done is not None and bool(done[:-1].any()):
        raise ValueError("PCP episode contains transitions after an earlier done flag.")
    contacts = values["any_contact"] > 0
    first = contacts.nonzero(as_tuple=False)
    contact_step = int(first[0, 0]) + 1 if first.numel() else steps
    terminated = rollout.get(("next", "terminated"), None)
    truncated = rollout.get(("next", "truncated"), None)
    reward_error = (values["predator_reward"] - values["expected_predator_reward"]).abs()
    metrics = {
        "episode_steps": float(steps),
        "complete": float(complete),
        "terminated": float(terminated is not None and bool(terminated[-1].all())),
        "truncated": float(truncated is not None and bool(truncated[-1].all())),
        "return": float(values["predator_reward"].sum()),
        "any_contact": float(contacts.any()),
        "first_contact_observed": float(contacts.any()),
        "first_contact_step_or_horizon": float(contact_step),
        "contact_duration_steps": float(contacts.sum()),
        "contact_pair_steps": float(values["contact_pairs"].sum()),
        "simultaneous_contact_steps": float(values["simultaneous_contact"].sum()),
        "any_simultaneous_contact": float((values["simultaneous_contact"] > 0).any()),
        "contacting_predators_mean": float(values["contacting_predators"].mean()),
        "contacting_predators_max": float(values["contacting_predators"].max()),
        "nearest_predator_prey_distance_mean": float(
            values["nearest_predator_prey_distance"].mean()
        ),
        "nearest_predator_prey_distance_min": float(
            values["nearest_predator_prey_distance"].min()
        ),
        "mean_predator_nearest_prey_distance": float(
            values["mean_predator_nearest_prey_distance"].mean()
        ),
        "predator_boundary_fraction": float(values["predator_boundary_fraction"].mean()),
        "prey_boundary_fraction": float(values["prey_boundary_fraction"].mean()),
        "predator_obstacle_collision_pair_steps": float(
            values["predator_obstacle_collision_pairs"].sum()
        ),
        "predator_teammate_collision_pair_steps": float(
            values["predator_teammate_collision_pairs"].sum()
        ),
        "sender_visible_receiver_blind_pairs_mean": float(
            values["sender_visible_receiver_blind_pairs"].mean()
        ),
        "sender_visible_receiver_blind_prey_triples_mean": float(
            values["sender_visible_receiver_blind_prey_triples"].mean()
        ),
        "reward_accounting_max_abs_error": float(reward_error.max()),
    }
    for name, value in values.items():
        if name.startswith("prey_visible_to_"):
            metrics[f"{name}_mean"] = float(value.mean())
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite PCP episode summary.")
    return metrics


def summarize_domain_episodes(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("At least one domain episode is required.")
    metrics = [episode["metrics"] for episode in episodes]
    if any(set(row) != set(metrics[0]) for row in metrics):
        raise ValueError("Domain episodes have inconsistent metric schemas.")
    means = {name: sum(row[name] for row in metrics) / len(metrics) for name in metrics[0]}
    observed = [row for row in metrics if row["first_contact_observed"]]
    return {
        "episode_count": len(episodes),
        "complete_episodes": sum(int(row["complete"]) for row in metrics),
        "transition_count": sum(int(row["episode_steps"]) for row in metrics),
        "means": means,
        "contact_episode_count": len(observed),
        "first_contact_step_conditional_mean": (
            sum(row["first_contact_step_or_horizon"] for row in observed) / len(observed)
            if observed else None
        ),
        "reward_accounting_max_abs_error": max(
            row["reward_accounting_max_abs_error"] for row in metrics
        ),
    }


def policy_domain_audit(
    experiment,
    *,
    episodes: int = 32,
    seed: int = 100000,
    explorations: Sequence[str] = ("DETERMINISTIC", "RANDOM"),
) -> dict[str, Any]:
    """Audit frozen PCP actions on an explicit, shared episode seed bank."""
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if experiment.model_config.is_rnn:
        raise ValueError("Recurrent domain audits require explicit hidden-state replay.")
    if set(experiment.group_map) != {"adversary", "agent"}:
        raise ValueError("Domain audit requires the PCP predator/prey groups.")
    result = {
        "schema_version": 1,
        "domain_protocol": PCP_DOMAIN_PROTOCOL,
        "group": "adversary",
        "analysis_provenance": getattr(experiment, "analysis_provenance", None),
        "modes": {},
    }
    for exploration in explorations:
        mode = ExplorationType[exploration.upper()]
        rows = []
        for index in range(episodes):
            episode_seed = seed + index
            with _measurement_context(experiment, episode_seed), set_exploration_type(mode):
                env = _fresh_env(experiment, episode_seed)
                try:
                    rollout = env.rollout(
                        max_steps=experiment.max_steps,
                        policy=experiment.policy,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                    if env.batch_size == torch.Size([1]):
                        rollout = rollout[0]
                    elif env.batch_size:
                        raise ValueError("Domain audit requires a single-world environment.")
                    metrics = domain_episode_metrics(rollout)
                    rows.append({
                        "episode_id": f"seed:{seed}/episode:{index}",
                        "seed": episode_seed,
                        "metrics": metrics,
                    })
                finally:
                    env.close()
        result["modes"][mode.name] = {"episodes": rows, **summarize_domain_episodes(rows)}
    return result
