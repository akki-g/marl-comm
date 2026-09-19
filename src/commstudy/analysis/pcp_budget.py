"""Held-out, episode-paired PCP budget measurements.

The semantic shuffle substitutes messages from another complete live episode
at the same transition index. It does not permute a permutation-invariant sum.
All interventions are evaluation-only and restore hooks, channels and RNG.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
from typing import Any

import numpy as np
import torch
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from commstudy.analysis.pcp_domain import domain_episode_metrics
from commstudy.analysis.saliency import (
    _fresh_env,
    _hash_inputs,
    _measurement_context,
    communication_modules,
)
from commstudy.communication.channel import ChannelOutput, channel_intervention
from commstudy.communication.identity import IdentityComm
from commstudy.communication.local_control import LocalCapacityComm
from commstudy.communication.budgeted_broadcast import BudgetedBroadcastComm


BUDGET_EVALUATION_PROTOCOL = "pcp_budget_evaluation_v1"
INTERVENTIONS = ("live", "severed", "suppress_0", "suppress_1", "suppress_2", "shuffle")


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def paired_episode_differences(left: list[dict], right: list[dict], metric: str) -> list[float]:
    """Pair only the same complete physical episodes, including hidden wander state."""
    if not left or len(left) != len(right):
        raise ValueError("Paired episode banks must be nonempty and equally sized.")
    ids = [row["episode_id"] for row in left]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicated episode IDs.")
    result = []
    for a, b in zip(left, right, strict=True):
        for key in ("episode_id", "seed", "initial_physical_state_sha256", "exogenous_initial"):
            if a[key] != b[key]:
                raise ValueError(f"Episode pairing mismatch: {key}.")
        for record in (a, b):
            if record["metrics"]["complete"] != 1 or record["metrics"]["episode_steps"] != 100:
                raise ValueError("Only complete 100-transition episodes may be paired.")
        values = [float(row["metrics"][metric]) for row in (a, b)]
        if not np.isfinite(values).all():
            raise ValueError("Non-finite paired measurements.")
        result.append(values[0] - values[1])
    return result


def paired_seed_interval(values: list[float], *, seed: int = 73191, samples: int = 10000) -> dict:
    """Descriptive percentile bootstrap over training seeds, never pooled episodes."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("At least two finite independent training-seed values are required.")
    rng = np.random.default_rng(seed)
    means = array[rng.integers(len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "training_seeds": len(array),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "unit": "training_seed",
        "method": "paired_seed_percentile_bootstrap",
        "caution": "Small seed allocation; descriptive interval, no multiplicity-adjusted test.",
    }


def _action_metrics(rollout) -> dict:
    values = {
        name: rollout.get(("adversary", name)).detach().to(torch.float64)
        for name in ("action", "loc", "scale")
    }
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("Non-finite predator action or distribution parameter.")
    if bool((values["scale"] <= 0).any()):
        raise ValueError("Nonpositive predator action scale.")
    return {
        "action_scalars": values["action"].numel(),
        "all_finite": True,
        "saturation_fraction": float((values["action"].abs() > 0.999).double().mean()),
        "max_abs_action": float(values["action"].abs().max()),
        "max_abs_loc": float(values["loc"].abs().max()),
        "min_scale": float(values["scale"].min()),
        "max_scale": float(values["scale"].max()),
    }


def _episode(
    experiment, *, seed: int, intervention: str, donor: torch.Tensor | None, deadline: int
) -> tuple[dict, torch.Tensor | None]:
    modules = communication_modules(experiment, "adversary")
    if len(modules) != 1:
        raise ValueError("Budget evaluation requires exactly one predator communication slot.")
    module = modules[0]
    channel = getattr(module, "channel", None)
    width = int(module.message_dim)
    packets, masks = [], []

    def packet_hook(_channel, _args, output):
        index = len(packets)
        if not isinstance(output, ChannelOutput):
            raise TypeError("Budget evaluator requires a ChannelOutput.")
        messages = output.messages
        if messages.dtype != torch.float32 or messages.shape != (1, 3, width):
            raise ValueError(f"Unexpected single-world packet shape/dtype: {messages.shape}.")
        if donor is not None:
            if index >= len(donor) or donor[index].shape != messages.shape:
                raise ValueError("Semantic donor packet bank does not match the episode.")
            messages = torch.where(output.sender_mask.unsqueeze(-1), donor[index].to(messages), 0.0)
        if not bool(torch.isfinite(messages).all()):
            raise ValueError("Non-finite transmitted packet.")
        packets.append(messages.detach().cpu().clone())
        masks.append(output.sender_mask.detach().cpu().clone())
        return ChannelOutput(messages, output.sender_mask)

    with (
        _measurement_context(experiment, seed),
        set_exploration_type(ExplorationType.DETERMINISTIC),
    ):
        with ExitStack() as stack:
            module.reset_stats()
            if channel is not None:
                if intervention == "severed":
                    stack.enter_context(channel_intervention([channel], False))
                elif intervention.startswith("suppress_"):
                    mask = torch.ones(3, dtype=torch.bool)
                    mask[int(intervention[-1])] = False
                    stack.enter_context(channel_intervention([channel], mask))
                handle = channel.register_forward_hook(packet_hook)
                stack.callback(handle.remove)
            env = _fresh_env(experiment, seed)
            stack.callback(env.close)
            initial = env.reset()
            physical_hash = _hash_inputs((initial.select("state"),))
            raw = env
            while isinstance(raw, TransformedEnv):
                raw = raw.base_env
            scenario = raw.scenario
            exogenous = {
                name: getattr(scenario, name).reshape(-1).tolist()
                for name in ("exogenous_episode_ids", "exogenous_steps")
            }
            rollout = env.rollout(
                max_steps=100,
                policy=experiment.policy,
                tensordict=initial,
                auto_reset=False,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )
            if env.batch_size != torch.Size([1]):
                raise ValueError("Budget evaluation requires one simulator instance.")
            rollout = rollout[0]
            metrics = domain_episode_metrics(rollout)
            if (
                metrics["episode_steps"] != 100
                or metrics["truncated"] != 1
                or metrics["terminated"]
            ):
                raise ValueError("Budget evaluation requires a complete PCP time-limit truncation.")
            if metrics["reward_accounting_max_abs_error"] != 0:
                raise ValueError("Repeated-contact reward accounting failed.")
            metrics["contact_by_deadline"] = float(
                metrics["first_contact_observed"]
                and metrics["first_contact_step_or_horizon"] <= deadline
            )
            costs = module.communication_stats()
            if channel is not None and packets:
                if len(packets) != 100:
                    raise ValueError("Exactly one message exchange per transition is required.")
                senders = torch.stack(masks).double().sum(-1).reshape(-1)
            else:
                if channel is not None and getattr(module, "sender_budget", None) != 0:
                    raise ValueError("A positive-budget channel emitted no packet records.")
                senders = torch.zeros(100, dtype=torch.float64)
            observed = {
                "sender_emissions": float(senders.sum()),
                "edge_deliveries": float((senders * 2).sum()),
                "sender_payload_bits": float(senders.sum()) * width * 32,
                "edge_delivery_payload_bits": float(senders.sum()) * 2 * width * 32,
                "transitions": 100,
                "packet_scalars": width,
                "bits_per_scalar": 32,
                "cost_model": "one_float32_broadcast_packet_per_available_sender; two_receivers",
                "network_headers_counted": False,
                "measured_network_traffic": False,
            }
            for stat, expected in (
                ("messages_per_step", observed["sender_emissions"] / 100),
                ("realized_sender_bits_per_step", observed["sender_payload_bits"] / 100),
                ("realized_bits_per_step", observed["edge_delivery_payload_bits"] / 100),
            ):
                if channel is not None and (
                    stat not in costs or abs(costs[stat] - expected) > 1e-6
                ):
                    raise ValueError(f"Packet-count and module cost disagreement: {stat}.")
            return {
                "episode_id": f"pcp-heldout:{seed}",
                "seed": seed,
                "initial_physical_state_sha256": physical_hash,
                "exogenous_initial": exogenous,
                "metrics": metrics,
                "action_health": _action_metrics(rollout),
                "costs": observed,
                "module_stats": costs,
            }, torch.stack(packets) if packets else None


def evaluate_budget_policy(
    experiment,
    *,
    episode_seeds: list[int],
    deadline: int = 50,
    interventions: tuple[str, ...] | None = None,
) -> tuple[dict, Any]:
    """Measure one saved feed-forward policy against a fixed held-out seed bank.

    The caller persists the returned live packet tensor when semantic shuffling
    is enabled. A cyclic donor index is fixed before observations; every donor
    is another episode at the same time index, with sender identity preserved.
    """
    modules = communication_modules(experiment, "adversary")
    if len(modules) != 1 or type(modules[0]) not in (
        IdentityComm, LocalCapacityComm, BudgetedBroadcastComm
    ):
        raise ValueError("Unsupported policy family for the declared budget evaluation.")
    module = modules[0]
    communicating = isinstance(module, BudgetedBroadcastComm) and module.sender_budget == 3
    allowed = INTERVENTIONS if communicating else ("live", "severed")
    interventions = allowed if interventions is None else interventions
    if (
        len(episode_seeds) < 2
        or len(set(episode_seeds)) != len(episode_seeds)
        or any(type(seed) is not int or seed < 0 for seed in episode_seeds)
    ):
        raise ValueError("Require at least two unique nonnegative episode seeds.")
    if type(deadline) is not int or not 1 <= deadline <= 100:
        raise ValueError("Deadline must be an integer in [1,100].")
    if (
        not interventions
        or interventions[0] != "live"
        or len(set(interventions)) != len(interventions)
        or set(interventions) - set(allowed)
    ):
        raise ValueError("Interventions must be unique, declared and start with live.")
    if experiment.model_config.is_rnn or experiment.max_steps != 100:
        raise ValueError("Only the declared feed-forward 100-step PCP task is supported.")
    report = {
        "schema_version": 1,
        "evaluation_protocol": BUDGET_EVALUATION_PROTOCOL,
        "group": "adversary",
        "exploration": "DETERMINISTIC",
        "deadline": deadline,
        "episode_seeds": episode_seeds,
        "analysis_provenance": experiment.analysis_provenance,
        "interventions": {},
        "shuffle_definition": "frozen_live_packets_from_cyclic_next_episode_at_same_step",
        "shuffle_distribution_shift": (
            "donor live trajectories substituted into receiver closed-loop trajectories"
        ),
    }
    bank = []
    for mode in interventions:
        rows = []
        for index, seed in enumerate(episode_seeds):
            donor = bank[(index + 1) % len(bank)] if mode == "shuffle" and bank else None
            row, packets = _episode(
                experiment, seed=seed, intervention=mode, donor=donor, deadline=deadline
            )
            if mode == "live" and packets is not None:
                bank.append(packets)
            if mode == "shuffle":
                row["donor_episode_seed"] = episode_seeds[(index + 1) % len(episode_seeds)]
                row["donor_packet_sha256"] = tensor_sha256(donor) if donor is not None else None
            rows.append(row)
        if mode != "live":
            paired_episode_differences(report["interventions"]["live"], rows, "return")
            if not communicating:
                for live, intervened in zip(report["interventions"]["live"], rows, strict=True):
                    if any(live[key] != intervened[key] for key in ("metrics", "action_health")):
                        raise ValueError(
                            "Zero-communication control failed the exact null intervention."
                        )
        report["interventions"][mode] = rows
    donor_bank = torch.stack(bank) if bank else None
    report["live_packet_bank_sha256"] = (
        tensor_sha256(donor_bank) if donor_bank is not None else None
    )
    return report, donor_bank
