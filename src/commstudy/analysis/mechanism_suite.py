"""Method-general paired PCP evaluation and seed-block statistics for the suite."""

from __future__ import annotations
from contextlib import ExitStack

import numpy as np
import torch
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from commstudy.analysis.pcp_budget import _action_metrics, paired_episode_differences, tensor_sha256
from commstudy.analysis.pcp_domain import domain_episode_metrics
from commstudy.analysis.saliency import (
    _fresh_env,
    _hash_inputs,
    _measurement_context,
    communication_modules,
)
from commstudy.communication.channel import channel_intervention


def contact_by_deadline(metrics, deadline=50):
    return float(
        bool(metrics["first_contact_observed"])
        and metrics["first_contact_step_or_horizon"] <= deadline
    )


def evaluate_pcp(experiment, episode_seeds, *, local_null=False, random_actions=False):

    if len(episode_seeds) < 2 or len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("Require a unique paired episode bank")
    modules = communication_modules(experiment, "adversary")
    if len(modules) != 1 or experiment.max_steps != 100 or experiment.model_config.is_rnn:
        raise ValueError("Require one feedforward predator slot and the declared 100-step task")
    module = modules[0]
    channel = getattr(module, "channel", None)
    before = actor_digest(experiment.policy)
    outcomes = {}
    try:
        for arm in ("live", "severed"):
            episodes = []
            for seed in episode_seeds:
                with (
                    _measurement_context(experiment, seed),
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                    ExitStack() as stack,
                ):
                    if channel is not None:
                        stack.enter_context(channel_intervention([channel], arm == "live"))
                    module.reset_stats()
                    env = _fresh_env(experiment, seed)
                    stack.callback(env.close)
                    initial = env.reset()
                    raw = env
                    while isinstance(raw, TransformedEnv):
                        raw = raw.base_env
                    initial_hash = _hash_inputs((initial.select("state"),))
                    exogenous = {
                        name: getattr(raw.scenario, name).reshape(-1).tolist()
                        for name in ("exogenous_episode_ids", "exogenous_steps")
                    }
                    rollout = env.rollout(
                        max_steps=100,
                        policy=None if random_actions else experiment.policy,
                        tensordict=initial,
                        auto_reset=False,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )[0]
                    metrics = domain_episode_metrics(rollout)
                    if (
                        metrics["episode_steps"] != 100
                        or metrics["complete"] != 1
                        or (metrics["reward_accounting_max_abs_error"] != 0)
                    ):
                        raise ValueError("PCP horizon or rewarded-contact accounting failed")
                    metrics["contact_by_deadline"] = contact_by_deadline(metrics)
                    episodes.append(
                        {
                            "episode_id": f"mechanisms_v1:{seed}",
                            "seed": seed,
                            "initial_physical_state_sha256": initial_hash,
                            "exogenous_initial": exogenous,
                            "metrics": metrics,
                            "action_sha256": tensor_sha256(rollout.get(("adversary", "action"))),
                            "actions": rollout.get(("adversary", "action")).cpu().tolist(),
                            "action_health": None if random_actions else _action_metrics(rollout),
                            "module_stats": module.communication_stats(),
                            "contact_trace": rollout.get(("next", "adversary", "reward"))
                            .cpu()
                            .tolist(),
                        }
                    )
            outcomes[arm] = episodes
        paired_episode_differences(outcomes["live"], outcomes["severed"], "return")
        exact = all(
            all(a[k] == b[k] for k in ("metrics", "actions", "action_sha256"))
            for a, b in zip(outcomes["live"], outcomes["severed"], strict=True)
        )
        if local_null and not exact:
            raise ValueError("Local control failed exact action/domain/return null")
        return {
            "schema": "pcp_mechanisms_paired_v1",
            "arms": outcomes,
            "actor_sha256": before,
            "exact_local_null": exact,
            "summaries": {
                arm: {
                    key: float(np.mean([e["metrics"][key] for e in episodes]))
                    for key in episodes[0]["metrics"]
                }
                for arm, episodes in outcomes.items()
            },
            "deadline_transition": 50,
            "transport_headers_counted": False,
        }
    finally:
        if actor_digest(experiment.policy) != before:
            raise ValueError("Evaluation modified the actor")


def paired_interval(values, seed, samples=10000):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Require a nonempty finite seed-level vector")
    if len(x) < 2:
        return {"mean": float(x.mean()), "ci95": None, "seed_differences": x.tolist(), "n": len(x)}
    indices = np.random.default_rng(seed).integers(len(x), size=(samples, len(x)))
    means = x[indices].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "seed_differences": x.tolist(),
        "n": len(x),
        "bootstrap_draws": samples,
        "rng": "numpy.PCG64",
        "bootstrap_seed": seed,
        "unit": "training_seed",
        "interval_scope": "descriptive pointwise; not multiplicity adjusted",
    }


def contrasts(outcomes, seeds, bootstrap_seed):
    """One value per policy/arm; shared seed order fixes joint bootstrap blocks."""
    from commstudy.experiments.mechanism_suite import METHODS, CONDITIONS

    lookup = {(r["task"], r["condition"], r["method"], r["seed"]): r for r in outcomes}
    effects, reliance, interactions = [], [], []
    for task, conditions in CONDITIONS.items():
        sign = 1 if task == "pcp" else -1
        for condition in conditions:
            for method in METHODS[2:]:
                for baseline in ("identity", "local_capacity", "broadcast"):
                    if method == baseline:
                        continue
                    differences = []
                    used = []
                    for seed in seeds:
                        a = lookup.get((task, condition, method, seed))
                        b = lookup.get((task, condition, baseline, seed))
                        if a and b and a["valid"] and b["valid"]:
                            differences.append(sign * (a["live"] - b["live"]))
                            used.append(seed)
                    effects.append(
                        {
                            "task": task,
                            "condition": condition,
                            "method": method,
                            "baseline": baseline,
                            "primary": baseline == "identity",
                            "seeds": used,
                            "complete": used == seeds,
                            "planned_n": len(seeds),
                            "estimate": paired_interval(differences, bootstrap_seed)
                            if differences
                            else None,
                        }
                    )
            for method in METHODS:
                rows = [lookup.get((task, condition, method, seed)) for seed in seeds]
                valid = [r for r in rows if r and r["valid"]]
                reliance.append(
                    {
                        "task": task,
                        "condition": condition,
                        "method": method,
                        "complete": len(valid) == len(seeds),
                        "seeds": [r["seed"] for r in valid],
                        "estimate": paired_interval(
                            [sign * (r["live"] - r["severed"]) for r in valid], bootstrap_seed
                        )
                        if valid
                        else None,
                    }
                )
    for method in METHODS[2:]:
        values, used = [], []
        for seed in seeds:
            rows = [
                lookup.get(("pcp", c, m, seed))
                for c in CONDITIONS["pcp"]
                for m in (method, "identity")
            ]
            if all(r and r["valid"] for r in rows):
                a, b, c, d = rows
                values.append((a["live"] - b["live"]) - (c["live"] - d["live"]))
                used.append(seed)
        interactions.append(
            {
                "method": method,
                "seeds": used,
                "complete": used == seeds,
                "estimate": paired_interval(values, bootstrap_seed) if values else None,
            }
        )
    supplement_effects = []
    comparisons = {
        "local_capacity64": ("identity", "local_capacity", "attention", "graph_full"),
        "graph_electrical": ("graph_full", "graph_random"),
        "graph_random": ("graph_full",),
    }
    for task, conditions in CONDITIONS.items():
        sign = 1 if task == "pcp" else -1
        for condition in conditions:
            for method, baselines in comparisons.items():
                if not any((task, condition, method, seed) in lookup for seed in seeds):
                    continue
                for baseline in baselines:
                    values, used = [], []
                    for seed in seeds:
                        a = lookup.get((task, condition, method, seed))
                        b = lookup.get((task, condition, baseline, seed))
                        if a and b and a["valid"] and b["valid"]:
                            values.append(sign * (a["live"] - b["live"]))
                            used.append(seed)
                    supplement_effects.append(
                        {
                            "task": task,
                            "condition": condition,
                            "method": method,
                            "baseline": baseline,
                            "primary": False,
                            "seeds": used,
                            "complete": used == seeds,
                            "planned_n": len(seeds),
                            "estimate": paired_interval(values, bootstrap_seed) if values else None,
                        }
                    )
    return {
        "effects": effects,
        "reliance": reliance,
        "visibility_interactions": interactions,
        "supplement_effects": supplement_effects,
    }


def actor_digest(policy):
    """Tensor bytes plus TorchRL grouped-MLP metadata; never drop a policy group."""
    import hashlib

    h = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        h.update(key.encode())
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError("Actor contains nonfinite state")
            h.update(tensor_sha256(value).encode())
        elif isinstance(value, torch.Size) or value is None:
            h.update(str(value).encode())
        else:
            raise ValueError("Unsupported actor state metadata")
    return h.hexdigest()


def verify_suite_checkpoint(experiment, path, expected_frames):
    """Strict native actor parity and all-group actor/critic restoration for PCP too."""
    from commstudy.experiments.mechanism_suite import file_sha
    from run_mapdn_confirmation import _validate_native_state

    before = actor_digest(experiment.policy)
    checksum = file_sha(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _validate_native_state(saved, experiment.losses, expected_frames)
    for group, policy in experiment.group_policies.items():
        native = {
            k.removeprefix("actor_network_params."): v
            for k, v in saved[f"loss_{group}"].items()
            if k.startswith("actor_network_params.") and isinstance(v, torch.Tensor)
        }
        current = {k: v for k, v in policy.state_dict().items() if isinstance(v, torch.Tensor)}
        if native.keys() != current.keys() or any(
            not torch.equal(v.cpu(), current[k].detach().cpu()) for k, v in native.items()
        ):
            raise ValueError("Native actor differs from managed final actor")
    for group, loss in experiment.losses.items():
        loss.load_state_dict(saved[f"loss_{group}"], strict=True)
    _validate_native_state(saved, experiment.losses, expected_frames, require_equal=True)
    if actor_digest(experiment.policy) != before or file_sha(path) != checksum:
        raise ValueError("Native reconstruction modified actor or checkpoint bytes")
    return {
        "path": str(path),
        "sha256": checksum,
        "frames": expected_frames,
        "actor_sha256": before,
        "all_groups": sorted(experiment.losses),
        "native_actor_matches_rollout": True,
        "actor_critic_restored": True,
    }
