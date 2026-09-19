"""MAPDN evaluation instrument invariants, using tiny synthetic feeder physics.

These are instrument tests, not evidence of a learned communication benefit.
"""

from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from commstudy.analysis import mapdn_paired as paired
from commstudy.communication.broadcast import BroadcastComm
from commstudy.communication.identity import IdentityComm
from commstudy.communication.local_control import LocalCapacityComm
from commstudy.experiments.metrics import CommunicationHealthObserver
from commstudy.tasks.torchrl.mapdn_data import (
    TrainingNormalizer, build_split_manifest, load_split_manifest, save_split_manifest,
)
from commstudy.tasks.torchrl.power_grids import _make_adapter
from commstudy.utils.rng import preserve_rng_state
from test_mapdn_integration import mapdn_config  # noqa: F401


@pytest.fixture(scope="module")
def config(mapdn_config, tmp_path_factory):  # noqa: F811
    result = deepcopy(mapdn_config)
    folder = tmp_path_factory.mktemp("paired_mapdn")
    result["manifest_path"] = str(folder / "manifest.json")
    settings = {key: value for key, value in result.items()
                if key not in {"data_path", "manifest_path", "normalization_path"}}
    manifest = build_split_manifest(result["data_path"], result["max_steps"], result["history"],
                                    settings=settings)
    save_split_manifest(manifest, result["manifest_path"])
    with preserve_rng_state():
        env = _make_adapter(result, seed=91, split="train")
        try:
            observations, _ = env.reset(seed=91, options={
                "start_row": manifest["splits"]["train"]["starts"][0]})
            normalizer = TrainingNormalizer.fit(
                np.stack(list(observations.values())), split="train",
                manifest_sha256=manifest["sha256"])
        finally:
            env.close()
    result["normalization_path"] = str(folder / "normalizer.json")
    normalizer.save(result["normalization_path"])
    return result


def _bank(config, split="validation", count=1):
    starts = load_split_manifest(config["manifest_path"])["splits"][split]["starts"]
    return paired.build_mapdn_bank(config, [
        {"episode_id": f"synthetic:{i}", "start_row": starts[i * 10], "seed": 900 + i}
        for i in range(count)], split=split, purpose="synthetic instrument test")


class InstrumentActor(torch.nn.Module):
    def __init__(self, kind="identity", stochastic=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.05))
        self.comm = None if kind == "mlp" else IdentityComm(2) if kind == "identity" else (
            LocalCapacityComm(2, message_dim=2) if kind == "local_capacity" else
            BroadcastComm(2, message_dim=2, channel={"type": "dropout", "p": 0.25}
                          if stochastic else None))
        if kind == "broadcast":
            with torch.no_grad():
                self.comm.message_encoder.weight.fill_(0.1)
                self.comm.message_decoder.weight.fill_(0.1)

    def forward(self, td):
        obs = td["agents", "observation"]
        h = torch.stack([obs.mean(-1), torch.ones_like(obs[:, 0])], dim=-1)
        if self.comm is not None:
            h = self.comm(h)
        td["agents", "action"] = torch.tanh(h[:, 1:2]) * self.scale
        return td


def _experiment(config, kind="identity", stochastic=False):
    policy = InstrumentActor(kind, stochastic)
    count = load_split_manifest(config["manifest_path"])["data"]["files"][
        "pv_active.csv"]["n_columns"]
    return SimpleNamespace(policy=policy, group_policies={"agents": policy},
                           group_map={"agents": [f"agent_{i}" for i in range(count)]},
                           task=SimpleNamespace(config=config))


def test_bank_binds_split_complete_window_source_runtime_and_train_normalizer(config):
    bank = _bank(config, count=2)
    paired.validate_mapdn_bank(bank, config)
    contract = bank["contract"]
    assert contract["source_sha256"] and contract["runtime"]["packages"]["pandapower"]
    assert contract["normalizer_sha256"] and len(contract["data"]["files"]) == 4
    for episode in bank["episodes"]:
        assert episode["split"] == "validation"
        assert episode["footprint"] == [episode["start_row"] - config["history"] + 1,
                                        episode["start_row"] + config["max_steps"] + 1]
        assert episode["contract_sha256"] == bank["contract_sha256"]
        assert episode["identity_sha256"]


@pytest.mark.parametrize("change", ["train", "empty", "duplicate_id", "duplicate_seed",
                                    "duplicate_row", "overlap", "out_of_split", "bool_seed"])
def test_invalid_episode_banks_fail_before_environment_construction(config, monkeypatch, change):
    def forbidden(*args, **kwargs):
        pytest.fail("A rejected bank must not construct a simulator")
    monkeypatch.setattr(paired, "_make_adapter", forbidden)
    starts = load_split_manifest(config["manifest_path"])["splits"]["validation"]["starts"]
    entries = [{"episode_id": "a", "seed": 5, "start_row": starts[0]},
               {"episode_id": "b", "seed": 6, "start_row": starts[10]}]
    split = "validation"
    if change == "train":
        split = "train"
    elif change == "empty":
        entries = []
    elif change.startswith("duplicate"):
        key = {"duplicate_id": "episode_id", "duplicate_seed": "seed",
               "duplicate_row": "start_row"}[change]
        entries[1][key] = entries[0][key]
    elif change == "overlap":
        entries[1]["start_row"] = starts[1]
    elif change == "out_of_split":
        entries[0]["start_row"] = 0
    else:
        entries[0]["seed"] = True
    with pytest.raises(ValueError):
        paired.build_mapdn_bank(config, entries, split=split, purpose="instrument rejection")


@pytest.mark.parametrize("change", ["seed", "source", "normalizer", "split", "runtime"])
def test_rehashed_tampering_does_not_bypass_bank_contract(config, change):
    bank = _bank(config)
    if change == "seed":
        bank["episodes"][0]["seed"] += 1
    elif change == "source":
        bank["contract"]["source_sha256"] = "0" * 64
    elif change == "normalizer":
        bank["contract"]["normalizer_sha256"] = "0" * 64
    elif change == "split":
        bank["split"] = "test"
    else:
        bank["contract"]["runtime"]["packages"]["torch"] = "wrong"
    bank["sha256"] = paired.canonical_sha({k: v for k, v in bank.items() if k != "sha256"})
    with pytest.raises(ValueError):
        paired.validate_mapdn_bank(bank, config)


@pytest.mark.parametrize("kind", ["identity", "mlp", "local_capacity"])
def test_real_synthetic_local_controls_exactly_null_and_repeatable(config, kind):
    experiment = _experiment(config, kind)
    bank = _bank(config)
    first = paired.evaluate_mapdn_paired(experiment, bank, require_exact_local_null=True)
    second = paired.evaluate_mapdn_paired(experiment, bank, require_exact_local_null=True)
    assert first == second
    assert first["exact_action_and_outcome_null"] is True
    assert first["summaries"]["live"]["transition_count"] == config["max_steps"]
    assert first["summaries"]["severed"]["solver_failure_count"] == 0


def test_real_synthetic_broadcast_intervention_changes_actions_with_paired_exogenous(config):
    result = paired.evaluate_mapdn_paired(_experiment(config, "broadcast"), _bank(config),
                                         arms=("live", "severed", "suppress_sender:0"))
    assert result["pairing_checks_passed"]
    assert result["exact_action_and_outcome_null"] is False
    assert result["paired"]["severed"][0]["shared_exogenous_steps"] == config["max_steps"]
    live = result["arms"]["live"][0]["communication_stats_by_module"][0]
    severed = result["arms"]["severed"][0]["communication_stats_by_module"][0]
    suppressed = result["arms"]["suppress_sender:0"][0]["communication_stats_by_module"][0]
    assert live["realized_bits_per_step"] > suppressed["realized_bits_per_step"] > 0
    assert severed["realized_bits_per_step"] == 0


def test_reused_adapter_matches_fresh_episode_physics_history_and_rng(config, monkeypatch):
    experiment = _experiment(config, "broadcast", stochastic=True)
    bank = _bank(config, count=2)
    original = paired._make_adapter
    creations = []

    def counted(*args, **kwargs):
        env = original(*args, **kwargs)
        creations.append(env)
        return env

    monkeypatch.setattr(paired, "_make_adapter", counted)
    reused = paired.evaluate_mapdn_paired(experiment, bank)
    assert len(creations) == 2  # One owned adapter for each complete arm.
    fresh = {arm: [paired._roll_episode(experiment.policy, config, entry, arm)
                   for entry in bank["episodes"]] for arm in ("live", "severed")}
    assert reused["arms"] == fresh


def test_calibration_can_use_live_only_without_claiming_a_null(config):
    result = paired.evaluate_mapdn_paired(_experiment(config), _bank(config), arms=("live",))
    assert result["paired"] == {}
    assert result["exact_action_and_outcome_null"] is None
    with pytest.raises(ValueError, match="requires a severed"):
        paired.evaluate_mapdn_paired(_experiment(config), _bank(config), arms=("live",),
                                     require_exact_local_null=True)


def test_managed_health_observer_keeps_evaluation_out_of_training_statistics(config):
    experiment = _experiment(config, "broadcast")
    module = experiment.policy.comm
    observer = CommunicationHealthObserver(module, lambda: False)
    experiment.callbacks = [SimpleNamespace(_comm_observers={id(module): observer})]
    try:
        with torch.no_grad():
            module(torch.ones(3, 6, 2))
        state = {name: deepcopy(getattr(observer, name))
                 for name in ("sums", "counts", "transitions", "agent_transitions")}
        stats = module.communication_stats()
        weight = module._stats_transition_weight
        result = paired.evaluate_mapdn_paired(experiment, _bank(config))
        assert result["pairing_checks_passed"]
        assert all(getattr(observer, name) == value for name, value in state.items())
        assert module.communication_stats() == stats
        assert module._stats_transition_weight == weight == 3
    finally:
        for handle in observer.handles:
            handle.remove()


def test_declared_actor_agent_order_is_checked_before_rollout(config):
    experiment = _experiment(config)
    experiment.group_map["agents"] = list(reversed(experiment.group_map["agents"]))
    with pytest.raises(ValueError, match="inverter order"):
        paired.evaluate_mapdn_paired(experiment, _bank(config))


def test_rng_modes_weights_stats_and_private_channel_rng_are_restored(config):
    experiment = _experiment(config, "broadcast", stochastic=True)
    policy = experiment.policy
    channel = policy.comm.channel
    policy.comm._record_stats(example=3.0)
    channel(torch.ones(6, 2))
    before_private = {k: v.get_state().clone() for k, v in channel._rng.items()}
    before_stats = policy.comm.communication_stats()
    before_actor = paired.actor_sha(policy)
    state = torch.random.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate()
    result = paired.evaluate_mapdn_paired(experiment, _bank(config))
    assert result["pairing_checks_passed"]
    assert torch.equal(state[0], torch.random.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(state[1], np.random.get_state(), strict=True))
    assert state[2] == random.getstate()
    assert policy.training and policy.comm.training
    assert paired.actor_sha(policy) == before_actor
    assert policy.comm.communication_stats() == before_stats
    assert set(before_private) == set(channel._rng)
    assert all(torch.equal(value, channel._rng[k].get_state())
               for k, value in before_private.items())
    assert channel._intervention_sender_mask is None


def test_solver_failures_remain_in_all_attempts_denominator(config, monkeypatch):
    original = paired._make_adapter

    def broken_after_reset(*args, **kwargs):
        env = original(*args, **kwargs)
        step = env.step

        def failed(actions):
            env._env._solve = lambda: False
            return step(actions)
        env.step = failed
        return env

    monkeypatch.setattr(paired, "_make_adapter", broken_after_reset)
    result = paired.evaluate_mapdn_paired(_experiment(config), _bank(config),
                                         require_exact_local_null=True)
    for summary in result["summaries"].values():
        assert summary["transition_count"] == summary["solver_failure_count"] == 1
        assert summary["domain_denominators"]["destroy"] == 1
        assert summary["domain_denominators"]["voltage_violation_magnitude"] == 0
        assert summary["domain_means"]["voltage_violation_magnitude"] is None
        assert summary["unattempted_transition_count"] == config["max_steps"] - 1
        assert summary["solver_failure_fraction_all_attempts"] == 1


def test_reset_failure_is_not_retried_and_restores_actor_context(config, monkeypatch):
    experiment = _experiment(config, "broadcast")
    original = paired._make_adapter
    reset_calls = []

    def broken(*args, **kwargs):
        env = original(*args, **kwargs)

        def failed(*args, **kwargs):
            reset_calls.append(1)
            raise RuntimeError("declared reset failure")
        env.reset = failed
        return env

    monkeypatch.setattr(paired, "_make_adapter", broken)
    before = paired.actor_sha(experiment.policy)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="declared reset failure"):
        paired.evaluate_mapdn_paired(experiment, _bank(config))
    assert reset_calls == [1]
    assert paired.actor_sha(experiment.policy) == before
    assert torch.equal(rng, torch.random.get_rng_state())
    assert experiment.policy.training


def test_exogenous_pairing_checks_private_rng_even_when_rows_match(config):
    result = paired.evaluate_mapdn_paired(_experiment(config), _bank(config))
    left, right = result["arms"]["live"][0], result["arms"]["severed"][0]
    right["transitions"][0]["exogenous"]["private_rng_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exogenous"):
        paired._pair(left, right)


def test_native_parity_rejects_wrong_frames_or_actor_without_selecting_it(tmp_path, config):
    experiment = _experiment(config)
    loss = torch.nn.Module()
    # Match actual native MAPPO: functional parameters retain __batch_size and
    # __device metadata which are absent from the plain rollout actor state.
    loss.actor_network_params = TensorDict.from_module(experiment.policy, as_module=True)
    loss.critic_network_params = TensorDict.from_module(torch.nn.Linear(2, 1), as_module=True)
    experiment.losses = {"agents": loss}
    path = Path(tmp_path) / "native.pt"
    saved = {"state": {"total_frames": 32}, "loss_agents": loss.state_dict()}
    assert "actor_network_params.__batch_size" in saved["loss_agents"]
    assert "actor_network_params.__device" in saved["loss_agents"]
    torch.save(saved, path)
    before = paired.actor_sha(experiment.policy)
    proof = paired.verify_native_checkpoint(experiment, path, expected_frames=32)
    assert proof["native_actor_matches_rollout"] and proof["actor_critic_restored"]
    with pytest.raises(ValueError, match="frame count"):
        paired.verify_native_checkpoint(experiment, path, expected_frames=64)
    saved["loss_agents"]["actor_network_params.scale"] = torch.tensor(9.0)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="selected rollout actor"):
        paired.verify_native_checkpoint(experiment, path, expected_frames=32)
    assert paired.actor_sha(experiment.policy) == before
