"""Independent real-PCP checks for the held-out budget evaluator."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from commstudy.analysis.pcp_budget import (
    INTERVENTIONS,
    _episode,
    evaluate_budget_policy,
    paired_episode_differences,
    tensor_sha256,
)
from commstudy.analysis.saliency import communication_modules
from commstudy.communication import BudgetedBroadcastComm, IdentityComm, LocalCapacityComm
from commstudy.experiments import build_experiment, load_experiment_spec


def _fake_experiment(module):
    return SimpleNamespace(
        group_policies={"adversary": module},
        model_config=SimpleNamespace(is_rnn=False),
        max_steps=100,
        analysis_provenance={"purpose": "synthetic negative-test fixture"},
    )


@pytest.mark.parametrize("arm", ["shuffle", "suppress_0", "suppress_1", "suppress_2"])
def test_zero_control_rejects_noop_budget_interventions_before_simulation(arm):
    experiment = _fake_experiment(LocalCapacityComm(8, message_dim=4))
    with pytest.raises(ValueError, match="Interventions must be"):
        evaluate_budget_policy(experiment, episode_seeds=[1, 2], interventions=("live", arm))


@pytest.mark.parametrize("changed_field", ["metrics", "action_health"])
def test_zero_control_rejects_nonnull_outcomes(monkeypatch, changed_field):
    experiment = _fake_experiment(LocalCapacityComm(8, message_dim=4))

    def episode(_experiment, *, seed, intervention, **kwargs):
        row = {
            "episode_id": str(seed), "seed": seed,
            "initial_physical_state_sha256": str(seed), "exogenous_initial": {},
            "metrics": {"complete": 1, "episode_steps": 100, "return": 0},
            "action_health": {"saturation_fraction": 0},
        }
        if intervention == "severed":
            row[changed_field]["unexpected_difference"] = 1
        return row, None

    monkeypatch.setattr("commstudy.analysis.pcp_budget._episode", episode)
    with pytest.raises(ValueError, match="exact null intervention"):
        evaluate_budget_policy(experiment, episode_seeds=[1, 2])


def test_evaluation_exception_restores_channel_hooks_masks_rng_and_stats(monkeypatch):
    module = BudgetedBroadcastComm(8, sender_budget=3, message_dim=4)
    experiment = _fake_experiment(module)
    module._record_stats(existing_probe=9.5)
    original_stats = copy.deepcopy(module.communication_stats())
    original_hooks = dict(module.channel._forward_hooks)
    original_rng = torch.get_rng_state().clone()

    def failed_factory(*args):
        torch.rand(4)
        raise RuntimeError("synthetic environment construction failure")

    monkeypatch.setattr("commstudy.analysis.pcp_budget._fresh_env", failed_factory)
    with pytest.raises(RuntimeError, match="synthetic environment"):
        _episode(experiment, seed=27, intervention="severed", donor=None, deadline=50)
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert module.communication_stats() == original_stats
    assert dict(module.channel._forward_hooks) == original_hooks
    assert module.channel._intervention_sender_mask is None


def _experiment(config_root, tmp_path, module, kwargs, *, radius=1.0):
    spec = load_experiment_spec(config_root, [
        "task=vmas_predator_capture_prey", "model=pcp_comm_identity", "critic_model=pcp_critic",
        "seed=5", "experiment.max_n_frames=6000", "experiment.evaluation=false",
        "experiment.loggers=[]", "experiment.create_json=false",
        "experiment.checkpoint_interval=0", "experiment.checkpoint_at_end=false",
        f"experiment.save_folder={tmp_path / 'benchmarl'}",
    ])
    params = spec.model_config["groups"]["adversary"]["params"]
    params["comm_class_path"] = f"{module.__module__}.{module.__name__}"
    params["comm_kwargs"] = dict(kwargs)
    params["comm_context_keys"] = {"sender_mask": "comm_sender_mask"}
    spec.task_config["params"]["predator_sensing_radius"] = radius
    experiment = build_experiment(spec)
    experiment.analysis_provenance = {
        "purpose": "temporary untrained real-PCP evaluator validation; no study claim",
    }
    return experiment


def _check_episode(row):
    assert row["metrics"]["episode_steps"] == 100
    assert row["metrics"]["complete"] == row["metrics"]["truncated"] == 1
    assert row["metrics"]["terminated"] == 0
    assert row["metrics"]["reward_accounting_max_abs_error"] == 0
    assert row["action_health"]["all_finite"]
    assert row["action_health"]["action_scalars"] == 600
    assert row["action_health"]["min_scale"] > 0


@pytest.mark.parametrize("module,kwargs", [
    (IdentityComm, {}),
    (LocalCapacityComm, {"message_dim": 32}),
    (BudgetedBroadcastComm, {"sender_budget": 0, "message_dim": 32}),
])
def test_real_two_episode_zero_controls_are_exact_null_and_preserve_state(
    config_root, tmp_path, module, kwargs,
):
    experiment = _experiment(config_root, tmp_path, module, kwargs)
    try:
        slot = communication_modules(experiment, "adversary")[0]
        slot._record_stats(existing_probe=3.2)
        old_stats = copy.deepcopy(slot.communication_stats())
        old_weights = {name: p.detach().clone() for name, p in experiment.policy.named_parameters()}
        old_rng = torch.get_rng_state().clone()
        report, bank = evaluate_budget_policy(
            experiment, episode_seeds=[900001, 900002], interventions=("live", "severed"),
        )
        assert bank is None
        assert torch.equal(old_rng, torch.get_rng_state())
        assert slot.communication_stats() == old_stats
        assert all(torch.equal(old_weights[name], p)
                   for name, p in experiment.policy.named_parameters())
        for live, severed in zip(
            report["interventions"]["live"], report["interventions"]["severed"], strict=True,
        ):
            _check_episode(live)
            _check_episode(severed)
            assert live == severed
            assert live["costs"]["sender_payload_bits"] == 0
            assert live["costs"]["edge_delivery_payload_bits"] == 0
            assert live["costs"]["sender_emissions"] == 0
        assert paired_episode_differences(
            report["interventions"]["live"], report["interventions"]["severed"], "return",
        ) == [0.0, 0.0]
    finally:
        experiment.close()


def test_real_two_episode_broadcast_budget_interventions_donors_costs_and_preservation(
    config_root, tmp_path,
):
    experiment = _experiment(
        config_root, tmp_path, BudgetedBroadcastComm, {"sender_budget": 3, "message_dim": 32},
    )
    try:
        slot = communication_modules(experiment, "adversary")[0]
        slot._record_stats(existing_probe=3.2)
        old_stats = copy.deepcopy(slot.communication_stats())
        old_weights = {name: p.detach().clone() for name, p in experiment.policy.named_parameters()}
        old_rng = torch.get_rng_state().clone()
        channel = slot.channel
        old_hooks = dict(channel._forward_hooks)
        report, bank = evaluate_budget_policy(
            experiment, episode_seeds=[900001, 900002], interventions=INTERVENTIONS,
        )
        assert bank.shape == (2, 100, 1, 3, 32)
        assert bank.dtype == torch.float32
        assert not torch.equal(bank[0], bank[1])
        assert report["live_packet_bank_sha256"] == tensor_sha256(bank)
        assert torch.equal(old_rng, torch.get_rng_state())
        assert slot.communication_stats() == old_stats
        assert all(torch.equal(old_weights[name], p)
                   for name, p in experiment.policy.named_parameters())
        assert dict(channel._forward_hooks) == old_hooks
        assert channel._intervention_sender_mask is None
        for mode, rows in report["interventions"].items():
            assert len(rows) == 2
            for index, row in enumerate(rows):
                _check_episode(row)
                senders = 0 if mode == "severed" else (2 if mode.startswith("suppress") else 3)
                assert row["costs"]["sender_emissions"] == 100 * senders
                assert row["costs"]["edge_deliveries"] == 200 * senders
                assert row["costs"]["sender_payload_bits"] == 100 * senders * 32 * 32
                assert row["costs"]["edge_delivery_payload_bits"] == 200 * senders * 32 * 32
                live_row = report["interventions"]["live"][index]
                assert row["initial_physical_state_sha256"] == live_row[
                    "initial_physical_state_sha256"]
                if mode == "shuffle":
                    donor_index = (index + 1) % 2
                    assert row["donor_episode_seed"] == [900001, 900002][donor_index]
                    assert row["donor_packet_sha256"] == tensor_sha256(bank[donor_index])
        # The substitution changes the complete trajectory, even when an
        # untrained policy receives zero sparse reward in both arms.
        assert any(live["metrics"] != shuffled["metrics"] for live, shuffled in zip(
            report["interventions"]["live"], report["interventions"]["shuffle"], strict=True,
        ))
    finally:
        experiment.close()


def test_real_visibility_pairing_uses_physics_not_changed_actor_observations(config_root, tmp_path):
    reports = []
    for radius in [1.0, None]:
        experiment = _experiment(
            config_root, tmp_path / str(radius), LocalCapacityComm,
            {"message_dim": 32}, radius=radius,
        )
        try:
            report, _ = evaluate_budget_policy(
                experiment, episode_seeds=[900001, 900002], interventions=("live", "severed"),
            )
            reports.append(report)
        finally:
            experiment.close()
    differences = paired_episode_differences(
        reports[0]["interventions"]["live"], reports[1]["interventions"]["live"], "return",
    )
    assert len(differences) == 2
    altered = copy.deepcopy(reports[1]["interventions"]["live"])
    altered[0]["initial_physical_state_sha256"] = "wrong physical episode"
    with pytest.raises(ValueError, match="pairing mismatch"):
        paired_episode_differences(reports[0]["interventions"]["live"], altered, "return")
