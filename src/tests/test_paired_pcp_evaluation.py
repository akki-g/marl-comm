"""Scientific regression checks for paired PCP and same-input interventions."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from commstudy.analysis.saliency import (
    EpisodeOutcome,
    communication_saliency,
    paired_episode_reliance,
    paired_return_deltas,
    same_input_influence,
)
from commstudy.communication.base import CommContext
from commstudy.communication.broadcast import BroadcastComm
from commstudy.experiments import build_experiment, load_experiment_spec


@pytest.fixture(params=[5, 128], ids=["five_existing_worlds", "128_existing_worlds"])
def pcp_experiment(request, tmp_path, config_root):
    spec = load_experiment_spec(
        config_root,
        [
            "task=vmas_predator_capture_prey",
            "model=pcp_comm_identity",
            "critic_model=pcp_critic",
        ],
    )
    spec = dataclasses.replace(
        spec,
        task_config={
            **spec.task_config,
            "params": {
                **spec.task_config["params"],
                "max_steps": 6,
                # Keep prey motion visible so trajectory equality tests the
                # exogenous disturbance stream even before any rewarded contact.
                "predator_sensing_radius": None,
            },
        },
        experiment={
            **spec.experiment,
            "save_folder": str(tmp_path / "experiment"),
            "evaluation": False,
            "evaluation_episodes": request.param,
            "on_policy_n_envs_per_worker": 1,
            "loggers": [],
            "create_json": False,
        },
    )
    experiment = build_experiment(spec)
    try:
        yield experiment
    finally:
        experiment.close()


@pytest.mark.parametrize("exploration", ["DETERMINISTIC", "RANDOM"])
def test_pcp_identity_is_an_exact_null_on_three_complete_paired_episodes(
    pcp_experiment, exploration
):
    """Requested episodes must not inherit the existing test environment width."""
    original_batch_size = pcp_experiment.test_env.batch_size
    result = communication_saliency(
        pcp_experiment,
        episodes=3,
        exploration=exploration,
        seed=47,
        return_groups=("adversary",),
    )

    assert result.episodes == 3
    assert len(result.episode_ids) == len(set(result.episode_ids)) == 3
    assert result.complete_episodes
    assert result.steps == 6
    assert result.group == "adversary"
    assert result.communicating_modules == 1
    assert result.return_delta == 0.0
    assert result.action_shift_mean == result.action_shift_max == 0.0
    assert result.policy_kl_mean == 0.0
    assert result.per_episode_returns_with == result.per_episode_returns_without
    assert result.input_count == 18
    assert result.input_distribution == "live_policy_trajectories"
    assert pcp_experiment.test_env.batch_size == original_batch_size
    assert len(result.outcomes_with) == len(result.outcomes_without) == 3
    assert len({item.initial_input_sha256 for item in result.outcomes_with}) == 3
    assert len({item.exogenous_episode_ids for item in result.outcomes_with}) == 3
    for live, severed in zip(result.outcomes_with, result.outcomes_without, strict=True):
        assert live.episode_id == severed.episode_id
        assert live.seed == severed.seed
        assert live.initial_input_sha256 == severed.initial_input_sha256
        assert len(live.exogenous_episode_ids) == 1
        assert live.exogenous_episode_ids == severed.exogenous_episode_ids
        assert live.exogenous_steps == severed.exogenous_steps == (0,)
        assert live.complete and severed.complete
        assert live.transitions == severed.transitions == 6
        assert live.terminated or live.truncated
        assert severed.terminated or severed.truncated

    # Zero reward alone would be a weak control on a short, untrained policy.
    # The entire visible trajectory, including disturbed prey motion, must
    # agree under a null intervention in both exploration modes.
    paired = paired_episode_reliance(
        pcp_experiment, group="adversary", episodes=3, exploration=exploration, seed=47
    )
    assert paired.with_comm == result.outcomes_with
    assert paired.without_comm == result.outcomes_without
    for live_inputs, severed_inputs in zip(
        paired.live_inputs, paired.severed_inputs, strict=True
    ):
        live_keys = set(live_inputs.keys(include_nested=True, leaves_only=True))
        assert live_keys == set(severed_inputs.keys(include_nested=True, leaves_only=True))
        for key in live_keys:
            assert torch.equal(live_inputs.get(key), severed_inputs.get(key))


def test_pcp_prefix_cannot_be_reported_as_a_complete_episode(pcp_experiment):
    with pytest.raises(ValueError, match="did not finish"):
        communication_saliency(
            pcp_experiment, episodes=1, steps=2, return_groups=("adversary",)
        )


def test_pcp_intervention_requires_an_explicit_measured_group(pcp_experiment):
    with pytest.raises(ValueError, match="explicit measured group"):
        communication_saliency(pcp_experiment, episodes=1)


class _ObservedPredatorPolicy(torch.nn.Module):
    """Record and mutate inputs to expose accidental sharing between both arms."""

    in_keys = [("adversary", "observation")]
    comm_context_keys = {
        "mask": ("adversary", "comm_mask"),
        "sender_mask": ("adversary", "sender_mask"),
    }

    def __init__(self):
        super().__init__()
        self.comm = BroadcastComm(hidden_dim=2, message_dim=2)
        with torch.no_grad():
            self.comm.message_encoder.weight.copy_(torch.eye(2))
            self.comm.message_decoder.weight.copy_(torch.eye(2))
        self.received = []
        self.input_references = []

    def forward(self, batch):
        self.received.append(batch.clone())
        self.input_references.append(batch)
        observation = batch.get(("adversary", "observation"))
        location = self.comm(
            observation,
            context=CommContext(
                mask=batch.get(self.comm_context_keys["mask"]),
                extras={"sender_mask": batch.get(self.comm_context_keys["sender_mask"])},
            ),
        )
        batch.set(("adversary", "action"), location.tanh())
        batch.set(("adversary", "loc"), location)
        batch.set(("adversary", "scale"), torch.ones_like(location))
        observation.add_(100)
        return batch


class _ForbiddenPreyPolicy(torch.nn.Module):
    def forward(self, batch):
        raise AssertionError("A predator influence measurement evaluated the prey policy.")


def test_same_input_influence_uses_identical_clones_and_only_the_selected_group():
    predator = _ObservedPredatorPolicy()
    experiment = SimpleNamespace(
        group_map={"adversary": ["p0", "p1", "p2"], "agent": ["prey"]},
        group_policies={"adversary": predator, "agent": _ForbiddenPreyPolicy()},
        model_config=SimpleNamespace(is_rnn=False),
    )
    observations = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10
    inputs = TensorDict(
        {
            ("adversary", "observation"): observations,
            ("adversary", "comm_mask"): ~torch.eye(3, dtype=torch.bool).expand(2, 3, 3),
            ("adversary", "sender_mask"): torch.ones(2, 3, dtype=torch.bool),
            ("adversary", "action"): torch.full((2, 3, 2), 1e20),
            ("agent", "action"): torch.full((2, 1, 2), float("nan")),
            ("agent", "loc"): torch.full((2, 1, 2), float("nan")),
            ("agent", "scale"): torch.zeros(2, 1, 2),
        },
        batch_size=[2],
    )
    original_observations = observations.clone()
    original_keys = set(inputs.keys(include_nested=True, leaves_only=True))

    result = same_input_influence(
        experiment, [inputs], group="adversary", input_distribution="held_out_state_bank"
    )

    assert result.group == "adversary"
    assert result.input_count == 2
    assert result.input_distribution == "held_out_state_bank"
    assert result.action_shift_mean > 0
    assert result.policy_kl_mean > 0
    assert len(result.input_sha256) == 64
    assert torch.equal(observations, original_observations)
    assert set(inputs.keys(include_nested=True, leaves_only=True)) == original_keys
    assert len(predator.received) == 2
    expected_keys = set(predator.in_keys) | set(predator.comm_context_keys.values())
    left, right = predator.received
    assert set(left.keys(include_nested=True, leaves_only=True)) == expected_keys
    assert set(right.keys(include_nested=True, leaves_only=True)) == expected_keys
    for key in expected_keys:
        assert torch.equal(left.get(key), right.get(key))
    first_observations, second_observations = (
        batch.get(("adversary", "observation")) for batch in predator.input_references
    )
    assert first_observations.data_ptr() != second_observations.data_ptr()
    assert first_observations.data_ptr() != observations.data_ptr()


def _outcome(episode_id, *, seed=0, initial_hash="same_initial_state"):
    return EpisodeOutcome(episode_id, seed, 1.0, 6, False, True, True, initial_hash)


@pytest.mark.parametrize(
    "without_ids",
    [("wrong", "b"), ("a",), ("b", "a"), ("a", "a")],
    ids=["mismatched", "missing", "reordered", "duplicated"],
)
def test_paired_returns_reject_invalid_episode_identifiers(without_ids):
    with pytest.raises(ValueError, match="episode IDs"):
        paired_return_deltas(
            [_outcome("a"), _outcome("b")],
            [_outcome(episode_id) for episode_id in without_ids],
        )


@pytest.mark.parametrize(
    "changed", [{"seed": 1}, {"initial_hash": "different_initial_state"}]
)
def test_paired_returns_reject_matching_ids_with_different_initial_conditions(changed):
    with pytest.raises(ValueError, match="Initial condition mismatch"):
        paired_return_deltas([_outcome("a")], [_outcome("a", **changed)])
