"""Analytic domain accounting and read-only instrument checks on real PCP."""

from contextlib import closing

import pytest
import torch

from commstudy.tasks import resolve_task
from commstudy.adapters.critic_state import pcp_critic_state
from commstudy.tasks.pcp_metrics import (
    PCP_DOMAIN_PROTOCOL,
    compute_pcp_domain_metrics,
    pcp_metric_specs,
)
from commstudy.utils.rng import RNGState


def _env(*, worlds=1, **params):
    task = resolve_task("vmas_predator_capture_prey", params)
    return closing(task.get_env_fun(worlds, True, 19, "cpu")())


def _place(scenario, predators, prey, landmarks=()):
    for agent, position in zip(scenario.adversaries(), predators, strict=True):
        agent.set_pos(
            torch.tensor(position, dtype=torch.float32).expand(scenario.world.batch_dim, 2),
            batch_index=None,
        )
        agent.set_vel(torch.zeros(scenario.world.batch_dim, 2), batch_index=None)
    for agent, position in zip(scenario.good_agents(), prey, strict=True):
        agent.set_pos(
            torch.tensor(position, dtype=torch.float32).expand(scenario.world.batch_dim, 2),
            batch_index=None,
        )
        agent.set_vel(torch.zeros(scenario.world.batch_dim, 2), batch_index=None)
    for landmark, position in zip(scenario.world.landmarks, landmarks, strict=True):
        landmark.set_pos(
            torch.tensor(position, dtype=torch.float32).expand(scenario.world.batch_dim, 2),
            batch_index=None,
        )


def test_hand_placed_contact_visibility_collision_and_distance_accounting():
    with _env(num_landmarks=1, predator_sensing_radius=0.2) as env:
        scenario = env.scenario
        _place(scenario, [(0, 0), (0.1, 0), (0.95, 0.95)], [(0.05, 0)], [(0.95, 0.95)])
        values = compute_pcp_domain_metrics(scenario)
        expected = {
            "contact_pairs": 2,
            "any_contact": 1,
            "simultaneous_contact": 1,
            "contacting_predators": 2,
            "predator_obstacle_collision_pairs": 1,
            "predator_teammate_collision_pairs": 1,
            "prey_visible_to_0_predators": 0,
            "prey_visible_to_1_predators": 0,
            "prey_visible_to_2_predators": 1,
            "prey_visible_to_3_predators": 0,
            "sender_visible_receiver_blind_pairs": 2,
            "sender_visible_receiver_blind_prey_triples": 2,
            "prey_boundary_fraction": 0,
        }
        for key, value in expected.items():
            assert values[key].item() == value, key
        assert values["predator_boundary_fraction"].item() == pytest.approx(1 / 3)
        assert values["nearest_predator_prey_distance"].item() == pytest.approx(0.05)
        far_distance = (0.9**2 + 0.95**2) ** 0.5
        assert values["mean_predator_nearest_prey_distance"].item() == pytest.approx(
            (0.05 + 0.05 + far_distance) / 3
        )
        assert scenario.domain_metrics_protocol == PCP_DOMAIN_PROTOCOL
        assert set(values) == set(pcp_metric_specs(3))
        for value in values.values():
            assert value.shape == (1, 1)
            assert value.dtype == torch.float32 and not value.requires_grad
        for predator in scenario.adversaries():
            assert scenario.reward(predator).item() == 10 * values["contact_pairs"].item()
        assert scenario.reward(scenario.good_agents()[0]).item() == -20


@pytest.mark.parametrize("shaping", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_expected_reward_accounts_for_shaping_and_sharing_without_reward_side_effects(
    shaping,
    shared,
):
    with _env(num_landmarks=0, shape_adversary_rew=shaping, adversaries_share_rew=shared) as env:
        scenario = env.scenario
        # Include positive contact rewards and a no-contact negative shaped case.
        for prey_position in ((0.05, 0), (-0.8, -0.8)):
            _place(scenario, [(0, 0), (0.1, 0), (0.9, 0.9)], [prey_position])
            before_state = pcp_critic_state(scenario).clone()
            before_rng = RNGState.capture()
            values = compute_pcp_domain_metrics(scenario)
            assert torch.equal(before_state, pcp_critic_state(scenario))
            assert torch.equal(before_rng.torch_cpu, RNGState.capture().torch_cpu)
            actual = torch.stack(
                [scenario.reward(agent) for agent in scenario.adversaries()], dim=-1
            ).mean(-1)
            torch.testing.assert_close(values["expected_predator_reward"].squeeze(-1), actual)
            if shaping and prey_position == (-0.8, -0.8):
                assert values["expected_predator_reward"].item() < 0
        assert pcp_metric_specs(3)["expected_predator_reward"]["kind"] == "signed_reward"


def test_touching_radii_are_not_contact_and_visibility_includes_radius_boundary():
    with _env(num_landmarks=0, predator_sensing_radius=0.125) as env:
        scenario = env.scenario
        _place(scenario, [(0, 0), (-0.5, 0.5), (-0.5, -0.5)], [(0.125, 0)])
        values = compute_pcp_domain_metrics(scenario)
        assert values["contact_pairs"].item() == 0
        assert values["any_contact"].item() == 0
        assert values["prey_visible_to_1_predators"].item() == 1
        assert values["sender_visible_receiver_blind_pairs"].item() == 2


@pytest.mark.parametrize("visible_count", range(4))
def test_visibility_histogram_covers_each_three_predator_case(visible_count):
    with _env(num_landmarks=0, predator_sensing_radius=0.2) as env:
        scenario = env.scenario
        positions = [(0.1, 0)] * visible_count + [(0.8, 0.8)] * (3 - visible_count)
        _place(scenario, positions, [(0, 0)])
        values = compute_pcp_domain_metrics(scenario)
        for count in range(4):
            assert values[f"prey_visible_to_{count}_predators"].item() == int(
                count == visible_count
            )
        assert values["sender_visible_receiver_blind_pairs"].item() == visible_count * (
            3 - visible_count
        )
        for predator in scenario.adversaries():
            actual_flag = scenario.observation(predator)[0, -1].item()
            expected_flag = float(
                (predator.state.pos - scenario.good_agents()[0].state.pos).norm() <= 0.2
            )
            assert actual_flag == expected_flag


def test_multi_prey_distinguishes_joint_contact_and_pair_vs_triple_opportunities():
    with _env(num_good_agents=2, num_landmarks=0, predator_sensing_radius=0.2) as env:
        scenario = env.scenario
        _place(scenario, [(0.05, 0), (-0.05, 0), (0.8, 0.8)], [(0.05, 0), (-0.05, 0)])
        values = compute_pcp_domain_metrics(scenario)
        assert values["contact_pairs"].item() == 4
        assert values["contacting_predators"].item() == 2
        assert values["simultaneous_contact"].item() == 1
        assert values["prey_visible_to_2_predators"].item() == 2
        assert values["sender_visible_receiver_blind_pairs"].item() == 2
        assert values["sender_visible_receiver_blind_prey_triples"].item() == 4
        _place(scenario, [(-0.5, 0), (0.5, 0), (0.8, 0.8)], [(-0.5, 0), (0.5, 0)])
        values = compute_pcp_domain_metrics(scenario)
        assert values["contact_pairs"].item() == 2
        assert values["contacting_predators"].item() == 2
        # Two predators catching different prey is not simultaneous cooperative contact.
        assert values["simultaneous_contact"].item() == 0


def test_global_visibility_and_boundary_band_are_explicit():
    with _env(num_landmarks=0, predator_sensing_radius=None) as env:
        scenario = env.scenario
        _place(scenario, [(0.9, 0), (0.899, 0), (-1.1, 0)], [(0, -0.9)])
        values = compute_pcp_domain_metrics(scenario)
        assert values["predator_boundary_fraction"].item() == pytest.approx(2 / 3)
        assert values["prey_boundary_fraction"].item() == 1
        assert values["prey_visible_to_3_predators"].item() == 1
        assert values["sender_visible_receiver_blind_pairs"].item() == 0
        assert values["sender_visible_receiver_blind_prey_triples"].item() == 0


def test_info_is_read_only_excluded_from_inputs_and_does_not_advance_rng_or_noise():
    with _env(worlds=2) as env:
        task = resolve_task("vmas_predator_capture_prey", {})
        scenario = env.scenario
        before_state = pcp_critic_state(scenario).clone()
        before_observations = [
            scenario.observation(agent).clone() for agent in scenario.world.agents
        ]
        before_rng = RNGState.capture()
        before_noise = scenario.exogenous_steps.clone()
        expected = scenario.info(scenario.adversaries()[0])
        for _ in range(3):
            for predator in scenario.adversaries():
                info = scenario.info(predator)
                assert all(torch.equal(info[key], value) for key, value in expected.items())
                info["contact_pairs"].fill_(99)
        assert not scenario.info(scenario.good_agents()[0])
        assert torch.equal(before_state, pcp_critic_state(scenario))
        assert all(
            torch.equal(old, scenario.observation(agent))
            for old, agent in zip(before_observations, scenario.world.agents, strict=True)
        )
        assert torch.equal(before_rng.torch_cpu, RNGState.capture().torch_cpu)
        assert torch.equal(before_rng.vmas[0], RNGState.capture().vmas[0])
        assert torch.equal(before_noise, scenario.exogenous_steps)
        assert set(task.state_spec(env).keys()) == {"state"}
        for group in task.group_map(env):
            assert set(task.observation_spec(env)[group].keys()) == {"observation"}
        assert set(task.info_spec(env)["adversary", "info"].keys()) == set(expected)


def test_transition_info_matches_reward_and_measurement_does_not_change_physics(monkeypatch):
    with _env(worlds=2) as measured, _env(worlds=2) as control:
        monkeypatch.setattr(control.scenario, "post_step", lambda: None)
        monkeypatch.setattr(control.scenario, "info", lambda agent: {})
        left, right = measured.reset(), control.reset()
        for _ in range(5):
            left.update(measured.full_action_spec.zero())
            right.update(control.full_action_spec.zero())
            left, right = measured.step(left)["next"], control.step(right)["next"]
            assert torch.equal(left["state"], right["state"])
            assert torch.equal(left["adversary", "observation"], right["adversary", "observation"])
            assert torch.equal(left["adversary", "reward"], right["adversary", "reward"])
            pairs = left["adversary", "info", "contact_pairs"]
            assert torch.equal(left["adversary", "reward"], 10 * pairs)


def test_contact_does_not_terminate_and_time_limit_is_a_truncation():
    with _env(num_landmarks=0, max_steps=2) as env:
        td = env.reset()
        _place(env.scenario, [(0, 0), (0.1, 0), (0.8, 0.8)], [(0.05, 0)])
        assert not env.scenario.done().any()
        td.update(env.full_action_spec.zero())
        first = env.step(td)["next"]
        assert not first["done"].any()
        first.update(env.full_action_spec.zero())
        last = env.step(first)["next"]
        assert last["done"].all()
        assert last["truncated"].all()
        assert not last["terminated"].any()


def test_post_step_snapshot_preserves_contact_accounting_when_reward_respawns_prey():
    with _env(num_landmarks=0, respawn_at_catch=True) as env:
        scenario = env.scenario
        _place(scenario, [(0, 0), (0.1, 0), (0.8, 0.8)], [(0.05, 0)])
        scenario.post_step()
        expected = scenario.info(scenario.adversaries()[0])["contact_pairs"].clone()
        reward = scenario.reward(scenario.adversaries()[0])
        assert expected.item() == 2 and reward.item() == 20
        assert torch.equal(scenario.info(scenario.adversaries()[0])["contact_pairs"], expected)
        # Reset removes cached transition information; a new reset is never
        # credited with contact from the previous episode.
        env.reset()
        assert scenario._transition_domain_metrics is None
